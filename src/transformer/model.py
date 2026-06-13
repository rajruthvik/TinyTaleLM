import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 256        # Default size, will be updated based on tokenizer
    block_size: int = 256        # Maximum context length
    n_layer: int = 4             # Number of transformer blocks
    n_head: int = 4              # Number of attention heads
    n_embd: int = 128            # Embedding dimension size
    dropout: float = 0.1         # Dropout probability

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer with support for KV Caching.
    """
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.layer_idx = layer_idx
        
        # Key, query, value projections in one linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask buffer (used when KV cache is disabled or during prompt pre-fill)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        B, T, C = x.size() # Batch size, sequence length, embedding dimension
        
        # Project x to queries, keys, and values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (Batch, Head, Seq_len, Head_size)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        
        # --- KV CACHE INTEGRATION ---
        if kv_cache is not None:
            # Retrieve cached past key and value tensors
            k_prev, v_prev = kv_cache.get(self.layer_idx)
            if k_prev is not None:
                # Concatenate past keys/values with current step keys/values
                k = torch.cat([k_prev, k], dim=-2)
                v = torch.cat([v_prev, v], dim=-2)
            # Save the updated key-value history back to the cache
            kv_cache.update(self.layer_idx, k, v)
        
        # Total context length we are attending to
        total_T = k.size(-2)
        
        # Dot-product attention: Q @ K^T / sqrt(head_size)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        
        # Apply causal masking
        # If we have a cache, and we are generating token-by-token (T == 1),
        # the new token can attend to all past tokens. No future masking is required!
        # If T > 1 (e.g. during prompt pre-fill or training), apply the lower-triangular causal mask.
        if T > 1:
            att = att.masked_fill(self.bias[:, :, :T, :total_T] == 0, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted sum: Attn @ V
        y = att @ v # (B, nh, T, hs)
        
        # Re-assemble head outputs side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """
    Simple Feed-Forward Layer (Multi-Layer Perceptron).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A single Transformer Block containing Self-Attention and MLP, pre-norm style.
    """
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, kv_cache=None) -> torch.Tensor:
        # Pre-LN residual connections
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """
    Decoder-Only Transformer (GPT) language model.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie embedding weights and output projection weights (weight tying)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Apply standard initialization
        self.apply(self._init_weights)
        print(f"GPT model initialized. Total Parameters: {self.get_num_params():,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, kv_cache=None):
        device = idx.device
        B, T = idx.size()
        
        # Assert input length is within bounds when not caching
        if kv_cache is None:
            assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, max block size is {self.config.block_size}"
        
        # Token Embeddings
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        
        # Position Embeddings (offsetted if generating with KV Cache)
        if kv_cache is not None:
            # We look up the cache's sequence length from layer 0
            past_length = kv_cache.get_seq_len(0)
            assert past_length + T <= self.config.block_size, f"Total generation length {past_length + T} exceeds max block size {self.config.block_size}"
            pos = torch.arange(past_length, past_length + T, dtype=torch.long, device=device)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        
        # Combine embeddings
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Process through transformer blocks
        for block in self.transformer.h:
            x = block(x, kv_cache=kv_cache)
            
        x = self.transformer.ln_f(x)
        
        # Logits projection
        if targets is not None:
            # Calculate loss (cross-entropy)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: only compute logits for the last token to save computation
            logits = self.lm_head(x[:, [-1], :]) # (B, 1, vocab_size)
            loss = None
            
        return logits, loss
