import time
import torch
from torch.nn import functional as F
from typing import Dict, Any, Generator, Tuple
from src.transformer.model import GPT
from src.training.tokenizer import CharTokenizer
from src.inference.kv_cache import KVCache

def sample_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter logits to only keep the top-k highest values."""
    v, ix = torch.topk(logits, min(top_k, logits.size(-1)))
    # Create mask of negative infinities
    out = torch.full_like(logits, float('-inf'))
    # Fill in top-k values at their respective indices
    out.scatter_(1, ix, v)
    return out

@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 1.0,
    top_k: int = 10,
    use_cache: bool = True,
    device: str = "cpu"
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate text autoregressively and return it along with profiling statistics.
    Supports running with or without key-value caching.
    """
    model.eval()
    
    # Initialize prompt tokens
    if not prompt:
        prompt = " "
    x = tokenizer.encode(prompt, return_tensors="pt").unsqueeze(0).to(device)
    prompt_len = x.size(1)
    
    generated_ids = []
    
    # Timing variables
    t_start = time.perf_counter()
    ttft = 0.0 # Time To First Token in seconds
    
    if use_cache:
        # Create a fresh KV cache container
        kv_cache = KVCache()
        
        # --- Pre-fill Phase ---
        # Run forward pass on the entire prompt to populate the KV Cache
        logits, _ = model(x, kv_cache=kv_cache)
        ttft = time.perf_counter() - t_start # TTFT measures the pre-fill latency
        
        # Process the final token logits
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            next_token_logits = sample_top_k(next_token_logits, top_k)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_ids.append(next_token.item())
        
        # --- Autoregressive Decoding Phase ---
        # Generate remaining tokens one-by-one, passing only the last generated token
        curr_token = next_token
        for _ in range(max_new_tokens - 1):
            # Model receives only a sequence of length 1 (curr_token) along with the cache
            logits, _ = model(curr_token, kv_cache=kv_cache)
            
            # Sample next token
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                next_token_logits = sample_top_k(next_token_logits, top_k)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids.append(next_token.item())
            curr_token = next_token
            
    else:
        # --- Standard No-Cache Pathway ---
        # Re-compute attention over the entire sequence at every step
        curr_x = x
        for i in range(max_new_tokens):
            # Crop current sequence if it exceeds the maximum context length
            # Note: GPT config has a maximum block_size limit
            inputs = curr_x[:, -model.config.block_size:]
            
            logits, _ = model(inputs)
            
            # For the first token, record the time-to-first-token
            if i == 0:
                ttft = time.perf_counter() - t_start
                
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                next_token_logits = sample_top_k(next_token_logits, top_k)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids.append(next_token.item())
            curr_x = torch.cat((curr_x, next_token), dim=1)
            
    t_end = time.perf_counter()
    total_time = t_end - t_start
    
    # Calculate throughput (tokens per second)
    # The first token is the pre-fill token (TTFT). Decoding throughput should measure decoding steps.
    num_generated = len(generated_ids)
    decoding_time = total_time - ttft
    tokens_per_sec = (num_generated - 1) / max(decoding_time, 1e-6) if num_generated > 1 else 0.0
    
    # Decode total generated tokens
    decoded_text = tokenizer.decode(generated_ids)
    
    metrics = {
        "use_cache": use_cache,
        "prompt_tokens": prompt_len,
        "generated_tokens": num_generated,
        "total_time_sec": total_time,
        "ttft_ms": ttft * 1000,
        "tokens_per_sec": tokens_per_sec,
        "device": device
    }
    
    return decoded_text, metrics


@torch.no_grad()
def generate_stream(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 1.0,
    top_k: int = 10,
    use_cache: bool = True,
    device: str = "cpu"
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator function that streams each generated token and character in real-time,
    accompanied by performance statistics.
    """
    model.eval()
    
    if not prompt:
        prompt = " "
    x = tokenizer.encode(prompt, return_tensors="pt").unsqueeze(0).to(device)
    prompt_len = x.size(1)
    
    t_start = time.perf_counter()
    tokens_yielded = 0
    
    if use_cache:
        kv_cache = KVCache()
        
        # Pre-fill
        logits, _ = model(x, kv_cache=kv_cache)
        ttft = time.perf_counter() - t_start
        
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            next_token_logits = sample_top_k(next_token_logits, top_k)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        char = tokenizer.decode([next_token.item()])
        tokens_yielded += 1
        
        # Yield first token
        yield {
            "char": char,
            "metrics": {
                "step": tokens_yielded,
                "ttft_ms": ttft * 1000,
                "tokens_per_sec": 0.0,
                "elapsed_sec": time.perf_counter() - t_start
            }
        }
        
        curr_token = next_token
        for i in range(max_new_tokens - 1):
            logits, _ = model(curr_token, kv_cache=kv_cache)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                next_token_logits = sample_top_k(next_token_logits, top_k)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            char = tokenizer.decode([next_token.item()])
            tokens_yielded += 1
            curr_token = next_token
            
            now = time.perf_counter()
            decoding_time = now - t_start - ttft
            tokens_per_sec = (tokens_yielded - 1) / max(decoding_time, 1e-6)
            
            yield {
                "char": char,
                "metrics": {
                    "step": tokens_yielded,
                    "ttft_ms": ttft * 1000,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_sec": now - t_start
                }
            }
    else:
        # Uncached streaming
        curr_x = x
        ttft = 0.0
        for i in range(max_new_tokens):
            inputs = curr_x[:, -model.config.block_size:]
            logits, _ = model(inputs)
            
            if i == 0:
                ttft = time.perf_counter() - t_start
                
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                next_token_logits = sample_top_k(next_token_logits, top_k)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            char = tokenizer.decode([next_token.item()])
            tokens_yielded += 1
            curr_x = torch.cat((curr_x, next_token), dim=1)
            
            now = time.perf_counter()
            decoding_time = now - t_start - ttft
            tokens_per_sec = (tokens_yielded - 1) / max(decoding_time, 1e-6) if tokens_yielded > 1 else 0.0
            
            yield {
                "char": char,
                "metrics": {
                    "step": tokens_yielded,
                    "ttft_ms": ttft * 1000,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_sec": now - t_start
                }
            }
