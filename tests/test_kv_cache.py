import os
import sys
import torch

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer.model import GPT, GPTConfig
from src.training.tokenizer import CharTokenizer
from src.inference.generator import generate

def test_equivalence():
    print("=" * 65)
    print("          KV CACHE MATHEMATICAL EQUIVALENCE TEST          ")
    print("=" * 65)
    
    # 1. Initialize config and model
    tokenizer = CharTokenizer()
    dummy_text = "Once upon a time, there was a little boy named Tim. He walked into the dark forest."
    tokenizer.build_from_text(dummy_text)
    
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0
    )
    
    # Seed torch for reproducibility
    torch.manual_seed(42)
    model = GPT(config)
    model.eval()
    
    prompt = "Once upon a time,"
    max_new_tokens = 15
    
    # 2. Run standard generation
    torch.manual_seed(123)
    standard_text, standard_metrics = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=5,
        use_cache=False,
        device="cpu"
    )
    
    # 3. Run KV Cache generation
    torch.manual_seed(123)
    cached_text, cached_metrics = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=5,
        use_cache=True,
        device="cpu"
    )
    
    # 4. Compare outputs
    print(f"Prompt: '{prompt}'")
    print(f"Standard generation output: '{prompt}{standard_text}'")
    print(f"KV-Cached generation output: '{prompt}{cached_text}'")
    
    assert standard_text == cached_text, f"EQUIVALENCE FAILED!\nStandard: {standard_text}\nCached: {cached_text}"
    print("\n[SUCCESS] Generated sequences are character-by-character identical.")
    
    # 5. Check logits equivalence directly step-by-step
    print("\nVerifying direct step-by-step logits numerical equivalence...")
    x = tokenizer.encode(prompt, return_tensors="pt").unsqueeze(0)
    
    # Run standard model forward to get base logits
    base_logits, _ = model(x)
    
    # Run KV Cache pre-fill forward
    from src.inference.kv_cache import KVCache
    cache = KVCache()
    cached_logits, _ = model(x, kv_cache=cache)
    
    # Compare pre-fill outputs (must be identical)
    diff_prefill = torch.max(torch.abs(base_logits - cached_logits)).item()
    print(f"Pre-fill maximum logit difference: {diff_prefill:.2e}")
    assert diff_prefill < 1e-5, f"Pre-fill logits differ by {diff_prefill:.2e}!"
    
    # Sample a token and feed it step-by-step
    next_token = torch.tensor([[tokenizer.char2idx[' ']]])
    
    # No cache: concatenate token to sequence and run forward
    x_combined = torch.cat([x, next_token], dim=1)
    logits_full, _ = model(x_combined)
    target_logit = logits_full[:, -1, :] # logit for the new token
    
    # Cached: feed only next_token
    logits_cached_step, _ = model(next_token, kv_cache=cache)
    cached_logit = logits_cached_step[:, -1, :]
    
    diff_step = torch.max(torch.abs(target_logit - cached_logit)).item()
    print(f"Autoregressive step maximum logit difference: {diff_step:.2e}")
    assert diff_step < 1e-5, f"Single-step logits differ by {diff_step:.2e}!"
    
    print("[SUCCESS] Logits are numerically identical within floating point tolerance.")
    print("=" * 65)

if __name__ == "__main__":
    test_equivalence()
