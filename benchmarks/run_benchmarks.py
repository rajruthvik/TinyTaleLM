import os
import sys
import time
import json
import torch
import torch.nn as nn

# Make sure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer.model import GPT, GPTConfig
from src.training.tokenizer import CharTokenizer
from src.inference.generator import generate
from src.optimization.quantize import quantize_model, get_model_size_mb

def load_or_create_model():
    """Load model from checkpoint if available; otherwise create a dummy model for benchmarking."""
    checkpoint_dir = "data"
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    
    tokenizer = CharTokenizer()
    
    # 1. Load or build a mock tokenizer
    if os.path.exists(vocab_path):
        tokenizer.load(vocab_path)
    else:
        print("No vocabulary found. Creating a dummy tokenizer vocabulary for benchmark...")
        # 65 unique characters typical of a TinyStories text subset
        dummy_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!'"
        tokenizer.build_from_text(dummy_text)
        os.makedirs(checkpoint_dir, exist_ok=True)
        tokenizer.save(vocab_path)
        
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1
    )
    
    model = GPT(config)
    
    # 2. Try loading weights from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded trained weights.")
        except Exception as e:
            print(f"Failed to load checkpoint weights: {e}. Benchmarking on randomly initialized model.")
    else:
        print("No checkpoint found at data/checkpoint.pt. Benchmarking on randomly initialized weights.")
        
    return model, tokenizer

def run_benchmarks():
    print("=" * 60)
    print("          MINILLM INFERENCE OPTIMIZATION BENCHMARK          ")
    print("=" * 60)
    
    # Load model and tokenizer
    fp32_model, tokenizer = load_or_create_model()
    
    # Pre-compute quantized model
    print("\nPreparing INT8 quantized model...")
    int8_model = quantize_model(fp32_model)
    
    # Define benchmark matrix
    cuda_available = torch.cuda.is_available()
    configs = [
        {"name": "FP32 CPU (No Cache)", "model": fp32_model, "device": "cpu", "use_cache": False, "quantized": False},
        {"name": "FP32 CPU (KV Cache)", "model": fp32_model, "device": "cpu", "use_cache": True, "quantized": False},
        {"name": "INT8 CPU (No Cache)", "model": int8_model, "device": "cpu", "use_cache": False, "quantized": True},
        {"name": "INT8 CPU (KV Cache)", "model": int8_model, "device": "cpu", "use_cache": True, "quantized": True},
    ]
    
    if cuda_available:
        configs.extend([
            {"name": "FP32 GPU (No Cache)", "model": fp32_model, "device": "cuda", "use_cache": False, "quantized": False},
            {"name": "FP32 GPU (KV Cache)", "model": fp32_model, "device": "cuda", "use_cache": True, "quantized": False},
        ])
    else:
        print("\nCUDA (GPU) is not available. Skipping GPU benchmarks.")
        
    prompt = "Once upon a time, there was a little boy named Tim. He loved to"
    tokens_to_generate = 100
    warmup_runs = 1
    eval_runs = 3
    
    results = []
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating {tokens_to_generate} tokens per run (averaged over {eval_runs} evaluations after {warmup_runs} warm-ups).\n")
    
    for config in configs:
        name = config["name"]
        model = config["model"]
        device = config["device"]
        use_cache = config["use_cache"]
        quantized = config["quantized"]
        
        # Move model to device
        if not quantized:
            model = model.to(device)
            
        print(f"Running: {name}...", end="", flush=True)
        
        # Measure size
        size_mb = get_model_size_mb(model)
        
        # Warmup
        try:
            for _ in range(warmup_runs):
                generate(model, tokenizer, prompt, max_new_tokens=20, use_cache=use_cache, device=device)
                
            # Eval
            total_ttft_ms = 0.0
            total_throughput = 0.0
            
            for _ in range(eval_runs):
                _, metrics = generate(
                    model, tokenizer, prompt, 
                    max_new_tokens=tokens_to_generate, 
                    use_cache=use_cache, 
                    device=device
                )
                total_ttft_ms += metrics["ttft_ms"]
                total_throughput += metrics["tokens_per_sec"]
                
            avg_ttft_ms = total_ttft_ms / eval_runs
            avg_throughput = total_throughput / eval_runs
            
            # Estimate peak VRAM if on CUDA
            vram_usage = "N/A"
            if device == "cuda":
                # Clear stats and run a test to check peak allocated memory
                torch.cuda.reset_peak_memory_stats()
                generate(model, tokenizer, prompt, max_new_tokens=tokens_to_generate, use_cache=use_cache, device=device)
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                vram_usage = f"{vram_mb:.2f} MB"
                
            results.append({
                "Configuration": name,
                "Model Size": f"{size_mb:.2f} MB",
                "TTFT (ms)": f"{avg_ttft_ms:.1f} ms",
                "Throughput": f"{avg_throughput:.1f} tokens/s",
                "GPU VRAM": vram_usage,
                "speedup": avg_throughput # saved for calculating improvements
            })
            print(" Done.")
            
        except Exception as e:
            print(f" Failed. (Error: {e})")
            
    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Configuration':<25} | {'Model Size':<10} | {'TTFT (ms)':<10} | {'Throughput':<15} | {'GPU VRAM':<10}")
    print("-" * 80)
    for res in results:
        print(f"{res['Configuration']:<25} | {res['Model Size']:<10} | {res['TTFT (ms)']:<10} | {res['Throughput']:<15} | {res['GPU VRAM']:<10}")
    print("=" * 80)
    
    # Print Key Insights
    if len(results) >= 2:
        # Compare FP32 CPU No Cache vs KV Cache
        cpu_no_cache = None
        cpu_cache = None
        cpu_quant_cache = None
        for res in results:
            if "FP32 CPU (No Cache)" in res["Configuration"]:
                cpu_no_cache = float(res["Throughput"].split()[0])
            elif "FP32 CPU (KV Cache)" in res["Configuration"]:
                cpu_cache = float(res["Throughput"].split()[0])
            elif "INT8 CPU (KV Cache)" in res["Configuration"]:
                cpu_quant_cache = float(res["Throughput"].split()[0])
                
        print("\nOptimization Insights:")
        if cpu_no_cache and cpu_cache:
            speedup = cpu_cache / cpu_no_cache
            print(f"[*] KV Cache Speedup (CPU): {speedup:.2f}x faster generation ({cpu_no_cache:.1f} -> {cpu_cache:.1f} tokens/s)")
            
        if cpu_cache and cpu_quant_cache:
            speedup = cpu_quant_cache / cpu_cache
            print(f"[*] INT8 Quantization + KV Cache Speedup: {cpu_quant_cache / cpu_no_cache:.2f}x faster than base model.")

if __name__ == "__main__":
    run_benchmarks()
