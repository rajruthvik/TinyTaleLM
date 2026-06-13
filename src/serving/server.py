import os
import json
import asyncio
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.transformer.model import GPT, GPTConfig
from src.training.tokenizer import CharTokenizer
from src.training.trainer import TrainManager
from src.inference.generator import generate, generate_stream
from src.optimization.quantize import quantize_model, get_model_size_mb

app = FastAPI(title="miniLLM serving layer", description="FastAPI server for training, serving, and benchmarking miniLLM.")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global training manager
trainer = TrainManager()

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Helper to get the active model for generation
def get_active_model(quantized: bool = False, device: str = "cpu"):
    """
    Get the current model. Uses the live training model if available,
    loads from checkpoint, or initializes a default model.
    """
    model = None
    tokenizer = trainer.tokenizer
    
    # 1. Use the live training model if it exists
    if trainer.model is not None:
        model = trainer.model
    # 2. Otherwise, check if checkpoint exists
    elif os.path.exists(trainer.checkpoint_path):
        try:
            print("Loading model from checkpoint for inference...")
            checkpoint = torch.load(trainer.checkpoint_path, map_location="cpu")
            config = checkpoint['config']
            model = GPT(config)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            model = None
            
    # 3. Fallback: Create a default model if nothing exists
    if model is None:
        print("No active model or checkpoint found. Initializing a default model for generation.")
        # Ensure tokenizer is built
        if tokenizer.vocab_size == 0:
            if os.path.exists(trainer.vocab_path):
                tokenizer.load(trainer.vocab_path)
            else:
                # Build a default fallback vocabulary
                fallback_text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!'"
                tokenizer.build_from_text(fallback_text)
                tokenizer.save(trainer.vocab_path)
                
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=256,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.1
        )
        model = GPT(config)
        
    # Apply dynamic quantization if requested
    if quantized:
        # Dynamic quantization is optimized for CPU
        model = quantize_model(model)
        device = "cpu"
    else:
        # Move model to requested device
        model = model.to(device)
        
    return model, tokenizer

# --- API ENDPOINTS ---

@app.get("/api/status")
def get_status():
    """Retrieve current training statuses and loss logs."""
    return trainer.get_status()

@app.post("/api/train/start")
def start_train(params: dict = Body(default={})):
    """Trigger the background training thread with parameters."""
    max_steps = params.get("max_steps", 2000)
    batch_size = params.get("batch_size", 64)
    learning_rate = params.get("learning_rate", 5e-4)
    
    if trainer.status == "training":
        return {"message": "Training is already running."}
        
    trainer.start(max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate)
    return {"message": "Training started successfully."}

@app.post("/api/train/pause")
def pause_train():
    """Signal training loop to pause."""
    trainer.pause()
    return {"message": "Pause signal sent."}

@app.post("/api/train/reset")
def reset_train():
    """Reset training parameters, checkpoint, and clear logs."""
    trainer.reset()
    return {"message": "Trainer reset successfully."}

@app.get("/api/dataset/preview")
def dataset_preview():
    """Return a preview snippet of the raw dataset text."""
    if not os.path.exists(trainer.data_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Please run download_data.py first.")
    try:
        with open(trainer.data_path, "r", encoding="utf-8") as f:
            preview = f.read(5000) # first 5000 chars
        return {"preview": preview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {e}")

@app.post("/api/generate")
def generate_text(payload: dict = Body(...)):
    """Generate text in a single block and return results & performance profile."""
    prompt = payload.get("prompt", "")
    max_new_tokens = payload.get("max_new_tokens", 150)
    temperature = payload.get("temperature", 1.0)
    top_k = payload.get("top_k", 10)
    use_cache = payload.get("use_cache", True)
    quantized = payload.get("quantized", False)
    device = payload.get("device", "cpu")
    
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    try:
        model, tokenizer = get_active_model(quantized=quantized, device=device)
        model_size = get_model_size_mb(model)
        
        text, metrics = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=use_cache,
            device=device
        )
        metrics["model_size_mb"] = model_size
        return {"text": text, "metrics": metrics}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generate/stream")
def stream_text(
    prompt: str = "",
    max_new_tokens: int = 150,
    temperature: float = 1.0,
    top_k: int = 10,
    use_cache: bool = True,
    quantized: bool = False,
    device: str = "cpu"
):
    """Stream generated text character-by-character along with real-time speed metrics."""
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    model, tokenizer = get_active_model(quantized=quantized, device=device)
    model_size = get_model_size_mb(model)
    
    def event_generator():
        # Yield initial size metric
        yield f"data: {json.dumps({'type': 'init', 'model_size_mb': model_size})}\n\n"
        
        generator = generate_stream(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=use_cache,
            device=device
        )
        
        for chunk in generator:
            yield f"data: {json.dumps(chunk)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/benchmark")
async def run_api_benchmark():
    """
    Run a mini-benchmark matrix asynchronously inside a threadpool to prevent
    blocking the main async loop, and return the comparison results.
    """
    import torch
    
    def sync_benchmark():
        # Get baseline model
        model, tokenizer = get_active_model(quantized=False, device="cpu")
        int8_model = quantize_model(model)
        
        prompt = "Once upon a time, a small child"
        tokens_to_generate = 50
        
        configs = [
            {"name": "FP32 CPU (No Cache)", "model": model, "device": "cpu", "use_cache": False, "quantized": False},
            {"name": "FP32 CPU (KV Cache)", "model": model, "device": "cpu", "use_cache": True, "quantized": False},
            {"name": "INT8 CPU (KV Cache)", "model": int8_model, "device": "cpu", "use_cache": True, "quantized": True},
        ]
        
        if torch.cuda.is_available():
            configs.append(
                {"name": "FP32 GPU (KV Cache)", "model": model, "device": "cuda", "use_cache": True, "quantized": False}
            )
            
        results = []
        for cfg in configs:
            name = cfg["name"]
            m = cfg["model"]
            device = cfg["device"]
            use_cache = cfg["use_cache"]
            quantized = cfg["quantized"]
            
            if not quantized:
                m = m.to(device)
                
            # Measure model size
            size_mb = get_model_size_mb(m)
            
            # Warm up
            try:
                generate(m, tokenizer, prompt, max_new_tokens=10, use_cache=use_cache, device=device)
                
                # Benchmark run
                _, metrics = generate(
                    m, tokenizer, prompt,
                    max_new_tokens=tokens_to_generate,
                    use_cache=use_cache,
                    device=device
                )
                
                results.append({
                    "name": name,
                    "model_size": f"{size_mb:.2f} MB",
                    "ttft_ms": f"{metrics['ttft_ms']:.1f}",
                    "throughput": f"{metrics['tokens_per_sec']:.1f}"
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "model_size": "N/A",
                    "ttft_ms": "Error",
                    "throughput": f"Error: {e}"
                })
        return results

    try:
        results = await run_in_threadpool(sync_benchmark)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- STATIC CONTENT ---

# Route to serve the main frontend file
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# Create static directory structure if it doesn't exist
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
