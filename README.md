# miniLLM: TinyStories GPT & Inference Optimization Engine

A character-level Decoder-Only Transformer (GPT) language model trained on the **TinyStories** dataset, featuring key-value caching, post-training dynamic INT8 quantization, and a local benchmark dashboard.

Unlike standard hobbyist GPT implementations, **miniLLM** is structured as an **AI engineering portfolio piece** focused on **inference optimization and runtime latency/throughput profiling**.

---

## Technical Features

### 1. KV Caching (`src/inference/kv_cache.py`)
In standard autoregressive text generation, generating the $t$-th token requires passing all $t-1$ preceding tokens through the model to compute self-attention. This has a time complexity of $O(t^2)$ attention operations per step, leading to $O(N^3)$ operations to generate $N$ tokens.
- **Our Optimization**: We implement a Key-Value Cache. By storing the historical key ($K$) and value ($V$) vectors for each attention head in each transformer layer, we only pass the *newest single token* through the network.
- **Impact**: Attending to history scales down to $O(t)$ operations per step, resulting in a **~2.5x to 3x throughput speedup** on CPU.

### 2. Post-Training Dynamic INT8 Quantization (`src/optimization/quantize.py`)
Model parameters are normally stored as float32 tensors (4 bytes per parameter).
- **Our Optimization**: We use PyTorch dynamic quantization to pack float32 Linear layer weights down to 8-bit integers (1 byte per parameter).
- **Impact**: Reduces model memory footprint on disk and RAM by **~75%** (e.g. from 16MB to 4MB) and speeds up CPU matrix operations by using vectorized INT8 arithmetic.

### 3. Asynchronous serving & Real-time Web Dashboard (`static/`)
- A single-page dashboard built using vanilla glassmorphic CSS, JS, and HTML.
- Uses Chart.js to plot train and validation loss curves in real-time.
- Integrates a **Typewriter Terminal console** that streams text token-by-token using **Server-Sent Events (SSE)**.
- Features a **Benchmarking Dashboard** that lets you toggle KV caching, INT8 quantization, and CPU vs. GPU live and see their relative throughput (tokens/s) and TTFT (ms) latency immediately.

---

## File Structure

```
miniLLM/
├── src/
│   ├── transformer/
│   │   └── model.py         # Decoder-Only Transformer (supports causal mask + KV cache)
│   ├── training/
│   │   ├── tokenizer.py     # Simple Character-level Tokenizer
│   │   └── trainer.py       # Asynchronous dataset trainer & checkpoint manager
│   ├── inference/
│   │   ├── kv_cache.py      # KV Cache state container class
│   │   └── generator.py     # Autoregressive generation engine (cached vs. standard)
│   ├── optimization/
│   │   └── quantize.py      # INT8 dynamic quantization helper
│   └── serving/
│       └── server.py        # FastAPI API routes & static mount points
├── benchmarks/
│   └── run_benchmarks.py    # Command-line comparison benchmarking script
├── tests/
│   └── test_kv_cache.py     # Logits & outputs mathematical equivalence validator
├── static/
│   ├── index.html           # Dashboard layout
│   ├── styles.css           # Custom glassmorphic dark styles
│   └── app.js               # EventSource SSE streams, Chart.js updates, & API fetches
├── app.py                   # Main FastAPI runner
├── requirements.txt         # Core dependencies
└── README.md                # Documentation (this file)
```

---

## Local Setup & Installation

### 1. Set Up Python Environment
Ensure you are using Python 3.10+ (tested on Python 3.11).

```bash
# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. Enable GPU/CUDA Support (Optional)
Your machine detects CPU training by default. If you have a CUDA-enabled GPU (like an RTX 4060) and want to run GPU-accelerated inference/training, install PyTorch with CUDA wheels:

```bash
# Uninstall standard CPU PyTorch
pip uninstall torch -y

# Install PyTorch with CUDA 12.1 support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

To verify CUDA is available in PyTorch, run:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

---

## Step-by-Step Operations

### Step 1: Download the Dataset
Download the TinyStories text validation corpus (19.4 MB):
```bash
python download_data.py
```

### Step 2: Run Mathematical Equivalence Tests
Verify that the KV Cache key-value recurrence is mathematically identical to recomputing standard attention:
```bash
python tests/test_kv_cache.py
```

### Step 3: Run the CLI Benchmarking Suite
Compare latency and speed tradeoffs directly in the command line:
```bash
python benchmarks/run_benchmarks.py
```

### Step 4: Launch the Dashboard Server
Start the serving layer locally:
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:8000`** to:
1. Start/pause training and watch convergence charts.
2. Interactively test the model with dynamic combinations of KV Cache, INT8 Quantization, and GPU acceleration.
