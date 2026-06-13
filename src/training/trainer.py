import os
import time
import threading
import torch
from typing import Dict, Any, Optional
from src.transformer.model import GPT, GPTConfig
from src.training.tokenizer import CharTokenizer

class TrainManager:
    """
    Manages the training state, dataset loading, and the training loop
    asynchronously inside a background thread.
    """
    def __init__(self, data_path: str = "data/tinystories.txt", checkpoint_dir: str = "data"):
        self.data_path = data_path
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        self.vocab_path = os.path.join(checkpoint_dir, "vocab.json")
        
        # Tokenizer & Data
        self.tokenizer = CharTokenizer()
        self.train_data: Optional[torch.Tensor] = None
        self.val_data: Optional[torch.Tensor] = None
        
        # Model & Optimization
        self.model: Optional[GPT] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.config: Optional[GPTConfig] = None
        
        # Threading & Control
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.stop_requested = False
        
        # Training metrics & status
        self.status = "idle"  # idle, training, paused, completed, error
        self.error_message = ""
        self.current_step = 0
        self.max_steps = 2000 # Short training for fast feedback
        self.eval_interval = 100
        self.eval_iters = 20
        self.batch_size = 64
        self.learning_rate = 5e-4
        
        # Metrics history
        self.train_losses = []
        self.val_losses = []
        self.steps_history = []
        self.tokens_per_sec = 0.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load dataset metadata (vocab) if it already exists
        if os.path.exists(self.vocab_path):
            self.tokenizer.load(self.vocab_path)
            
    def prepare_data(self) -> bool:
        """Load text data, fit tokenizer if needed, and create train/val splits."""
        if not os.path.exists(self.data_path):
            print(f"Data file not found at {self.data_path}. Please run download_data.py first.")
            return False
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read(10_000_000) # Load up to 10MB to keep memory consumption low and training snappy
            
        # Initialize vocab if empty
        if self.tokenizer.vocab_size == 0:
            self.tokenizer.build_from_text(text)
            self.tokenizer.save(self.vocab_path)
            
        # Encode dataset to tensors
        print("Encoding dataset...")
        data_ids = self.tokenizer.encode(text, return_tensors="pt")
        
        # 90/10 Train/Validation Split
        n = int(0.9 * len(data_ids))
        self.train_data = data_ids[:n]
        self.val_data = data_ids[n:]
        print(f"Dataset splits prepared: Train={len(self.train_data):,} tokens, Val={len(self.val_data):,} tokens")
        return True

    def get_batch(self, split: str) -> tuple:
        """Sample a random batch of inputs (X) and targets (Y) from dataset splits."""
        data = self.train_data if split == 'train' else self.val_data
        block_size = self.config.block_size
        
        # Generate random starting indices for batches
        ix = torch.randint(len(data) - block_size, (self.batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        # Move tensors to active device
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """Compute average loss over multiple validation batches to track convergence."""
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                x, y = self.get_batch(split)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def get_status(self) -> Dict[str, Any]:
        """Return the current metrics and status of the trainer."""
        with self.lock:
            return {
                "status": self.status,
                "current_step": self.current_step,
                "max_steps": self.max_steps,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "steps_history": self.steps_history,
                "tokens_per_sec": self.tokens_per_sec,
                "device": self.device,
                "error_message": self.error_message,
                "vocab_size": self.tokenizer.vocab_size
            }

    def reset(self):
        """Reset training state, history, and delete local checkpoints."""
        self.pause()
        with self.lock:
            self.current_step = 0
            self.train_losses.clear()
            self.val_losses.clear()
            self.steps_history.clear()
            self.tokens_per_sec = 0.0
            self.status = "idle"
            self.error_message = ""
            
            # Recreate model & optimizer
            self.model = None
            self.optimizer = None
            
            # Clean up disk checkpoints
            if os.path.exists(self.checkpoint_path):
                try:
                    os.remove(self.checkpoint_path)
                    print(f"Deleted old checkpoint: {self.checkpoint_path}")
                except Exception as e:
                    print(f"Error removing checkpoint: {e}")

    def start(self, max_steps: int = 2000, batch_size: int = 64, learning_rate: float = 5e-4):
        """Spawn background training thread."""
        with self.lock:
            if self.status == "training":
                return
            self.stop_requested = False
            self.max_steps = max_steps
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.status = "training"
            self.error_message = ""
            
        self.thread = threading.Thread(target=self._run_training)
        self.thread.daemon = True
        self.thread.start()

    def pause(self):
        """Signal background thread to pause training loop."""
        with self.lock:
            if self.status == "training":
                self.stop_requested = True

    def _run_training(self):
        """Core training loop running on the background thread."""
        try:
            # 1. Prepare data splits if they aren't loaded
            if self.train_data is None or self.val_data is None:
                success = self.prepare_data()
                if not success:
                    with self.lock:
                        self.status = "error"
                        self.error_message = "Failed to load/prepare training dataset."
                    return

            # 2. Model & Optimizer Initialization
            with self.lock:
                if self.config is None:
                    self.config = GPTConfig(
                        vocab_size=self.tokenizer.vocab_size,
                        block_size=256,
                        n_layer=4,
                        n_head=4,
                        n_embd=128,
                        dropout=0.1
                    )
                
                # Check for existing checkpoint
                best_val_loss = float('inf')
                if self.model is None:
                    self.model = GPT(self.config).to(self.device)
                    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
                    
                    if os.path.exists(self.checkpoint_path):
                        print(f"Loading checkpoint from {self.checkpoint_path}")
                        try:
                            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            self.current_step = checkpoint.get('step', 0)
                            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                            self.train_losses = checkpoint.get('train_losses', [])
                            self.val_losses = checkpoint.get('val_losses', [])
                            self.steps_history = checkpoint.get('steps_history', [])
                            print(f"Resumed training from step {self.current_step}")
                        except Exception as e:
                            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

            self.model.train()
            step = self.current_step
            accumulated_tokens = 0
            start_time = time.time()
            
            while step < self.max_steps:
                # Check if stop has been requested
                with self.lock:
                    if self.stop_requested:
                        self.status = "paused"
                        self.current_step = step
                        self._save_checkpoint(best_val_loss)
                        print(f"Training paused at step {step}")
                        return

                # Sample training batch
                x, y = self.get_batch('train')
                
                # Forward, backward, optimize
                logits, loss = self.model(x, y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                step += 1
                accumulated_tokens += x.numel()
                
                # Estimate throughput
                now = time.time()
                elapsed = now - start_time
                if elapsed >= 2.0: # Recalculate speed every 2 seconds
                    with self.lock:
                        self.tokens_per_sec = accumulated_tokens / elapsed
                    accumulated_tokens = 0
                    start_time = now
                
                # Periodic evaluation and validation logging
                if step % self.eval_interval == 0 or step == self.max_steps:
                    losses = self.estimate_loss()
                    print(f"Step {step}/{self.max_steps}: Train Loss = {losses['train']:.4f}, Val Loss = {losses['val']:.4f}")
                    
                    with self.lock:
                        self.train_losses.append(losses['train'])
                        self.val_losses.append(losses['val'])
                        self.steps_history.append(step)
                        self.current_step = step
                        
                        # Save if validation loss improves
                        if losses['val'] < best_val_loss:
                            best_val_loss = losses['val']
                            self._save_checkpoint(best_val_loss)

            # Training completed successfully
            with self.lock:
                self.status = "completed"
                self.current_step = step
                self._save_checkpoint(best_val_loss)
                print("Training completed successfully!")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            with self.lock:
                self.status = "error"
                self.error_message = str(e)
                print(f"Error in training thread: {e}")

    def _save_checkpoint(self, best_val_loss: float):
        """Write model weights, optimizer state, and loss history to file."""
        if self.model is None or self.optimizer is None:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step': self.current_step,
            'best_val_loss': best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'steps_history': self.steps_history
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Saved training checkpoint to {self.checkpoint_path}")
