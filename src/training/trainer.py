import time
import math
import torch
from torch.nn import functional as F
from contextlib import nullcontext

class TrainerConfig:
    # Optimization parameters
    max_iters: int = 1000        # Total number of training steps
    batch_size: int = 64         # Micro-batch size per device
    gradient_accumulation_steps: int = 1  # Simulate larger batches
    learning_rate: float = 6e-4  # Max learning rate
    
    # System parameters
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile: bool = False        # PyTorch 2.0 compilation (free speedup)
    
    # Logging
    log_interval: int = 10       # How often to print/log metrics
    eval_interval: int = 100     # How often to run full evaluation
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, config: TrainerConfig, model, optimizer, train_dataloader, val_dataloader=None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        
        # System Setup
        self.ctx = nullcontext() if self.config.device == 'cpu' else torch.amp.autocast(device_type=self.config.device, dtype=getattr(torch, self.config.dtype))
        
        # Initialize Gradient Scaler for Mixed Precision (prevents underflow in float16)
        # Note: bfloat16 doesn't strictly need a scaler, but it's good practice for compatibility.
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == 'float16'))

        # Move model to device
        self.model.to(self.config.device)
        
        # Compile model (PyTorch 2.0+) - Essential for modern research speed
        if self.config.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model)

    def train(self):
        """
        The Research Loop:
        We manually control the forward/backward/update cadence to allow for 
        instrumentation and gradient accumulation.
        """
        model, optimizer = self.model, self.optimizer
        model.train()
        
        # Data Iterator
        data_iter = iter(self.train_loader)
        
        # Timing vars
        t0 = time.time()
        running_mfu = -1.0
        
        print(f"Starting training on {self.config.device}...")
        
        for iter_num in range(self.config.max_iters):
            
            # --- MICRO-BATCH LOOP (Gradient Accumulation) ---
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    X, Y = next(data_iter)
                except StopIteration:
                    # Restart loader if exhausted
                    data_iter = iter(self.train_loader)
                    X, Y = next(data_iter)
                    
                X, Y = X.to(self.config.device), Y.to(self.config.device)
                
                # Forward Pass (with Mixed Precision Context)
                with self.ctx:
                    logits, loss = model(X, targets=Y)
                    # Scale loss for accumulation
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward Pass (Accumulate gradients)
                # We use the scaler to handle float16 dynamic range
                self.scaler.scale(loss).backward()
            
            # --- OPTIMIZATION STEP ---
            # Unscale gradients before clipping
            self.scaler.unscale_(optimizer)
            
            # Research Note: Clip the global norm of the gradient at 1.0. 
            # This prevents "exploding gradients" which ruin long training runs.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step the optimizer and update the scaler
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Flush gradients (set_to_none=True is faster than .zero_grad())
            optimizer.zero_grad(set_to_none=True)

            # --- LOGGING & DIAGNOSTICS ---
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            
            if iter_num % self.config.log_interval == 0:
                # Calculate tokens per second (throughput is a key research metric)
                tokens_per_sec = (self.config.batch_size * self.config.gradient_accumulation_steps * model.config.block_size) / dt
                
                # Extract loss as float
                lossf = loss.item() * self.config.gradient_accumulation_steps
                
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tokens/sec {tokens_per_sec:.2f}, grad_norm {norm:.4f}")
            
            # --- EVALUATION ---
            if self.val_loader and iter_num % self.config.eval_interval == 0:
                self.evaluate(iter_num)

    @torch.no_grad()
    def evaluate(self, iter_num):
        """
        Minimal evaluation loop to track generalization gap.
        """
        self.model.eval()
        losses = []
        for _ in range(20): # Estimate over 20 batches
            try:
                X, Y = next(iter(self.val_loader))
            except StopIteration:
                break
            X, Y = X.to(self.config.device), Y.to(self.config.device)
            with self.ctx:
                _, loss = self.model(X, targets=Y)
            losses.append(loss.item())
        
        mean_loss = sum(losses) / len(losses)
        print(f"--- EVAL: iter {iter_num} | val_loss {mean_loss:.4f} ---")
        self.model.train()