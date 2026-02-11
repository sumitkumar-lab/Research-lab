import torch
import math

class Metrics:
    @staticmethod
    def estimate_loss(model, dataloader, eval_iters, device, ctx):
        """
        Estimates the loss on a given split (train or val) more accurately 
        by averaging over multiple batches. This smooths out noise.
        """
        out = {}
        model.eval()
        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            # Fetch a random batch
            try:
                X, Y = next(iter(dataloader))
            except StopIteration:
                break
                
            X, Y = X.to(device), Y.to(device)
            
            with ctx:
                logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
            
        out['loss'] = losses.mean()
        # Perplexity is exp(cross_entropy_loss)
        out['perplexity'] = torch.exp(out['loss'])
        
        model.train()
        return out

    @staticmethod
    def estimate_mfu(model_config, batch_size, dt, device_flops_peak=None):
        """
        Estimates Model FLOPs Utilization (MFU) in units of A100 bfloat16 peak FLOPS.
        Even if you don't have an A100, this number tells you how efficient your code is.
        
        Formula from PaLM paper (Appendix B).
        """
        # Get number of parameters in the model
        # We assume standard Transformer architecture for FLOP counting
        N = model_config.n_layer
        H = model_config.n_head
        D = model_config.n_embd
        # approximate number of parameters
        # 12 * N * D^2 (for attention + MLP) usually dominates
        params_num = 12 * N * (D**2) 
        
        # We need FLOPs per token per forward/backward pass
        # generally ~6N for forward+backward
        flops_per_token = 6 * params_num 
        
        # Total flops for one iteration
        flops_per_iter = flops_per_token * batch_size * model_config.block_size
        
        # FLOPs achieved per second
        flops_achieved = flops_per_iter / dt 
        
        # If we know the GPU peak, calculate utilization percentage
        # Example: A100 bfloat16 peak is ~312 TFLOPS
        if device_flops_peak is None:
            # Default to A100 40GB/80GB (312e12) or approx 3090/4090 numbers
            # You should configure this based on YOUR hardware.
            # RTX 4090 ~ 83 TFLOPS (FP16/BF16 tensor core)
            # A100 ~ 312 TFLOPS
            # T4 ~ 65 TFLOPS
            device_flops_peak = 312e12 
            
        mfu = flops_achieved / device_flops_peak
        return mfu

    @staticmethod
    @torch.no_grad()
    def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Takes a conditioning sequence (idx) and completes it.
        Strictly for qualitative evaluation (human review).
        """
        model.eval()
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            
            # Optional: Top-K sampling (truncates the tail of the distribution)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        model.train()
        return idx