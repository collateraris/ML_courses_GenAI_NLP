# train.py
# This script trains the MiniDeepSeek model on the tokenized TinyStories dataset.
# It is designed to run on a single GPU.

import argparse
import os
import json
import time
from contextlib import nullcontext
import math # Added for learning rate scheduler

import numpy as np
import torch

# Import the model definition from model.py
from model import LOSS_VERSION, MiniDeepSeek, ModelArgs
from checkpoints import load_latest_checkpoint, save_checkpoint

# --- Configuration ---
# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=pt_dtype) if 'cuda' in device else nullcontext()

# Training
out_dir = 'out'
data_dir = 'data/tinystories_tokenized' # Points to the generic output directory
max_iters = 30000            # Total iteration limit, including resumed steps
eval_interval = 250          # Evaluate less frequently on a longer run
log_interval = 20
eval_iters = 100
batch_size = 24              # A good starting batch size for a 4090
block_size = 256

# AdamW Optimizer
learning_rate = 4e-4         # A slightly higher LR can work well for this size
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

# --- Data Loading ---
def get_batch(split: str):
    """
    Loads a batch of data from the memory-mapped .bin files.
    """
    # The data files are now uint16, as created by the new prepare.py
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    # Generate random starting points for each sequence in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create input sequences (x)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # Create target sequences (y), which are shifted by one
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if 'cuda' in device:
        # Pin memory helps speed up CPU-to-GPU data transfer
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        return x, y

# Learning rate scheduler: persist its horizon and resume without an LR jump.
def make_lr_schedule(total_iters, checkpoint=None):
    if checkpoint is not None:
        saved_schedule = checkpoint.get('lr_schedule')
        if saved_schedule is not None and saved_schedule['end_iter'] == total_iters:
            return dict(saved_schedule)
        if (saved_schedule is not None
                and checkpoint['iter_num'] < saved_schedule['start_iter'] + saved_schedule['warmup_iters']):
            # Extending a short run must finish warmup, including after its
            # first zero-LR step, rather than freezing at the current LR.
            return dict(saved_schedule, end_iter=total_iters)
        # An old checkpoint or a changed horizon starts a new decay segment
        # from the LR actually used by the saved optimizer, without warmup.
        start_lr = float(checkpoint['optimizer']['param_groups'][0]['lr'])
        if start_lr == 0.0 and checkpoint['iter_num'] + 1 < 200:
            return make_lr_schedule(total_iters)
        return {
            'start_iter': checkpoint['iter_num'] + 1,
            'end_iter': total_iters,
            'start_lr': start_lr,
            'min_lr': min(learning_rate / 10, start_lr),
            'warmup_iters': 0,
        }
    return {
        'start_iter': 0,
        'end_iter': total_iters,
        'start_lr': learning_rate,
        'min_lr': learning_rate / 10,
        'warmup_iters': 200,
    }


def get_lr(it, schedule=None):
    if schedule is None:
        schedule = make_lr_schedule(max_iters)
    elapsed = max(0, it - schedule['start_iter'])
    warmup_iters = schedule['warmup_iters']
    if warmup_iters > 0 and elapsed < warmup_iters:
        return schedule['start_lr'] * elapsed / warmup_iters
    decay_iters = schedule['end_iter'] - schedule['start_iter'] - warmup_iters
    if decay_iters <= 0 or it >= schedule['end_iter']:
        return schedule['min_lr']
    decay_ratio = min(1.0, max(0.0, (elapsed - warmup_iters) / decay_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return schedule['min_lr'] + coeff * (schedule['start_lr'] - schedule['min_lr'])


def make_checkpoint(model, optimizer, model_args, iter_num, val_loss, best_val_loss,
                    *, lr_schedule, val_main_loss, val_mtp_loss, best_val_main_loss):
    return {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'val_main_loss': val_main_loss,
        'val_mtp_loss': val_mtp_loss,
        'best_val_main_loss': best_val_main_loss,
        'loss_version': LOSS_VERSION,
        'lr_schedule': dict(lr_schedule),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if 'cuda' in device else None,
    }


# --- Main Training Script ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train, automatically resuming the latest checkpoint.')
    parser.add_argument('--max-iters', type=int, default=max_iters,
                        help='Total iteration limit, including steps already saved (default: %(default)s).')
    cli_args = parser.parse_args()
    if cli_args.max_iters <= 0:
        parser.error('--max-iters must be positive')
    max_iters = cli_args.max_iters
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Metadata and Initialize Model
    meta_path = os.path.join(data_dir, 'meta.json')
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        vocab_size = meta['vocab_size']
    except FileNotFoundError:
        print(f"Error: meta.json not found in {data_dir}.")
        print("Please run prepare.py first to tokenize the dataset.")
        exit(1)

    checkpoint, checkpoint_path = load_latest_checkpoint(out_dir)
    start_iter = checkpoint['iter_num'] + 1 if checkpoint is not None else 0
    if start_iter >= max_iters:
        print(f"Checkpoint {checkpoint_path} already contains {start_iter} completed steps.")
        print(f"Use --max-iters greater than {start_iter} to continue training.")
        raise SystemExit(0)

    lr_schedule = make_lr_schedule(max_iters, checkpoint)

    if checkpoint is not None:
        model_args = checkpoint['model_args']
        if isinstance(model_args, dict):
            model_args = ModelArgs(**model_args)
        if model_args.vocab_size != vocab_size:
            raise ValueError('Checkpoint vocabulary size does not match the tokenized dataset.')
        block_size = model_args.max_seq_len
        print(f"Resuming from {checkpoint_path} at iteration {start_iter} (limit: {max_iters}).")
    else:
        print('No checkpoint found; starting training from scratch.')
        # Default configuration for a new model on RTX 4090.
        model_args = ModelArgs(
            d_model=768,                # Standard "small" model dimension
            n_layers=12,                # 12 layers deep
            num_heads=12,               # 12 attention heads (d_head=64)
            d_latent=192,               # Latent dim for MLA (3 * d_head)
            d_rope=64,                  # RoPE dim, same as d_head
            moe_n_routed_experts=16,    # A good number of experts to allow specialization
            moe_n_shared_experts=1,     # One generalist expert for common knowledge
            moe_top_k=4,                # Route to the best 4 experts per token
            moe_routed_hidden=512,      # Each expert is reasonably sized
            vocab_size=vocab_size,      # 50257 from our data
            max_seq_len=block_size
        )
    
    model = MiniDeepSeek(model_args)
    if checkpoint is not None:
        prefix = '_orig_mod.'
        state_dict = {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in checkpoint['model'].items()
        }
        model.load_state_dict(state_dict)
        del state_dict
    model.to(device)
    
    # --- Calculate and Print Total Parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel initialized on {device}")
    print(f"  -> Total parameters: {total_params/1e6:.2f}M")
    print(f"  -> Trainable parameters: {trainable_params/1e6:.2f}M\n")

    # 2. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    best_val_loss = float('inf')
    best_val_main_loss = float('inf')
    val_loss = float('inf')
    val_main_loss = float('inf')
    val_mtp_loss = float('inf')
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('loss_version', 1) == LOSS_VERSION:
            best_val_loss = checkpoint['best_val_loss']
            best_val_main_loss = float(checkpoint.get('best_val_main_loss', float('inf')))
            val_loss = float(checkpoint.get('val_loss', float('inf')))
            val_main_loss = float(checkpoint.get('val_main_loss', float('inf')))
            val_mtp_loss = float(checkpoint.get('val_mtp_loss', float('inf')))
        else:
            print('Checkpoint uses the old target alignment; resetting validation metrics for the corrected loss.')
        if checkpoint.get('rng_state') is not None:
            torch.set_rng_state(checkpoint['rng_state'])
        if 'cuda' in device and checkpoint.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        del checkpoint

    print(f"Learning rate at iteration {start_iter}: {get_lr(start_iter, lr_schedule):.8f}")

    # 3. Training Loop
    t0 = time.time()
    for iter_num in range(start_iter, max_iters):
        # --- Update Learning Rate ---
        lr = get_lr(iter_num, lr_schedule)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- Training Step ---
        model.train()
        X, Y = get_batch('train')
        
        # Forward pass with Automatic Mixed Precision
        with ctx:
            outputs = model(X, targets=Y)
            loss = outputs['loss']
        
        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # --- Logging ---
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num}: loss {loss.item():.4f}, main {outputs['loss_main'].item():.4f}, "
                  f"mtp {outputs['loss_mtp'].item():.4f}, time {dt*1000:.2f}ms, lr {lr:.8f}")

        # --- Evaluation Step ---
        if iter_num % eval_interval == 0 and iter_num > 0:
            model.eval()
            losses = torch.zeros(eval_iters)
            main_losses = torch.zeros(eval_iters)
            mtp_losses = torch.zeros(eval_iters)
            print("Running evaluation...")
            with torch.no_grad():
                for k in range(eval_iters):
                    X, Y = get_batch('val')
                    with ctx:
                        outputs = model(X, targets=Y)
                        losses[k] = outputs['loss'].item()
                        main_losses[k] = outputs['loss_main'].item()
                        mtp_losses[k] = outputs['loss_mtp'].item()
            
            val_loss = losses.mean().item()
            val_main_loss = main_losses.mean().item()
            val_mtp_loss = mtp_losses.mean().item()
            print(f"iter {iter_num}: validation loss {val_loss:.4f}, "
                  f"main {val_main_loss:.4f}, mtp {val_mtp_loss:.4f}")

            # --- Checkpointing ---
            is_best = val_main_loss < best_val_main_loss
            best_val_loss = min(best_val_loss, val_loss)
            best_val_main_loss = min(best_val_main_loss, val_main_loss)
            checkpoint = make_checkpoint(
                model, optimizer, model_args, iter_num, val_loss, best_val_loss,
                lr_schedule=lr_schedule, val_main_loss=val_main_loss,
                val_mtp_loss=val_mtp_loss, best_val_main_loss=best_val_main_loss,
            )
            latest_ckpt_path = os.path.join(out_dir, 'last_ckpt.pt')
            print(f"Saving latest checkpoint to {latest_ckpt_path}")
            save_checkpoint(checkpoint, latest_ckpt_path)

            # 1. Save the best model so far
            if is_best:
                best_ckpt_path = os.path.join(out_dir, 'best_ckpt.pt')
                print(f"New best validation main loss: {val_main_loss:.4f}. Saving best checkpoint to {best_ckpt_path}")
                save_checkpoint(checkpoint, best_ckpt_path)

            # 2. Save a periodic checkpoint every 500 iterations
            save_interval = 500
            if iter_num % save_interval == 0 and iter_num > 0:
                periodic_ckpt_path = os.path.join(out_dir, f'ckpt_iter_{iter_num}.pt')
                print(f"Saving periodic checkpoint to {periodic_ckpt_path}")
                save_checkpoint(checkpoint, periodic_ckpt_path)

    # Keep the final updates even when the last iteration is not an evaluation.
    final_ckpt_path = os.path.join(out_dir, 'last_ckpt.pt')
    checkpoint = make_checkpoint(
        model, optimizer, model_args, iter_num, val_loss, best_val_loss,
        lr_schedule=lr_schedule, val_main_loss=val_main_loss,
        val_mtp_loss=val_mtp_loss, best_val_main_loss=best_val_main_loss,
    )
    print(f"Saving final checkpoint to {final_ckpt_path}")
    save_checkpoint(checkpoint, final_ckpt_path)
    print("\nTraining complete.")
