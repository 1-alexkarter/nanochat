"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disable torch.compile / TorchDynamo / Inductor globally.
# This avoids Inductor trying to compile Muon/AdamW on XLA devices.
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import time
from contextlib import nullcontext

import wandb
import torch

# Optional TPU support (torch-xla). On pure GPU setups this will just set xm/xmp = None.
try:
    import torch_xla.core.xla_model as xm  # type: ignore
    import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore
    from torch_xla import runtime as xr  # <--- add this
except ImportError:
    xm = None
    xmp = None
    xr = None


from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
    autodetect_device_type,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model

print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy"  # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = ""  # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = (
    20  # the depth of the Transformer model to train, rest of the kwargs are derived
)
max_seq_len = 2048  # max context length
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1  # explicit number of steps of the optimization (-1 = disable)
target_flops = (
    -1.0
)  # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20  # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32  # per-device batch size (set to not OOM)
total_batch_size = 524288  # total desired batch size, in #tokens
embedding_lr = 0.2  # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004  # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0  # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02  # learning rate for the matrix parameters (Muon)
grad_clip = 1.0  # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0  # ratio of iterations for LR warmup
warmdown_ratio = 0.2  # ratio of iterations for LR warmdown
final_lr_frac = 0.0  # final LR is this fraction of the initial LR
# Evaluation
eval_every = 250  # every how many steps to evaluate the model for val bpb
eval_tokens = 20 * 524288  # number of tokens to evaluate val loss on
core_metric_every = (
    2000  # every how many steps to evaluate the core metric (-1 = disable)
)
core_metric_max_per_task = 500  # examples per task in estimating the core metric
sample_every = 2000  # every how many steps to sample from the model
# Output
model_tag = (
    ""  # optionally override the model tag for the output checkpoint directory name
)
# now allow CLI to override the settings via the configurator lol
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(
    open(os.path.join("nanochat", "configurator.py")).read()
)  # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Main training function (single or multi-process)
def _mp_train_fn(index: int):
    """
    index: process index when launched via xmp.spawn.
           For non-XLA runs we just pass 0.
    """
    global device_type, num_iterations  # we let configurator override this at module level

    # Decide device_type once more, honoring CLI overrides
    device_type = autodetect_device_type() if device_type == "" else device_type

    if device_type == "xla":
        # Multi-process XLA: each process gets its own TPU core.
        assert xm is not None, "torch_xla is required for device_type='xla'"
        assert xr is not None, "torch_xla.runtime is required for PJRT"

        # World size and rank come from XLA's view of the mesh

        device = xm.xla_device()
        ddp_world_size = xr.global_runtime_device_count()
        ddp_rank = xr.global_ordinal()
        ddp_local_rank = ddp_rank  # single host, 1 process per device
        ddp = ddp_world_size > 1

        # Make the rest of nanochat infra happy (print0, dataloader, etc.)
        os.environ["RANK"] = str(ddp_rank)
        os.environ["LOCAL_RANK"] = str(ddp_local_rank)
        os.environ["WORLD_SIZE"] = str(ddp_world_size)

    else:
        # Original CUDA/MPS/CPU path, using nanochat.common.compute_init
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(
            device_type
        )

    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

    # Autocast / sync / memory helpers per device type
    if device_type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        synchronize = torch.cuda.synchronize
        get_max_memory = torch.cuda.max_memory_allocated
    elif device_type == "xla":
        # XLA/TPU: no torch.cuda, and bfloat16 is handled natively
        from contextlib import nullcontext as _nullcontext  # shadow just to be explicit

        autocast_ctx = _nullcontext()
        # mark_step acts as a sync point + "flush" for pending ops
        synchronize = xm.mark_step
        get_max_memory = (
            lambda: 0
        )  # nothing convenient like torch.cuda.max_memory_allocated
    else:
        # CPU / MPS: no special sync/autocast
        from contextlib import nullcontext as _nullcontext

        autocast_ctx = _nullcontext()
        synchronize = lambda: None
        get_max_memory = lambda: 0

    # wandb logging init
    use_dummy_wandb = run == "dummy" or not master_process
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(project="nanochat", name=run, config=user_config)
    )

    # Tokenizer will be useful for evaluation, also we need the vocab size
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # Model kwargs are derived from the desired depth of the model
    num_layers = depth
    model_dim = depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim 128 (ceil div)
    num_kv_heads = num_heads  # default 1:1 GQA
    print0(f"num_layers: {num_layers}")
    print0(f"model_dim: {model_dim}")
    print0(f"num_heads: {num_heads}")
    print0(f"num_kv_heads: {num_kv_heads}")

    # Optimizer / data / training length related hyperparameters
    tokens_per_fwdbwd = device_batch_size * max_seq_len  # per rank
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
    print0(
        f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}"
    )
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(
        f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
    )

    # -----------------------------------------------------------------------------
    # Initialize the Model
    model_config_kwargs = dict(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim,
    )
    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    orig_model = model  # original, uncompiled model, for saving raw model state_dict

    if device_type == "xla":
        print0(
            "device_type=xla: skipping torch.compile (not supported on XLA in this setup)"
        )
    else:
        model = torch.compile(model, dynamic=False)

    num_params = sum(p.numel() for p in model.parameters())
    print0(f"Number of parameters: {num_params:,}")
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    # Calculate number of iterations
    assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0

    if num_iterations > 0:
        num_iters = num_iterations
        print0(f"Using user-provided number of iterations: {num_iters:,}")
    elif target_flops > 0:
        # calculate the number of iterations from the target flops
        num_iters = round(target_flops / (num_flops_per_token * total_batch_size))
        print0(f"Calculated number of iterations from target FLOPs: {num_iters:,}")
    elif target_param_data_ratio > 0:
        # calculate the number of iterations from the target param data ratio
        target_tokens = target_param_data_ratio * num_params
        num_iters = target_tokens // total_batch_size
        print0(
            f"Calculated number of iterations from target data:param ratio: {num_iters:,}"
        )
    else:
        raise ValueError("No training horizon specified")

    # keep the global in sync for logging / report at the end
    num_iterations = num_iters

    total_tokens = total_batch_size * num_iters
    print0(f"Total number of training tokens: {total_tokens:,}")
    print0(f"Tokens : Params ratio: {total_tokens / num_params:.2f}")  # Chinchilla ~20
    print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

    # -----------------------------------------------------------------------------
    # Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
    optimizers = model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )
    adamw_optimizer, muon_optimizer = optimizers

    if device_type == "xla":
        print0("device_type=xla: disabling fused optimizers (not supported on XLA)")
        for opt in optimizers:
            for group in opt.param_groups:
                if "fused" in group and group["fused"]:
                    group["fused"] = False

    # Initialize the DataLoaders for train/val
    base_dir = get_base_dir()
    tokens_dir = os.path.join(base_dir, "tokenized_data")
    train_loader = tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split="train", device=device
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split="val", device=device
    )
    x, y = next(train_loader)  # kick off load of the very first batch of data

    # -----------------------------------------------------------------------------
    # Schedulers
    def get_lr_multiplier(it):
        warmup_iters = round(warmup_ratio * num_iterations)
        warmdown_iters = round(warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac

    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        momentum = (1 - frac) * 0.85 + frac * 0.95
        return momentum

    # -----------------------------------------------------------------------------
    # Training loop
    min_val_bpb = float("inf")
    smooth_train_loss = 0.0
    ema_beta = 0.9
    total_training_time = 0.0
    results = {}

    for step in range(num_iterations + 1):
        last_step = step == num_iterations
        flops_so_far = num_flops_per_token * total_batch_size * step

        # Eval
        if last_step or step % eval_every == 0:
            model.eval()
            val_loader = build_val_loader()
            eval_steps = eval_tokens // (
                device_batch_size * max_seq_len * ddp_world_size
            )
            with autocast_ctx:
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "val/bpb": val_bpb,
                }
            )
            model.train()

        # CORE metric
        if core_metric_every > 0 and (
            last_step or (step > 0 and step % core_metric_every == 0)
        ):
            model.eval()
            with autocast_ctx:
                results = evaluate_model(
                    orig_model, tokenizer, device, max_per_task=core_metric_max_per_task
                )
            print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "core_metric": results["core_metric"],
                    "centered_results": results["centered_results"],
                }
            )
            model.train()

        # Sampling (skip on XLA)
        if master_process and (last_step or (step > 0 and step % sample_every == 0)):
            if device_type == "xla":
                print0(
                    "device_type=xla: skipping sampling (Engine.generate uses a CPU/GPU-only torch.Generator)"
                )
            else:
                model.eval()
                prompts = [
                    "The capital of France is",
                    "The chemical symbol of gold is",
                    "If yesterday was Friday, then tomorrow will be",
                    "The opposite of hot is",
                    "The planets of the solar system are:",
                    "My favorite color is",
                    "If 5*x + 3 = 13, then x is",
                ]
                engine = Engine(orig_model, tokenizer)
                for prompt in prompts:
                    tokens = tokenizer(prompt, prepend="<|bos|>")
                    with autocast_ctx:
                        sample, _ = engine.generate_batch(
                            tokens, num_samples=1, max_tokens=16, temperature=0
                        )
                    print0(tokenizer.decode(sample[0]))
                model.train()

        # Save checkpoint at the very end (master only)
        if master_process and last_step:
            output_dirname = model_tag if model_tag else f"d{depth}"  # e.g. d20
            checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                [opt.state_dict() for opt in optimizers],
                {
                    "step": step,
                    "val_bpb": val_bpb,
                    "model_config": model_config_kwargs,
                    "user_config": user_config,
                    "device_batch_size": device_batch_size,
                    "max_seq_len": max_seq_len,
                },
            )

        if last_step:
            break

        # ---------------------------------------------------------------------
        # Single training step
        synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y = next(train_loader)

        grad_clip_enabled = grad_clip > 0.0
        if grad_clip_enabled:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                orig_model.parameters(), grad_clip
            )
            grad_norm = grad_norm_tensor.item()
        lrm = get_lr_multiplier(step)

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        muon_momentum = get_muon_momentum(step)
        for group in muon_optimizer.param_groups:
            group["momentum"] = muon_momentum

        if device_type == "xla":
            for opt in optimizers:
                xm.optimizer_step(opt, barrier=True)
        else:
            for opt in optimizers:
                opt.step()
        model.zero_grad(set_to_none=True)
        synchronize()
        t1 = time.time()
        dt = t1 - t0

        # Logging
        smooth_train_loss = (
            ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
        )
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * step / num_iterations
        tok_per_sec = int(total_batch_size / dt)
        flops_per_sec = num_flops_per_token * total_batch_size / dt
        promised_flops_per_sec_h100 = 989e12 * ddp_world_size
        mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
        if step > 10:
            total_training_time += dt
        print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
        print0(
            f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
            f"loss: {debiased_smooth_loss:.6f} |{print_grad_norm} "
            f"lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | "
            f"mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m"
        )
        if step % 100 == 0 and master_process:
            log_data = {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "train/loss": debiased_smooth_loss,
                "train/lrm": lrm,
                "train/dt": dt,
                "train/tok_per_sec": tok_per_sec,
                "train/mfu": mfu,
            }
            if grad_clip_enabled:
                log_data["train/grad_norm"] = grad_norm
            wandb_run.log(log_data)

    # print a few more stats (master only)
    if master_process:
        print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
        print0(f"Total training time: {total_training_time/60:.2f}m")
        print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

        # Log to report
        from nanochat.report import get_report

        get_report().log(
            section="Base model training",
            data=[
                user_config,
                {
                    "Number of parameters": num_params,
                    "Number of FLOPs per token": f"{num_flops_per_token:e}",
                    "Calculated number of iterations": num_iterations,
                    "Number of training tokens": total_tokens,
                    "Tokens : Params ratio": total_batch_size
                    * num_iterations
                    / num_params,
                    "DDP world size": ddp_world_size,
                    "warmup_ratio": warmup_ratio,
                    "warmdown_ratio": warmdown_ratio,
                    "final_lr_frac": final_lr_frac,
                },
                {
                    "Minimum validation bpb": min_val_bpb,
                    "Final validation bpb": val_bpb,
                    "CORE metric estimate": results.get("core_metric", None),
                    "MFU %": f"{mfu:.2f}%",
                    "Total training flops": f"{flops_so_far:e}",
                    "Total training time": f"{total_training_time/60:.2f}m",
                    "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
                },
            ],
        )

    wandb_run.finish()
    if device_type != "xla":
        compute_cleanup()


if __name__ == "__main__":
    # Multi-core TPU via PJRT
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        # Use xmp.spawn directly, one process per TPU core
        import torch_xla.distributed.xla_multiprocessing as xmp

        nprocs = int(os.environ.get("TPU_NUM_DEVICES", "8"))
        xmp.spawn(_mp_train_fn, args=())

    else:
        # Single-process path (GPU/CPU/MPS)
        _mp_train_fn(0)

#    torch_xla.launch version
# if __name__ == "__main__":
#     if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
#         import torch_xla

#         # One process per TPU chip, all managed by PJRT.
#         torch_xla.launch(_mp_train_fn, args=())

#     else:
#         _mp_train_fn(0)

# # device_type may have been overridden from CLI via configurator
# device_type = autodetect_device_type() if device_type == "" else device_type

# if device_type == "xla" and xmp is not None:
#     # Use all 8 TPU cores on v6e-8
#     xmp.spawn(_mp_train_fn, args=(), nprocs=8)
# else:
#     # Normal (single-process or torchrun-DDP) path
#     _mp_train_fn(index=0)

# OLD VERSION
# # Compute init
# device_type = autodetect_device_type() if device_type == "" else device_type
# if device_type == "xla":
#     assert (
#         xm is not None
#     ), "torch_xla is not installed, but device_type='xla' was requested"

#     # Under xla_spawn, this gives you the global world size and rank.
#     ddp_world_size = xm.xrt_world_size()
#     ddp_rank = xm.get_ordinal()
#     ddp_local_rank = int(os.environ.get("LOCAL_RANK", ddp_rank))
#     ddp = ddp_world_size > 1

#     # Make sure any code that uses RANK/WORLD_SIZE (e.g. dataloader sharding) sees correct values.
#     os.environ["RANK"] = str(ddp_rank)
#     os.environ["WORLD_SIZE"] = str(ddp_world_size)

#     device = xm.xla_device()
# else:
#     # Original path: CUDA/MPS/CPU, handled by nanochat.common.compute_init
#     ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


# master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

# # Autocast / sync / memory helpers per device type
# if device_type == "cuda":
#     autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
#     synchronize = torch.cuda.synchronize
#     get_max_memory = torch.cuda.max_memory_allocated
# elif device_type == "xla":
#     # XLA/TPU: no torch.cuda, and bfloat16 is handled natively
#     from contextlib import nullcontext as _nullcontext  # shadow just to be explicit

#     autocast_ctx = _nullcontext()
#     synchronize = xm.mark_step
#     get_max_memory = (
#         lambda: 0
#     )  # nothing convenient like torch.cuda.max_memory_allocated

# else:
#     # CPU / MPS: no special sync/autocast
#     from contextlib import nullcontext as _nullcontext

#     autocast_ctx = _nullcontext()
#     synchronize = lambda: None
#     get_max_memory = lambda: 0

# # ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# # master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
# # autocast_ctx = (
# #     torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
# #     if device_type == "cuda"
# #     else nullcontext()
# # )
# # synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
# # get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# # wandb logging init
# use_dummy_wandb = run == "dummy" or not master_process
# wandb_run = (
#     DummyWandb()
#     if use_dummy_wandb
#     else wandb.init(project="nanochat", name=run, config=user_config)
# )

# # Tokenizer will be useful for evaluation, also we need the vocab size
# tokenizer = get_tokenizer()
# token_bytes = get_token_bytes(device=device)
# vocab_size = tokenizer.get_vocab_size()
# print0(f"Vocab size: {vocab_size:,}")

# # Model kwargs are derived from the desired depth of the model
# num_layers = depth
# model_dim = (
#     depth * 64
# )  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
# num_heads = max(
#     1, (model_dim + 127) // 128
# )  # head dim 128 (the division here is ceil div)
# num_kv_heads = (
#     num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
# )
# print0(f"num_layers: {num_layers}")
# print0(f"model_dim: {model_dim}")
# print0(f"num_heads: {num_heads}")
# print0(f"num_kv_heads: {num_kv_heads}")

# # Optimizer / data / training length related hyperparameters
# # figure out the needed gradient accumulation to reach the desired total batch size
# tokens_per_fwdbwd = (
#     device_batch_size * max_seq_len
# )  # tokens per iteration for a single rank
# world_tokens_per_fwdbwd = (
#     tokens_per_fwdbwd * ddp_world_size
# )  # total tokens per iteration for all ranks
# assert total_batch_size % world_tokens_per_fwdbwd == 0
# grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
# print0(
#     f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}"
# )
# print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
# print0(
#     f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
# )
# # -----------------------------------------------------------------------------
# # Initialize the Model
# model_config_kwargs = dict(
#     sequence_len=max_seq_len,
#     vocab_size=vocab_size,
#     n_layer=num_layers,
#     n_head=num_heads,
#     n_kv_head=num_kv_heads,
#     n_embd=model_dim,
# )
# with torch.device("meta"):
#     model_config = GPTConfig(**model_config_kwargs)
#     model = GPT(model_config)
# model.to_empty(device=device)
# model.init_weights()
# orig_model = model  # original, uncompiled model, for saving raw model state_dict
# # model = torch.compile(model, dynamic=False)  # TODO: dynamic True/False think through
# if device_type == "xla":
#     print0(
#         "device_type=xla: skipping torch.compile (not supported on XLA in this setup)"
#     )
# else:
#     model = torch.compile(
#         model, dynamic=False
#     )  # TODO: dynamic True/False think through

# num_params = sum(p.numel() for p in model.parameters())
# print0(f"Number of parameters: {num_params:,}")
# num_flops_per_token = model.estimate_flops()
# print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# # Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
# assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
# if num_iterations > 0:
#     print0(f"Using user-provided number of iterations: {num_iterations:,}")
# elif target_flops > 0:
#     # calculate the number of iterations from the target flops
#     num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
#     print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
# elif target_param_data_ratio > 0:
#     # calculate the number of iterations from the target param data ratio
#     target_tokens = target_param_data_ratio * num_params
#     num_iterations = target_tokens // total_batch_size
#     print0(
#         f"Calculated number of iterations from target data:param ratio: {num_iterations:,}"
#     )
# else:
#     raise ValueError("No training horizon specified")
# total_tokens = total_batch_size * num_iterations
# print0(f"Total number of training tokens: {total_tokens:,}")
# print0(
#     f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}"
# )  # Chinchilla is ~20
# print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# # -----------------------------------------------------------------------------
# # Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
# optimizers = model.setup_optimizers(
#     unembedding_lr=unembedding_lr,
#     embedding_lr=embedding_lr,
#     matrix_lr=matrix_lr,
#     weight_decay=weight_decay,
# )
# adamw_optimizer, muon_optimizer = optimizers
# # Disable fused optimizers on XLA (they are only supported on cuda/mps/cpu/etc.)
# if device_type == "xla":
#     from nanochat.common import print0

#     print0("device_type=xla: disabling fused optimizers (not supported on XLA)")
#     for opt in optimizers:
#         for group in opt.param_groups:
#             if "fused" in group and group["fused"]:
#                 group["fused"] = False

# # Initialize the DataLoaders for train/val
# base_dir = get_base_dir()
# tokens_dir = os.path.join(base_dir, "tokenized_data")
# train_loader = tokenizing_distributed_data_loader(
#     device_batch_size, max_seq_len, split="train", device=device
# )
# build_val_loader = lambda: tokenizing_distributed_data_loader(
#     device_batch_size, max_seq_len, split="val", device=device
# )
# x, y = next(train_loader)  # kick off load of the very first batch of data

# # -----------------------------------------------------------------------------
# # Set up hyperparameter schedulers


# # Learning rate scheduler
# def get_lr_multiplier(it):
#     warmup_iters = round(warmup_ratio * num_iterations)
#     warmdown_iters = round(warmdown_ratio * num_iterations)
#     if it < warmup_iters:
#         return (it + 1) / warmup_iters
#     elif it <= num_iterations - warmdown_iters:
#         return 1.0
#     else:
#         progress = (num_iterations - it) / warmdown_iters
#         return progress * 1.0 + (1 - progress) * final_lr_frac


# # Momentum scheduler for Muon optimizer
# def get_muon_momentum(it):
#     frac = min(it / 300, 1)
#     momentum = (1 - frac) * 0.85 + frac * 0.95
#     return momentum


# # -----------------------------------------------------------------------------
# # Training loop
# min_val_bpb = float("inf")
# smooth_train_loss = 0  # EMA of training loss
# ema_beta = 0.9  # EMA decay factor
# total_training_time = 0  # total wall-clock time of training
# # note that we run +1 steps only so that we can eval and save at the end
# for step in range(num_iterations + 1):
#     last_step = step == num_iterations
#     flops_so_far = num_flops_per_token * total_batch_size * step

#     # once in a while: evaluate the val bpb (all ranks participate)
#     if last_step or step % eval_every == 0:
#         model.eval()
#         val_loader = build_val_loader()
#         eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
#         with autocast_ctx:
#             val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
#         print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
#         if val_bpb < min_val_bpb:
#             min_val_bpb = val_bpb
#         wandb_run.log(
#             {
#                 "step": step,
#                 "total_training_flops": flops_so_far,
#                 "total_training_time": total_training_time,
#                 "val/bpb": val_bpb,
#             }
#         )
#         model.train()

#     # once in a while: estimate the CORE metric (all ranks participate)
#     # use the original uncompiled model because the inputs keep changing shape
#     results = {}
#     if core_metric_every > 0 and (
#         last_step or (step > 0 and step % core_metric_every == 0)
#     ):
#         model.eval()
#         with autocast_ctx:
#             results = evaluate_model(
#                 orig_model, tokenizer, device, max_per_task=core_metric_max_per_task
#             )
#         print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
#         wandb_run.log(
#             {
#                 "step": step,
#                 "total_training_flops": flops_so_far,
#                 "core_metric": results["core_metric"],
#                 "centered_results": results["centered_results"],
#             }
#         )
#         model.train()

#     # once in a while: sample from the model (only on master process)
#     # use the original uncompiled model because the inputs keep changing shape
#     if master_process and (last_step or (step > 0 and step % sample_every == 0)):
#         if device_type == "xla":
#             # Engine.generate uses torch.Generator(device=...), which does not support XLA.
#             # Sampling is just for human inspection; we can skip it on TPU.
#             print0(
#                 "device_type=xla: skipping sampling (Engine.generate uses a CPU/GPU-only torch.Generator)"
#             )
#         else:
#             model.eval()
#             prompts = [
#                 "The capital of France is",
#                 "The chemical symbol of gold is",
#                 "If yesterday was Friday, then tomorrow will be",
#                 "The opposite of hot is",
#                 "The planets of the solar system are:",
#                 "My favorite color is",
#                 "If 5*x + 3 = 13, then x is",
#             ]
#             engine = Engine(
#                 orig_model, tokenizer
#             )  # use orig_model to avoid recompilation
#             for prompt in prompts:
#                 tokens = tokenizer(prompt, prepend="<|bos|>")
#                 with autocast_ctx:
#                     sample, _ = engine.generate_batch(
#                         tokens, num_samples=1, max_tokens=16, temperature=0
#                     )
#                 print0(tokenizer.decode(sample[0]))
#             model.train()

#     # save checkpoint at the end of the run (only on master process)
#     if master_process and last_step:
#         output_dirname = model_tag if model_tag else f"d{depth}"  # e.g. d12
#         checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
#         save_checkpoint(
#             checkpoint_dir,
#             step,
#             orig_model.state_dict(),
#             [
#                 opt.state_dict() for opt in optimizers
#             ],  # TODO: make sure saving across ranks is done correctly
#             {
#                 "step": step,
#                 "val_bpb": val_bpb,  # loss at last step
#                 "model_config": model_config_kwargs,
#                 "user_config": user_config,  # inputs to the training script
#                 "device_batch_size": device_batch_size,
#                 "max_seq_len": max_seq_len,
#             },
#         )

#     if last_step:
#         break

#     # -------------------------------------------------------------------------
#     # single training step
#     # evaluate the gradient
#     synchronize()
#     t0 = time.time()
#     for micro_step in range(grad_accum_steps):
#         with autocast_ctx:
#             loss = model(x, y)
#         train_loss = loss.detach()  # for logging
#         loss = (
#             loss / grad_accum_steps
#         )  # each .backward() is a grad sum => normalize loss here
#         loss.backward()
#         x, y = next(
#             train_loader
#         )  # prefetch the next batch while the GPU is busy with forward/backward
#     # gradient clipping
#     grad_clip_enabled = grad_clip > 0.0
#     if grad_clip_enabled:
#         grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
#             orig_model.parameters(), grad_clip
#         )
#         grad_norm = (
#             grad_norm_tensor.item()
#         )  # GPU tensor -> CPU float (note: cpu-gpu sync point)
#     # step the optimizers
#     lrm = get_lr_multiplier(step)

#     for opt in optimizers:
#         for group in opt.param_groups:
#             group["lr"] = group["initial_lr"] * lrm
#     muon_momentum = get_muon_momentum(step)
#     for group in muon_optimizer.param_groups:
#         group["momentum"] = muon_momentum

#     if device_type == "xla":
#         # XLA-friendly optimizer step
#         for opt in optimizers:
#             xm.optimizer_step(opt, barrier=True)
#     else:
#         for opt in optimizers:
#             opt.step()
#     model.zero_grad(set_to_none=True)
#     synchronize()
#     t1 = time.time()
#     dt = t1 - t0
#     # -------------------------------------------------------------------------

#     # logging
#     smooth_train_loss = (
#         ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
#     )  # EMA the training loss
#     debiased_smooth_loss = smooth_train_loss / (
#         1 - ema_beta ** (step + 1)
#     )  # debias the EMA
#     pct_done = 100 * step / num_iterations
#     tok_per_sec = int(total_batch_size / dt)
#     flops_per_sec = num_flops_per_token * total_batch_size / dt
#     promised_flops_per_sec_h100 = (
#         989e12 * ddp_world_size
#     )  # bfloat16 H100 SXM and without 2:4 sparsity
#     mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # in %
#     if step > 10:
#         total_training_time += dt  # only count the time after the first 10 steps
#     print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
#     print0(
#         f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m"
#     )
#     if step % 100 == 0:
#         log_data = {
#             "step": step,
#             "total_training_flops": flops_so_far,
#             "total_training_time": total_training_time,
#             "train/loss": debiased_smooth_loss,
#             "train/lrm": lrm,
#             "train/dt": dt,
#             "train/tok_per_sec": tok_per_sec,
#             "train/mfu": mfu,
#         }
#         if grad_clip_enabled:
#             log_data["train/grad_norm"] = grad_norm
#         wandb_run.log(log_data)

# # print a few more stats
# print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
# print0(f"Total training time: {total_training_time/60:.2f}m")
# print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# # Log to report
# from nanochat.report import get_report

# get_report().log(
#     section="Base model training",
#     data=[
#         user_config,  # CLI args
#         {  # stats about the training setup
#             "Number of parameters": num_params,
#             "Number of FLOPs per token": f"{num_flops_per_token:e}",
#             "Calculated number of iterations": num_iterations,
#             "Number of training tokens": total_tokens,
#             "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
#             "DDP world size": ddp_world_size,
#             "warmup_ratio": warmup_ratio,
#             "warmdown_ratio": warmdown_ratio,
#             "final_lr_frac": final_lr_frac,
#         },
#         {  # stats about training outcomes
#             "Minimum validation bpb": min_val_bpb,
#             "Final validation bpb": val_bpb,
#             "CORE metric estimate": results.get("core_metric", None),
#             "MFU %": f"{mfu:.2f}%",
#             "Total training flops": f"{flops_so_far:e}",
#             "Total training time": f"{total_training_time/60:.2f}m",
#             "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
#         },
#     ],
# )

# # cleanup
# wandb_run.finish()  # wandb run finish
# if device_type != "xla":
#     compute_cleanup()
