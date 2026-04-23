"""
Timing benchmark: vanilla ViT vs Flash Attention ViT.

Measures forward and backward wall-clock time across a range of sequence
lengths (controlled by image size).  Memory benchmarking is deliberately
omitted because the pynvml poller does not work reliably in this environment.

Usage (from repo root, on a GPU node):
    python benchmarking/benchmark_flash_vs_vanilla.py

Output:
    - printed table to stdout
    - benchmarking/results/flash_vs_vanilla_time.png
"""

import matplotlib
matplotlib.use("Agg")

import sys, os, gc, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_ops import TensorBackend
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.transformer import ViT
from minitorch.flash_transformer import FlashViT

import pycuda.autoinit  # sets up CUDA context; needed for .synchronize()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BACKEND       = TensorBackend(CudaKernelOps)
BATCH_SIZE    = 2
N_EMBD        = 64     # keep small so sequence-length scaling is the bottleneck
N_HEAD        = 4      # d = N_EMBD // N_HEAD = 16  (<= FA_D_MAX=64)
PATCH_SIZE    = 16
N_CHANNELS    = 3
N_CLASSES     = 10
N_TRANS_LAYERS = 1     # single layer to isolate attention cost
N_WARMUP      = 3      # throw-away iterations (JIT warm-up)
N_ITERS       = 5      # timed iterations

# Image sizes → number of patches (sequence lengths to sweep over)
IMG_SIZES = [64, 128, 256]      # patches: 16, 64, 256
# Uncomment to go larger (very slow for vanilla):
# IMG_SIZES = [64, 128, 256, 512]


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper (time only, no memory)
# ─────────────────────────────────────────────────────────────────────────────
def _sync():
    """Synchronise the CUDA device so perf_counter captures true GPU time."""
    pycuda.autoinit.context.synchronize()


def time_forward_backward(model, input_fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    """
    Returns (fwd_ms, bwd_ms) — average over n_iters timed iterations.
    Runs n_warmup un-timed iterations first so Numba / CUDA JIT is warm.
    """
    inp = input_fn()

    # Warm-up
    for _ in range(n_warmup):
        out = model(inp)
        out.sum().backward()

    fwd_times, bwd_times = [], []

    for _ in range(n_iters):
        inp = input_fn()   # fresh input each iteration

        # Forward
        _sync()
        t0 = time.perf_counter()
        out = model(inp)
        _sync()
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)

        # Backward
        loss = out.sum()
        _sync()
        t0 = time.perf_counter()
        loss.backward()
        _sync()
        t1 = time.perf_counter()
        bwd_times.append(t1 - t0)

    fwd_ms = sum(fwd_times) / len(fwd_times) * 1000
    bwd_ms = sum(bwd_times) / len(bwd_times) * 1000
    return fwd_ms, bwd_ms


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────
def run():
    os.makedirs("benchmarking/results", exist_ok=True)

    results = {
        "vanilla": {"n_patches": [], "fwd_ms": [], "bwd_ms": []},
        "flash":   {"n_patches": [], "fwd_ms": [], "bwd_ms": []},
    }

    header = f"{'img':>6}  {'patches':>8}  {'vanilla_fwd':>12}  {'vanilla_bwd':>12}  {'flash_fwd':>10}  {'flash_bwd':>10}  {'speedup_fwd':>12}"
    print("\n" + "=" * len(header))
    print(f"  Vanilla ViT vs Flash ViT  —  time benchmark")
    print(f"  n_embd={N_EMBD}, n_head={N_HEAD}, d={N_EMBD//N_HEAD}, "
          f"batch={BATCH_SIZE}, patch={PATCH_SIZE}, layers={N_TRANS_LAYERS}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for img_sz in IMG_SIZES:
        n_patches = (img_sz // PATCH_SIZE) ** 2
        max_patches = n_patches + 1  # +1 for CLS token

        # ── Vanilla ViT ──────────────────────────────────────────────────────
        vanilla = ViT(
            n_embd=N_EMBD, n_head=N_HEAD, p_dropout=0.0,
            patch_size=PATCH_SIZE, n_trans_layers=N_TRANS_LAYERS,
            n_classes=N_CLASSES, max_patches=max_patches,
            n_channels=N_CHANNELS, backend=BACKEND,
        )
        vanilla.train()

        input_fn = lambda sz=img_sz: tensor_from_numpy(
            np.random.randn(BATCH_SIZE, N_CHANNELS, sz, sz).astype(np.float32),
            backend=BACKEND,
        )

        v_fwd, v_bwd = time_forward_backward(vanilla, input_fn)
        results["vanilla"]["n_patches"].append(n_patches)
        results["vanilla"]["fwd_ms"].append(v_fwd)
        results["vanilla"]["bwd_ms"].append(v_bwd)
        del vanilla; gc.collect()

        # ── Flash ViT ─────────────────────────────────────────────────────────
        flash = FlashViT(
            n_embd=N_EMBD, n_head=N_HEAD, p_dropout=0.0,
            patch_size=PATCH_SIZE, n_trans_layers=N_TRANS_LAYERS,
            n_classes=N_CLASSES, max_patches=max_patches,
            n_channels=N_CHANNELS, backend=BACKEND,
        )
        flash.train()

        f_fwd, f_bwd = time_forward_backward(flash, input_fn)
        results["flash"]["n_patches"].append(n_patches)
        results["flash"]["fwd_ms"].append(f_fwd)
        results["flash"]["bwd_ms"].append(f_bwd)
        del flash; gc.collect()

        speedup = v_fwd / f_fwd if f_fwd > 0 else float("nan")
        print(
            f"{img_sz:>6}  {n_patches:>8}  "
            f"{v_fwd:>10.1f}ms  {v_bwd:>10.1f}ms  "
            f"{f_fwd:>8.1f}ms  {f_bwd:>8.1f}ms  "
            f"{speedup:>10.2f}x"
        )

    print("-" * len(header))

    # ─────────────────────────────────────────────────────────────────────────
    # Plot
    # ─────────────────────────────────────────────────────────────────────────
    n_patches_v = results["vanilla"]["n_patches"]
    n_patches_f = results["flash"]["n_patches"]
    x = range(len(n_patches_v))
    labels = [str(n) for n in n_patches_v]

    fig, (ax_fwd, ax_bwd) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Vanilla ViT vs Flash ViT — Forward & Backward Time\n"
        f"n_embd={N_EMBD}, n_head={N_HEAD}, d={N_EMBD//N_HEAD}, "
        f"batch={BATCH_SIZE}, patch={PATCH_SIZE}, layers={N_TRANS_LAYERS}"
    )

    for ax, v_times, f_times, title in [
        (ax_fwd, results["vanilla"]["fwd_ms"], results["flash"]["fwd_ms"], "Forward Time (ms)"),
        (ax_bwd, results["vanilla"]["bwd_ms"], results["flash"]["bwd_ms"], "Backward Time (ms)"),
    ]:
        ax.plot(x, v_times, marker="o", label="Vanilla ViT", color="steelblue")
        ax.plot(x, f_times, marker="s", label="Flash ViT",   color="darkorange")
        ax.set_title(title)
        ax.set_xlabel("Sequence length (# patches)")
        ax.set_ylabel("Time (ms)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = "benchmarking/results/flash_vs_vanilla_time.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    run()
