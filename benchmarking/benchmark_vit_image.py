"""
Estimates full ViT forward/backward time per attention variant by profiling
each layer individually and summing with the layer-count multipliers.
This avoids running a full ViT end-to-end backward pass, which hangs at large sizes.
"""
import os
import json
import gc
import numpy as np
import matplotlib.pyplot as plt

import minitorch
from minitorch import bench_utils
from minitorch.transformer import MultiHeadAttention, FeedForward, Patchify
from minitorch.modules_basic import Linear, LayerNorm1d, Embedding
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_ops import TensorBackend
from minitorch.tensor_functions import tensor_from_numpy

backend  = TensorBackend(CudaKernelOps)
datatype = np.float32

IMAGE_SIZES    = [128, 256, 512, 1024, 2048]
PATCH_SIZE     = 16
N_EMBD         = 64
N_HEAD         = 1
N_TRANS_LAYERS = 2
N_CLASSES      = 10
N_CHANNELS     = 3
BATCH_SIZE     = 4
N_ITERS        = 5
N_WARMUP       = 2

ATTN_TYPES  = ["standard", "fa1", "fa2"]
ATTN_COLORS = {"standard": "#D62728", "fa1": "#38A8A8", "fa2": "#5B2C8D"}
ATTN_LABELS = {"standard": "Standard", "fa1": "FA1", "fa2": "FA2"}

# How many times each layer appears in a ViT with N_TRANS_LAYERS blocks
LAYER_MULTS = {
    "Patchify":    1,
    "PatchProj":   1,
    "PosEmbed":    1,
    "LayerNorm":   N_TRANS_LAYERS * 2,
    "FeedForward": N_TRANS_LAYERS,
    "ClassHead":   1,
    "Attn":        N_TRANS_LAYERS,
}


def profile_image_size(img_size):
    N           = (img_size // PATCH_SIZE) ** 2
    num_patches = N + 1
    patch_dim   = N_CHANNELS * PATCH_SIZE * PATCH_SIZE

    shared_layers = {
        "Patchify": (
            Patchify(PATCH_SIZE),
            lambda: tensor_from_numpy(
                np.random.randn(BATCH_SIZE, N_CHANNELS, img_size, img_size).astype(datatype),
                backend=backend,
            ),
        ),
        "PatchProj": (
            Linear(patch_dim, N_EMBD, False, backend),
            lambda: tensor_from_numpy(
                np.random.randn(BATCH_SIZE * N, patch_dim).astype(datatype),
                backend=backend,
            ),
        ),
        "PosEmbed": (
            Embedding(num_patches, N_EMBD, backend),
            lambda: minitorch.tensor(
                [list(range(num_patches))], backend=backend, requires_grad=False,
            ),
        ),
        "LayerNorm": (
            LayerNorm1d(N_EMBD, 1e-5, backend),
            lambda: tensor_from_numpy(
                np.random.randn(BATCH_SIZE * num_patches, N_EMBD).astype(datatype),
                backend=backend,
            ),
        ),
        "FeedForward": (
            FeedForward(N_EMBD, 4 * N_EMBD, p_dropout=0.0, bias=False, backend=backend),
            lambda: tensor_from_numpy(
                np.random.randn(BATCH_SIZE, num_patches, N_EMBD).astype(datatype),
                backend=backend,
            ),
        ),
        "ClassHead": (
            Linear(N_EMBD, N_CLASSES, False, backend),
            lambda: tensor_from_numpy(
                np.random.randn(BATCH_SIZE, N_EMBD).astype(datatype),
                backend=backend,
            ),
        ),
    }

    attn_input_fn = lambda: tensor_from_numpy(
        np.random.randn(BATCH_SIZE, num_patches, N_EMBD).astype(datatype),
        backend=backend,
    )

    timings = {}  # layer_name -> {"fwd": ms, "bwd": ms, "fwd_mem": MB, "bwd_mem": MB}

    for name, (module, input_fn) in shared_layers.items():
        print(f"    {name}...", end=" ", flush=True)
        res = bench_utils.benchmark_module(module, input_fn, n_iters=N_ITERS, n_warmup=N_WARMUP)
        timings[name] = {
            "fwd": res["fwd_time_ms"],
            "bwd": res["bwd_time_ms"],
            "fwd_mem": res["fwd_peak_mem_mb"],
            "bwd_mem": res["bwd_peak_mem_mb"],
        }
        print(f"fwd={res['fwd_time_ms']:.1f}ms  bwd={res['bwd_time_ms']:.1f}ms")
        del module
        gc.collect()

    attn_timings = {}
    for attn_type in ATTN_TYPES:
        print(f"    Attn ({attn_type})...", end=" ", flush=True)
        layer = MultiHeadAttention(
            N_EMBD, N_HEAD, causal=False, p_dropout=0.0,
            bias=False, backend=backend, attn_type=attn_type,
        )
        res = bench_utils.benchmark_module(layer, attn_input_fn, n_iters=N_ITERS, n_warmup=N_WARMUP)
        attn_timings[attn_type] = {
            "fwd": res["fwd_time_ms"],
            "bwd": res["bwd_time_ms"],
            "fwd_mem": res["fwd_peak_mem_mb"],
            "bwd_mem": res["bwd_peak_mem_mb"],
        }
        print(f"fwd={res['fwd_time_ms']:.1f}ms  bwd={res['bwd_time_ms']:.1f}ms")
        del layer
        gc.collect()

    return timings, attn_timings


def estimate_vit(timings, attn_timings, attn_type, metric):
    total = sum(timings[l][metric] * LAYER_MULTS[l] for l in timings)
    total += attn_timings[attn_type][metric] * LAYER_MULTS["Attn"]
    return total


def run_benchmarks():
    all_timings      = {}  # img_size -> shared layer timings
    all_attn_timings = {}  # img_size -> {attn_type -> timings}

    for img_size in IMAGE_SIZES:
        N = (img_size // PATCH_SIZE) ** 2
        print(f"\nImage {img_size}x{img_size}  (N={N})")
        t, at = profile_image_size(img_size)
        all_timings[img_size]      = t
        all_attn_timings[img_size] = at

    # Build results dict
    results = {attn: {"img": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []}
               for attn in ATTN_TYPES}
    for img_size in IMAGE_SIZES:
        for attn_type in ATTN_TYPES:
            results[attn_type]["img"].append(img_size)
            for metric, key in [("fwd", "fwd_time"), ("bwd", "bwd_time"),
                                 ("fwd_mem", "fwd_mem"), ("bwd_mem", "bwd_mem")]:
                results[attn_type][key].append(
                    estimate_vit(all_timings[img_size], all_attn_timings[img_size], attn_type, metric)
                )

    # Save JSON
    os.makedirs("benchmarking/layers", exist_ok=True)
    log_path = "benchmarking/layers/vit_image_benchmark.json"
    with open(log_path, "w") as f:
        json.dump({
            "results": results,
            "layer_timings": {str(s): {k: v for k, v in t.items()} for s, t in all_timings.items()},
            "attn_timings":  {str(s): at for s, at in all_attn_timings.items()},
        }, f, indent=2)
    print(f"\nResults saved to {log_path}")

    # Plot
    xlabels = [f"{s}×{s}\n(N={(s//PATCH_SIZE)**2})" for s in IMAGE_SIZES]
    x = range(len(IMAGE_SIZES))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Estimated ViT Fwd & Bwd by Image Resolution  "
        f"(patch={PATCH_SIZE}, B={BATCH_SIZE}, d={N_EMBD}, layers={N_TRANS_LAYERS})  "
        f"[per-layer estimate]"
    )

    metrics = [
        ("fwd_time", "Forward Time (ms)",  "Estimated Total Fwd Time",  axes[0, 0]),
        ("bwd_time", "Backward Time (ms)", "Estimated Total Bwd Time",  axes[0, 1]),
        ("fwd_mem",  "Peak Memory (MB)",   "Estimated Total Fwd Memory", axes[1, 0]),
        ("bwd_mem",  "Peak Memory (MB)",   "Estimated Total Bwd Memory", axes[1, 1]),
    ]

    for key, ylabel, title, ax in metrics:
        for attn_type in ATTN_TYPES:
            ax.plot(x, results[attn_type][key],
                    label=ATTN_LABELS[attn_type],
                    marker="o",
                    color=ATTN_COLORS[attn_type])
        ax.set_title(title)
        ax.set_xlabel("Image Resolution")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "benchmarking/layers/vit_image_benchmark.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    run_benchmarks()
