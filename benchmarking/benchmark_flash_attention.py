"""
Benchmark: Standard Attention vs Flash Attention (layer-level + kernel-level)

Layer-level: sweeps sequence length N through MultiHeadAttention with
  use_flash_attn=False vs True, reporting fwd/bwd time and peak GPU memory.

Kernel-level: directly times the raw kernels:
  Standard fwd, FA1 forward, FA2 forward,
  Standard bwd (fwd+bwd cycle), FA1 backward, FA2 backward.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import minitorch
from minitorch import bench_utils
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_ops import TensorBackend
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.nn import softmax as minitorch_softmax
import pycuda.autoinit

backend = TensorBackend(CudaKernelOps)
datatype = np.float32

BATCH      = 4
N_EMBD     = 64   # d = n_embd / n_head = 64
N_HEAD     = 1
SEQ_LENS   = [64, 128] #256, 512, 1024, 2048, 4096]
N_ITERS    = 5
N_WARMUP   = 2


# ─────────────────────────────────────────────────────────────────────────────
# Layer-level benchmark (Standard vs Flash MultiHeadAttention)
# ─────────────────────────────────────────────────────────────────────────────

def make_layers(n_embd, n_head):
    std_layer = minitorch.MultiHeadAttention(
        n_embd, n_head, causal=False, p_dropout=0.0,
        bias=False, backend=backend, use_flash_attn=False,
    )
    flash_layer = minitorch.MultiHeadAttention(
        n_embd, n_head, causal=False, p_dropout=0.0,
        bias=False, backend=backend, use_flash_attn=True,
    )
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n_embd, n_embd)).astype(datatype)
    for layer in (std_layer, flash_layer):
        for proj in (layer.q_projection, layer.k_projection,
                     layer.v_projection, layer.out_projection):
            t = tensor_from_numpy(w.copy(), backend=backend)
            t.requires_grad_(True)
            proj.weights.value = t
    return std_layer, flash_layer


def run_layer_benchmarks():
    results = {
        "std":   {"N": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []},
        "flash": {"N": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []},
    }

    for N in SEQ_LENS:
        print(f"\n[layer] N={N}")
        std_layer, flash_layer = make_layers(N_EMBD, N_HEAD)

        input_fn = lambda: tensor_from_numpy(
            np.random.randn(BATCH, N, N_EMBD).astype(datatype),
            backend=backend,
        )

        for tag, layer in (("std", std_layer), ("flash", flash_layer)):
            print(f"  [{tag}]", end="", flush=True)
            res = bench_utils.benchmark_module(layer, input_fn, n_iters=N_ITERS, n_warmup=N_WARMUP)
            results[tag]["N"].append(N)
            results[tag]["fwd_time"].append(res["fwd_time_ms"])
            results[tag]["bwd_time"].append(res["bwd_time_ms"])
            results[tag]["fwd_mem"].append(res["fwd_peak_mem_mb"])
            results[tag]["bwd_mem"].append(res["bwd_peak_mem_mb"])
            print(f"  fwd={res['fwd_time_ms']:.2f}ms  bwd={res['bwd_time_ms']:.2f}ms"
                  f"  fwd_mem={res['fwd_peak_mem_mb']:.1f}MB  bwd_mem={res['bwd_peak_mem_mb']:.1f}MB")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Kernel-level benchmark (FA1 vs FA2 forward and backward)
# ─────────────────────────────────────────────────────────────────────────────

def _time_kernel(fn, n_iters, n_warmup):
    """Run fn() n_warmup times, then time n_iters calls. Returns (ms, peak_mb)."""
    for _ in range(n_warmup):
        fn()

    times = []
    peak_mbs = []
    for _ in range(n_iters):
        pycuda.autoinit.context.synchronize()
        poller = bench_utils.NvmlMemPoller()
        poller.start()
        t0 = time.perf_counter()
        fn()
        pycuda.autoinit.context.synchronize()
        t1 = time.perf_counter()
        poller.stop()
        times.append(t1 - t0)
        peak_mbs.append(poller.peak_mb)

    return sum(times) / len(times) * 1000, max(peak_mbs)


def run_kernel_benchmarks():
    # BH = BATCH * N_HEAD = 4
    BH = BATCH * N_HEAD

    # Build a standard (non-flash) attention layer for std_fwd / std_bwd timing.
    # We call layer.self_attention() directly so projections are excluded and
    # only the core softmax(Q @ K^T) @ V compute is measured.
    std_layer, _ = make_layers(N_EMBD, N_HEAD)

    results = {
        "std_fwd": {"N": [], "time": [], "mem": []},
        "fa1_fwd": {"N": [], "time": [], "mem": []},
        "fa2_fwd": {"N": [], "time": [], "mem": []},
        "std_bwd": {"N": [], "time": [], "mem": []},
        "fa1_bwd": {"N": [], "time": [], "mem": []},
        "fa2_bwd": {"N": [], "time": [], "mem": []},
    }

    rng = np.random.default_rng(42)

    for N in SEQ_LENS:
        print(f"\n[kernel] N={N}")
        scale = 64 ** -0.5

        q_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
        k_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
        v_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
        do_np = rng.standard_normal((BH, N, 64)).astype(datatype)

        q  = tensor_from_numpy(q_np,  backend=backend)
        k  = tensor_from_numpy(k_np,  backend=backend)
        v  = tensor_from_numpy(v_np,  backend=backend)
        do = tensor_from_numpy(do_np, backend=backend)

        # Reshape (BH, N, 64) → (BATCH, N_HEAD, N, 64) for self_attention()
        q_4d  = q.view(BATCH, N_HEAD, N, 64)
        kT_4d = k.view(BATCH, N_HEAD, N, 64).permute(0, 1, 3, 2)  # (B, H, 64, N)
        v_4d  = v.view(BATCH, N_HEAD, N, 64)

        # Run FA2 forward once to get O and L for flash backward benchmarks
        o, L = CudaKernelOps.flash_attention_forward_fa2(q, k, v, scale)

        # Standard forward (uses layer.self_attention — no projections)
        ms, mb = _time_kernel(
            lambda: std_layer.self_attention(q_4d, kT_4d, v_4d),
            N_ITERS, N_WARMUP,
        )
        results["std_fwd"]["N"].append(N)
        results["std_fwd"]["time"].append(ms)
        results["std_fwd"]["mem"].append(mb)
        print(f"  [std_fwd]  {ms:.2f}ms  {mb:.1f}MB")

        # FA1 forward
        ms, mb = _time_kernel(
            lambda: CudaKernelOps.flash_attention_forward_fa1(q, k, v, scale),
            N_ITERS, N_WARMUP,
        )
        results["fa1_fwd"]["N"].append(N)
        results["fa1_fwd"]["time"].append(ms)
        results["fa1_fwd"]["mem"].append(mb)
        print(f"  [fa1_fwd]  {ms:.2f}ms  {mb:.1f}MB")

        # FA2 forward
        ms, mb = _time_kernel(
            lambda: CudaKernelOps.flash_attention_forward_fa2(q, k, v, scale),
            N_ITERS, N_WARMUP,
        )
        results["fa2_fwd"]["N"].append(N)
        results["fa2_fwd"]["time"].append(ms)
        results["fa2_fwd"]["mem"].append(mb)
        print(f"  [fa2_fwd]  {ms:.2f}ms  {mb:.1f}MB")

        # Standard backward — fwd+bwd cycle (graph rebuilt each call)
        def _std_bwd_cycle():
            q_g  = tensor_from_numpy(q_np,  backend=backend)
            k_g  = tensor_from_numpy(k_np,  backend=backend)
            v_g  = tensor_from_numpy(v_np,  backend=backend)
            q_g.requires_grad_(True); k_g.requires_grad_(True); v_g.requires_grad_(True)
            q_g4  = q_g.view(BATCH, N_HEAD, N, 64)
            kT_g4 = k_g.view(BATCH, N_HEAD, N, 64).permute(0, 1, 3, 2)
            v_g4  = v_g.view(BATCH, N_HEAD, N, 64)
            out = std_layer.self_attention(q_g4, kT_g4, v_g4)
            do_t = tensor_from_numpy(do_np, backend=backend)
            (out * do_t).sum().backward()

        ms, mb = _time_kernel(_std_bwd_cycle, N_ITERS, N_WARMUP)
        results["std_bwd"]["N"].append(N)
        results["std_bwd"]["time"].append(ms)
        results["std_bwd"]["mem"].append(mb)
        print(f"  [std_bwd]  {ms:.2f}ms  {mb:.1f}MB  (fwd+bwd cycle)")

        # FA1 backward
        ms, mb = _time_kernel(
            lambda: CudaKernelOps.flash_attention_backward_fa1(q, k, v, o, do, L, scale),
            N_ITERS, N_WARMUP,
        )
        results["fa1_bwd"]["N"].append(N)
        results["fa1_bwd"]["time"].append(ms)
        results["fa1_bwd"]["mem"].append(mb)
        print(f"  [fa1_bwd]  {ms:.2f}ms  {mb:.1f}MB")

        # FA2 backward
        ms, mb = _time_kernel(
            lambda: CudaKernelOps.flash_attention_backward_fa2(q, k, v, o, do, L, scale),
            N_ITERS, N_WARMUP,
        )
        results["fa2_bwd"]["N"].append(N)
        results["fa2_bwd"]["time"].append(ms)
        results["fa2_bwd"]["mem"].append(mb)
        print(f"  [fa2_bwd]  {ms:.2f}ms  {mb:.1f}MB")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_layer(results):
    os.makedirs("benchmarking/layers", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Standard vs Flash Attention  (B={BATCH}, d={N_EMBD}, n_head={N_HEAD})")

    std   = results["std"]
    flash = results["flash"]

    axes[0].plot(std["N"],   std["fwd_time"],   marker="o", label="Standard fwd")
    axes[0].plot(flash["N"], flash["fwd_time"], marker="s", label="Flash fwd")
    axes[0].set_title("Forward Time")
    axes[0].set_xlabel("Sequence Length N")
    axes[0].set_ylabel("Time (ms)")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.7)

    axes[1].plot(std["N"],   std["bwd_time"],   marker="o", label="Standard bwd")
    axes[1].plot(flash["N"], flash["bwd_time"], marker="s", label="Flash bwd")
    axes[1].set_title("Backward Time")
    axes[1].set_xlabel("Sequence Length N")
    axes[1].set_ylabel("Time (ms)")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.7)

    axes[2].plot(std["N"],   std["fwd_mem"],   marker="o", linestyle="-",  label="Standard fwd mem")
    axes[2].plot(std["N"],   std["bwd_mem"],   marker="o", linestyle="--", label="Standard bwd mem")
    axes[2].plot(flash["N"], flash["fwd_mem"], marker="s", linestyle="-",  label="Flash fwd mem")
    axes[2].plot(flash["N"], flash["bwd_mem"], marker="s", linestyle="--", label="Flash bwd mem")
    axes[2].set_title("Peak GPU Memory")
    axes[2].set_xlabel("Sequence Length N")
    axes[2].set_ylabel("Memory (MB)")
    axes[2].legend(); axes[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = "benchmarking/layers/flash_attention_benchmark.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"\nLayer plot saved to {out_path}")


def _plot_kernels(results):
    os.makedirs("benchmarking/layers", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Standard vs FA1 vs FA2 Kernel Benchmarks  (BH={BATCH * N_HEAD}, d=64)")

    Ns = results["fa1_fwd"]["N"]

    # Forward time
    axes[0].plot(Ns, results["std_fwd"]["time"], marker="^", label="Std fwd")
    axes[0].plot(Ns, results["fa1_fwd"]["time"], marker="o", label="FA1 fwd")
    axes[0].plot(Ns, results["fa2_fwd"]["time"], marker="s", label="FA2 fwd")
    axes[0].set_title("Forward Time")
    axes[0].set_xlabel("Sequence Length N")
    axes[0].set_ylabel("Time (ms)")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.7)

    # Backward time
    axes[1].plot(Ns, results["std_bwd"]["time"], marker="^", label="Std bwd (fwd+bwd)")
    axes[1].plot(Ns, results["fa1_bwd"]["time"], marker="o", label="FA1 bwd (atomicAdd)")
    axes[1].plot(Ns, results["fa2_bwd"]["time"], marker="s", label="FA2 bwd (two-kernel)")
    axes[1].set_title("Backward Time")
    axes[1].set_xlabel("Sequence Length N")
    axes[1].set_ylabel("Time (ms)")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.7)

    # Peak memory — all kernels
    axes[2].plot(Ns, results["std_fwd"]["mem"], marker="^", linestyle="-",  label="Std fwd mem")
    axes[2].plot(Ns, results["fa1_fwd"]["mem"], marker="o", linestyle="-",  label="FA1 fwd mem")
    axes[2].plot(Ns, results["fa2_fwd"]["mem"], marker="s", linestyle="-",  label="FA2 fwd mem")
    axes[2].plot(Ns, results["std_bwd"]["mem"], marker="^", linestyle="--", label="Std bwd mem")
    axes[2].plot(Ns, results["fa1_bwd"]["mem"], marker="o", linestyle="--", label="FA1 bwd mem")
    axes[2].plot(Ns, results["fa2_bwd"]["mem"], marker="s", linestyle="--", label="FA2 bwd mem")
    axes[2].set_title("Peak GPU Memory")
    axes[2].set_xlabel("Sequence Length N")
    axes[2].set_ylabel("Memory (MB)")
    axes[2].legend(); axes[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = "benchmarking/layers/flash_attention_kernel_benchmark.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Kernel plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Image-resolution benchmark (Std vs FA1 vs FA2 MHA, sweeping image size)
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_SIZES = [128, 256, 512, 1024]   # square image side length
PATCH_SIZE  = 16                       # ViT patch size → N = (img // patch)^2


# Thin autograd wrapper for FA1 (mirrors flash_attention_fn.py for FA2)
from minitorch.autodiff import Context
from minitorch.tensor_functions import Function as _Fn

class _FA1Fn(_Fn):
    @staticmethod
    def forward(ctx: Context, q, k, v):
        scale = q.shape[2] ** -0.5
        o, L  = CudaKernelOps.flash_attention_forward_fa1(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, L, scale)
        return o

    @staticmethod
    def backward(ctx: Context, do):
        q, k, v, o, L, scale = ctx.saved_values
        dq, dk, dv = CudaKernelOps.flash_attention_backward_fa1(q, k, v, o, do, L, scale)
        return dq, dk, dv


class _FA1AttentionLayer(minitorch.Module):
    """Wraps an existing MHA layer but swaps the attention kernel to FA1."""
    def __init__(self, mha_layer):
        super().__init__()
        self._mha = mha_layer

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape
        q, kT, v = self._mha.project_to_query_key_value(x)
        # q: (B,H,N,d)  kT: (B,H,d,N)  v: (B,H,N,d)
        k   = kT.permute(0, 1, 3, 2)
        BH  = batch_size * self._mha.n_head
        d   = self._mha.attn_hidden_dim
        q_f = q.contiguous().view(BH, seq_len, d)
        k_f = k.contiguous().view(BH, seq_len, d)
        v_f = v.contiguous().view(BH, seq_len, d)
        out = _FA1Fn.apply(q_f, k_f, v_f)
        out = out.view(batch_size, self._mha.n_head, seq_len, d)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.view(batch_size, seq_len, n_embd)


def run_image_benchmarks():
    results = {
        "std":   {"img": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []},
        "fa1":   {"img": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []},
        "flash": {"img": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []},
    }

    for img_size in IMAGE_SIZES:
        N = (img_size // PATCH_SIZE) ** 2
        print(f"\n[image] {img_size}x{img_size}  (N={N})")

        std_layer, flash_layer = make_layers(N_EMBD, N_HEAD)
        fa1_layer = _FA1AttentionLayer(std_layer)

        input_fn = lambda: tensor_from_numpy(
            np.random.randn(BATCH, N, N_EMBD).astype(datatype),
            backend=backend,
        )

        for tag, layer in (("std", std_layer), ("fa1", fa1_layer), ("flash", flash_layer)):
            print(f"  [{tag}]", end="", flush=True)
            res = bench_utils.benchmark_module(layer, input_fn, n_iters=N_ITERS, n_warmup=N_WARMUP)
            results[tag]["img"].append(img_size)
            results[tag]["fwd_time"].append(res["fwd_time_ms"])
            results[tag]["bwd_time"].append(res["bwd_time_ms"])
            results[tag]["fwd_mem"].append(res["fwd_peak_mem_mb"])
            results[tag]["bwd_mem"].append(res["bwd_peak_mem_mb"])
            print(f"  fwd={res['fwd_time_ms']:.2f}ms  bwd={res['bwd_time_ms']:.2f}ms"
                  f"  fwd_mem={res['fwd_peak_mem_mb']:.1f}MB  bwd_mem={res['bwd_peak_mem_mb']:.1f}MB")

    return results


def _plot_image_benchmarks(results):
    os.makedirs("benchmarking/layers", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Std vs FA1 vs FA2 — MHA by Image Resolution  "
        f"(B={BATCH}, patch={PATCH_SIZE}, d={N_EMBD}, n_head={N_HEAD})"
    )

    xlabels = [f"{s}×{s}" for s in results["std"]["img"]]
    xs      = results["std"]["img"]

    for tag, marker, label in (
        ("std",   "^", "Standard"),
        ("fa1",   "o", "FA1"),
        ("flash", "s", "FA2 (flash)"),
    ):
        axes[0].plot(xs, results[tag]["fwd_time"], marker=marker, label=label)
        axes[1].plot(xs, results[tag]["bwd_time"], marker=marker, label=label)

    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    for ax, title in zip(axes[:2], ["Forward Time (ms)", "Backward Time (ms)"]):
        ax.set_title(title); ax.set_xlabel("Image Resolution")
        ax.set_xticks(xs); ax.set_xticklabels(xlabels)
        ax.legend(); ax.grid(True, linestyle="--", alpha=0.7)

    # Speedup: std / fa1 and std / fa2
    std_fwd  = np.array(results["std"]["fwd_time"])
    std_bwd  = np.array(results["std"]["bwd_time"])
    for tag, marker, label in (("fa1", "o", "vs FA1"), ("flash", "s", "vs FA2")):
        axes[2].plot(xs, std_fwd / np.array(results[tag]["fwd_time"]),
                     marker=marker, linestyle="-",  label=f"fwd speedup {label}")
        axes[2].plot(xs, std_bwd / np.array(results[tag]["bwd_time"]),
                     marker=marker, linestyle="--", label=f"bwd speedup {label}")
    axes[2].axhline(1.0, color="gray", linestyle=":", linewidth=1)
    axes[2].set_title("Speedup over Standard (std / flash)")
    axes[2].set_xlabel("Image Resolution")
    axes[2].set_xticks(xs); axes[2].set_xticklabels(xlabels)
    axes[2].legend(); axes[2].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = "benchmarking/layers/image_resolution_benchmark.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nImage resolution plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("Layer-level: Standard vs Flash MultiHeadAttention")
    print("=" * 60)
    layer_results = run_layer_benchmarks()
    _plot_layer(layer_results)

    print("\n" + "=" * 60)
    print("Kernel-level: Standard vs FA1 vs FA2 forward and backward")
    print("=" * 60)
    kernel_results = run_kernel_benchmarks()
    _plot_kernels(kernel_results)

    print("\n" + "=" * 60)
    print("Image-resolution: Std vs FA1 vs FA2 MHA by image size")
    print("=" * 60)
    image_results = run_image_benchmarks()
    _plot_image_benchmarks(image_results)


if __name__ == "__main__":
    run()
