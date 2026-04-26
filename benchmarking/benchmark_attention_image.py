import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import minitorch
from minitorch import bench_utils
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_ops import TensorBackend
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.autodiff import Context
from minitorch.tensor_functions import Function
from minitorch.nn import softmax as minitorch_softmax
import pycuda.autoinit

backend = TensorBackend(CudaKernelOps)
datatype = np.float32

IMAGE_SIZES = [128, 256, 512, 1024, 2048]
PATCH_SIZE  = 16   # N = (img_size // PATCH_SIZE) ** 2 → 64, 256, 1024, 4096
BATCH       = 4
N_EMBD      = 64
N_HEAD      = 1
N_ITERS     = 5
N_WARMUP    = 5

STD_OPS = ["QK^T/√d", "softmax", "dropout", "(QK^T/√d)V"]
STD_COLORS = {
    "QK^T/√d":    "#3B4CC0",
    "softmax":         "#C94040",
    "dropout":         "#7B68B0",
    "(QK^T/√d)V": "#C8A838",
}
FA1_COLOR = "#38A8A8"
FA2_COLOR = "#5B2C8D"


# ── FA1 autograd wrapper ──────────────────────────────────────────────────────

class _FA1Fn(Function):
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


class FA1MultiHeadAttention(minitorch.Module):
    def __init__(self, mha_layer):
        super().__init__()
        self._mha = mha_layer

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q, kT, v = self._mha.project_to_query_key_value(x)
        k   = kT.permute(0, 1, 3, 2)
        BH  = batch_size * self._mha.n_head
        d   = self._mha.attn_hidden_dim
        out = _FA1Fn.apply(
            q.contiguous().view(BH, seq_len, d),
            k.contiguous().view(BH, seq_len, d),
            v.contiguous().view(BH, seq_len, d),
        )
        out = out.view(batch_size, self._mha.n_head, seq_len, d)
        out = out.permute(0, 2, 1, 3).contiguous()
        return out.view(batch_size, seq_len, N_EMBD)


# ── Layer factory ─────────────────────────────────────────────────────────────

def make_layers():
    rng = np.random.default_rng(0)
    w   = rng.standard_normal((N_EMBD, N_EMBD)).astype(datatype)

    std_layer = minitorch.MultiHeadAttention(
        N_EMBD, N_HEAD, causal=False, p_dropout=0.0,
        bias=False, backend=backend, use_flash_attn=False,
    )
    flash_layer = minitorch.MultiHeadAttention(
        N_EMBD, N_HEAD, causal=False, p_dropout=0.0,
        bias=False, backend=backend, use_flash_attn=True,
    )
    for layer in (std_layer, flash_layer):
        for proj in (layer.q_projection, layer.k_projection,
                     layer.v_projection, layer.out_projection):
            t = tensor_from_numpy(w.copy(), backend=backend)
            t.requires_grad_(True)
            proj.weights.value = t

    fa1_layer = FA1MultiHeadAttention(std_layer)
    return std_layer, fa1_layer, flash_layer


# ── Timing helper ─────────────────────────────────────────────────────────────

def _time_op(fn):
    for _ in range(N_WARMUP):
        fn()
    times = []
    for _ in range(N_ITERS):
        pycuda.autoinit.context.synchronize()
        t0 = time.perf_counter()
        fn()
        pycuda.autoinit.context.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times) * 1000


# ── Per-image profiling ───────────────────────────────────────────────────────

def profile_image_size(img_size):
    N = (img_size // PATCH_SIZE) ** 2
    d = N_EMBD // N_HEAD

    std_layer, fa1_layer, flash_layer = make_layers()

    x_np = np.random.randn(BATCH, N, N_EMBD).astype(datatype)
    x    = tensor_from_numpy(x_np, backend=backend)

    # Pre-compute intermediates for standard breakdown
    x_flat = x.view(BATCH * N, N_EMBD)
    q_out  = std_layer.q_projection(x_flat).view(BATCH, N, N_HEAD, d).permute(0, 2, 1, 3)
    kT_out = std_layer.k_projection(x_flat).view(BATCH, N, N_HEAD, d).permute(0, 2, 3, 1)
    v_out  = std_layer.v_projection(x_flat).view(BATCH, N, N_HEAD, d).permute(0, 2, 1, 3)
    S      = (q_out @ kT_out) * (d ** -0.5)
    P      = minitorch_softmax(S, dim=3)
    attn   = (P @ v_out).permute(0, 2, 1, 3).contiguous().view(BATCH * N, N_EMBD)

    scale = d ** -0.5
    q_flat_bh = q_out.contiguous().view(BATCH * N_HEAD, N, d)
    k_flat_bh = kT_out.permute(0, 1, 3, 2).contiguous().view(BATCH * N_HEAD, N, d)
    v_flat_bh = v_out.contiguous().view(BATCH * N_HEAD, N, d)

    std_ops = {
        "QK^T/√d":    _time_op(lambda: (q_out @ kT_out) * scale),
        "softmax":         _time_op(lambda: minitorch_softmax(S, dim=3)),
        "dropout":         0.0,   # p_dropout=0.0, negligible
        "(QK^T/√d)V": _time_op(lambda: P @ v_out),
    }

    # Raw kernel calls — core attention only, no projections
    if N >= 64:
        fa1_total = _time_op(
            lambda: CudaKernelOps.flash_attention_forward_fa1(q_flat_bh, k_flat_bh, v_flat_bh, scale)
        )
        fa2_total = _time_op(
            lambda: CudaKernelOps.flash_attention_forward_fa2(q_flat_bh, k_flat_bh, v_flat_bh, scale)
        )
    else:
        fa1_total = None
        fa2_total = None

    std_total = sum(v for v in std_ops.values() if isinstance(v, float))
    fa1_str   = f"{fa1_total:.1f}ms" if fa1_total is not None else "N/A"
    fa2_str   = f"{fa2_total:.1f}ms" if fa2_total is not None else "N/A"
    print(f"  std={std_total:.1f}ms  fa1={fa1_str}  fa2={fa2_str}")
    for op, ms in std_ops.items():
        print(f"    {op}: {ms:.1f}ms" if isinstance(ms, float) else f"    {op}: {ms}")

    return std_ops, fa1_total, fa2_total


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmarks():
    all_std_ops = []   # list of dicts, one per image size
    fa1_totals  = []
    fa2_totals  = []

    for img_size in IMAGE_SIZES:
        N = (img_size // PATCH_SIZE) ** 2
        print(f"\nImage {img_size}x{img_size}  (N={N})")
        std_ops, fa1_t, fa2_t = profile_image_size(img_size)
        all_std_ops.append(std_ops)
        fa1_totals.append(fa1_t)
        fa2_totals.append(fa2_t)

    # Save to JSON
    os.makedirs("benchmarking/layers", exist_ok=True)
    log = {
        "image_sizes": IMAGE_SIZES,
        "standard":    all_std_ops,
        "fa1_total":   fa1_totals,
        "fa2_total":   fa2_totals,
    }
    log_path = "benchmarking/layers/attention_image_benchmark.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nResults saved to {log_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    n     = len(IMAGE_SIZES)
    xs    = np.arange(n)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        f"Attention Forward Time by Image Resolution  "
        f"(patch={PATCH_SIZE}, B={BATCH}, d={N_EMBD})"
    )

    # Standard — stacked bars
    bottoms = np.zeros(n)
    for op in STD_OPS:
        heights = np.array([all_std_ops[i][op] for i in range(n)])
        ax.bar(xs - bar_w, heights, width=bar_w, bottom=bottoms,
               label=op, color=STD_COLORS[op])
        bottoms += heights

    # FA1 and FA2 — single bars (None → 0 for image sizes where N < 64)
    fa1_vals = [v if v is not None else 0.0 for v in fa1_totals]
    fa2_vals = [v if v is not None else 0.0 for v in fa2_totals]
    ax.bar(xs,          fa1_vals, width=bar_w, label="flash-attention (FA1)", color=FA1_COLOR)
    ax.bar(xs + bar_w,  fa2_vals, width=bar_w, label="flash-attention-2 (FA2)", color=FA2_COLOR)

    xlabels = [f"{s}×{s}\n(N={( s//PATCH_SIZE)**2})" for s in IMAGE_SIZES]
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Image Resolution")
    ax.set_ylabel("Time (ms)")
    ax.set_yscale("log")
    ax.set_title("Forward Pass: Standard (stacked ops) vs FA1 vs FA2")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    save_path = "benchmarking/layers/attention_image_benchmark.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    run_benchmarks()
