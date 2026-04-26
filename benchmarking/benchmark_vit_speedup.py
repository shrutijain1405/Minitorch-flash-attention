import os
import json
import gc
import numpy as np
import matplotlib.pyplot as plt

import minitorch
from minitorch import bench_utils
from minitorch.transformer import ViT, MultiHeadAttention, FeedForward, Patchify
from minitorch.modules_basic import Linear, LayerNorm1d, Embedding
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_ops import TensorBackend
from minitorch.tensor_functions import tensor_from_numpy
from minitorch.autodiff import Context
from minitorch.tensor_functions import Function

backend    = TensorBackend(CudaKernelOps)
datatype   = np.float32

IMAGE_SIZES    = [128, 256]
PATCH_SIZE     = 16
N_EMBD         = 64
N_HEAD         = 1
N_TRANS_LAYERS = 2
N_CLASSES      = 10
N_CHANNELS     = 3
BATCH_SIZE     = 4
N_ITERS        = 5
N_WARMUP       = 2


# ── FA1 autograd wrapper ──────────────────────────────────────────────────────

class _FA1Fn(Function):
    @staticmethod
    def forward(ctx: Context, q, k, v):
        from minitorch.cuda_kernel_ops import CudaKernelOps as K
        scale = q.shape[2] ** -0.5
        o, L  = K.flash_attention_forward_fa1(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, L, scale)
        return o

    @staticmethod
    def backward(ctx: Context, do):
        from minitorch.cuda_kernel_ops import CudaKernelOps as K
        q, k, v, o, L, scale = ctx.saved_values
        dq, dk, dv = K.flash_attention_backward_fa1(q, k, v, o, do, L, scale)
        return dq, dk, dv


class FA1MultiHeadAttention(minitorch.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self._mha = MultiHeadAttention(
            n_embd, n_head, causal=False, p_dropout=0.0,
            bias=False, backend=backend, use_flash_attn=False,
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q, kT, v = self._mha.project_to_query_key_value(x)
        k  = kT.permute(0, 1, 3, 2)
        BH = batch_size * self._mha.n_head
        d  = self._mha.attn_hidden_dim
        out = _FA1Fn.apply(
            q.contiguous().view(BH, seq_len, d),
            k.contiguous().view(BH, seq_len, d),
            v.contiguous().view(BH, seq_len, d),
        )
        out = out.view(batch_size, self._mha.n_head, seq_len, d)
        out = out.permute(0, 2, 1, 3).contiguous()
        out_flat = out.contiguous().view(batch_size * seq_len, N_EMBD)
        return self._mha.out_projection(out_flat).view(batch_size, seq_len, N_EMBD)


# ── Profile one image size ────────────────────────────────────────────────────

def profile_image_size(img_size):
    N           = (img_size // PATCH_SIZE) ** 2
    num_patches = N + 1
    patch_dim   = N_CHANNELS * PATCH_SIZE * PATCH_SIZE

    layers = {
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

    # Attention variants
    attn_layers = {
        "Attn (Std)": MultiHeadAttention(
            N_EMBD, N_HEAD, causal=False, p_dropout=0.0,
            bias=False, backend=backend, use_flash_attn=False,
        ),
        "Attn (FA1)": FA1MultiHeadAttention(N_EMBD, N_HEAD),
        "Attn (FA2)": MultiHeadAttention(
            N_EMBD, N_HEAD, causal=False, p_dropout=0.0,
            bias=False, backend=backend, use_flash_attn=True,
        ),
    }

    attn_input_fn = lambda: tensor_from_numpy(
        np.random.randn(BATCH_SIZE, num_patches, N_EMBD).astype(datatype),
        backend=backend,
    )

    timings = {}

    # Time non-attention layers (same for all variants)
    for name, (module, input_fn) in layers.items():
        print(f"    {name}...", end=" ", flush=True)
        res = bench_utils.benchmark_module(module, input_fn, n_iters=N_ITERS, n_warmup=N_WARMUP)
        timings[name] = res["fwd_time_ms"]
        print(f"{res['fwd_time_ms']:.1f}ms")
        del module
        gc.collect()

    # Time attention variants
    for name, module in attn_layers.items():
        print(f"    {name}...", end=" ", flush=True)
        res = bench_utils.benchmark_module(module, attn_input_fn, n_iters=N_ITERS, n_warmup=N_WARMUP)
        timings[name] = res["fwd_time_ms"]
        print(f"{res['fwd_time_ms']:.1f}ms")
        del module
        gc.collect()

    return timings


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmarks():
    all_timings = {}

    for img_size in IMAGE_SIZES:
        N = (img_size // PATCH_SIZE) ** 2
        print(f"\nImage {img_size}x{img_size}  (N={N})")
        all_timings[img_size] = profile_image_size(img_size)

    # Save JSON
    os.makedirs("benchmarking/layers", exist_ok=True)
    log_path = "benchmarking/layers/vit_speedup_benchmark.json"
    with open(log_path, "w") as f:
        json.dump({str(k): v for k, v in all_timings.items()}, f, indent=2)
    print(f"\nResults saved to {log_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    other_layers = ["Patchify", "PatchProj", "PosEmbed", "LayerNorm", "FeedForward", "ClassHead"]
    # LayerNorm and FeedForward appear N_TRANS_LAYERS * 2 and N_TRANS_LAYERS times
    layer_multipliers = {
        "Patchify":    1,
        "PatchProj":   1,
        "PosEmbed":    1,
        "LayerNorm":   N_TRANS_LAYERS * 2,
        "FeedForward": N_TRANS_LAYERS,
        "ClassHead":   1,
    }
    attn_multiplier = N_TRANS_LAYERS

    n      = len(IMAGE_SIZES)
    xs     = np.arange(n)
    bar_w  = 0.25
    xlabels = [f"{s}×{s}\n(N={(s//PATCH_SIZE)**2})" for s in IMAGE_SIZES]

    other_colors = ["#AEC6E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94"]
    attn_colors  = {"Attn (Std)": "#D62728", "Attn (FA1)": "#38A8A8", "Attn (FA2)": "#5B2C8D"}

    fig, (ax_bar, ax_speedup) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"ViT Layer Breakdown & Flash Attention Speedup  "
        f"(patch={PATCH_SIZE}, B={BATCH_SIZE}, d={N_EMBD}, layers={N_TRANS_LAYERS})"
    )

    # ── Stacked bar: other layers + each attn variant ─────────────────────
    for ai, attn_key in enumerate(["Attn (Std)", "Attn (FA1)", "Attn (FA2)"]):
        offset = (ai - 1) * bar_w
        bottoms = np.zeros(n)

        for li, layer in enumerate(other_layers):
            heights = np.array([
                all_timings[s][layer] * layer_multipliers[layer]
                for s in IMAGE_SIZES
            ])
            ax_bar.bar(xs + offset, heights, width=bar_w, bottom=bottoms,
                       color=other_colors[li],
                       label=layer if ai == 0 else None)
            bottoms += heights

        attn_heights = np.array([
            all_timings[s][attn_key] * attn_multiplier for s in IMAGE_SIZES
        ])
        ax_bar.bar(xs + offset, attn_heights, width=bar_w, bottom=bottoms,
                   color=attn_colors[attn_key],
                   label=attn_key if ai == 0 else None)
        bottoms += attn_heights

        # Label bar group
        ax_bar.text(xs[0] + offset, -ax_bar.get_ylim()[1] * 0.05,
                    attn_key.replace("Attn ", ""), ha="center", fontsize=7)

    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(xlabels)
    ax_bar.set_xlabel("Image Resolution")
    ax_bar.set_ylabel("Forward Time (ms)")
    ax_bar.set_title("Layer Breakdown: Std vs FA1 vs FA2")
    ax_bar.legend(loc="upper left", fontsize=8)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)

    # ── Speedup plot ──────────────────────────────────────────────────────
    def total_time(img_size, attn_key):
        t = all_timings[img_size]
        other = sum(t[l] * layer_multipliers[l] for l in other_layers)
        return other + t[attn_key] * attn_multiplier

    std_totals = np.array([total_time(s, "Attn (Std)") for s in IMAGE_SIZES])
    fa1_totals = np.array([total_time(s, "Attn (FA1)") for s in IMAGE_SIZES])
    fa2_totals = np.array([total_time(s, "Attn (FA2)") for s in IMAGE_SIZES])

    ax_speedup.plot(xs, std_totals / fa1_totals, marker="o", color="#38A8A8", label="FA1 speedup")
    ax_speedup.plot(xs, std_totals / fa2_totals, marker="s", color="#5B2C8D", label="FA2 speedup")
    ax_speedup.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax_speedup.set_xticks(xs)
    ax_speedup.set_xticklabels(xlabels)
    ax_speedup.set_xlabel("Image Resolution")
    ax_speedup.set_ylabel("Speedup (std total / flash total)")
    ax_speedup.set_title("ViT Speedup from Flash Attention")
    ax_speedup.legend()
    ax_speedup.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "benchmarking/layers/vit_speedup_benchmark.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    run_benchmarks()
