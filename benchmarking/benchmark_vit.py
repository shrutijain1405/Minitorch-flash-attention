import matplotlib
matplotlib.use("Agg")
import minitorch
import matplotlib.pyplot as plt
import numpy as np
import os
import gc

from minitorch import bench_utils
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.module import Module
from minitorch.transformer import ViT, MultiHeadAttention, FeedForward, Patchify
from minitorch.modules_basic import Linear, LayerNorm1d, Embedding, Dropout
from minitorch.tensor_ops import TensorBackend
from minitorch.tensor_functions import zeros

backend = TensorBackend(CudaKernelOps)

# ViT-Base/16 config
N_EMBD = 768
N_HEAD = 12
PATCH_SIZE = 16
N_CHANNELS = 3
N_CLASSES = 1000
N_TRANS_LAYERS = 12
BATCH_SIZE = 2
IMG_SIZE = 224

HIDDEN_SIZES = [768]
# IMG_SIZES = [256, 512, 1024, 2048]  # patches: 256, 1024, 4096, 16384
IMG_SIZES = [64, 128, 256, 512]  


def benchmark_forward_only(module, input_fn, n_iters=10, n_warmup=3):
    """Benchmark only the forward pass (no backward) using the same memory poller."""
    import time
    import pycuda.autoinit

    inp = input_fn()
    for _ in range(n_warmup):
        module(inp)

    fwd_times = []
    fwd_peak_mems = []

    for _ in range(n_iters):
        pycuda.autoinit.context.synchronize()

        poller = bench_utils.NvmlMemPoller()
        poller.start()
        t0 = time.perf_counter()
        out = module(inp)
        pycuda.autoinit.context.synchronize()
        t1 = time.perf_counter()
        poller.stop()

        fwd_times.append(t1 - t0)
        fwd_peak_mems.append(poller.peak_mb)

    return {
        "fwd_time_ms": sum(fwd_times) / len(fwd_times) * 1000,
        "fwd_peak_mem_mb": max(fwd_peak_mems),
    }


def make_layers_for_img_size(img_sz):
    """Create each ViT sub-layer and its input function for a given image size."""
    N = (img_sz // PATCH_SIZE) ** 2
    patch_dim = N_CHANNELS * PATCH_SIZE * PATCH_SIZE
    num_patches = N + 1  # with CLS token

    return {
        "Patchify": (
            Patchify(PATCH_SIZE),
            lambda img_sz=img_sz: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE, N_CHANNELS, img_sz, img_sz).astype(np.float32),
                backend=backend,
            ),
        ),
        "PatchProj": (
            Linear(patch_dim, N_EMBD, True, backend),
            lambda N=N, patch_dim=patch_dim: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE * N, patch_dim).astype(np.float32),
                backend=backend,
            ),
        ),
        "PosEmbedding": (
            Embedding(max(num_patches, 1), N_EMBD, backend),
            lambda num_patches=num_patches: minitorch.tensor(
                [list(range(num_patches))], backend=backend, requires_grad=False,
            ),
        ),
        "LayerNorm": (
            LayerNorm1d(N_EMBD, 1e-5, backend),
            lambda num_patches=num_patches: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE * num_patches, N_EMBD).astype(np.float32),
                backend=backend,
            ),
        ),
        "Attention": (
            MultiHeadAttention(N_EMBD, N_HEAD, causal=False, p_dropout=0.0, bias=True, backend=backend),
            lambda num_patches=num_patches: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE, num_patches, N_EMBD).astype(np.float32),
                backend=backend,
            ),
        ),
        "FeedForward": (
            FeedForward(N_EMBD, 4 * N_EMBD, p_dropout=0.0, bias=True, backend=backend),
            lambda num_patches=num_patches: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE, num_patches, N_EMBD).astype(np.float32),
                backend=backend,
            ),
        ),
        "Classification Head": (
            Linear(N_EMBD, N_CLASSES, True, backend),
            lambda: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE, N_EMBD).astype(np.float32),
                backend=backend,
            ),
        ),
    }


def run_layer_breakdown():
    """Time each sub-layer at every image size and produce a grouped bar chart."""
    print("=" * 60)
    print("Layer-by-layer breakdown across image sizes")
    print(f"  n_embd={N_EMBD}, n_head={N_HEAD}, batch={BATCH_SIZE}, patch={PATCH_SIZE}")
    print(f"  image sizes: {IMG_SIZES}")
    print("=" * 60)

    layer_names = None
    # {img_sz: {layer_name: fwd_time_ms}}
    all_results = {}

    for img_sz in IMG_SIZES:
        n_patches = (img_sz // PATCH_SIZE) ** 2
        print(f"\n  img={img_sz}x{img_sz} ({n_patches} patches):")
        layers = make_layers_for_img_size(img_sz)
        if layer_names is None:
            layer_names = list(layers.keys())

        all_results[img_sz] = {}
        for name, (module, input_fn) in layers.items():
            module.train()
            print(f"    {name}...", end=" ", flush=True)
            res = benchmark_forward_only(module, input_fn, n_iters=10, n_warmup=3)
            all_results[img_sz][name] = res["fwd_time_ms"]
            print(f"{res['fwd_time_ms']:.3f} ms")
            del module, input_fn
        del layers
        gc.collect()

    # Grouped bar chart: x-axis = image sizes, groups = layers
    n_groups = len(IMG_SIZES)
    n_bars = len(layer_names)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f"ViT-Base/16 Layer Timing by Image Size (n_embd={N_EMBD}, patch={PATCH_SIZE})")

    for i, layer_name in enumerate(layer_names):
        times = [all_results[img_sz][layer_name] for img_sz in IMG_SIZES]
        offset = (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, times, bar_width, label=layer_name, color=colors[i])

    patch_labels = [f"{sz}x{sz}\n({(sz // PATCH_SIZE) ** 2} patches)" for sz in IMG_SIZES]
    ax.set_xticks(x)
    ax.set_xticklabels(patch_labels)
    ax.set_xlabel("Image Size (patches)")
    ax.set_ylabel("Forward Time (ms)")
    ax.set_yscale("log")
    ax.set_title("Per-Layer Forward Time")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "benchmarking/layers/vit_layer_breakdown.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nLayer breakdown saved to {save_path}")


class QKVProjection(Module):
    """Wraps MultiHeadAttention.project_to_query_key_value for benchmarking."""
    def __init__(self, attn):
        super().__init__()
        self.attn = attn

    def forward(self, x):
        q, kT, v = self.attn.project_to_query_key_value(x)
        return q


class QKMatmul(Module):
    """Benchmarks Q @ K^T / sqrt(d)."""
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, q_kT):
        B, H, S, D = q_kT.shape
        q = q_kT
        kT = q_kT.permute(0, 1, 3, 2).contiguous()
        return (q @ kT) / (self.scale ** 0.5)


class SoftmaxModule(Module):
    """Benchmarks softmax over last dim."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        from minitorch.nn import softmax
        return softmax(x, dim=3)


class AttnVMatmul(Module):
    """Benchmarks attn_weights @ V."""
    def __init__(self, n_head, attn_dim, num_patches, backend):
        super().__init__()
        self.v = minitorch.tensor_from_numpy(
            np.random.randn(BATCH_SIZE, n_head, num_patches, attn_dim).astype(np.float32),
            backend=backend,
        )

    def forward(self, weights):
        return weights @ self.v


class OutProjection(Module):
    """Wraps the output projection linear layer."""
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)


def make_attn_ops_for_img_size(img_sz):
    """Create each attention sub-operation and its input for a given image size."""
    N = (img_sz // PATCH_SIZE) ** 2
    num_patches = N + 1
    attn_dim = N_EMBD // N_HEAD

    attn = MultiHeadAttention(N_EMBD, N_HEAD, causal=False, p_dropout=0.0, bias=True, backend=backend)

    return {
        "QKV Projection": (
            QKVProjection(attn),
            lambda num_patches=num_patches: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE, num_patches, N_EMBD).astype(np.float32),
                backend=backend,
            ),
        ),
        "Q @ K^T (scaled)": (
            QKMatmul(attn_dim),
            lambda num_patches=num_patches, attn_dim=attn_dim: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE, N_HEAD, num_patches, attn_dim).astype(np.float32),
                backend=backend,
            ),
        ),
        "Softmax": (
            SoftmaxModule(),
            lambda num_patches=num_patches: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE, N_HEAD, num_patches, num_patches).astype(np.float32),
                backend=backend,
            ),
        ),
        "Attn @ V": (
            AttnVMatmul(N_HEAD, attn_dim, num_patches, backend),
            lambda num_patches=num_patches: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE, N_HEAD, num_patches, num_patches).astype(np.float32),
                backend=backend,
            ),
        ),
        "Out Projection": (
            OutProjection(attn.out_projection),
            lambda num_patches=num_patches: minitorch.tensor_from_numpy(
                np.random.randn(BATCH_SIZE * num_patches, N_EMBD).astype(np.float32),
                backend=backend,
            ),
        ),
    }


def run_attention_breakdown():
    """Time each attention sub-operation at every image size."""
    print("\n" + "=" * 60)
    print("Attention breakdown across image sizes")
    print(f"  n_embd={N_EMBD}, n_head={N_HEAD}, batch={BATCH_SIZE}, patch={PATCH_SIZE}")
    print(f"  image sizes: {IMG_SIZES}")
    print("=" * 60)

    op_names = None
    all_results = {}

    for img_sz in IMG_SIZES:
        n_patches = (img_sz // PATCH_SIZE) ** 2
        print(f"\n  img={img_sz}x{img_sz} ({n_patches} patches):")
        ops = make_attn_ops_for_img_size(img_sz)
        if op_names is None:
            op_names = list(ops.keys())

        all_results[img_sz] = {}
        for name, (module, input_fn) in ops.items():
            module.train()
            print(f"    {name}...", end=" ", flush=True)
            res = benchmark_forward_only(module, input_fn, n_iters=10, n_warmup=3)
            all_results[img_sz][name] = res["fwd_time_ms"]
            print(f"{res['fwd_time_ms']:.3f} ms")
            del module, input_fn
        del ops
        gc.collect()

    # Grouped bar chart
    n_groups = len(IMG_SIZES)
    n_bars = len(op_names)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    colors = plt.cm.Set2(np.linspace(0, 1, n_bars))

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f"Attention Layer Breakdown by Image Size (n_embd={N_EMBD}, n_head={N_HEAD})")

    for i, op_name in enumerate(op_names):
        times = [all_results[img_sz][op_name] for img_sz in IMG_SIZES]
        offset = (i - n_bars / 2 + 0.5) * bar_width
        ax.bar(x + offset, times, bar_width, label=op_name, color=colors[i])

    patch_labels = [f"{sz}x{sz}\n({(sz // PATCH_SIZE) ** 2} patches)" for sz in IMG_SIZES]
    ax.set_xticks(x)
    ax.set_xticklabels(patch_labels)
    ax.set_xlabel("Image Size (patches)")
    ax.set_ylabel("Forward Time (ms)")
    ax.set_yscale("log")
    ax.set_title("Per-Operation Forward Time (Single Attention Layer)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "benchmarking/layers/vit_attention_breakdown.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nAttention breakdown saved to {save_path}")


def run_hidden_size_sweep():
    """Benchmark the full ViT across different hidden sizes."""
    print("\n" + "=" * 60)
    print("Full ViT — Hidden size sweep")
    print(f"  batch={BATCH_SIZE}, img={IMG_SIZE}x{IMG_SIZE}, patch={PATCH_SIZE}")
    print(f"  n_trans_layers={N_TRANS_LAYERS}, n_classes={N_CLASSES}")
    print("=" * 60)

    data = {"hs": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []}

    for hs in HIDDEN_SIZES:
        n_head = max(1, hs // 32)
        print(f"  n_embd={hs}, n_head={n_head}...", end=" ", flush=True)

        model = ViT(
            n_embd=hs,
            n_head=n_head,
            n_trans_layers=N_TRANS_LAYERS,
            patch_size=PATCH_SIZE,
            n_classes=N_CLASSES,
            max_patches=256,
            n_channels=N_CHANNELS,
            backend=backend,
        )
        model.train()

        input_fn = lambda: minitorch.tensor_from_numpy(
            np.random.randn(BATCH_SIZE, N_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32),
            backend=backend,
        )

        res = bench_utils.benchmark_module(model, input_fn, n_iters=10, n_warmup=3)
        data["hs"].append(hs)
        data["fwd_time"].append(res["fwd_time_ms"])
        data["bwd_time"].append(res["bwd_time_ms"])
        data["fwd_mem"].append(res["fwd_peak_mem_mb"])
        data["bwd_mem"].append(res["bwd_peak_mem_mb"])
        print(f"fwd={res['fwd_time_ms']:.1f}ms  bwd={res['bwd_time_ms']:.1f}ms")

    # Plot
    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("ViT — Hidden Size Scaling")

    ax_time.plot(data["hs"], data["fwd_time"], label="Fwd Time (ms)", marker="o")
    ax_time.plot(data["hs"], data["bwd_time"], label="Bwd Time (ms)", marker="x")
    ax_time.set_title("Execution Time")
    ax_time.set_xlabel("Hidden Size")
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_xscale("log", base=2)
    ax_time.set_xticks(data["hs"])
    ax_time.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_time.grid(True, linestyle="--", alpha=0.7)
    ax_time.legend()

    ax_mem.plot(data["hs"], data["fwd_mem"], label="Fwd Peak Mem (MB)", marker="o", color="green")
    ax_mem.plot(data["hs"], data["bwd_mem"], label="Bwd Peak Mem (MB)", marker="x", color="red")
    ax_mem.set_title("Peak GPU Memory Usage")
    ax_mem.set_xlabel("Hidden Size")
    ax_mem.set_ylabel("Memory (MB)")
    ax_mem.set_xscale("log", base=2)
    ax_mem.set_xticks(data["hs"])
    ax_mem.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_mem.grid(True, linestyle="--", alpha=0.7)
    ax_mem.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "benchmarking/layers/vit_hidden_size_sweep.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nHidden size sweep saved to {save_path}")


def run_image_size_sweep():
    """Benchmark the full ViT across different image sizes (patch count scaling)."""
    print("\n" + "=" * 60)
    print("Full ViT — Image size sweep")
    print(f"  n_embd={N_EMBD}, n_head={N_HEAD}, batch={BATCH_SIZE}")
    print(f"  patch={PATCH_SIZE}, n_trans_layers={N_TRANS_LAYERS}")
    print("=" * 60)

    data = {
        "img_size": [], "num_patches": [],
        "fwd_time": [], "bwd_time": [],
        "fwd_mem": [], "bwd_mem": [],
    }

    for img_sz in IMG_SIZES:
        n_patches = (img_sz // PATCH_SIZE) ** 2
        max_patches = n_patches + 1  # +1 for CLS token
        print(f"  img={img_sz}x{img_sz}  ({n_patches} patches)...", end=" ", flush=True)

        model = ViT(
            n_embd=N_EMBD,
            n_head=N_HEAD,
            n_trans_layers=N_TRANS_LAYERS,
            patch_size=PATCH_SIZE,
            n_classes=N_CLASSES,
            max_patches=max_patches,
            n_channels=N_CHANNELS,
            backend=backend,
        )
        model.train()

        input_fn = lambda img_sz=img_sz: minitorch.tensor_from_numpy(
            np.random.randn(BATCH_SIZE, N_CHANNELS, img_sz, img_sz).astype(np.float32),
            backend=backend,
        )

        res = bench_utils.benchmark_module(model, input_fn, n_iters=10, n_warmup=3)
        data["img_size"].append(img_sz)
        data["num_patches"].append(n_patches)
        data["fwd_time"].append(res["fwd_time_ms"])
        data["bwd_time"].append(res["bwd_time_ms"])
        data["fwd_mem"].append(res["fwd_peak_mem_mb"])
        data["bwd_mem"].append(res["bwd_peak_mem_mb"])
        print(f"fwd={res['fwd_time_ms']:.1f}ms  bwd={res['bwd_time_ms']:.1f}ms")
        del model, input_fn
        gc.collect()

    # Plot with patch count as x-axis labels
    labels = [f"{sz}\n({n} patches)" for sz, n in zip(data["img_size"], data["num_patches"])]

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("ViT — Image Size Scaling (patch count)")

    x = range(len(labels))

    ax_time.plot(x, data["fwd_time"], label="Fwd Time (ms)", marker="o")
    ax_time.plot(x, data["bwd_time"], label="Bwd Time (ms)", marker="x")
    ax_time.set_title("Execution Time")
    ax_time.set_xlabel("Image Size (patches)")
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels)
    ax_time.grid(True, linestyle="--", alpha=0.7)
    ax_time.legend()

    ax_mem.plot(x, data["fwd_mem"], label="Fwd Peak Mem (MB)", marker="o", color="green")
    ax_mem.plot(x, data["bwd_mem"], label="Bwd Peak Mem (MB)", marker="x", color="red")
    ax_mem.set_title("Peak GPU Memory Usage")
    ax_mem.set_xlabel("Image Size (patches)")
    ax_mem.set_ylabel("Memory (MB)")
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(labels)
    ax_mem.grid(True, linestyle="--", alpha=0.7)
    ax_mem.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "benchmarking/layers/vit_image_size_sweep.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nImage size sweep saved to {save_path}")


if __name__ == "__main__":
    os.makedirs("benchmarking/layers", exist_ok=True)
    run_layer_breakdown()
    run_attention_breakdown()
    # run_hidden_size_sweep()
    run_image_size_sweep()
