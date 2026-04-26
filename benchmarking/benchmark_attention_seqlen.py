import os
import json
import gc
import numpy as np
import matplotlib.pyplot as plt

import minitorch
from minitorch import bench_utils
from minitorch.transformer import MultiHeadAttention
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_ops import TensorBackend
from minitorch.tensor_functions import tensor_from_numpy

backend  = TensorBackend(CudaKernelOps)
datatype = np.float32

SEQ_LENS = [32, 64128, 256, 512, 1024]
N_EMBD   = 64
N_HEAD   = 1
BATCH    = 4
N_ITERS  = 5
N_WARMUP = 5

ATTN_TYPES  = ["standard", "fa1", "fa2"]
ATTN_COLORS = {"standard": "#D62728", "fa1": "#38A8A8", "fa2": "#5B2C8D"}
ATTN_LABELS = {"standard": "Standard", "fa1": "FA1", "fa2": "FA2"}


def run_benchmarks():
    results = {attn: {"seq_len": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []}
               for attn in ATTN_TYPES}

    for seq_len in SEQ_LENS:
        print(f"\nSeq len {seq_len}")

        for attn_type in ATTN_TYPES:
            print(f"  [{attn_type}]...", end=" ", flush=True)
            layer = MultiHeadAttention(
                N_EMBD, N_HEAD,
                causal=False, p_dropout=0.0, bias=False,
                backend=backend, attn_type=attn_type,
            )
            layer.train()

            input_fn = lambda seq_len=seq_len: tensor_from_numpy(
                np.random.randn(BATCH, seq_len, N_EMBD).astype(datatype),
                backend=backend,
            )

            res = bench_utils.benchmark_module(layer, input_fn, n_iters=N_ITERS, n_warmup=N_WARMUP)
            results[attn_type]["seq_len"].append(seq_len)
            results[attn_type]["fwd_time"].append(res["fwd_time_ms"])
            results[attn_type]["bwd_time"].append(res["bwd_time_ms"])
            results[attn_type]["fwd_mem"].append(res["fwd_peak_mem_mb"])
            results[attn_type]["bwd_mem"].append(res["bwd_peak_mem_mb"])
            print(f"fwd={res['fwd_time_ms']:.1f}ms  bwd={res['bwd_time_ms']:.1f}ms"
                  f"  fwd_mem={res['fwd_peak_mem_mb']:.1f}MB  bwd_mem={res['bwd_peak_mem_mb']:.1f}MB")

            del layer, input_fn
            gc.collect()

    # Save JSON
    os.makedirs("benchmarking/layers", exist_ok=True)
    log_path = "benchmarking/layers/attention_seqlen_benchmark.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {log_path}")

    # Plot
    x = range(len(SEQ_LENS))
    xlabels = [str(s) for s in SEQ_LENS]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Attention Fwd & Bwd by Sequence Length  "
        f"(B={BATCH}, d={N_EMBD}, H={N_HEAD})"
    )

    metrics = [
        ("fwd_time", "Forward Time (ms)",  "Execution Time (Fwd)", axes[0, 0]),
        ("bwd_time", "Backward Time (ms)", "Execution Time (Bwd)", axes[0, 1]),
        ("fwd_mem",  "Peak Memory (MB)",   "Peak GPU Memory (Fwd)", axes[1, 0]),
        ("bwd_mem",  "Peak Memory (MB)",   "Peak GPU Memory (Bwd)", axes[1, 1]),
    ]

    for key, ylabel, title, ax in metrics:
        for attn_type in ATTN_TYPES:
            ax.plot(x, results[attn_type][key],
                    label=ATTN_LABELS[attn_type],
                    marker="o",
                    color=ATTN_COLORS[attn_type])
        ax.set_title(title)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "benchmarking/layers/attention_seqlen_benchmark.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    run_benchmarks()
