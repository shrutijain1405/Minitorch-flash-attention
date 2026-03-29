import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os

HIDDEN_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZE = 32
SEQ_LEN = 64
DEVICE = "cuda"


def benchmark_pytorch_module(module, input_fn, n_iters=10, n_warmup=5):
    inp = input_fn()

    for _ in range(n_warmup):
        out = module(inp)
        if out.requires_grad:
            out.sum().backward()

    fwd_times, bwd_times = [], []
    fwd_peak_mems, bwd_peak_mems = [], []

    for _ in range(n_iters):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Forward
        baseline = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        out = module(inp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_peak = torch.cuda.max_memory_allocated() - baseline

        fwd_times.append(t1 - t0)
        fwd_peak_mems.append(fwd_peak / 1024**2)

        # Backward
        out_sum = out.sum()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        out_sum.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bwd_peak = torch.cuda.max_memory_allocated() - baseline

        bwd_times.append(t1 - t0)
        bwd_peak_mems.append(bwd_peak / 1024**2)

    return {
        "fwd_time_ms": sum(fwd_times) / len(fwd_times) * 1000,
        "bwd_time_ms": sum(bwd_times) / len(bwd_times) * 1000,
        "fwd_peak_mem_mb": max(fwd_peak_mems),
        "bwd_peak_mem_mb": max(bwd_peak_mems),
    }


def run_benchmarks():
    results = {}

    layers = {
        "Linear": lambda hs: (
            nn.Linear(hs, hs, bias=True).to(DEVICE),
            lambda: torch.randn(BATCH_SIZE, hs, device=DEVICE, requires_grad=True),
        ),
        "Embedding": lambda hs: (
            nn.Embedding(1000, hs).to(DEVICE),
            lambda: torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), device=DEVICE),
        ),
        "LayerNorm": lambda hs: (
            nn.LayerNorm(hs, eps=1e-5).to(DEVICE),
            lambda: torch.randn(BATCH_SIZE, hs, device=DEVICE, requires_grad=True),
        ),
        "Dropout": lambda hs: (
            nn.Dropout(0.1).to(DEVICE),
            lambda: torch.randn(BATCH_SIZE, SEQ_LEN, hs, device=DEVICE, requires_grad=True),
        ),
    }

    for name, factory in layers.items():
        print(f"Benchmarking PyTorch {name}...")
        results[name] = {"hs": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []}
        for hs in HIDDEN_SIZES:
            print(f"  Hidden Size: {hs}")
            module, input_fn = factory(hs)
            if name == "Dropout":
                module.train()

            res = benchmark_pytorch_module(module, input_fn, n_iters=10, n_warmup=5)

            results[name]["hs"].append(hs)
            results[name]["fwd_time"].append(res["fwd_time_ms"])
            results[name]["bwd_time"].append(res["bwd_time_ms"])
            results[name]["fwd_mem"].append(res["fwd_peak_mem_mb"])
            results[name]["bwd_mem"].append(res["bwd_peak_mem_mb"])

    # Plotting
    os.makedirs("benchmarking/layers_torch", exist_ok=True)

    for name, data in results.items():
        fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"PyTorch {name} Performance Benchmark")

        # Time plot
        ax_time.plot(data["hs"], data["fwd_time"], label="Fwd Time (ms)", marker='o')
        ax_time.plot(data["hs"], data["bwd_time"], label="Bwd Time (ms)", marker='x')
        ax_time.set_title("Execution Time")
        ax_time.set_xlabel("Hidden Size")
        ax_time.set_ylabel("Time (ms)")
        ax_time.set_xscale('log', base=2)
        ax_time.set_xticks(data["hs"])
        ax_time.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_time.grid(True, linestyle='--', alpha=0.7)
        ax_time.legend()

        # Memory plot
        ax_mem.plot(data["hs"], data["fwd_mem"], label="Fwd Peak Mem (MB)", marker='o', color='green')
        ax_mem.plot(data["hs"], data["bwd_mem"], label="Bwd Peak Mem (MB)", marker='x', color='red')
        ax_mem.set_title("Peak GPU Memory Usage")
        ax_mem.set_xlabel("Hidden Size")
        ax_mem.set_ylabel("Memory (MB)")
        ax_mem.set_xscale('log', base=2)
        ax_mem.set_xticks(data["hs"])
        ax_mem.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_mem.grid(True, linestyle='--', alpha=0.7)
        ax_mem.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = f"benchmarking/layers/pytorch_{name.lower()}_benchmark.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Results for PyTorch {name} saved to {save_path}")


if __name__ == "__main__":
    run_benchmarks()
