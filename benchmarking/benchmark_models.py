import minitorch
import matplotlib.pyplot as plt
import numpy as np
import os

from minitorch import bench_utils
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.transformer import DecoderLM, ViT
from minitorch.tensor_ops import TensorBackend

# Use CUDA backend instance
backend = TensorBackend(CudaKernelOps)

HIDDEN_SIZES = [128, 256, 512, 1024]
BATCH_SIZE = 8
SEQ_LEN = 32
N_VOCAB = 10000
N_HEAD = 8

def run_benchmarks():
    results = {}
    
    models = {
        # "DecoderLM": lambda hs: (
        #     DecoderLM(
        #         n_vocab=N_VOCAB,
        #         n_embd=hs,
        #         n_head=N_HEAD,
        #         n_positions=SEQ_LEN,
        #         p_dropout=0.1,
        #         ln_eps=1e-5,
        #         bias=True,
        #         backend=backend
        #     ),
        #     lambda: minitorch.tensor_from_numpy(np.random.randint(0, N_VOCAB, (BATCH_SIZE, SEQ_LEN)), backend=backend)
        # ),
        "ViT": lambda hs: (
            ViT(
                n_embd=hs,
                n_head=N_HEAD,
                n_trans_layers=4,
                patch_size=16,
                n_classes=10,
                max_patches=256,
                n_channels=3,
                backend=backend
            ),
            lambda: minitorch.tensor_from_numpy(np.random.randn(BATCH_SIZE, 3, 64, 64).astype(np.float32), backend=backend)
        )
    }

    for name, factory in models.items():
        print(f"Benchmarking {name}...")
        results[name] = {"hs": [], "fwd_time": [], "bwd_time": [], "fwd_mem": [], "bwd_mem": []}
        for hs in HIDDEN_SIZES:
            print(f"  Hidden Size: {hs}")
            module, input_fn = factory(hs)
            module.train()
            
            res = bench_utils.benchmark_module(module, input_fn, n_iters=10, n_warmup=5)
            
            results[name]["hs"].append(hs)
            results[name]["fwd_time"].append(res["fwd_time_ms"])
            results[name]["bwd_time"].append(res["bwd_time_ms"])
            results[name]["fwd_mem"].append(res["fwd_peak_mem_mb"])
            results[name]["bwd_mem"].append(res["bwd_peak_mem_mb"])

    # Plotting
    os.makedirs("benchmarking/layers", exist_ok=True)

    for name, data in results.items():
        fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"{name} Performance Benchmark")
        
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
        save_path = f"benchmarking/layers/{name.lower()}_benchmark.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Results for {name} saved to {save_path}")

if __name__ == "__main__":
    run_benchmarks()
