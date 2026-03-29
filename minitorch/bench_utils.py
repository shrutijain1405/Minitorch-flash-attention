import time
import threading
import os
import pynvml
import pycuda.driver as cuda_driver
import pycuda.autoinit


class NvmlMemPoller(threading.Thread):
    """
    A more accurate poller that uses NVML to track the specific memory usage
    of the current process.
    """

    def __init__(self, interval_s: float = 0.00001):
        super().__init__(daemon=True)
        self._interval = interval_s
        self._stop_event = threading.Event()
        self.pid = os.getpid()
        self.baseline_bytes: int = 0
        self.peak_bytes: int = 0

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _get_process_memory(self) -> int:
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            for proc in processes:
                if proc.pid == self.pid:
                    return proc.usedGpuMemory
        except pynvml.NVMLError:
            pass
        return 0

    def run(self) -> None:
        self.baseline_bytes = self._get_process_memory()
        self.peak_bytes = self.baseline_bytes
        while not self._stop_event.is_set():
            used = self._get_process_memory()
            if used > self.peak_bytes:
                self.peak_bytes = used
            time.sleep(self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()

    @property
    def peak_mb(self) -> float:
        return max(0, self.peak_bytes - self.baseline_bytes) / 1024**2


class GpuMemPoller(threading.Thread):
    def __init__(self, interval_s: float = 0.00001):
        super().__init__(daemon=True)
        self._interval = interval_s
        self._stop_event = threading.Event()
        self._ctx = pycuda.autoinit.context
        self.baseline_bytes: int = 0
        self.peak_bytes: int = 0

    def run(self) -> None:
        self._ctx.push()
        try:
            free, total = cuda_driver.mem_get_info()
            self.baseline_bytes = total - free
            self.peak_bytes = self.baseline_bytes
            while not self._stop_event.is_set():
                free, total = cuda_driver.mem_get_info()
                used = total - free
                if used > self.peak_bytes:
                    self.peak_bytes = used
                time.sleep(self._interval)
        finally:
            self._ctx.pop()

    def stop(self) -> None:
        self._stop_event.set()
        self.join()

    @property
    def peak_mb(self) -> float:
        return max(0, self.peak_bytes - self.baseline_bytes) / 1024**2


def benchmark_module(module, input_fn, n_iters=10, n_warmup=2):
    inp = input_fn()

    for _ in range(n_warmup):
        out = module(inp)
        out.sum().backward()

    fwd_times, bwd_times = [], []
    fwd_peak_mems, bwd_peak_mems = [], []

    for _ in range(n_iters):
        pycuda.autoinit.context.synchronize()

        # Forward
        poller = NvmlMemPoller()
        poller.start()
        t0 = time.perf_counter()
        out = module(inp)
        pycuda.autoinit.context.synchronize()
        t1 = time.perf_counter()
        poller.stop()

        fwd_times.append(t1 - t0)
        fwd_peak_mems.append(poller.peak_mb)

        # Backward
        out_sum = out.sum()
        poller = NvmlMemPoller()
        poller.start()
        t0 = time.perf_counter()
        out_sum.backward()
        pycuda.autoinit.context.synchronize()
        t1 = time.perf_counter()
        poller.stop()

        bwd_times.append(t1 - t0)
        bwd_peak_mems.append(poller.peak_mb)

    res = {
        "fwd_time_ms": sum(fwd_times) / len(fwd_times) * 1000,
        "bwd_time_ms": sum(bwd_times) / len(bwd_times) * 1000,
        "fwd_peak_mem_mb": max(fwd_peak_mems),
        "bwd_peak_mem_mb": max(bwd_peak_mems),
    }

    return res
