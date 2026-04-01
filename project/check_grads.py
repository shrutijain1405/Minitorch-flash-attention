"""Quick test: verify which ViT parameters receive gradients after one backward pass."""

import sys
sys.path.append("./")
import numpy as np
import minitorch
from minitorch import TensorBackend
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.transformer import ViT

backend = TensorBackend(CudaKernelOps)

model = ViT(
    n_embd=64,
    n_head=4,
    n_channels=3,
    patch_size=8,
    n_trans_layers=2,
    n_classes=10,
    max_patches=256,
    backend=backend,
)
model.train()

x = minitorch.tensor_from_numpy(
    np.random.randn(1, 3, 16, 16).astype(np.float32), backend=backend
)
y = minitorch.tensor([0], backend=backend)

logits = model(x)
loss = minitorch.nn.softmax_loss(logits, y)
loss.sum().backward()

total = 0
got_grad = 0
print(f"{'Parameter':<50} {'Has grad?':<12} {'Grad norm'}")
print("-" * 80)
for name, param in model.named_parameters():
    total += 1
    grad = param.value.grad
    if grad is not None:
        grad_norm = np.abs(grad.to_numpy()).sum()
        has_grad = grad_norm > 1e-12
        if has_grad:
            got_grad += 1
        print(f"{name:<50} {'YES' if has_grad else 'ZERO':<12} {grad_norm:.6e}")
    else:
        print(f"{name:<50} {'NO (None)':<12} ---")

print(f"\n{got_grad}/{total} parameters received gradients")