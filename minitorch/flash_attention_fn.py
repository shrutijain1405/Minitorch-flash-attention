from typing import Tuple
from .autodiff import Context
from .tensor_functions import Function
import minitorch


class FlashAttentionFnFA1(Function):
    """FA1 autograd wrapper — uses FA1 forward/backward kernels."""

    @staticmethod
    def forward(ctx: Context, q: "minitorch.Tensor", k: "minitorch.Tensor", v: "minitorch.Tensor") -> "minitorch.Tensor":
        from .cuda_kernel_ops import CudaKernelOps
        scale = q.shape[2] ** -0.5
        o, L = CudaKernelOps.flash_attention_forward_fa1(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, L, scale)
        return o

    @staticmethod
    def backward(ctx: Context, do: "minitorch.Tensor") -> Tuple["minitorch.Tensor", "minitorch.Tensor", "minitorch.Tensor"]:
        from .cuda_kernel_ops import CudaKernelOps
        q, k, v, o, L, scale = ctx.saved_values
        dq, dk, dv = CudaKernelOps.flash_attention_backward_fa1(q, k, v, o, do, L, scale)
        return dq, dk, dv


class FlashAttentionFn(Function):
    """
    Autograd-integrated Flash Attention (forward + backward).

    Inputs : q, k, v — each (BH, N, 64)
    Output : o       — (BH, N, 64)

    Saves q, k, v, o, L (logsumexp) and scale in ctx for the backward pass.
    """

    @staticmethod
    def forward(ctx: Context, q: "minitorch.Tensor", k: "minitorch.Tensor", v: "minitorch.Tensor") -> "minitorch.Tensor":
        from .cuda_kernel_ops import CudaKernelOps
        scale = q.shape[2] ** -0.5
        o, L = CudaKernelOps.flash_attention_forward_fa2(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, L, scale)
        return o

    @staticmethod
    def backward(ctx: Context, do: "minitorch.Tensor") -> Tuple["minitorch.Tensor", "minitorch.Tensor", "minitorch.Tensor"]:
        from .cuda_kernel_ops import CudaKernelOps
        q, k, v, o, L, scale = ctx.saved_values
        dq, dk, dv = CudaKernelOps.flash_attention_backward_fa2(q, k, v, o, do, L, scale)
        return dq, dk, dv
