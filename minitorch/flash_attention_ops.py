"""
Flash Attention 1 — Python interface to the compiled CUDA shared library
(flash_attention.so built from src/flash_attention.cu).

Provides
--------
FA_D_MAX        : int  — maximum head-dimension supported by the kernel (64)
flash_attn_available : bool — True when the .so was successfully loaded

flash_attention_forward(Q_np, K_np, V_np, scale, causal) -> (O_np, L_np)
flash_attention_backward(Q_np, K_np, V_np, O_np, L_np, dO_np, scale, causal)
    -> (dQ_np, dK_np, dV_np)

FlashAttentionFunction  : minitorch autodiff Function that wires the above
                          into the minitorch computation graph.

All numpy arrays must be contiguous float32 of shape (batch, T, d).
"""

from __future__ import annotations

import ctypes
import os
from typing import Tuple

import numpy as np

import minitorch
from .autodiff import Context
from .tensor_functions import Function
from .tensor_ops import TensorBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FA_D_MAX = 64  # kernel supports d <= FA_D_MAX


# ---------------------------------------------------------------------------
# Library loading — lazy, so import of this module does not crash when the
# .so has not been compiled yet.
# ---------------------------------------------------------------------------
_lib = None
flash_attn_available = False


def _load_lib() -> bool:
    """Try to load flash_attention.so; return True on success."""
    global _lib, flash_attn_available
    if _lib is not None:
        return flash_attn_available

    so_path = os.path.join(
        os.path.dirname(__file__), "cuda_kernels", "flash_attention.so"
    )
    if not os.path.exists(so_path):
        return False

    try:
        lib = ctypes.CDLL(so_path)

        # --- Forward ---
        lib.flashAttentionForward.restype = None
        lib.flashAttentionForward.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Q
            ctypes.POINTER(ctypes.c_float),  # K
            ctypes.POINTER(ctypes.c_float),  # V
            ctypes.POINTER(ctypes.c_float),  # O  (output)
            ctypes.POINTER(ctypes.c_float),  # L  (lse output)
            ctypes.c_int,    # batch
            ctypes.c_int,    # T
            ctypes.c_int,    # d
            ctypes.c_float,  # scale
            ctypes.c_int,    # causal
        ]

        # --- Backward ---
        lib.flashAttentionBackward.restype = None
        lib.flashAttentionBackward.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Q
            ctypes.POINTER(ctypes.c_float),  # K
            ctypes.POINTER(ctypes.c_float),  # V
            ctypes.POINTER(ctypes.c_float),  # O
            ctypes.POINTER(ctypes.c_float),  # L
            ctypes.POINTER(ctypes.c_float),  # dO
            ctypes.POINTER(ctypes.c_float),  # dQ (output)
            ctypes.POINTER(ctypes.c_float),  # dK (output)
            ctypes.POINTER(ctypes.c_float),  # dV (output)
            ctypes.c_int,    # batch
            ctypes.c_int,    # T
            ctypes.c_int,    # d
            ctypes.c_float,  # scale
            ctypes.c_int,    # causal
        ]

        _lib = lib
        flash_attn_available = True
    except Exception as e:
        print(f"[flash_attention_ops] Could not load flash_attention.so: {e}")
        flash_attn_available = False

    return flash_attn_available


def _fp(arr: np.ndarray) -> ctypes.POINTER:
    """Return a ctypes float* pointer to a contiguous float32 numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ---------------------------------------------------------------------------
# Numpy-level interfaces (no autograd)
# ---------------------------------------------------------------------------

def flash_attention_forward(
    Q_np: np.ndarray,
    K_np: np.ndarray,
    V_np: np.ndarray,
    scale: float,
    causal: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Flash Attention forward.

    Parameters
    ----------
    Q_np, K_np, V_np : ndarray, shape (batch, T, d), float32, contiguous
    scale            : softmax scale factor (typically 1/sqrt(d))
    causal           : whether to apply causal (autoregressive) mask

    Returns
    -------
    O_np : ndarray, shape (batch, T, d), float32
    L_np : ndarray, shape (batch, T),    float32  — log-sum-exp per row
    """
    if not _load_lib():
        raise RuntimeError(
            "flash_attention.so not found. Run compile_cuda.sh first."
        )
    batch, T, d = Q_np.shape
    assert d <= FA_D_MAX, (
        f"Flash Attention kernel supports d <= {FA_D_MAX}, got d={d}. "
        "Use vanilla MultiHeadAttention for larger head dimensions."
    )

    Q_c = np.ascontiguousarray(Q_np, dtype=np.float32)
    K_c = np.ascontiguousarray(K_np, dtype=np.float32)
    V_c = np.ascontiguousarray(V_np, dtype=np.float32)
    O_np = np.zeros((batch, T, d), dtype=np.float32)
    L_np = np.zeros((batch, T),    dtype=np.float32)

    _lib.flashAttentionForward(
        _fp(Q_c), _fp(K_c), _fp(V_c),
        _fp(O_np), _fp(L_np),
        ctypes.c_int(batch), ctypes.c_int(T), ctypes.c_int(d),
        ctypes.c_float(scale), ctypes.c_int(int(causal)),
    )
    return O_np, L_np


def flash_attention_backward(
    Q_np:  np.ndarray,
    K_np:  np.ndarray,
    V_np:  np.ndarray,
    O_np:  np.ndarray,
    L_np:  np.ndarray,
    dO_np: np.ndarray,
    scale: float,
    causal: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Flash Attention backward.

    Parameters
    ----------
    Q_np, K_np, V_np : saved from forward, shape (batch, T, d)
    O_np             : saved output from forward, shape (batch, T, d)
    L_np             : saved log-sum-exp from forward, shape (batch, T)
    dO_np            : upstream gradient, shape (batch, T, d)
    scale            : same scale used in forward
    causal           : same flag used in forward

    Returns
    -------
    dQ_np, dK_np, dV_np : ndarray, shape (batch, T, d), float32
    """
    if not _load_lib():
        raise RuntimeError(
            "flash_attention.so not found. Run compile_cuda.sh first."
        )
    batch, T, d = Q_np.shape

    Q_c  = np.ascontiguousarray(Q_np,  dtype=np.float32)
    K_c  = np.ascontiguousarray(K_np,  dtype=np.float32)
    V_c  = np.ascontiguousarray(V_np,  dtype=np.float32)
    O_c  = np.ascontiguousarray(O_np,  dtype=np.float32)
    L_c  = np.ascontiguousarray(L_np,  dtype=np.float32)
    dO_c = np.ascontiguousarray(dO_np, dtype=np.float32)

    dQ_np = np.zeros((batch, T, d), dtype=np.float32)
    dK_np = np.zeros((batch, T, d), dtype=np.float32)
    dV_np = np.zeros((batch, T, d), dtype=np.float32)

    _lib.flashAttentionBackward(
        _fp(Q_c), _fp(K_c), _fp(V_c),
        _fp(O_c), _fp(L_c), _fp(dO_c),
        _fp(dQ_np), _fp(dK_np), _fp(dV_np),
        ctypes.c_int(batch), ctypes.c_int(T), ctypes.c_int(d),
        ctypes.c_float(scale), ctypes.c_int(int(causal)),
    )
    return dQ_np, dK_np, dV_np


# ---------------------------------------------------------------------------
# Minitorch autodiff Function
# ---------------------------------------------------------------------------

class FlashAttentionFunction(Function):
    """
    Minitorch autodiff wrapper for Flash Attention 1.

    Inputs
    ------
    Q      : Tensor (batch, T, d)  — batch = B * n_heads
    K      : Tensor (batch, T, d)
    V      : Tensor (batch, T, d)
    scale  : Tensor scalar  (1,)   — 1 / sqrt(d); no gradient
    causal : Tensor scalar  (1,)   — 0.0 or 1.0;  no gradient

    Output
    ------
    O : Tensor (batch, T, d)

    The intermediate log-sum-exp L is stored on the context and is used
    in the backward pass to recompute attention weights without storing
    the full T × T matrix.
    """

    @staticmethod
    def forward(
        ctx: Context,
        Q: "minitorch.Tensor",
        K: "minitorch.Tensor",
        V: "minitorch.Tensor",
        scale: "minitorch.Tensor",
        causal: "minitorch.Tensor",
    ) -> "minitorch.Tensor":
        batch, T, d = Q.shape
        scale_val  = float(scale.item())
        causal_val = bool(causal.item())

        # Extract contiguous float32 numpy arrays
        Q_np  = np.ascontiguousarray(
            Q.contiguous()._tensor._storage.reshape(batch, T, d), dtype=np.float32
        )
        K_np  = np.ascontiguousarray(
            K.contiguous()._tensor._storage.reshape(batch, T, d), dtype=np.float32
        )
        V_np  = np.ascontiguousarray(
            V.contiguous()._tensor._storage.reshape(batch, T, d), dtype=np.float32
        )

        O_np, L_np = flash_attention_forward(Q_np, K_np, V_np, scale_val, causal_val)

        # Save tensors for backward
        ctx.save_for_backward(Q, K, V)
        # Store numpy arrays directly — they are not minitorch tensors but
        # Python objects, which the Context can hold as plain attributes.
        ctx._O_np    = O_np
        ctx._L_np    = L_np
        ctx._scale   = scale_val
        ctx._causal  = causal_val
        ctx._shape   = (batch, T, d)

        return minitorch.Tensor.make(
            O_np.flatten().tolist(), (batch, T, d), backend=Q.backend
        )

    @staticmethod
    def backward(
        ctx: Context,
        dO: "minitorch.Tensor",
    ) -> tuple:
        Q, K, V     = ctx.saved_values
        batch, T, d = ctx._shape
        scale_val   = ctx._scale
        causal_val  = ctx._causal

        Q_np  = np.ascontiguousarray(
            Q.contiguous()._tensor._storage.reshape(batch, T, d), dtype=np.float32
        )
        K_np  = np.ascontiguousarray(
            K.contiguous()._tensor._storage.reshape(batch, T, d), dtype=np.float32
        )
        V_np  = np.ascontiguousarray(
            V.contiguous()._tensor._storage.reshape(batch, T, d), dtype=np.float32
        )
        dO_np = np.ascontiguousarray(
            dO.contiguous()._tensor._storage.reshape(batch, T, d), dtype=np.float32
        )

        dQ_np, dK_np, dV_np = flash_attention_backward(
            Q_np, K_np, V_np,
            ctx._O_np, ctx._L_np,
            dO_np, scale_val, causal_val,
        )

        backend = Q.backend

        dQ = minitorch.Tensor.make(dQ_np.flatten().tolist(), (batch, T, d), backend=backend)
        dK = minitorch.Tensor.make(dK_np.flatten().tolist(), (batch, T, d), backend=backend)
        dV = minitorch.Tensor.make(dV_np.flatten().tolist(), (batch, T, d), backend=backend)

        # Gradients for Q, K, V, scale (0.0), causal (0.0)
        zero = minitorch.Tensor.make([0.0], (1,), backend=backend)
        return dQ, dK, dV, zero, zero


def flash_attention(
    Q: "minitorch.Tensor",
    K: "minitorch.Tensor",
    V: "minitorch.Tensor",
    scale: float,
    causal: bool = False,
) -> "minitorch.Tensor":
    """
    Functional interface for Flash Attention.

    Parameters
    ----------
    Q, K, V : Tensors of shape (batch, T, d), where batch = B * n_heads.
              Must be contiguous (call .contiguous() before if needed).
    scale   : Attention scale, typically ``1 / math.sqrt(d)``.
    causal  : Whether to mask future positions.

    Returns
    -------
    O : Tensor of shape (batch, T, d)
    """
    backend     = Q.backend
    scale_t     = minitorch.Tensor.make([scale],         (1,), backend=backend)
    causal_t    = minitorch.Tensor.make([float(causal)], (1,), backend=backend)
    return FlashAttentionFunction.apply(Q, K, V, scale_t, causal_t)


# Eagerly attempt to load the library when the module is first imported so
# that `flash_attn_available` reflects the true state immediately.
_load_lib()
