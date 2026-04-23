"""
Correctness tests for Flash Attention forward pass.

Strategy: run the same Q/K/V through both standard attention and flash
attention, then compare outputs.  We test at multiple (batch, heads, seq_len)
combinations that are all compatible with d=64.
"""

import pytest
import numpy as np
import numba

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.tensor_functions import tensor_from_numpy

datatype = np.float32

_BACKEND = pytest.param(
    minitorch.TensorBackend(CudaKernelOps),
    marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU"),
)


def standard_attention(q_np, k_np, v_np):
    """Pure numpy reference: scaled dot-product attention.

    Args:
        q_np, k_np, v_np : (BH, N, d) float32 numpy arrays
    Returns:
        (BH, N, d) numpy array
    """
    scale = q_np.shape[2] ** -0.5
    scores = np.matmul(q_np, k_np.transpose(0, 2, 1)) * scale  # (BH, N, N)
    scores -= scores.max(axis=-1, keepdims=True)                # numerical stability
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, v_np)                             # (BH, N, d)


def flash_attention(q_np, k_np, v_np, backend):
    """Flash attention via CudaKernelOps kernel.

    Args:
        q_np, k_np, v_np : (BH, N, d) float32 numpy arrays
    Returns:
        (BH, N, d) numpy array
    """
    d = q_np.shape[2]
    scale = d ** -0.5
    q = tensor_from_numpy(q_np, backend=backend)
    k = tensor_from_numpy(k_np, backend=backend)
    v = tensor_from_numpy(v_np, backend=backend)
    out = CudaKernelOps.flash_attention_forward(q, k, v, scale)
    return out.to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: kernel correctness across shapes
# N=100 is deliberately not a multiple of Bc=64 to stress padding logic
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("BH", [1, 4, 16])
@pytest.mark.parametrize("N",  [64, 128, 100])
@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_flash_attention_forward_correctness(BH, N, backend):
    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    k_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    v_np = rng.standard_normal((BH, N, 64)).astype(datatype)

    ref = standard_attention(q_np, k_np, v_np)
    got = flash_attention(q_np, k_np, v_np, backend)

    np.testing.assert_allclose(
        got, ref, atol=1e-3, rtol=1e-3,
        err_msg=f"Flash attention mismatch for BH={BH}, N={N}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: end-to-end MultiHeadAttention with identical weights
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_flash_attention_via_multihead(backend):
    B, N, n_embd, n_head = 2, 64, 64, 1   # d = n_embd / n_head = 64
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((B, N, n_embd)).astype(datatype)

    x_std   = tensor_from_numpy(x_np, backend=backend)
    x_flash = tensor_from_numpy(x_np, backend=backend)

    std_layer = minitorch.MultiHeadAttention(
        n_embd, n_head, causal=False, p_dropout=0.0,
        bias=False, backend=backend, use_flash_attn=False,
    )
    flash_layer = minitorch.MultiHeadAttention(
        n_embd, n_head, causal=False, p_dropout=0.0,
        bias=False, backend=backend, use_flash_attn=True,
    )

    # Share identical weights so outputs are directly comparable
    w = rng.standard_normal((n_embd, n_embd)).astype(datatype)
    for layer in (std_layer, flash_layer):
        layer.q_projection.weights.value   = tensor_from_numpy(w.copy(), backend=backend)
        layer.k_projection.weights.value   = tensor_from_numpy(w.copy(), backend=backend)
        layer.v_projection.weights.value   = tensor_from_numpy(w.copy(), backend=backend)
        layer.out_projection.weights.value = tensor_from_numpy(w.copy(), backend=backend)

    out_std   = std_layer(x_std).to_numpy()
    out_flash = flash_layer(x_flash).to_numpy()

    np.testing.assert_allclose(
        out_flash, out_std, atol=1e-3, rtol=1e-3,
        err_msg="MultiHeadAttention flash vs standard output mismatch",
    )
