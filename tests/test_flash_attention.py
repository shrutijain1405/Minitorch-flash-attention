"""
Correctness tests for Flash Attention forward and backward passes.

Strategy: run the same Q/K/V through both standard attention and flash
attention, then compare outputs and gradients.  We test at multiple
(batch, heads, seq_len) combinations that are all compatible with d=64.
"""

import pytest
import numpy as np
import numba

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.flash_attention_fn import FlashAttentionFn
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


def flash_attention_fa2(q_np, k_np, v_np, backend):
    """FA2 forward via CudaKernelOps kernel."""
    d = q_np.shape[2]
    scale = d ** -0.5
    q = tensor_from_numpy(q_np, backend=backend)
    k = tensor_from_numpy(k_np, backend=backend)
    v = tensor_from_numpy(v_np, backend=backend)
    out, _ = CudaKernelOps.flash_attention_forward_fa2(q, k, v, scale)
    return out.to_numpy()


def flash_attention_fa1(q_np, k_np, v_np, backend):
    """FA1 forward via CudaKernelOps kernel."""
    d = q_np.shape[2]
    scale = d ** -0.5
    q = tensor_from_numpy(q_np, backend=backend)
    k = tensor_from_numpy(k_np, backend=backend)
    v = tensor_from_numpy(v_np, backend=backend)
    out, _ = CudaKernelOps.flash_attention_forward_fa1(q, k, v, scale)
    return out.to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1a: FA2 forward correctness across shapes
# N=100 is deliberately not a multiple of Bc=64 to stress padding logic
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("BH", [1, 4, 16])
@pytest.mark.parametrize("N",  [64, 128, 100])
@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_fa2_forward_correctness(BH, N, backend):
    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    k_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    v_np = rng.standard_normal((BH, N, 64)).astype(datatype)

    ref = standard_attention(q_np, k_np, v_np)
    got = flash_attention_fa2(q_np, k_np, v_np, backend)

    np.testing.assert_allclose(
        got, ref, atol=1e-3, rtol=1e-3,
        err_msg=f"FA2 forward mismatch for BH={BH}, N={N}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1b: FA1 forward correctness across shapes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("BH", [1, 4, 16])
@pytest.mark.parametrize("N",  [64, 128, 100])
@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_fa1_forward_correctness(BH, N, backend):
    rng = np.random.default_rng(42)
    q_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    k_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    v_np = rng.standard_normal((BH, N, 64)).astype(datatype)

    ref = standard_attention(q_np, k_np, v_np)
    got = flash_attention_fa1(q_np, k_np, v_np, backend)

    np.testing.assert_allclose(
        got, ref, atol=1e-3, rtol=1e-3,
        err_msg=f"FA1 forward mismatch for BH={BH}, N={N}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1c: FA1 and FA2 produce identical outputs
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("BH", [1, 4, 16])
@pytest.mark.parametrize("N",  [64, 128, 100])
@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_fa1_fa2_forward_agree(BH, N, backend):
    rng = np.random.default_rng(99)
    q_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    k_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    v_np = rng.standard_normal((BH, N, 64)).astype(datatype)

    fa1_out = flash_attention_fa1(q_np, k_np, v_np, backend)
    fa2_out = flash_attention_fa2(q_np, k_np, v_np, backend)

    np.testing.assert_allclose(
        fa1_out, fa2_out, atol=1e-3, rtol=1e-3,
        err_msg=f"FA1 vs FA2 output mismatch for BH={BH}, N={N}",
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


def standard_attention_backward(q_np, k_np, v_np, do_np):
    """Pure numpy reference: gradients of scaled dot-product attention.

    Args:
        q_np, k_np, v_np : (BH, N, d) float32 arrays
        do_np            : (BH, N, d) upstream gradient
    Returns:
        (dq, dk, dv) : each (BH, N, d)
    """
    scale = q_np.shape[2] ** -0.5
    scores = np.matmul(q_np, k_np.transpose(0, 2, 1)) * scale  # (BH, N, N)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)              # P = softmax

    dv = np.matmul(weights.transpose(0, 2, 1), do_np)          # (BH, N, d)
    dp = np.matmul(do_np, v_np.transpose(0, 2, 1))             # (BH, N, N)
    # softmax backward: dS_ij = P_ij * (dP_ij - sum_k P_ik * dP_ik)
    ds = weights * (dp - (weights * dp).sum(axis=-1, keepdims=True))
    dq = np.matmul(ds, k_np) * scale                           # (BH, N, d)
    dk = np.matmul(ds.transpose(0, 2, 1), q_np) * scale        # (BH, N, d)
    return dq, dk, dv


# ─────────────────────────────────────────────────────────────────────────────
# Test 3a: FA1 backward kernel correctness across shapes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("BH", [1, 4, 16])
@pytest.mark.parametrize("N",  [64, 128, 100])
@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_fa1_backward_correctness(BH, N, backend):
    rng = np.random.default_rng(7)
    q_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    k_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    v_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    do_np = rng.standard_normal((BH, N, 64)).astype(datatype)

    scale = 64 ** -0.5

    dq_ref, dk_ref, dv_ref = standard_attention_backward(q_np, k_np, v_np, do_np)

    q  = tensor_from_numpy(q_np,  backend=backend)
    k  = tensor_from_numpy(k_np,  backend=backend)
    v  = tensor_from_numpy(v_np,  backend=backend)
    do = tensor_from_numpy(do_np, backend=backend)
    o, L = CudaKernelOps.flash_attention_forward_fa2(q, k, v, scale)
    dq, dk, dv = CudaKernelOps.flash_attention_backward_fa1(q, k, v, o, do, L, scale)

    np.testing.assert_allclose(
        dq.to_numpy(), dq_ref, atol=1e-3, rtol=1e-3,
        err_msg=f"FA1 dQ mismatch for BH={BH}, N={N}",
    )
    np.testing.assert_allclose(
        dk.to_numpy(), dk_ref, atol=1e-3, rtol=1e-3,
        err_msg=f"FA1 dK mismatch for BH={BH}, N={N}",
    )
    np.testing.assert_allclose(
        dv.to_numpy(), dv_ref, atol=1e-3, rtol=1e-3,
        err_msg=f"FA1 dV mismatch for BH={BH}, N={N}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3b: FA2 backward kernel correctness across shapes
# Two-kernel split (DKV + DQ), no atomicAdd.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("BH", [1, 4, 16])
@pytest.mark.parametrize("N",  [64, 128, 100])
@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_fa2_backward_correctness(BH, N, backend):
    rng = np.random.default_rng(7)
    q_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    k_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    v_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    do_np = rng.standard_normal((BH, N, 64)).astype(datatype)

    scale = 64 ** -0.5

    dq_ref, dk_ref, dv_ref = standard_attention_backward(q_np, k_np, v_np, do_np)

    q  = tensor_from_numpy(q_np,  backend=backend)
    k  = tensor_from_numpy(k_np,  backend=backend)
    v  = tensor_from_numpy(v_np,  backend=backend)
    do = tensor_from_numpy(do_np, backend=backend)
    o, L = CudaKernelOps.flash_attention_forward_fa2(q, k, v, scale)
    dq, dk, dv = CudaKernelOps.flash_attention_backward_fa2(q, k, v, o, do, L, scale)

    np.testing.assert_allclose(
        dq.to_numpy(), dq_ref, atol=1e-3, rtol=1e-3,
        err_msg=f"FA2 dQ mismatch for BH={BH}, N={N}",
    )
    np.testing.assert_allclose(
        dk.to_numpy(), dk_ref, atol=1e-3, rtol=1e-3,
        err_msg=f"FA2 dK mismatch for BH={BH}, N={N}",
    )
    np.testing.assert_allclose(
        dv.to_numpy(), dv_ref, atol=1e-3, rtol=1e-3,
        err_msg=f"FA2 dV mismatch for BH={BH}, N={N}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3c: FA1 and FA2 backward produce identical gradients
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("BH", [1, 4, 16])
@pytest.mark.parametrize("N",  [64, 128, 100])
@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_fa1_fa2_backward_agree(BH, N, backend):
    rng = np.random.default_rng(55)
    q_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    k_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    v_np  = rng.standard_normal((BH, N, 64)).astype(datatype)
    do_np = rng.standard_normal((BH, N, 64)).astype(datatype)

    scale = 64 ** -0.5

    q  = tensor_from_numpy(q_np,  backend=backend)
    k  = tensor_from_numpy(k_np,  backend=backend)
    v  = tensor_from_numpy(v_np,  backend=backend)
    do = tensor_from_numpy(do_np, backend=backend)
    o, L = CudaKernelOps.flash_attention_forward_fa2(q, k, v, scale)

    dq1, dk1, dv1 = CudaKernelOps.flash_attention_backward_fa1(q, k, v, o, do, L, scale)
    dq2, dk2, dv2 = CudaKernelOps.flash_attention_backward_fa2(q, k, v, o, do, L, scale)

    np.testing.assert_allclose(
        dq1.to_numpy(), dq2.to_numpy(), atol=1e-3, rtol=1e-3,
        err_msg=f"FA1 vs FA2 dQ mismatch for BH={BH}, N={N}",
    )
    np.testing.assert_allclose(
        dk1.to_numpy(), dk2.to_numpy(), atol=1e-3, rtol=1e-3,
        err_msg=f"FA1 vs FA2 dK mismatch for BH={BH}, N={N}",
    )
    np.testing.assert_allclose(
        dv1.to_numpy(), dv2.to_numpy(), atol=1e-3, rtol=1e-3,
        err_msg=f"FA1 vs FA2 dV mismatch for BH={BH}, N={N}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: autograd integration — gradients flow through FlashAttentionFn
# dO = ones (from .sum()), so reference uses do_np = ones_like(O)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", [_BACKEND], ids=["CudaKernelOps"])
def test_flash_attention_autograd(backend):
    BH, N = 2, 64
    rng = np.random.default_rng(13)
    q_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    k_np = rng.standard_normal((BH, N, 64)).astype(datatype)
    v_np = rng.standard_normal((BH, N, 64)).astype(datatype)

    scale = 64 ** -0.5

    # Build reference: dO = ones because loss = output.sum()
    o_ref  = standard_attention(q_np, k_np, v_np)
    do_ref = np.ones_like(o_ref)
    dq_ref, dk_ref, dv_ref = standard_attention_backward(q_np, k_np, v_np, do_ref)

    # Autograd path through FlashAttentionFn
    q = tensor_from_numpy(q_np, backend=backend)
    k = tensor_from_numpy(k_np, backend=backend)
    v = tensor_from_numpy(v_np, backend=backend)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    o = FlashAttentionFn.apply(q, k, v)
    o.sum().backward()

    np.testing.assert_allclose(
        q.grad.to_numpy(), dq_ref, atol=1e-3, rtol=1e-3,
        err_msg="Autograd dQ mismatch",
    )
    np.testing.assert_allclose(
        k.grad.to_numpy(), dk_ref, atol=1e-3, rtol=1e-3,
        err_msg="Autograd dK mismatch",
    )
    np.testing.assert_allclose(
        v.grad.to_numpy(), dv_ref, atol=1e-3, rtol=1e-3,
        err_msg="Autograd dV mismatch",
    )
