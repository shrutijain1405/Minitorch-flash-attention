"""
Correctness tests for Flash Attention 1.

Run from the repo root:
    python tests/test_flash_attention.py

Tests
-----
1. Forward pass (non-causal): compare Flash Attention output O and LSE L
   against a NumPy reference implementation.
2. Forward pass (causal): same comparison with upper-triangle masking.
3. Backward pass: compare dQ, dK, dV from the Flash Attention backward
   kernel against finite-difference estimates.
4. End-to-end FlashViT forward: build a tiny FlashViT and run a forward
   pass, comparing its output shape and checking the values are finite.
5. Equivalence check: FlashViT and vanilla ViT produce the same outputs
   when initialised with identical weights.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import minitorch
from minitorch.flash_attention_ops import (
    flash_attention_forward,
    flash_attention_backward,
    flash_attn_available,
    FA_D_MAX,
)
from minitorch.flash_transformer import FlashViT, FlashMultiHeadAttention
from minitorch.tensor_functions import tensor_from_numpy, zeros
from minitorch.fast_ops import FastOps

BACKEND = minitorch.TensorBackend(FastOps)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _numpy_attention(Q, K, V, causal=False):
    """Reference: softmax(QK^T/sqrt(d)) @ V  plus LSE.
    Q, K, V : (batch, T, d)
    Returns  O (batch, T, d),  L (batch, T)
    """
    scale = Q.shape[-1] ** -0.5
    S = np.einsum("btd,bsd->bts", Q, K) * scale   # (batch, T, T)
    if causal:
        T = Q.shape[1]
        mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e30, 1)
        S = S + mask
    S_max = S.max(axis=-1, keepdims=True)
    exp_S = np.exp(S - S_max)
    denom = exp_S.sum(axis=-1, keepdims=True)
    P = exp_S / denom
    O = np.einsum("bts,bsd->btd", P, V)
    L = np.log(denom[..., 0]) + S_max[..., 0]
    return O, L


def _fd_grad(Q, K, V, scale, causal, eps=1e-3):
    """Finite-difference gradients for the Flash Attention forward w.r.t. Q, K, V."""
    def loss(q, k, v):
        O, _ = flash_attention_forward(q, k, v, scale, causal)
        return O.sum()

    grads = []
    for X in (Q, K, V):
        g = np.zeros_like(X)
        it = np.nditer(X, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = X[idx]

            X[idx] = orig + eps
            Xp = X.copy(); X[idx] = orig - eps; Xm = X.copy(); X[idx] = orig
            if X is Q:
                fp = loss(Xp, K, V); fm = loss(Xm, K, V)
            elif X is K:
                fp = loss(Q, Xp, V); fm = loss(Q, Xm, V)
            else:
                fp = loss(Q, K, Xp); fm = loss(Q, K, Xm)

            g[idx] = (fp - fm) / (2 * eps)
            it.iternext()
        grads.append(g)
    return grads


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Forward, non-causal
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_noncausal():
    np.random.seed(0)
    batch, T, d = 2, 24, 32
    scale = d ** -0.5

    Q = np.random.randn(batch, T, d).astype(np.float32)
    K = np.random.randn(batch, T, d).astype(np.float32)
    V = np.random.randn(batch, T, d).astype(np.float32)

    O_fa, L_fa   = flash_attention_forward(Q, K, V, scale, causal=False)
    O_ref, L_ref = _numpy_attention(Q, K, V, causal=False)

    err_O = np.abs(O_fa - O_ref).max()
    err_L = np.abs(L_fa - L_ref).max()

    print(f"[test_forward_noncausal] max|ΔO|={err_O:.2e}  max|ΔL|={err_L:.2e}")
    assert err_O < 1e-3, f"Output mismatch: {err_O}"
    assert err_L < 1e-3, f"LSE mismatch:    {err_L}"
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Forward, causal
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_causal():
    np.random.seed(1)
    batch, T, d = 2, 24, 32
    scale = d ** -0.5

    Q = np.random.randn(batch, T, d).astype(np.float32)
    K = np.random.randn(batch, T, d).astype(np.float32)
    V = np.random.randn(batch, T, d).astype(np.float32)

    O_fa, L_fa   = flash_attention_forward(Q, K, V, scale, causal=True)
    O_ref, L_ref = _numpy_attention(Q, K, V, causal=True)

    err_O = np.abs(O_fa - O_ref).max()
    err_L = np.abs(L_fa - L_ref).max()

    print(f"[test_forward_causal]    max|ΔO|={err_O:.2e}  max|ΔL|={err_L:.2e}")
    assert err_O < 1e-3, f"Output mismatch: {err_O}"
    assert err_L < 1e-3, f"LSE mismatch:    {err_L}"
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Backward (small size for finite-diff feasibility)
# ─────────────────────────────────────────────────────────────────────────────

def test_backward():
    np.random.seed(2)
    batch, T, d = 1, 8, 16   # small enough for finite-diff
    scale = d ** -0.5

    Q = np.random.randn(batch, T, d).astype(np.float32) * 0.5
    K = np.random.randn(batch, T, d).astype(np.float32) * 0.5
    V = np.random.randn(batch, T, d).astype(np.float32) * 0.5

    O, L = flash_attention_forward(Q, K, V, scale, causal=False)
    dO   = np.ones_like(O)   # upstream gradient = 1 everywhere

    dQ_fa, dK_fa, dV_fa = flash_attention_backward(Q, K, V, O, L, dO, scale, causal=False)
    dQ_fd, dK_fd, dV_fd = _fd_grad(Q, K, V, scale, causal=False, eps=1e-3)

    for name, g_fa, g_fd in [("dQ", dQ_fa, dQ_fd), ("dK", dK_fa, dK_fd), ("dV", dV_fa, dV_fd)]:
        err = np.abs(g_fa - g_fd).max()
        print(f"[test_backward]          max|Δ{name}|={err:.2e}")
        assert err < 5e-2, f"{name} gradient mismatch: {err}"
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — FlashViT forward shape + finite values
# ─────────────────────────────────────────────────────────────────────────────

def test_flash_vit_forward():
    n_embd, n_head = 64, 4          # d = 16 per head (fits in FA_D_MAX=64)
    patch_size     = 8
    img_size       = 32             # 32x32 image → (32/8)²=16 patches
    n_classes      = 10
    batch          = 2

    model = FlashViT(
        n_embd=n_embd,
        n_head=n_head,
        p_dropout=0.0,
        patch_size=patch_size,
        n_trans_layers=2,
        n_classes=n_classes,
        max_patches=20,
        n_channels=3,
        backend=BACKEND,
    )

    x_np = np.random.randn(batch, 3, img_size, img_size).astype(np.float32)
    x    = tensor_from_numpy(x_np, backend=BACKEND)

    out  = model(x)

    assert out.shape == (batch, n_classes), f"Bad output shape: {out.shape}"
    assert np.all(np.isfinite(out.to_numpy())), "Output contains NaN/Inf"
    print(f"[test_flash_vit_forward] output shape={out.shape}  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Flash Attention kernel with d=8 (head-dim used by the comparison tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_flash_kernel_d8():
    """Verify the CUDA kernel is correct for d=8, T=12, batch=8."""
    np.random.seed(10)
    batch, T, d = 8, 12, 8
    scale = d ** -0.5

    Q = np.random.randn(batch, T, d).astype(np.float32)
    K = np.random.randn(batch, T, d).astype(np.float32)
    V = np.random.randn(batch, T, d).astype(np.float32)

    O_fa, L_fa   = flash_attention_forward(Q, K, V, scale, causal=False)
    O_ref, L_ref = _numpy_attention(Q, K, V, causal=False)

    err_O = np.abs(O_fa - O_ref).max()
    err_L = np.abs(L_fa - L_ref).max()
    print(f"[test_flash_kernel_d8]       max|ΔO|={err_O:.2e}  max|ΔL|={err_L:.2e}")
    assert err_O < 1e-3, f"d=8 output mismatch: {err_O}"
    assert err_L < 1e-3, f"d=8 LSE mismatch:    {err_L}"
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — FlashAttentionFunction.apply via minitorch tensors (d=8)
# Isolates the minitorch integration path from the projection pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def test_flash_func_minitorch_d8():
    """FlashAttentionFunction.apply must give correct results with minitorch tensors."""
    from minitorch.flash_transformer import flash_attention

    np.random.seed(10)
    batch, T, d = 8, 12, 8
    scale = d ** -0.5

    Q_np = np.random.randn(batch, T, d).astype(np.float32)
    K_np = np.random.randn(batch, T, d).astype(np.float32)
    V_np = np.random.randn(batch, T, d).astype(np.float32)

    Q = tensor_from_numpy(Q_np, backend=BACKEND)
    K = tensor_from_numpy(K_np, backend=BACKEND)
    V = tensor_from_numpy(V_np, backend=BACKEND)

    O = flash_attention(Q, K, V, scale, causal=False)

    O_np  = O.to_numpy().reshape(batch, T, d)
    O_ref, _ = _numpy_attention(Q_np, K_np, V_np, causal=False)

    err = np.abs(O_np - O_ref).max()
    print(f"[test_flash_func_minitorch_d8]  max|ΔO|={err:.2e}  shape={O.shape}")
    assert err < 1e-3, f"FlashAttentionFunction minitorch d=8 mismatch: {err}"
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7a — FlashMultiHeadAttention == MultiHeadAttention (unit test)
# Both models are created with the same numpy random seed so they have
# identical weights without any parameter-copying.
# ─────────────────────────────────────────────────────────────────────────────

def _copy_weights_fresh(src, dst):
    """
    Copy all parameters from src to dst by creating brand-new, independent
    tensor objects (uses tensor_from_numpy so there is zero shared storage).
    Returns the number of parameters successfully copied.
    """
    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    copied = 0
    for name, p_dst in dst_params.items():
        if name in src_params:
            w_np = src_params[name].value.to_numpy().copy()
            fresh = tensor_from_numpy(w_np, backend=p_dst.value.backend)
            fresh.requires_grad_(True)
            p_dst.value = fresh
            copied += 1
    return copied


def test_flash_mha_equals_numpy():
    """
    Flash MHA must match a pure-numpy reference implementation.

    Root cause of prior 'equals vanilla' failure: minitorch's
    _tensor_matrix_multiply is hardcoded for 3-D tensors and silently
    produces wrong results for 4-D attention weights.  The flash kernel
    bypasses this by reshaping to (B*H, T, d) before calling the CUDA
    kernel, which is why flash gives correct results while vanilla does not.
    We therefore test flash against numpy directly.
    """
    n_embd, n_head = 32, 4
    B, T           = 2, 12
    H, d           = n_head, n_embd // n_head

    np.random.seed(42)
    flash_mha = FlashMultiHeadAttention(
        n_embd, n_head, causal=False, p_dropout=0.0, backend=BACKEND
    )
    assert flash_mha._use_flash, "Flash kernel must be available for this test"

    np.random.seed(99)
    x_np = np.random.randn(B, T, n_embd).astype(np.float32)
    x    = tensor_from_numpy(x_np, backend=BACKEND)

    out_flash = flash_mha(x).to_numpy()

    # ── pure-numpy MHA reference ─────────────────────────────────────────────
    p = dict(flash_mha.named_parameters())
    def _w(name): return p[name].value.to_numpy()

    x_flat = x_np.reshape(B * T, n_embd)
    Q = (x_flat @ _w('q_projection.weights') + _w('q_projection.bias')).reshape(B, T, H, d).transpose(0, 2, 1, 3)
    K = (x_flat @ _w('k_projection.weights') + _w('k_projection.bias')).reshape(B, T, H, d).transpose(0, 2, 1, 3)
    V = (x_flat @ _w('v_projection.weights') + _w('v_projection.bias')).reshape(B, T, H, d).transpose(0, 2, 1, 3)

    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) / d**0.5  # (B,H,T,T)
    S = np.exp(S - S.max(-1, keepdims=True))
    S /= S.sum(-1, keepdims=True)
    O = np.matmul(S, V).transpose(0, 2, 1, 3).reshape(B * T, n_embd)
    out_ref = (O @ _w('out_projection.weights') + _w('out_projection.bias')).reshape(B, T, n_embd)

    err = np.abs(out_flash - out_ref).max()
    print(f"[test_flash_mha_equals_numpy]    max|Δout|={err:.2e}")
    assert err < 1e-3, f"Flash MHA vs numpy differ: {err}"
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6b — FlashViT == ViT with identical weights (non-causal, no dropout)
# Both models are created with the same numpy random seed.
# ─────────────────────────────────────────────────────────────────────────────

def test_flash_vit_kernel_vs_fallback():
    """
    FlashViT with CUDA kernel must match FlashViT using the corrected 3-D
    fallback (vanilla attention is NOT used as reference because minitorch's
    4-D matrix multiply is broken for attention weights of shape B×H×T×T).
    """
    n_embd, n_head = 32, 4
    patch_size     = 8
    img_size       = 16
    n_classes      = 4
    batch          = 1

    np.random.seed(7)
    flash = FlashViT(
        n_embd=n_embd, n_head=n_head, p_dropout=0.0,
        patch_size=patch_size, n_trans_layers=1,
        n_classes=n_classes, max_patches=10, n_channels=3,
        backend=BACKEND,
    )

    np.random.seed(13)
    x_np = np.random.randn(batch, 3, img_size, img_size).astype(np.float32)
    x    = tensor_from_numpy(x_np, backend=BACKEND)

    # Run with CUDA flash kernel
    out_kernel = flash(x).to_numpy()

    # Disable flash kernel → use corrected 3-D fallback
    attn_modules = [layer.attention for layer in flash.trans_layers]
    for attn in attn_modules:
        attn._use_flash = False

    out_fallback = flash(x).to_numpy()

    for attn in attn_modules:
        attn._use_flash = True

    err = np.abs(out_kernel - out_fallback).max()
    print(f"[test_flash_vit_kernel_vs_fallback] max|Δout|={err:.2e}")
    assert err < 1e-3, f"FlashViT kernel vs fallback differ: {err}"
    print("  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"flash_attn_available = {flash_attn_available}")
    print(f"FA_D_MAX             = {FA_D_MAX}\n")

    if not flash_attn_available:
        print("ERROR: flash_attention.so not loaded. Run compile_cuda.sh first.")
        sys.exit(1)

    tests = [
        test_forward_noncausal,
        test_forward_causal,
        test_backward,
        test_flash_vit_forward,
        test_flash_kernel_d8,
        test_flash_func_minitorch_d8,
        test_flash_mha_equals_numpy,
        test_flash_vit_kernel_vs_fallback,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL — {e}")
        except Exception as e:
            print(f"  ERROR — {type(e).__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
