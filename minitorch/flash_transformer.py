"""
Flash Attention 1 variants of the ViT building blocks.

This module is a drop-in replacement for the corresponding classes in
transformer.py.  The public API is identical so existing training loops
need only swap the class names:

    # vanilla ViT
    from minitorch.transformer import ViT, TransformerLayer, MultiHeadAttention

    # Flash Attention ViT
    from minitorch.flash_transformer import FlashViT, FlashTransformerLayer, FlashMultiHeadAttention

All original classes in transformer.py are left completely untouched.

Design
------
* FlashMultiHeadAttention  — replaces the softmax(QK^T/√d)·V call with the
  Flash Attention 1 CUDA kernel (via FlashAttentionFunction).  Projects Q,K,V
  exactly as MultiHeadAttention does but passes K (not Kᵀ) to the kernel.

* FlashTransformerLayer    — same pre-norm residual structure as
  TransformerLayer, but uses FlashMultiHeadAttention.

* FlashViT                 — same Patchify → patch_proj → CLS → pos_emb →
  TransformerLayers → CLS readout → classifier as ViT, but uses
  FlashTransformerLayer throughout.

Fallback
--------
If flash_attention.so has not been compiled yet, or if the head dimension d
exceeds FA_D_MAX (64), FlashMultiHeadAttention silently falls back to
vanilla attention so the model can still be constructed and used on CPU.
"""

import math
import numpy as np

from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import Embedding, Dropout, LayerNorm1d, Linear
from .tensor_ops import TensorBackend
from .nn import softmax, dropout, GELU
from .tensor_functions import zeros, ones, rand, cat, select

from .flash_attention_ops import (
    flash_attention,
    flash_attn_available,
    FA_D_MAX,
    _load_lib,
)

from typing import List, Optional

datatype = np.float32


# ---------------------------------------------------------------------------
# Helper: vanilla self-attention (exact same math as MultiHeadAttention)
# used as fallback when Flash Attention is unavailable or d > FA_D_MAX.
# ---------------------------------------------------------------------------
def _vanilla_self_attention(q, kT, v, causal: bool, backend):
    """
    Compute softmax((q @ kT) / sqrt(d)) @ v.

    q  : (B, H, T, d)
    kT : (B, H, d, T)
    v  : (B, H, T, d)

    Returns (B, T, H*d) after merging heads.
    """
    batch_size, num_head, queries_len, q_dim = q.shape
    scale = q_dim ** 0.5

    attn_weights = (q @ kT) / scale
    if causal:
        mask = -np.finfo(datatype).max * np.triu(
            np.ones((1, 1, queries_len, queries_len), dtype=datatype), 1
        )
        attn_weights = attn_weights + tensor_from_numpy(mask, backend=backend)

    result = softmax(attn_weights, dim=3) @ v
    result = result.permute(0, 2, 1, 3).contiguous()
    result = result.view(batch_size, queries_len, num_head * q_dim)
    return result


# ---------------------------------------------------------------------------
# FlashMultiHeadAttention
# ---------------------------------------------------------------------------

class FlashMultiHeadAttention(Module):
    """
    Multi-Head Self-Attention using Flash Attention 1.

    Identical constructor signature to MultiHeadAttention so it can be
    swapped in without touching calling code.

    Parameters
    ----------
    n_embd    : total embedding dimension
    n_head    : number of attention heads  (d = n_embd // n_head must be ≤ 64)
    causal    : apply causal mask (for decoder; ViT uses causal=False)
    p_dropout : dropout probability on the output projection
    bias      : include bias in projection layers
    backend   : minitorch TensorBackend
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        causal: bool = False,
        p_dropout: float = 0.1,
        bias: bool = True,
        backend: TensorBackend = None,
    ):
        super().__init__()
        self.backend        = backend
        self.n_embd         = n_embd
        self.n_head         = n_head
        self.causal         = causal
        self.attn_hidden_dim = n_embd // n_head

        self.q_projection  = Linear(n_embd, n_embd, bias, backend)
        self.k_projection  = Linear(n_embd, n_embd, bias, backend)
        self.v_projection  = Linear(n_embd, n_embd, bias, backend)
        self.out_projection = Linear(n_embd, n_embd, bias, backend)
        self.dropout        = Dropout(p_dropout)

        # Determine at construction time whether the kernel can be used.
        _load_lib()
        self._use_flash = flash_attn_available and (self.attn_hidden_dim <= FA_D_MAX)

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def project_to_query_key_value(self, x):
        """
        Project input to Q, K, V.

        Parameters
        ----------
        x : Tensor (B, T, n_embd)

        Returns
        -------
        q : Tensor (B, H, T, d)
        k : Tensor (B, H, T, d)   ← note: K, not Kᵀ
        v : Tensor (B, H, T, d)
        """
        B, T, _ = x.shape
        H, d    = self.n_head, self.attn_hidden_dim

        x_flat = x.view(B * T, self.n_embd)

        q = self.q_projection(x_flat).view(B, T, H, d).permute(0, 2, 1, 3)
        k = self.k_projection(x_flat).view(B, T, H, d).permute(0, 2, 1, 3)
        v = self.v_projection(x_flat).view(B, T, H, d).permute(0, 2, 1, 3)
        return q, k, v

    # ------------------------------------------------------------------
    # Flash Attention path
    # ------------------------------------------------------------------

    def _flash_self_attention(self, q, k, v):
        """
        Compute attention via the Flash Attention 1 CUDA kernel.

        q, k, v : (B, H, T, d)

        Returns (B, T, H*d).
        """
        B, H, T, d = q.shape
        scale = d ** -0.5

        # Merge batch and heads: (B*H, T, d)
        q_fa = q.contiguous().view(B * H, T, d)
        k_fa = k.contiguous().view(B * H, T, d)
        v_fa = v.contiguous().view(B * H, T, d)

        # Flash attention kernel call (autodiff-aware)
        O_fa = flash_attention(q_fa, k_fa, v_fa, scale, self.causal)

        # (B*H, T, d) → (B, H, T, d) → (B, T, H, d) → (B, T, H*d)
        result = O_fa.view(B, H, T, d).permute(0, 2, 1, 3).contiguous()
        result = result.view(B, T, H * d)
        return result

    # ------------------------------------------------------------------
    # Fallback path (vanilla attention)
    # ------------------------------------------------------------------

    def _vanilla_self_attention(self, q, k, v):
        """
        Vanilla attention fallback used when Flash Attention is unavailable
        or d > FA_D_MAX.

        q, k, v : (B, H, T, d)

        Returns (B, T, H*d).

        NOTE: minitorch's _tensor_matrix_multiply is only correct for 3-D
        tensors.  We therefore reshape to (B*H, T, d) before every matmul so
        that the 3-D kernel is used, then reshape the result back.
        """
        B, H, T, d = q.shape
        scale = float(d) ** 0.5

        # Merge batch and heads: (B*H, T, d) — ensures 3-D matmul path
        q3 = q.contiguous().view(B * H, T, d)
        k3 = k.contiguous().view(B * H, T, d)
        v3 = v.contiguous().view(B * H, T, d)

        # kT3: (B*H, d, T)  via a non-copying permute of the TensorData
        kT3 = k3._new(k3._tensor.permute(0, 2, 1))

        attn_weights = (q3 @ kT3) / scale   # (B*H, T, T) — 3-D ✓
        if self.causal:
            mask = -np.finfo(datatype).max * np.triu(
                np.ones((1, T, T), dtype=datatype), 1
            )
            attn_weights = attn_weights + tensor_from_numpy(mask, backend=self.backend)

        O3 = softmax(attn_weights, dim=2) @ v3   # (B*H, T, d) — 3-D ✓
        result = O3.view(B, H, T, d).permute(0, 2, 1, 3).contiguous()
        result = result.view(B, T, H * d)
        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def self_attention(self, q, k, v):
        """
        Dispatch to Flash Attention or vanilla attention.

        Parameters
        ----------
        q, k, v : Tensors of shape (B, H, T, d)

        Returns
        -------
        Tensor of shape (B, T, n_embd)
        """
        if self._use_flash:
            return self._flash_self_attention(q, k, v)
        return self._vanilla_self_attention(q, k, v)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, T, n_embd)

        Returns
        -------
        Tensor (B, T, n_embd)
        """
        B, T, n_embd = x.shape
        q, k, v = self.project_to_query_key_value(x)
        attn_out = self.self_attention(q, k, v)
        attn_out = (
            self.out_projection(attn_out.view(B * T, n_embd))
            .view(B, T, n_embd)
        )
        attn_out = self.dropout(attn_out)
        return attn_out


# ---------------------------------------------------------------------------
# FlashTransformerLayer
# ---------------------------------------------------------------------------

class FlashTransformerLayer(Module):
    """
    Transformer encoder layer (pre-norm) using Flash Attention.

    Drop-in replacement for TransformerLayer.

    Parameters
    ----------
    n_embd    : embedding / model dimension
    n_head    : number of attention heads
    p_dropout : dropout probability
    ln_eps    : LayerNorm epsilon
    bias      : bias in linear layers
    backend   : minitorch TensorBackend
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        p_dropout: float = 0.1,
        ln_eps: float = 1e-5,
        bias: bool = True,
        backend: TensorBackend = None,
    ):
        super().__init__()
        self.ln_1      = LayerNorm1d(n_embd, ln_eps, backend)
        self.ln_2      = LayerNorm1d(n_embd, ln_eps, backend)
        self.attention = FlashMultiHeadAttention(
            n_embd, n_head, causal=False, p_dropout=p_dropout,
            bias=bias, backend=backend,
        )
        self.ff = _FeedForward(n_embd, 4 * n_embd, p_dropout, bias, backend)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, T, n_embd)

        Returns
        -------
        Tensor (B, T, n_embd)
        """
        B, T, D = x.shape

        ln1_out  = self.ln_1(x.view(B * T, D)).view(B, T, D)
        attn_out = self.attention(ln1_out)
        x        = attn_out + x

        ln2_out = self.ln_2(x.view(B * T, D)).view(B, T, D)
        ff_out  = self.ff(ln2_out)
        x       = ff_out + x
        return x


# ---------------------------------------------------------------------------
# Private helper: FeedForward (identical to transformer.FeedForward)
# ---------------------------------------------------------------------------

class _FeedForward(Module):
    def __init__(self, n_embd, middle_dim, p_dropout, bias, backend):
        super().__init__()
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        B, T, D = x.shape
        x = GELU(self.linear_in(x.view(B * T, D)))
        x = self.dropout(self.linear_out(x)).view(B, T, D)
        return x


# ---------------------------------------------------------------------------
# Patchify (identical to transformer.Patchify — reproduced to keep this
# module self-contained and avoid cross-module coupling)
# ---------------------------------------------------------------------------

class _Patchify(Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        """x : (B, C, H, W)  →  (B, nH*nW, C*P*P)"""
        B, C, H, W = x.shape
        P  = self.patch_size
        nH = H // P
        nW = W // P
        x = x.view(B, C, nH, P, nW, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, nH * nW, C * P * P)
        return x


# ---------------------------------------------------------------------------
# FlashViT
# ---------------------------------------------------------------------------

class FlashViT(Module):
    """
    Vision Transformer using Flash Attention 1.

    Drop-in replacement for ViT (same constructor signature and forward
    signature).

    The only architectural difference is that each TransformerLayer uses
    FlashMultiHeadAttention internally.  If the .so is not compiled or d
    exceeds the kernel limit, each layer falls back to vanilla attention
    transparently.

    Parameters
    ----------
    n_embd         : embedding dimension
    n_head         : number of attention heads per layer (d = n_embd // n_head)
    p_dropout      : dropout probability
    ln_eps         : layer-norm epsilon
    bias           : use bias in linear projections
    patch_size     : image patch size P  (image dims must be divisible by P)
    n_trans_layers : number of transformer layers
    n_classes      : number of output classes
    max_patches    : maximum sequence length (N_patches + 1 for CLS token)
    n_channels     : input image channels (3 for RGB)
    backend        : minitorch TensorBackend
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        p_dropout: float = 0.1,
        ln_eps: float = 1e-5,
        bias: bool = True,
        patch_size: int = 16,
        n_trans_layers: int = 12,
        n_classes: int = 1000,
        max_patches: int = 1000,
        n_channels: int = 3,
        backend: TensorBackend = None,
    ):
        super().__init__()
        self.backend      = backend
        self.n_embd       = n_embd
        self.n_head       = n_head
        self.patch_size   = patch_size
        self.max_patches  = max_patches

        self.patchify     = _Patchify(patch_size)
        self.patch_proj   = Linear(patch_size ** 2 * n_channels, n_embd, bias, backend)
        self.position_embeddings = Embedding(max_patches, n_embd, backend)
        self.cls_token    = Parameter(rand((1, 1, n_embd), backend=backend, requires_grad=True))

        self.n_classes    = n_classes
        self.trans_layers: List[FlashTransformerLayer] = []
        for i in range(n_trans_layers):
            layer = FlashTransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
            self.trans_layers.append(layer)
            setattr(self, f"trans_layer_{i}", layer)

        self.lm_head = Linear(n_embd, n_classes, bias, backend)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, C, H, W)   — input images

        Returns
        -------
        Tensor (B, n_classes)     — class logits
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            "Image dimensions must be divisible by patch_size"
        )

        # Patchify and project
        x = self.patchify(x)                             # (B, N, P²·C)
        B, N, D_in = x.shape
        x = self.patch_proj(x.view(B * N, D_in)).view(B, N, self.n_embd)

        # Prepend CLS token
        cls = self.cls_token.value + zeros((B, 1, self.n_embd), backend=self.backend)
        x   = cat([cls, x], dim=1, backend=self.backend)  # (B, N+1, D)

        num_patches = N + 1

        # Add positional embeddings
        pos_ids = tensor(
            [list(range(num_patches))], backend=self.backend, requires_grad=False
        )
        pos_enc = self.position_embeddings(pos_ids)                           # (1, N+1, D)
        pos_enc = pos_enc + zeros((B, num_patches, self.n_embd), backend=self.backend)
        x = x + pos_enc

        # Transformer layers
        for layer in self.trans_layers:
            x = layer(x)

        # CLS readout → classifier
        x = select(x, dim=1, index=0)  # (B, D)
        x = self.lm_head(x)            # (B, n_classes)
        return x
