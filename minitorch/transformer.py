import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from .tensor_functions import (zeros, ones, rand,cat,select)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        self.q_projection = Linear(n_embd, n_embd,bias,backend)
        self.k_projection = Linear(n_embd, n_embd,bias,backend)
        self.v_projection = Linear(n_embd, n_embd,bias,backend)
        self.out_projection = Linear(n_embd, n_embd, bias, backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        q = (self.q_projection(x.view(batch_size*seq_len,n_embd))).view(batch_size, seq_len, n_embd)
        q = (q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)).permute(0,2,1,3)
        
        k = (self.k_projection(x.view(batch_size*seq_len,n_embd))).view(batch_size, seq_len, n_embd)
        kT = (k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)).permute(0,2,3,1)
        
        v= (self.v_projection(x.view(batch_size*seq_len,n_embd))).view(batch_size, seq_len, n_embd)
        v = (v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)).permute(0,2,1,3)
        ### END ASSIGN3_3
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        ### BEGIN ASSIGN3_3
        
        attn_weights = ((q @ kT) / (q_dim ** 0.5))
        if(self.causal):
            mask = self.create_causal_mask(queries_len)
            attn_weights += mask
        result =  (softmax(attn_weights, dim = 3)) @ v
        result = (result.permute(0,2,1,3)).contiguous()
        result = result.view(batch_size, queries_len, num_head*q_dim)
        ### END ASSIGN3_3

        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        q, kT, v = self.project_to_query_key_value(x)
        attn_out = self.self_attention(q, kT, v)
        attn_out = self.out_projection(attn_out.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        attn_out = self.dropout(attn_out)
        return attn_out
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with  activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.ln_1 = LayerNorm1d(n_embd,ln_eps,backend)
        self.ln_2 = LayerNorm1d(n_embd,ln_eps,backend)
        self.attention = MultiHeadAttention(n_embd, n_head, False, p_dropout, bias, backend)
        self.ff = FeedForward(n_embd,  4 * n_embd, p_dropout, bias, backend)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        ln_1_out = (self.ln_1(x.view(batch_size*seq_len,n_embd))).view(batch_size, seq_len, n_embd)
        attn_out = self.attention(ln_1_out)
        residual_out_1 = attn_out + x
        ln_2_out = (self.ln_2(residual_out_1.view(batch_size*seq_len, n_embd))).view(batch_size, seq_len, n_embd)
        ff_out = self.ff(ln_2_out)
        residual_out_2 = ff_out + residual_out_1
        return residual_out_2
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        ### BEGIN ASSIGN3_3
        self.token_embeddings = Embedding(n_vocab, n_embd, backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend)
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.dropout = Dropout(p_dropout)
        self.ln = LayerNorm1d(n_embd, ln_eps, backend)
        self.lm_head = Linear(n_embd, n_vocab, bias, backend)
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
        # 1. Get token embeddings of shape (batch_size, seq_len, n_embd)
        # 2. Create positional embeddings of shape (1, seq_len, n_embd):
        #    - Create position ids tensor [0, 1, 2, ..., seq_len-1] of shape (1, seq_len)
        #    - Pass through positional embedding layer
        #    - Ensure output shape is (1, seq_len, n_embd)
        # 3. Add token and positional embeddings
        # 4. Apply dropout
        # 5. Pass through transformer layers (t_layer_1 to t_layer_4)
        # 6. Apply final layer normalization
        # 7. Project to vocabulary size using lm_head
        tok_embd = self.token_embeddings(idx)
        pos_ids = tensor( [list(range(seq_len))], backend=self.backend, requires_grad=False )
        pos_embd = self.position_embeddings(pos_ids)
        x = tok_embd + pos_embd
        x_dropout = self.dropout(x)
        trans_out = self.t_layer_1(x_dropout)
        trans_out = self.t_layer_2(trans_out)
        trans_out = self.t_layer_3(trans_out)
        trans_out = self.t_layer_4(trans_out)
        l_norm_out = self.ln(trans_out.view(batch_size*seq_len, self.n_embd)).view(batch_size, seq_len, self.n_embd)
        output  = self.lm_head(l_norm_out.view(batch_size*seq_len,self.n_embd)).view(batch_size, seq_len, self.n_vocab)
        return output
        ### END ASSIGN3_3

class Patchify(Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W)
        returns: (B, N_patches, patch_dim)
        """
        B, C, H, W = x.shape
        P = self.patch_size

        assert H % P == 0 and W % P == 0, "Image dimensions must be divisible by patch size"

        nH = H // P
        nW = W // P
        # print(nH, nW, P)
        x = x.view(B, C, nH, P, nW, P)
        # print(x.shape)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        # print(x.shape)

        # (B, nH, nW, C, P, P)  
        x = x.view(B, nH * nW, C * P * P)
        # print(x.shape)
        return x

        
class ViT(Module):
    def __init__(
        self, 
        n_embd: int,
        n_head: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        patch_size: int=16,
        n_trans_layers: int=12,
        n_classes: int=1000,
        max_patches: int=1000,
        n_channels: int=3,
        backend: TensorBackend=None
    ):
        super().__init__()
        self.backend = backend
        self.n_embd = n_embd
        self.n_head = n_head
        self.p_dropout = p_dropout
        self.ln_eps = ln_eps
        self.bias = bias
        self.backend = backend
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.patchify = Patchify(self.patch_size)
        self.patch_proj = Linear(patch_size**2 * n_channels, n_embd, bias, backend)  #(P^2 * C, D)
        self.position_embeddings = Embedding(self.max_patches, n_embd, backend)
        self.cls_token = Parameter(rand((1, 1, n_embd), backend=backend, requires_grad=True))
        self.n_classes = n_classes
        self.trans_layers = []
        for i in range(n_trans_layers):
            layer = TransformerLayer(
                n_embd, n_head, p_dropout, ln_eps, bias, backend
            )
            self.trans_layers.append(layer)

            setattr(self, f"trans_layer_{i}", layer)
        self.lm_head = Linear(n_embd, n_classes, bias, backend)
        
    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W)
 
        """
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        x = self.patchify(x)  #(B, C, nH, nW) -> (B, nH * nW, P^2 * C)
        B, N, D_in = x.shape    
        print(B, N, D_in)
        x = self.patch_proj(x.view(B*N, D_in)).view(B, N, self.n_embd)  #(B, nH * nW, P^2 * C) -> (B, nH * nW, D)
        print(x.shape)
        B, N, D = x.shape #B: batch size, N: number of patches, D: embedding dimension
        
        cls_token = self.cls_token.value + zeros((B, 1, self.n_embd), backend=self.backend)  # (1, 1, D)->(B, 1, D)
        
        x = cat([cls_token, x], dim=1, backend=self.backend)  # (B, N+1, D)
        num_patches = N + 1

        pos_ids = tensor([list(range(num_patches))], backend=self.backend, requires_grad=False)  # (1, N+1)
        pos_enc = self.position_embeddings(pos_ids)  # (1, N+1, D)
        pos_enc = pos_enc + zeros((B, num_patches, self.n_embd), backend=self.backend)  # (B, N+1, D)
        x = x + pos_enc
        for layer in self.trans_layers:
            x = layer(x)
        x = select(x, dim=1, index=0)  # (B, D)
        x = self.lm_head(x)
        return x
