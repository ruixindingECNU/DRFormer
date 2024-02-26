"""
Code from: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""


import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from itertools import chain


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, nums_head, seq_len = input_ids_shape[:3]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
        positions = positions.unsqueeze(0).unsqueeze(0)
        positions = positions.repeat(bsz, nums_head, 1)
        return super().forward(positions)
    
    
    

class RoFormerSinusoidalPositionalEmbeddingSequenceInGroup(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, group_positions: list, padding_idx: Optional[int] = None) -> None:
        
        super().__init__(len(sum(group_positions,[])), embedding_dim)
        self.weight = self._init_weight(self.weight, group_positions)

    @staticmethod
    def _init_weight(out: nn.Parameter, group_positions: list) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        
            
        L = len(group_positions[0])
        position_enc = []
        for positions_one_group in group_positions:
            position_enc = position_enc + [[(j/len(positions_one_group))*L / np.power(10000, 2 * (i // 2) / dim) for i in range(dim)] for j in range(len(positions_one_group))]
        position_enc = np.array(position_enc)
        
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))    
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, nums_head, seq_len = input_ids_shape[:3]
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
        positions = positions.unsqueeze(0).unsqueeze(0)
        positions = positions.repeat(bsz, nums_head, 1)
        return super().forward(positions)
    
    
class RoFormerSinusoidalPositionalEmbeddingSequence(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, group_positions: list, padding_idx: Optional[int] = None) -> None:
        
        super().__init__(len(sum(group_positions,[])), embedding_dim)
        self.weight = self._init_weight(self.weight, group_positions)


    @staticmethod
    def _init_weight(out: nn.Parameter, group_positions: list) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = []
        
        for positions_one_group in group_positions:
            position_enc = position_enc + [[positions_one_group[0] / np.power(10000, j / len(positions_one_group)) for i in range(dim)] for j in range(len(positions_one_group))]

        position_enc = np.array(position_enc)
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
            
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, nums_head, seq_len = input_ids_shape[:3]
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device)
        positions = positions.unsqueeze(0).unsqueeze(0)
        positions = positions.repeat(bsz, nums_head, 1)
        return super().forward(positions)
    
    
class ROPEAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(ROPEAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        self.embed_key_positions = RoFormerSinusoidalPositionalEmbedding(num_positions=100, embedding_dim=d_keys)
        self.embed_query_positions = RoFormerSinusoidalPositionalEmbedding(num_positions=100, embedding_dim=d_keys)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        t_q = torch.arange(queries.shape[2]).to(queries.device)
        t_k = torch.arange(keys.shape[2]).to(keys.device)
                
        sin_q_pos = self.embed_query_positions(queries.shape[:])
        sin_k_pos = self.embed_key_positions(keys.shape[:])
        
        queries = self.apply_rotary_position_embeddings(sin_q_pos, queries)
        keys = self.apply_rotary_position_embeddings(sin_k_pos, keys)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
    def apply_rotary_position_embeddings(self, sinusoidal_pos, input_sequence):
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

        rotate_half_query_layer = torch.stack([-input_sequence[..., 1::2], input_sequence[..., ::2]],
                                              dim=-1).reshape_as(
            input_sequence
        )
        input_layer = input_sequence * cos_pos + rotate_half_query_layer * sin_pos

        return input_layer
    
    
    
class ROPEAttentionGroupSequenceIngroupLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, group_positions, d_keys=None,
                 d_values=None):
        super(ROPEAttentionGroupSequenceIngroupLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
        
        self.embed_key_positions = RoFormerSinusoidalPositionalEmbeddingSequenceInGroup(num_positions=100, embedding_dim=d_keys, group_positions = group_positions)
        self.embed_query_positions = RoFormerSinusoidalPositionalEmbeddingSequenceInGroup(num_positions=100, embedding_dim=d_keys, group_positions = group_positions)
        
        self.embed_key_group_positions = RoFormerSinusoidalPositionalEmbeddingSequence(num_positions=150, embedding_dim=d_keys, group_positions = group_positions)
        self.embed_query_group_positions = RoFormerSinusoidalPositionalEmbeddingSequence(num_positions=150, embedding_dim=d_keys, group_positions = group_positions)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        t_q = torch.arange(queries.shape[2]).to(queries.device)
        t_k = torch.arange(keys.shape[2]).to(keys.device)
                
        sin_q_pos = self.embed_query_positions(queries.shape[:])
        sin_k_pos = self.embed_key_positions(keys.shape[:])
        
        sin_q_group_pos = self.embed_query_group_positions(queries.shape[:])
        sin_k_group_pos = self.embed_key_group_positions(keys.shape[:])
        
        queries = self.apply_rotary_position_embeddings(sin_q_pos, queries)
        keys = self.apply_rotary_position_embeddings(sin_k_pos, keys)
        
        queries = self.apply_rotary_position_embeddings(sin_q_group_pos, queries)
        keys = self.apply_rotary_position_embeddings(sin_k_group_pos, keys)
        
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
    def apply_rotary_position_embeddings(self, sinusoidal_pos, input_sequence):
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)

        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)

        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

        rotate_half_query_layer = torch.stack([-input_sequence[..., 1::2], input_sequence[..., ::2]],
                                              dim=-1).reshape_as(
            input_sequence
        )
        input_layer = input_sequence * cos_pos + rotate_half_query_layer * sin_pos

        return input_layer