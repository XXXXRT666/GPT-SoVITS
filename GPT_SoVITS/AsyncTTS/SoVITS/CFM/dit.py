"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from collections.abc import MutableSequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding

from GPT_SoVITS.AsyncTTS.SoVITS.CFM.dit_modules import (
    AdaLayerNormZero,
    AdaLayerNormZero_Final,
    Attention,
    FeedForward,
    InputEmbedding,
    TextEmbedding,
    TimestepEmbedding,
    sequence_mask,
)

Tensor = torch.Tensor


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1) -> None:
        super().__init__()

        self.attn = Attention(dim, heads, dim_head, dropout)
        self.feed_forward = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        self.attn_norm = AdaLayerNormZero(dim)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix: str, *args):
        for key in list(state_dict.keys()):
            new_key = key.replace("ff_norm", "ffn_nrom")
            if new_key == "ff":
                new_key = "feed_forward"
            state_dict[new_key] = state_dict.pop(key)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        mask: Tensor,
        rope: tuple[Tensor, float] | tuple[Tensor, Tensor],
    ):
        """_

        Args:
            x (Tensor): Noised Input
            t (Tensor): Time Embedding
            mask (Tensor): [B, N]
            rope (tuple[Tensor, float] | tuple[Tensor, Tensor]): Rotary Position Embedding for x

        Returns:
            Tensor
        """

        # Pre-Norm & Modulation for Attention Input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm.forward(x, emb=t)

        attn_output = self.attn.forward(x=norm, mask=mask, rope=rope)

        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ffn_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.feed_forward.forward(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_layer=8,
        n_head=8,
        head_dim=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_dim: Optional[int] = None,  # type: ignore
        n_conv_layer=0,
        **kwds,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.n_layer = n_layer
        if text_dim is None:
            text_dim = mel_dim

        self.time_embedding = TimestepEmbedding(dim)
        self.d_embedding = TimestepEmbedding(dim)
        self.text_embedding = TextEmbedding(text_dim, conv_layers=n_conv_layer)
        self.input_embedding = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embedding = RotaryEmbedding(head_dim)

        self.layers: MutableSequence[DiTBlock] = nn.ModuleList(
            [
                DiTBlock(dim=dim, heads=n_head, dim_head=head_dim, ff_mult=ff_mult, dropout=dropout)
                for _ in range(n_layer)
            ]
        )  # type: ignore

        self.norm = AdaLayerNormZero_Final(dim)  # final modulation
        self.out_proj = nn.Linear(dim, mel_dim)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix: str, *args):
        for key in list(state_dict.keys()):
            new_key = (
                key.replace("proj_out", "out_proj")
                .replace("transformer_blocks", "layers")
                .replace("embed", "embedding")
                .replace("norm_out", "norm")
            )
            state_dict[new_key] = state_dict.pop(key)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        x_lens: Tensor,
        time: Tensor,
        dt_base_bootstrap: Tensor,
        text: Tensor,
    ):
        """
        D: Channel
        Args:
            x (Tensor): Nosied Input Audio [B, N, D]
            cond (Tensror): Masked Cond Audio [B, N, D]
            x_lens (Tensor): [B]
            time (Tensor): Time Step [B]
            dt_base_bootstrap: [B]
            text (Tensor): Condition Feature, [B, N, D]
        """
        seq_len = x.size(1)
        mask = sequence_mask(x_lens, max_length=x.size(1)).to(x.device)

        t = self.time_embedding.forward(time)
        dt = self.d_embedding.forward(dt_base_bootstrap)
        t += dt
        text_emb = self.text_embedding.forward(text, seq_len)
        x = self.input_embedding.forward(x, cond, text_emb)

        rope = self.rotary_embedding.forward_from_seq_len(seq_len)

        for block in self.layers:
            x = block.forward(x, t, mask=mask, rope=rope)

        x = self.norm.forward(x, t)
        output = self.out_proj.forward(x)

        return output


class DiT1(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        self.d_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(  # x, prompt_x, x_lens, t, style,cond
        self,  # d is channel,n is T
        x0: Tensor,  # nosied input audio  [B, N, D]
        cond0: Tensor,  # masked cond audio [B, N, D]
        x_lens,
        time: Tensor,  # time step  [B]
        dt_base_bootstrap,
        text0: Tensor,  # condition feature [B, NT]
    ):
        x = x0.transpose(2, 1)
        cond = cond0.transpose(2, 1)
        text = text0.transpose(2, 1)
        mask = sequence_mask(x_lens, max_length=x.size(1)).to(x.device)

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        dt = self.d_embed(dt_base_bootstrap)
        t += dt
        text_embed = self.text_embed(text, seq_len)  ###need to change
        x = self.input_embed(x, cond, text_embed)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        residual = x

        for block in self.transformer_blocks:
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
