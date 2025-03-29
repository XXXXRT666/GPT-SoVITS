"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

from GPT_SoVITS.f5_tts.model.modules import (
    AdaLayerNormZero,
    AdaLayerNormZero_Final,
    TimestepEmbedding,
)
from GPT_SoVITS.module.dit_modules import InputEmbedding, TextEmbedding

Tensor = torch.Tensor


class Attention1(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.in_proj = nn.Linear(dim, self.inner_dim * 3, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "to_q.weight" in state_dict:
            wq = state_dict.pop(prefix + "to_q.weight")
            wk = state_dict.pop(prefix + "to_k.weight")
            wv = state_dict.pop(prefix + "to_v.weight")
            bq = state_dict.pop(prefix + "to_q.bias")
            bk = state_dict.pop(prefix + "to_k.bias")
            bv = state_dict.pop(prefix + "to_v.bias")
            state_dict[prefix + "in_proj.weight"] = torch.cat([wq, wk, wv])
            state_dict[prefix + "in_proj.bias"] = torch.cat([bq, bk, bv])
        state_dict[prefix + "out_proj.bias"] = state_dict.pop(prefix + "to_out.0.bias")
        state_dict[prefix + "out_proj.bias"] = state_dict.pop(prefix + "to_out.0.bias")

    def forward(
        self,
        x: Tensor[float["b n d"]],  # noised input x  # noqa: F722 # type: ignore
        mask: Tensor[bool["b n"]],  # noqa: F722 # type: ignore
        rope: tuple[Tensor, float] | tuple[Tensor, Tensor],  # rotary position embedding for x
    ) -> torch.Tensor:
        bs = x.shape[0]

        q, k, v = self.in_proj.forward(x).chunk(3, dim=-1)

        freqs, xpos_scale = rope

        q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0)

        q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)  # type: ignore

        k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)  # type: ignore


# Attention processor


class AttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x  # noqa: F722
        mask: bool["b n"] | None = None,  # noqa: F722
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections.
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            # print(3433333333,attn_mask.shape)
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(
            processor=AttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class DiT(nn.Module):
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

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(  # x, prompt_x, x_lens, t, style,cond
        self,  # d is channel,n is T
        x0: float["b n d"],  # nosied input audio  # noqa: F722
        cond0: float["b n d"],  # masked cond audio  # noqa: F722
        x_lens,
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        dt_base_bootstrap,
        text0,  # : int["b nt"]  # noqa: F722#####condition feature
        use_grad_ckpt=False,  # bool
        ###no-use
        drop_audio_cond=False,  # cfg for cond audio
        drop_text=False,  # cfg for text
        # mask: bool["b n"] | None = None,  # noqa: F722
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
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)  ###need to change
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if use_grad_ckpt:
                x = checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
