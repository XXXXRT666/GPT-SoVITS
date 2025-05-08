import math
from collections.abc import MutableSequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import apply_rotary_pos_emb

Tensor = torch.Tensor


def sequence_mask(length: Tensor, max_length: Optional[int] = None):
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start: Tensor, length: int, max_pos: int, scale=1.0):
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim: int, kernel_size=31, groups=16):
        super().__init__()

        assert kernel_size % 2 != 0

        self.conv1 = nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2)
        self.act = nn.Mish()

        self.register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix: str, *args):
        if prefix + "conv1d.0.weight" in state_dict:
            state_dict[prefix + "conv1.weight"] = state_dict.pop(prefix + "conv1d.0.weight")
            state_dict[prefix + "conv1.bias"] = state_dict.pop(prefix + "conv1d.0.bias")
        if prefix + "conv1d.2.weight" in state_dict:
            state_dict[prefix + "conv2.weight"] = state_dict.pop(prefix + "conv1d.2.weight")
            state_dict[prefix + "conv2.bias"] = state_dict.pop(prefix + "conv1d.2.bias")

    def forward(self, x: Tensor, mask: Tensor | None = None):
        """
        Args:
            x (Tensor): [B, N, D]
            mask (Tensor | None, optional): [B, N]. Defaults to None.

        Returns:
            Tensor
        """

        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1.forward(x)
        x = self.act.forward(x)
        x = self.conv2.forward(x)
        x = self.act.forward(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


class GRN(nn.Module):
    gamma: Tensor
    beta: Tensor

    def __init__(self, dim: int):
        super().__init__()

        self.register_parameter("gamma", nn.Parameter(torch.zeros(1, 1, dim)))
        self.register_parameter("beta", nn.Parameter(torch.zeros(1, 1, dim)))

    def forward(self, x: Tensor) -> Tensor:
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()

        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): [B, N, D]

        Returns:
            Tensor: [B, N, D]
        """
        residual = x
        x = x.transpose(1, 2)  # B N D -> B D N
        x = self.dwconv.forward(x)
        x = x.transpose(1, 2)  # B D N -> B N D
        x = self.norm.forward(x)
        x = self.pwconv1.forward(x)
        x = self.act.forward(x)
        x = self.grn.forward(x)
        x = self.pwconv2.forward(x)
        return residual + x


class TextEmbedding(nn.Module):
    freqs_cis: Tensor

    def __init__(self, text_dim: int, conv_layers: int, conv_mult=2):
        super().__init__()

        self.precompute_max_pos = 4096  # ~44s of 24khz audio
        self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
        self.text_blocks: MutableSequence[ConvNeXtV2Block] = nn.ModuleList(
            [ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
        )  # type: ignore

    def forward(self, text: Tensor, seq_len: int):
        """
        Args:
            text (Tensor): [B, N, D]
            seq_len (int)

        Returns:
            Tensor
        """
        batch, _ = text.shape[0], text.shape[1]

        # sinus pos emb
        batch_start = torch.zeros((batch,), dtype=torch.long)
        pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
        text_pos_embed = self.freqs_cis[pos_idx]
        text = text + text_pos_embed

        # convnextv2 blocks
        for block in self.text_blocks:
            text = block.forward(text)

        return text


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()

        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: Tensor, cond: Tensor, text_embed: Tensor):
        """
        Args:
            x (Tensor): [B, N, D]
            cond (Tensor): [B, N, D]
            text_embed (Tensor): [B, N, D]

        Returns:
            Tensor
        """

        x = self.proj.forward(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed.forward(x) + x
        return x


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x: Tensor, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim=256):
        super().__init__()

        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()
        self.linear1 = nn.Linear(freq_embed_dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        self.register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix: str, *args):
        if prefix + "time_mlp.0.weight" in state_dict:
            state_dict[prefix + "linear1.weight"] = state_dict.pop(prefix + "time_mlp.0.weight")
            state_dict[prefix + "linear1.bias"] = state_dict.pop(prefix + "time_mlp.0.bias")
        if prefix + "time_mlp.2.weight" in state_dict:
            state_dict[prefix + "linear2.weight"] = state_dict.pop(prefix + "time_mlp.2.weight")
            state_dict[prefix + "linear2.bias"] = state_dict.pop(prefix + "time_mlp.2.bias")

    def forward(self, timestep: Tensor) -> Tensor:
        """
        Args:
            timestep (Tensor): [B]

        Returns:
            Tensor
        """
        time_hidden = self.time_embed.forward(timestep).to(timestep.dtype)
        time = self.time_mlp.forward(time_hidden)  # [B, D]
        return time


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor, emb: Tensor):
        emb = self.linear.forward(self.silu.forward(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        x = self.norm.forward(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: Tensor, emb: Tensor):
        emb = self.linear.forward(self.silu.forward(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm.forward(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.activation = nn.GELU(approximate=approximate)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, dim_out)

        self.register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix: str, *args):
        if prefix + "ff.0.0.weight" in state_dict:
            state_dict[prefix + "linear1.weight"] = state_dict.pop(prefix + "ff.0.0.weight")
            state_dict[prefix + "linear1.bias"] = state_dict.pop(prefix + "ff.0.0.bias")
        if prefix + "ff.2.weight" in state_dict:
            state_dict[prefix + "linear2.weight"] = state_dict.pop(prefix + "ff.2.weight")
            state_dict[prefix + "linear2.bias"] = state_dict.pop(prefix + "ff.2.bias")

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation.forward(self.linear1.forward(x))
        return self.linear2.forward(self.dropout.forward(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.n_head = heads
        self.hidden_dim = head_dim * heads
        self.head_dim = head_dim

        self.in_proj = nn.Linear(dim, self.hidden_dim * 3, bias=True)
        self.out_proj = nn.Linear(self.hidden_dim, dim)

        self.dropout = nn.Dropout(dropout)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix: str, *args):
        if prefix + "to_q.weight" in state_dict:
            wq = state_dict.pop(prefix + "to_q.weight")
            wk = state_dict.pop(prefix + "to_k.weight")
            wv = state_dict.pop(prefix + "to_v.weight")
            bq = state_dict.pop(prefix + "to_q.bias")
            bk = state_dict.pop(prefix + "to_k.bias")
            bv = state_dict.pop(prefix + "to_v.bias")
            state_dict[prefix + "in_proj.weight"] = torch.cat([wq, wk, wv])
            state_dict[prefix + "in_proj.bias"] = torch.cat([bq, bk, bv])
        if prefix + "to_out.weight" in state_dict:
            state_dict[prefix + "out_proj.weight"] = state_dict.pop(prefix + "to_out.0.weight")
            state_dict[prefix + "out_proj.bias"] = state_dict.pop(prefix + "to_out.0.bias")

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        rope: tuple[Tensor, float] | tuple[Tensor, Tensor],
    ) -> Tensor:
        """

        Args:
            x (Tensor): Noised Input [B, N, D]
            mask (Tensor): [B, N]
            rope (tuple[Tensor, float] | tuple[Tensor, Tensor]): Rotary Position Embedding for x
        Returns:
            Tensor
        """
        bs = x.shape[0]

        q, k, v = self.in_proj.forward(x).chunk(3, dim=-1)

        freqs, xpos_scale = rope

        q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0)

        q: Tensor = apply_rotary_pos_emb(q, freqs, q_xpos_scale)  # type: ignore
        k: Tensor = apply_rotary_pos_emb(k, freqs, k_xpos_scale)  # type: ignore

        q = q.view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bs, -1, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        attn_mask = mask.unsqueeze(1).unsqueeze(1).expand(bs, self.n_head, q.shape[-2], -1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).contiguous().view(bs, -1, self.dim)

        attn = self.out_proj.forward(attn)

        return self.dropout.forward(attn)
