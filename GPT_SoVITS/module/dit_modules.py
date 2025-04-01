import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import apply_rotary_pos_emb

from GPT_SoVITS.f5_tts.model.modules import (
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

Tensor = torch.Tensor


class TextEmbedding(nn.Module):
    def __init__(self, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.freqs_cis: Tensor
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def forward(self, text: Tensor[int["b nt"]], seq_len, drop_text=False):  # type: ignore # noqa: F722
        batch, _ = text.shape[0], text.shape[1]

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: Tensor[float["b n d"]], cond: Tensor[float["b n d"]], text_embed: Tensor[float["b n d"]], drop_audio_cond=False):  # type: ignore # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


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
        x: Tensor[float["b n d"]],  # noised input x  # noqa: F722 # type: ignore
        mask: Tensor[bool["b n"]],  # noqa: F722 # type: ignore
        rope: tuple[Tensor, float] | tuple[Tensor, Tensor],  # rotary position embedding for x
    ) -> torch.Tensor:
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
