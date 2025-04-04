from abc import ABC, abstractmethod
from typing import List, MutableSequence, Optional, Tuple

import torch
import torch._inductor.config
import torch.nn as nn
import torch.nn.functional as F

from GPT_SoVITS.AR.modules.embedding import (
    SinePositionalEmbeddingNested as SinePositionalEmbedding,
)
from GPT_SoVITS.AR.modules.embedding import TokenEmbedding

Tensor = torch.Tensor


class KVCacheABC(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_cache: Tensor
        self.v_cache: Tensor
        self.n_head: int
        self.head_dim: int
        self.max_batch_size: int
        self.max_seq_length: int

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    @abstractmethod
    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor, *args, **kwds) -> Tuple[Tensor, Tensor]: ...

    @abstractmethod
    def prefill_kv(self, k_val: Tensor, v_val: Tensor, *args, **kwds) -> None: ...

    def forward(self):
        raise NotImplementedError()


class KVCacheNHD(KVCacheABC):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim):
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.n_head = n_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length

        self.register_buffer("k_cache", torch.zeros(size=cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(size=cache_shape), persistent=False)

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: [B, ], k_val: [B, 1, H, D]

        index = (
            (input_pos - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                self.n_head,
                self.head_dim,
            )
            .to(torch.int64)
        )  # (bs, 1, num_head, head_dim)

        k_out = self.k_cache
        v_out = self.v_cache
        k_out.scatter_(1, index, k_val)
        v_out.scatter_(1, index, v_val)

        return k_out, v_out

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def prefill_kv(self, k_val: Tensor, v_val: Tensor, bs: int):
        # input_pos: int, k_val: [B, S, H, D]

        self.k_cache[[bs], : k_val.shape[1]] = k_val
        self.v_cache[[bs], : v_val.shape[1]] = v_val


class KVCacheHND(KVCacheABC):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.n_head = n_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length

        self.register_buffer("k_cache", torch.zeros(size=cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(size=cache_shape), persistent=False)

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: [B, ], k_val: [B, H, 1, D]

        index = (
            (input_pos - 1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                self.n_head,
                -1,
                self.head_dim,
            )
            .to(torch.int64)
        )  # (bs, num_head, 1, head_dim)

        k_out = self.k_cache
        v_out = self.v_cache
        k_out.scatter_(2, index, k_val)
        v_out.scatter_(2, index, v_val)

        return k_out, v_out

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def prefill_kv(self, k_val: Tensor, v_val: Tensor):
        # input_pos: int, k_val: [B, H, S, D]

        self.k_cache[:, :, : k_val.shape[2]] = k_val
        self.v_cache[:, :, : v_val.shape[2]] = v_val


class AttentionABC(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head: int
        self.hidden_dim: int
        self.head_dim: int

        # key, query, value projections for all heads, but in a batch
        self.in_proj: nn.Linear
        self.out_proj: nn.Linear

        self.dropout: nn.Dropout

        self.kv_cache: KVCacheABC

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict, prefix, *args):
        keys_to_modify = [key for key in state_dict if "in_proj_" in key]
        for key in keys_to_modify:
            new_key = key.replace("in_proj_", "in_proj.")  # in_proj_ -> in_proj.
            state_dict[new_key] = state_dict.pop(key)

    @abstractmethod
    def forward(self, x: Tensor, input_pos: Tensor, *args, **kwds) -> Tensor: ...

    def prefill(self, x: Tensor, mask: Tensor) -> Tensor:
        bsz = x.size(0)

        outputs = []

        for bs in range(bsz):
            x_b = x[bs].unsqueeze(0)

            q, k, v = self.in_proj.forward(x_b.unsqueeze(0)).chunk(3, dim=-1)

            q = q.contiguous().view(1, -1, self.n_head, self.head_dim)
            k = k.contiguous().view(1, -1, self.n_head, self.head_dim)
            v = v.contiguous().view(1, -1, self.n_head, self.head_dim)

            self.kv_cache.prefill_kv(k, v, bs)

            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

            attn_mask = mask[bs].unsqueeze(0).unsqueeze(0).expand(1, self.n_head, -1, -1)

            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            attn = self.dropout.forward(attn)

            attn = attn.transpose(1, 2).contiguous().view(1, -1, self.hidden_dim)

            output = self.out_proj.forward(attn)

            outputs.append(output.squeeze(0))

        return torch.nested.nested_tensor(outputs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout.forward(self.linear2(self.dropout.forward(F.relu(self.linear1(x)))))


class TransformerBlockABC(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim: int
        self.attention: AttentionABC
        self.feed_forward: FeedForward
        self.attention_norm: nn.LayerNorm
        self.ffn_norm: nn.LayerNorm
        self.dropout: nn.Dropout

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict[str, Tensor], prefix, *args):
        for key in list(state_dict.keys()):
            new_key = (
                key.replace("self_attn", "attention")
                .replace("linear", "feed_forward.linear")
                .replace("norm1", "attention_norm")
                .replace("norm2", "ffn_norm")
            )
            state_dict[new_key] = state_dict.pop(key)

    def forward(self, x: Tensor, input_pos: Tensor, *args, **kwds) -> Tensor:
        h = self.attention_norm.forward(x + self.dropout.forward(self.attention.forward(x, input_pos)))
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out

    def prefill(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.attention_norm.forward(x + self.dropout.forward(self.attention.prefill(x, mask)))
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out


class TransformerDecoderABC(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.hidden_dim: int
        self.n_head: int
        self.head_dim: int
        self.vocab_size: int
        self.n_layer: int

        self.layers: MutableSequence[TransformerBlockABC]

        self.max_seq_length: int
        self.max_batch_size: int

        self.input_pos: Tensor
        self.xy_pos: Tensor
        self.xy_dec: Tensor

    @abstractmethod
    def setup_caches(self, max_batch_size: int, max_seq_length: int) -> None: ...

    def forward(self, input_pos: Tensor, x: Tensor, *args, **kwds):
        for layer in self.layers:
            x = layer.forward(x, input_pos)
        return x

    def prefill(self, x: Tensor, mask: Tensor):
        for layer in self.layers:
            x = layer.prefill(x, mask)
        return x


class T2SDecoderABC(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_first: bool

        self.hidden_dim: int
        self.n_head: int

        self.head_dim: int
        self.embedding_dim: int
        self.vocab_size: int
        self.phoneme_vocab_size: int
        self.p_dropout: float
        self.max_seq_length: int
        self.max_batch_size: int
        self.EOS: int

        self.bert_proj: nn.Linear
        self.ar_text_embedding: TokenEmbedding
        self.ar_text_position: SinePositionalEmbedding
        self.ar_audio_embedding: TokenEmbedding
        self.ar_audio_position: SinePositionalEmbedding
        self.ar_predict_layer: nn.Linear
        self.h: TransformerDecoderABC

        self.__CUDAGraph: Optional[torch.cuda.CUDAGraph] = None

    def load_hook(self, state_dict, prefix, *args):
        model_keys = [key for key in state_dict if key.startswith("model.")]
        for key in model_keys:
            new_key = key[len("model.") :]
            state_dict[new_key] = state_dict.pop(key)

    def empty_cache(self):
        for layer in self.h.layers:
            layer.attention.kv_cache.empty()
        self.h.input_pos.zero_()
        self.h.xy_pos.zero_()
        self.h.xy_dec.zero_()
        self.__CUDAGraph = None

    @abstractmethod
    def embed(
        self, x: List[torch.LongTensor], y: torch.LongTensor, bert_features: List[torch.LongTensor]
    ) -> Tensor: ...

    @abstractmethod
    def forward(
        self,
        x: List[torch.LongTensor],
        x_lens: torch.Tensor,
        prompts: torch.LongTensor,
        bert_feature: List[torch.LongTensor],
        top_k: int,
        top_p: int,
        early_stop_num: int,
        temperature: float,
        repetition_penalty: float,
        **kwargs,
    ) -> List[Tensor]: ...

    def compile(self, *args, **kwargs):
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        # Experimental features to reduce compilation times, will be on by default in future
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.triton.cudagraph_trees = True
        torch._inductor.config.triton.cudagraph_support_input_mutation = True

        self.h.forward = torch.compile(
            self.h.forward,
            fullgraph=True,
            mode="reduce-overhead",
            # mode="max-autotune",
        )

    def capture(self, input_pos: Tensor, x: Tensor):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        self.h.xy_pos.copy_(x)
        self.h.input_pos.copy_(input_pos)

        with torch.cuda.stream(s):  # type: ignore
            for _ in range(5):
                self.h.forward(self.h.input_pos, self.h.xy_pos)
        torch.cuda.current_stream().wait_stream(s)

        self.__CUDAGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.__CUDAGraph):
            self.h.xy_dec = self.h.forward(self.h.input_pos, self.h.xy_pos)
        torch.cuda.synchronize()
