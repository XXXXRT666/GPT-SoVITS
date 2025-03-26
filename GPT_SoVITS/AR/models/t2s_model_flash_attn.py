import contextlib
import time
from typing import List, Optional, Sequence

import flash_attn  # type: ignore
import torch
import torch.nested._internal.nested_tensor
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from GPT_SoVITS.AR.models.t2s_model_abc import T2SDecoderABC
from GPT_SoVITS.AR.models.utils import sample
from GPT_SoVITS.AR.modules.embedding import (
    SinePositionalEmbeddingNested as SinePositionalEmbedding,
)
from GPT_SoVITS.AR.modules.embedding import TokenEmbedding

Tensor = torch.Tensor
dtype = torch.dtype


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim):
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, n_heads, head_dim)
        self.num_heads = n_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length

        self.register_buffer("k_cache", torch.zeros(size=cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(size=cache_shape), persistent=False)
        self.k_cache: Tensor
        self.v_cache: Tensor

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
                self.num_heads,
                self.head_dim,
            )
            .to(torch.int64)
        )  # (bs, 1, num_head, head_dim)

        k_out = self.k_cache
        v_out = self.v_cache
        k_out.scatter_(1, index, k_val)
        v_out.scatter_(1, index, v_val)

        return k_out, v_out

    def forward(self):
        pass

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def prefill_kv(self, bs: int, k_val: Tensor, v_val: Tensor):
        # input_pos: int, k_val: [B, S, H, D]

        self.k_cache[[bs], : k_val.shape[1]] = k_val
        self.v_cache[[bs], : v_val.shape[1]] = v_val


class Attention(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # key, query, value projections for all heads, but in a batch
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.kv_cache: KVCache

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict, prefix, *args):
        keys_to_modify = [key for key in state_dict if "in_proj_" in key]
        for key in keys_to_modify:
            new_key = key.replace("in_proj_", "in_proj.")  # in_proj_ -> in_proj.
            state_dict[new_key] = state_dict.pop(key)

    def forward(self, x: Tensor, input_pos: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj.forward(x).chunk(3, dim=-1)

        q = q.view(bsz, -1, self.num_heads, self.head_dim)
        k = k.view(bsz, -1, self.num_heads, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim)

        k, v = self.kv_cache.update(input_pos, k, v)

        attn: Tensor = flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=input_pos)

        attn = attn.view(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj.forward(attn)

        return attn

    def prefill(self, x: Tensor, mask: Tensor) -> Tensor:
        bsz = x.size(0)

        outputs = []

        for bs in range(bsz):
            x_b = x[bs].unsqueeze(0)

            q, k, v = self.in_proj.forward(x_b.unsqueeze(0)).chunk(3, dim=-1)

            q = q.contiguous().view(1, -1, self.num_heads, self.head_dim)
            k = k.contiguous().view(1, -1, self.num_heads, self.head_dim)
            v = v.contiguous().view(1, -1, self.num_heads, self.head_dim)

            self.kv_cache.prefill_kv(bs, k, v)

            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

            attn_mask = mask[bs].unsqueeze(0).unsqueeze(0).expand(1, self.num_heads, -1, -1)

            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            attn = attn.transpose(1, 2).contiguous().view(1, -1, self.hidden_dim)

            output = self.out_proj.forward(attn)

            outputs.append(output.squeeze(0))

        return torch.nested.nested_tensor(outputs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, ffn_dim, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = Attention(num_heads, hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, ffn_dim)
        self.attention_norm = nn.LayerNorm([self.hidden_dim])
        self.ffn_norm = nn.LayerNorm([self.hidden_dim])

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

    def forward(self, x: Tensor, input_pos: Tensor) -> Tensor:
        h = self.attention_norm.forward(x + self.attention.forward(x, input_pos))
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out

    def prefill(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.attention_norm.forward(x + self.attention.prefill(x, mask))
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layer,
        num_heads,
        ffn_dim,
        vocab_size,
        max_seq_length,
        max_batch_size,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0

        self.head_dim = hidden_dim // num_heads
        self.vocab_size = vocab_size

        self.n_layer = n_layer

        self.layers: Sequence[TransformerBlock] = nn.ModuleList(TransformerBlock(num_heads, ffn_dim, hidden_dim) for _ in range(n_layer))  # type: ignore

        self.max_seq_length: int = max_seq_length
        self.max_batch_size: int = max_batch_size

        self.register_buffer("input_pos", torch.zeros((self.max_batch_size,)).to(torch.int32), persistent=False)
        self.register_buffer("xy_pos", torch.zeros((self.max_batch_size, 1, self.hidden_dim)), persistent=False)
        self.register_buffer("xy_dec", torch.zeros((self.max_batch_size, 1, self.hidden_dim)), persistent=False)

        self.input_pos: Tensor
        self.xy_pos: Tensor
        self.xy_dec: Tensor

        self.setup_caches(self.max_batch_size, self.max_seq_length)

    def setup_caches(self, max_batch_size=10, max_seq_length=2500):
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(self.max_batch_size, self.max_seq_length, self.num_heads, self.head_dim)

    def forward(self, input_pos: Tensor, x: Tensor):
        for layer in self.layers:
            x = layer.forward(x, input_pos)
        return x

    def prefill(self, x: Tensor, mask: Tensor):
        for layer in self.layers:
            x = layer.prefill(x, mask)
        return x


class T2SDecoder(T2SDecoderABC):
    def __init__(
        self,
        config,
        *args,
        norm_first=False,
        max_seq_length=2500,
        max_batch_size=10,
        **kwds,
    ) -> None:
        super().__init__()

        hidden_dim = config["model"]["hidden_dim"]
        embedding_dim = config["model"]["embedding_dim"]
        num_heads = config["model"]["head"]
        n_layer = config["model"]["n_layer"]
        vocab_size = config["model"]["vocab_size"]
        phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        p_dropout = config["model"]["dropout"]
        EOS = config["model"]["EOS"]
        ffn_dim = hidden_dim * 4
        self.norm_first = norm_first

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0

        self.head_dim = hidden_dim // num_heads
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.phoneme_vocab_size = phoneme_vocab_size
        self.p_dropout = p_dropout
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.EOS = EOS
        assert self.EOS == self.vocab_size - 1

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True, max_batch_size=max_batch_size, max_seq_len=max_seq_length
        )
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True, max_batch_size=max_batch_size, max_seq_len=max_seq_length
        )
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.h = TransformerDecoder(hidden_dim, n_layer, num_heads, ffn_dim, vocab_size, max_seq_length, max_batch_size)

        self.__CUDAGraph: Optional[torch.cuda.CUDAGraph] = None

        # self._register_load_state_dict_pre_hook(self.load_hook)

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

    def embed(
        self,
        x: List[torch.LongTensor],
        y: torch.LongTensor,
        bert_features: List[torch.LongTensor],
    ):
        x_nested = torch.nested.nested_tensor(x)
        assert x_nested.size(0) == self.max_batch_size
        bert_features_nested = torch.nested.nested_tensor(list(map(lambda x: x.transpose(0, 1), bert_features)))

        x_emb = self.ar_text_embedding.forward(x_nested)
        bert = self.bert_proj.forward(bert_features_nested)
        x_emb = x_emb + bert
        x_pos = self.ar_text_position.prefill(x_emb)

        y_emb = self.ar_audio_embedding.forward(y)
        y_emb = torch.nested.as_nested_tensor(y_emb)
        y_pos = self.ar_audio_position.prefill(y_emb)

        xy_pos = torch.nested.nested_tensor([torch.cat([x_pos[i], y_pos[i]]) for i in range(len(x))])

        return xy_pos

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

    def forward(
        self,
        x: List[torch.LongTensor],  #####全部文本token
        x_lens: torch.Tensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: List[torch.LongTensor],
        top_k: int = 5,
        top_p: int = 1,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        use_cuda_graph=False,
        **kwargs,
    ):
        self.empty_cache()

        bsz = len(x)
        y = prompts
        x_lens = x_lens.to(torch.int64)
        y_len = y.shape[-1]
        prefill_len = x_lens + y_len
        xy_pos = self.embed(x, y, bert_feature)

        xy_attn_mask = []
        for bs in range(bsz):
            pos = int(x_lens[bs].item())
            mask = torch.zeros(pos + y_len, pos + y_len, device=xy_pos.device).bool()
            mask[:, :pos].fill_(True)
            mask[-y_len:, -y_len:] = ~torch.triu(torch.ones(y_len, y_len, device=xy_pos.device, dtype=torch.bool), diagonal=1)
            xy_attn_mask.append(mask)
        xy_attn_mask_nested = torch.nested.nested_tensor(xy_attn_mask)

        completed = [False] * bsz
        y_results: List[Tensor] = [None] * bsz  # type: ignore
        self.h.input_pos += prefill_len
        input_pos = self.h.input_pos

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True
        # ) as prof:
        with contextlib.nullcontext():
            for idx in tqdm(range(1500)):
                if idx == 0:
                    xy_dec = self.h.prefill(xy_pos, xy_attn_mask_nested)
                    xy_dec = torch.stack([t[[-1]] for t in xy_dec.unbind()])
                    input_pos = input_pos.to(torch.int32)
                else:
                    if torch.cuda.is_available() and use_cuda_graph and self.__CUDAGraph is None:
                        self.capture(input_pos, xy_pos)

                    # with torch.profiler.record_function("AR"):
                    with contextlib.nullcontext():
                        if self.__CUDAGraph is not None:
                            self.h.xy_pos.copy_(xy_pos)
                            self.__CUDAGraph.replay()
                            xy_dec = self.h.xy_dec.clone()
                        else:
                            xy_dec = self.h.forward(input_pos, xy_pos)

                logits = self.ar_predict_layer(xy_dec[:, -1])
                input_pos += 1

                if idx == 0:
                    t1 = time.perf_counter()
                    logits = logits[:, :-1]

                samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]

                y = torch.concat([y, samples], dim=1)

                tokens = torch.argmax(logits, dim=-1)

                EOS_mask = (samples[:, 0] == self.EOS) | (tokens == self.EOS)
                EOS_indices: List[int] = torch.where(EOS_mask)[0].tolist()

                for i in EOS_indices:
                    if not completed[i]:
                        y_results[i] = y[i, y_len:-1]  # type: ignore
                        completed[i] = True

                if (early_stop_num != -1 and (y.shape[1] - y_len) > early_stop_num) or idx == 1499:
                    tqdm.write(f"Reached early stop limit: {early_stop_num}")
                    for i in range(bsz):
                        if not completed[i]:
                            y_results[i] = y[i, y_len:]  # type: ignore
                            completed[i] = True
                    break

                if all(completed):
                    if y.shape[1] == 0:
                        y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                        tqdm.write("bad zero prediction")
                    else:
                        tqdm.write(f"T2S Decoding EOS {prefill_len.tolist()} -> {y.shape[1]}")
                        tqdm.write(f"{idx / (time.perf_counter() - t1):.2f}")  # type: ignore
                    break

                y_emb = self.ar_audio_embedding(y[:, -1:])
                xy_pos = self.ar_audio_position.forward(input_pos - x_lens, y_emb)

                # if idx == 50:
                #     break

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

        # exit()

        return y_results
