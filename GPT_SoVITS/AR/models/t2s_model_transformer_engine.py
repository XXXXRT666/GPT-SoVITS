import contextlib
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te  # type: ignore
from tqdm import tqdm

from GPT_SoVITS.AR.models.t2s_model_abc import (
    AttentionABC,
    Sampler,
    T2SDecoderABC,
    TransformerBlockABC,
    TransformerDecoderABC,
)
from GPT_SoVITS.AR.models.t2s_model_abc import (
    FeedForward as ffn,
)
from GPT_SoVITS.AR.models.t2s_model_abc import KVCacheNHD as KVCache
from GPT_SoVITS.AR.modules.embedding import (
    SinePositionalEmbeddingNested as SinePositionalEmbedding,
)
from GPT_SoVITS.AR.modules.embedding import TokenEmbedding

Tensor = torch.Tensor
dtype = torch.dtype


class FeedForward(ffn):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__(dim, hidden_dim)
        self.linear1 = te.Linear(dim, hidden_dim, bias=True)
        self.linear2 = te.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout.forward(self.linear2(self.dropout.forward(F.relu(self.linear1(x)))))


class Attention(AttentionABC):
    def __init__(self, n_head: int, hidden_dim: int):
        super().__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        assert hidden_dim % n_head == 0
        self.head_dim = hidden_dim // n_head

        # key, query, value projections for all heads, but in a batch
        self.in_proj = te.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.out_proj = te.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn = te.DotProductAttention(
            num_attention_heads=self.n_head, kv_channels=self.head_dim, qkv_format="bshd"
        )

    def forward(self, x: Tensor, input_pos: Tensor, cu_seqlens_q: Tensor, cu_seqlens_kv: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj.forward(x).chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        k, v = self.kv_cache.update(input_pos, k, v)

        attn: Tensor = self.attn(
            query_layer=q, key_layer=k, value_layer=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv
        )

        attn = self.dropout.forward(attn)

        attn = attn.view(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj.forward(attn)

        return attn


class TransformerBlock(TransformerBlockABC):
    def __init__(self, n_head, ffn_dim, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = Attention(n_head, hidden_dim)
        self.feed_forward = FeedForward(hidden_dim, ffn_dim)
        self.attention_norm = te.LayerNorm([self.hidden_dim])
        self.ffn_norm = te.LayerNorm([self.hidden_dim])


class TransformerDecoder(TransformerDecoderABC):
    def __init__(
        self,
        hidden_dim,
        n_layer,
        n_head,
        ffn_dim,
        vocab_size,
        max_seq_length,
        max_batch_size,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_head = n_head
        assert hidden_dim % n_head == 0

        self.head_dim = hidden_dim // n_head
        self.vocab_size = vocab_size

        self.n_layer = n_layer

        self.layers = nn.ModuleList(  # type: ignore
            TransformerBlock(n_head, ffn_dim, hidden_dim) for _ in range(n_layer)
        )

        self.max_seq_length: int = max_seq_length
        self.max_batch_size: int = max_batch_size

        self.register_buffer("input_pos", torch.zeros((self.max_batch_size,)).to(torch.int32), persistent=False)
        self.register_buffer("xy_pos", torch.zeros((self.max_batch_size, 1, self.hidden_dim)), persistent=False)
        self.register_buffer("xy_dec", torch.zeros((self.max_batch_size, 1, self.hidden_dim)), persistent=False)

        self.setup_caches(self.max_batch_size, self.max_seq_length)

        self.cu_seqlens_q: Tensor
        self.cu_seqlens_kv: Tensor

        self.register_buffer(
            "cu_seqlens_q", torch.arange(0, self.max_batch_size + 1, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "cu_seqlens_kv", torch.arange(0, self.max_batch_size + 1, dtype=torch.int32), persistent=False
        )

    def setup_caches(self, max_batch_size=10, max_seq_length=2500):
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(self.max_batch_size, self.max_seq_length, self.n_head, self.head_dim)


class T2SDecoder(T2SDecoderABC):
    h: TransformerDecoder

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
        n_head = config["model"]["head"]
        n_layer = config["model"]["n_layer"]
        vocab_size = config["model"]["vocab_size"]
        phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        p_dropout = config["model"]["dropout"]
        EOS = config["model"]["EOS"]
        ffn_dim = hidden_dim * 4
        self.norm_first = norm_first

        self.hidden_dim = hidden_dim
        self.n_head = n_head
        assert hidden_dim % n_head == 0

        self.head_dim = hidden_dim // n_head
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.phoneme_vocab_size = phoneme_vocab_size
        self.p_dropout = p_dropout
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.EOS = EOS
        assert self.EOS == self.vocab_size - 1

        self.bert_proj = te.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_length,
        )
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim,
            dropout=0.1,
            scale=False,
            alpha=True,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_length,
        )
        self.ar_predict_layer = te.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.h = TransformerDecoder(hidden_dim, n_layer, n_head, ffn_dim, vocab_size, max_seq_length, max_batch_size)  # type: ignore
        self.sampler = Sampler(max_batch_size, vocab_size)

        self.__CUDAGraph: Optional[torch.cuda.CUDAGraph] = None

        # self._register_load_state_dict_pre_hook(self.load_hook)

    def empty_cache(self):
        super().empty_cache()
        self.h.cu_seqlens_kv.zero_()

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

    def capture(self, input_pos: Tensor, x: Tensor, cu_seqlens_q: Tensor, cu_seqlens_kv: Tensor):
        return super().capture(input_pos, x, cu_seqlens_q, cu_seqlens_kv)

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
            mask[-y_len:, -y_len:] = ~torch.triu(
                torch.ones(y_len, y_len, device=xy_pos.device, dtype=torch.bool), diagonal=1
            )
            xy_attn_mask.append(mask)
        xy_attn_mask_nested = torch.nested.nested_tensor(xy_attn_mask)

        completed = [False] * bsz
        y_results: List[Tensor] = [None] * bsz  # type: ignore
        t1 = 0.0

        self.h.input_pos.add_(prefill_len)
        self.h.cu_seqlens_kv.copy_(
            torch.cat([torch.tensor(0, device=xy_pos.device, dtype=torch.int32), self.h.input_pos])
        )

        input_pos = self.h.input_pos
        cu_seqlens_q = self.h.cu_seqlens_q
        cu_seqlens_kv = self.h.cu_seqlens_kv

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
                        self.capture(input_pos, xy_pos, cu_seqlens_q, cu_seqlens_kv)

                    with torch.profiler.record_function("AR"):
                        # with contextlib.nullcontext():
                        if self.__CUDAGraph is not None:
                            self.h.xy_pos.copy_(xy_pos)
                            self.__CUDAGraph.replay()
                            xy_dec = self.h.xy_dec.clone()
                        else:
                            xy_dec = self.h.forward(input_pos, xy_pos, cu_seqlens_q, cu_seqlens_kv)

                with torch.profiler.record_function("Logits"):
                    # with contextlib.nullcontext():
                    logits = self.ar_predict_layer(xy_dec[:, -1])
                    input_pos.add_(1)
                    self.h.cu_seqlens_kv.add_(self.h.cu_seqlens_q)

                if idx == 0:
                    logits = logits[:, :-1]

                with torch.profiler.record_function("Sampling"):
                    # with contextlib.nullcontext():
                    samples = self.sampler.sample(
                        logits=logits,
                        previous_tokens=y,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature,
                        use_cuda_graph=use_cuda_graph,
                        idx=idx,
                    )

                    y = torch.concat([y, samples], dim=1)

                with torch.profiler.record_function("EOS"):
                    # with contextlib.nullcontext():
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
                        tqdm.write(f"T2S Decoding EOS {prefill_len.tolist()} -> {[i.shape[0] for i in y_results]}")
                        tqdm.write(f"{(idx - 1) / (time.perf_counter() - t1):.2f}")
                    break

                with torch.profiler.record_function("Next xy_pos"):
                    # with contextlib.nullcontext():

                    y_emb = self.ar_audio_embedding(y[:, -1:])
                    xy_pos = self.ar_audio_position.forward(input_pos - x_lens, y_emb)

                if idx == 2:
                    t1 = time.perf_counter()

        #         if idx == 50:
        #             break

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

        # exit()

        return y_results
