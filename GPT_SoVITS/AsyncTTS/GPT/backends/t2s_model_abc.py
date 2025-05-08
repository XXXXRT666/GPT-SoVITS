import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import List, MutableSequence, Optional, Tuple

import torch
import torch._inductor.config
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.graphs import CUDAGraph
from torch.profiler import ProfilerAction, tensorboard_trace_handler

from GPT_SoVITS.AR.modules.embedding import (
    SinePositionalEmbeddingNested as SinePositionalEmbedding,
)
from GPT_SoVITS.AR.modules.embedding import TokenEmbedding

Tensor = torch.Tensor


class Sampler(nn.Module):
    def __init__(self, batch_size: int, vocab_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.logits: Tensor
        self.samples: Tensor
        self.register_buffer("logits", torch.zeros((batch_size, vocab_size)), persistent=False)
        self.register_buffer("samples", torch.zeros((batch_size,), dtype=torch.int32), persistent=False)

        self.__CUDAGraph: Optional[CUDAGraph] = None

    def empty_cache(self):
        self.logits.zero_()
        self.__CUDAGraph = None

    @staticmethod
    def multinomial_sample_one_no_sync(probs_sort: Tensor):  # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int32)

    @staticmethod
    def logits_to_probs(
        logits: Tensor,
        previous_tokens: Tensor,
        temperature: float,
        top_k: int,
        top_p: int,
        repetition_penalty: float,
    ):
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(dim=1, index=previous_tokens, src=score)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        logits = logits / max(temperature, 1e-5)

        v, _ = torch.topk(logits, top_k)
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    @staticmethod
    def apply_repetition_penalty(logits: Tensor, previous_tokens: Tensor, repetition_penalty: float):
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(dim=1, index=previous_tokens, src=score)
        return logits

    @staticmethod
    def logits_to_probs_cuda_graph(
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: int,
    ):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        logits = logits / max(temperature, 1e-5)

        v, _ = torch.topk(logits, top_k)
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def __sample(
        self,
        logits: Tensor,
        previous_tokens: Tensor,
        temperature: float,
        top_k: int,
        top_p: int,
        repetition_penalty: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.logits_to_probs(
            logits=logits,
            previous_tokens=previous_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def __sample_cuda_graph(
        self,
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: int,
    ):
        probs = self.logits_to_probs_cuda_graph(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next

    def capture(self, temperature: float, top_k: int, top_p: int):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        logits = self.logits

        with torch.cuda.stream(s):  # type: ignore
            for _ in range(5):
                self.__sample_cuda_graph(logits, temperature, top_k, top_p)
        torch.cuda.current_stream().wait_stream(s)

        self.__CUDAGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.__CUDAGraph):
            self.samples = self.__sample_cuda_graph(logits, temperature, top_k, top_p)
        torch.cuda.synchronize()

    def sample(
        self,
        logits: Tensor,
        previous_tokens: Tensor,
        temperature: float,
        top_k: int,
        top_p: int,
        repetition_penalty: float,
        use_cuda_graph=False,
        idx=-1,
    ) -> Tensor:
        self.logits.copy_(logits)
        if use_cuda_graph and torch.cuda.is_available() and self.__CUDAGraph is None and idx > 0:
            self.capture(temperature, top_k, top_p)
        if self.__CUDAGraph is not None:
            self.apply_repetition_penalty(logits, previous_tokens, repetition_penalty)
            self.__CUDAGraph.replay()
            samples = self.samples.clone()
        else:
            samples = self.__sample(logits, previous_tokens, temperature, top_k, top_p, repetition_penalty)[0]

        return samples


class KVCacheABC(ABC, nn.Module):
    def __init__(self, *args, **kwds) -> None:
        super().__init__()
        self.k_cache: Tensor
        self.v_cache: Tensor
        self.n_head: int
        self.head_dim: int
        self.batch_size: int
        self.max_seq_length: int

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    @abstractmethod
    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor, *args, **kwds) -> Tuple[Tensor, Tensor]: ...

    @abstractmethod
    def prefill_kv(self, k_val: Tensor, v_val: Tensor, bs: int) -> None: ...

    def forward(self):
        raise NotImplementedError()


class KVCacheNHD(KVCacheABC):
    def __init__(self, batch_size, max_seq_length, n_heads, head_dim):
        super().__init__()
        assert batch_size > 0
        cache_shape = (batch_size, max_seq_length, n_heads, head_dim)
        self.n_head = n_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
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
    def __init__(self, batch_size, max_seq_length, n_heads, head_dim):
        super().__init__()
        assert batch_size > 0
        cache_shape = (batch_size, n_heads, max_seq_length, head_dim)
        self.n_head = n_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
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

    def prefill_kv(self, k_val: Tensor, v_val: Tensor, bs: int):
        # input_pos: int, k_val: [B, S, H, D]

        self.k_cache[[bs], :, : k_val.shape[1]] = k_val.transpose(1, 2)
        self.v_cache[[bs], :, : v_val.shape[1]] = v_val.transpose(1, 2)


class AttentionABC(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head: int
        self.hidden_dim: int
        self.head_dim: int

        # key, query, value projections for all heads, but in a batch
        self.in_proj: nn.Linear
        self.out_proj: nn.Linear

        self.dropout = nn.Dropout(0.1)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict: dict, prefix, *args):
        keys_to_modify = [key for key in state_dict if "in_proj_" in key]
        for key in keys_to_modify:
            new_key = key.replace("in_proj_", "in_proj.")  # in_proj_ -> in_proj.
            state_dict[new_key] = state_dict.pop(key)

    @abstractmethod
    def forward(self, x: Tensor, input_pos: Tensor, kv_cache: KVCacheABC, *args, **kwds) -> Tensor: ...

    def prefill(self, x: Tensor, mask: Tensor, kv_cache: KVCacheABC) -> Tensor:
        bsz = x.size(0)

        outputs = []

        for bs in range(bsz):
            x_b = x[bs].unsqueeze(0)

            q, k, v = self.in_proj.forward(x_b.unsqueeze(0)).chunk(3, dim=-1)

            q = q.contiguous().view(1, -1, self.n_head, self.head_dim)
            k = k.contiguous().view(1, -1, self.n_head, self.head_dim)
            v = v.contiguous().view(1, -1, self.n_head, self.head_dim)

            kv_cache.prefill_kv(k, v, bs)

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
        self.dropout = nn.Dropout(0.1)

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

    def forward(self, x: Tensor, input_pos: Tensor, kv_cache: KVCacheABC, *args, **kwds) -> Tensor:
        h = self.attention_norm.forward(
            x
            + self.dropout.forward(
                self.attention.forward(
                    x,
                    input_pos,
                    kv_cache,
                    *args,
                    **kwds,
                )
            )
        )
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out

    def prefill(self, x: Tensor, mask: Tensor, kv_cache: KVCacheABC) -> Tensor:
        h = self.attention_norm.forward(
            x
            + self.dropout.forward(
                self.attention.prefill(
                    x,
                    mask,
                    kv_cache,
                )
            )
        )
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

    def forward(self, input_pos: Tensor, x: Tensor, kv_caches: MutableSequence[KVCacheABC], *args, **kwds):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.forward(x, input_pos, kv_cache, *args, **kwds)
        return x

    def prefill(self, x: Tensor, mask: Tensor, kv_caches: MutableSequence[KVCacheABC]):
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer.prefill(x, mask, kv_cache)
        return x


class T2SDecoderABC(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm_first: bool

        self.n_layer: int
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

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        model_keys = [key for key in state_dict if key.startswith("model.")]
        for key in model_keys:
            new_key = key[len("model.") :]
            state_dict[new_key] = state_dict.pop(key)

    def init_cache(self, kvclass: KVCacheABC, bsz: int = 0) -> MutableSequence[KVCacheABC]:
        bsz = bsz or self.h.max_batch_size
        assert bsz < self.h.max_batch_size
        seq_lens = self.h.max_seq_length
        device = self.bert_proj.bias.device
        dtype = self.bert_proj.bias.dtype
        return nn.ModuleList(
            [kvclass(bsz, seq_lens, self.n_head, self.head_dim) for _ in range(self.n_layer)],
        ).to(device, dtype)  # type: ignore

    @abstractmethod
    def embed(self, x: List[torch.Tensor], y: torch.Tensor, bert_features: List[Tensor]) -> Tensor: ...

    # @abstractmethod
    # def forward(
    #     self,
    #     x: List[torch.LongTensor],
    #     x_lens: torch.Tensor,
    #     prompts: torch.LongTensor,
    #     bert_feature: List[torch.LongTensor],
    #     top_k: int,
    #     top_p: int,
    #     early_stop_num: int,
    #     temperature: float,
    #     repetition_penalty: float,
    #     use_cuda_graph: bool,
    #     debug: bool,
    #     *args,
    #     **kwds,
    # ) -> List[Tensor]: ...

    def compile(self, *args, **kwds):
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

    def capture(self, input_pos: Tensor, x: Tensor, x_dec: Tensor, *args, **kwds) -> CUDAGraph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.stream(s):  # type: ignore
            for _ in range(5):
                self.h.forward(input_pos, x, *args, **kwds)
        torch.cuda.current_stream().wait_stream(s)

        with torch.cuda.graph(graph):
            x_dec.copy_(self.h.forward(input_pos, x, *args, **kwds))
        torch.cuda.synchronize()

        return graph


class TorchProfiler:
    def __init__(self, debug: bool, log_dir: str = "./profiler") -> None:
        self.debug = debug
        self.log_dir = log_dir
        self.__profiler: torch.profiler.profile

        if self.debug and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.tensorboard_handler = tensorboard_trace_handler(self.log_dir)

    def profiler_callback(self, prof: torch.profiler.profile):
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        self.tensorboard_handler(prof)

    @staticmethod
    def three_step_schedule(step: int) -> ProfilerAction:
        if step == 0:
            return ProfilerAction.NONE
        elif step == 1:
            return ProfilerAction.RECORD
        elif step == 2:
            return ProfilerAction.RECORD_AND_SAVE
        else:
            return ProfilerAction.NONE

    def start(self):
        if not self.debug:
            return
        assert self.__profiler is not None
        self.__profiler.step()

    def end(self):
        if not self.debug:
            return
        assert self.__profiler is not None
        self.__profiler.step()

    def profiler(self):
        if self.debug:
            activities_list = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities_list.append(torch.profiler.ProfilerActivity.CUDA)

            self.__profiler = torch.profiler.profile(
                activities=activities_list,
                record_shapes=True,
                with_stack=True,
                with_modules=True,
                profile_memory=True,
                schedule=self.three_step_schedule,
                on_trace_ready=self.profiler_callback,
            )
            return self.__profiler
        else:
            return nullcontext()

    def record(self, func_name: str):
        if self.debug:
            return torch.profiler.record_function(func_name)
        else:
            return nullcontext()
