import contextlib
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from GPT_SoVITS.AR.models.utils import sample
from GPT_SoVITS.AR.modules.embedding import SinePositionalEmbedding, TokenEmbedding

Tensor = torch.Tensor
dtype = torch.dtype


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos - 1 : input_pos] = k_val
        v_out[:, :, input_pos - 1 : input_pos] = v_val

        return k_out[:, :, :input_pos], v_out[:, :, :input_pos]

    def update_static(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos - 1 : input_pos] = k_val
        v_out[:, :, input_pos - 1 : input_pos] = v_val

        return k_out, v_out

    def forward(self):
        pass

    def empty(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def init_kv(self, k, v):
        self.k_cache[:, :, : k.shape[-2], :] = k
        self.v_cache[:, :, : v.shape[-2], :] = v


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

        self.kv_cache: Optional[KVCache] = None

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        keys_to_modify = [key for key in state_dict if "in_proj_" in key]
        for key in keys_to_modify:
            new_key = key.replace("in_proj_", "in_proj.")  # in_proj_ -> in_proj.
            state_dict[new_key] = state_dict.pop(key)

    def forward(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:

        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj.forward(x).chunk(3, dim=-1)

        q = q.view(bsz, -1, self.num_heads, self.head_dim)
        k = k.view(bsz, -1, self.num_heads, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj.forward(attn)

        return attn

    def forward_static(self, x: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:

        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj.forward(x).chunk(3, dim=-1)

        q = q.view(bsz, -1, self.num_heads, self.head_dim)
        k = k.view(bsz, -1, self.num_heads, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update_static(input_pos, k, v)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj.forward(attn)

        return attn

    def forward_prefill(self, x: Tensor, mask: Tensor) -> Tensor:

        bsz, seqlen, _ = x.shape

        q, k, v = self.in_proj.forward(x).chunk(3, dim=-1)

        q = q.view(bsz, -1, self.num_heads, self.head_dim)
        k = k.view(bsz, -1, self.num_heads, self.head_dim)
        v = v.view(bsz, -1, self.num_heads, self.head_dim)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        self.kv_cache.init_kv(k, v)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj.forward(attn)

        return attn


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

    def load_hook(self, state_dict, prefix, *args):

        for key in list(state_dict.keys()):
            new_key = (
                key.replace("self_attn", "attention")
                .replace("linear", "feed_forward.linear")
                .replace("norm1", "attention_norm")
                .replace("norm2", "ffn_norm")
            )
            state_dict[new_key] = state_dict.pop(key)

    def forward(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        h = self.attention_norm.forward(x + self.attention.forward(x, mask, input_pos))
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out

    def forward_static(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        h = self.attention_norm.forward(x + self.attention.forward_static(x, mask, input_pos))
        out = self.ffn_norm.forward(h + self.feed_forward.forward(h))
        return out

    def forward_prefill(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.attention_norm.forward(x + self.attention.forward_prefill(x, mask))
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

        self.layers: List[TransformerBlock] = nn.ModuleList(TransformerBlock(num_heads, ffn_dim, hidden_dim) for _ in range(n_layer))

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        self.register_buffer("static_xy_pos", torch.zeros((self.max_batch_size, 1, self.hidden_dim)), persistent=False)
        self.register_buffer("static_xy_pos_", torch.zeros((self.max_batch_size, 1, self.hidden_dim)), persistent=False)
        self.register_buffer("static_out", torch.zeros((self.max_batch_size, 1, self.hidden_dim)), persistent=False)
        self.register_buffer("static_out_", torch.zeros((self.max_batch_size, 1, self.hidden_dim)), persistent=False)
        self.register_buffer("static_attn_mask", torch.ones(max_batch_size, num_heads, 1, max_seq_length).bool(), persistent=False)
        self.register_buffer("static_attn_mask_", torch.zeros(max_batch_size, num_heads, 1, max_seq_length).bool(), persistent=False)

    def setup_caches(self, max_batch_size=5, max_seq_length=2500, dtype=None):
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(self.max_batch_size, self.max_seq_length, self.num_heads, self.head_dim, dtype)

    def forward(self, input_pos: Optional[Tensor] = None) -> Tensor:
        x = self.static_xy_pos
        mask = self.static_attn_mask[:, :, :, :input_pos]
        for layer in self.layers:
            x = layer.forward(x, input_pos, mask)
        return x

    def forward_prefill(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer.forward_prefill(x, mask)
        return x

    def forward_static(self, input_pos: Optional[Tensor] = None) -> Tensor:
        x = self.static_xy_pos_
        mask = self.static_attn_mask_
        for layer in self.layers:
            x = layer.forward_static(x, input_pos, mask)
        return x


class T2SDecoder(nn.Module):
    def __init__(
        self,
        config,
        *args,
        norm_first=False,
        max_seq_length=2500,
        max_batch_size=5,
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

        self._CUDA_GRAPH = None

        self._CUDA_GRAPH_STATIC = None

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(self.embedding_dim, self.phoneme_vocab_size, self.p_dropout)
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_predict_layer = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        self.h = TransformerDecoder(hidden_dim, n_layer, num_heads, ffn_dim, vocab_size, max_seq_length, max_batch_size)

        self.register_buffer("xy_attn_mask", torch.ones(max_batch_size, num_heads, 1, max_seq_length).bool(), persistent=False)
        self.register_buffer("xy_attn_mask_", torch.zeros(max_batch_size, num_heads, 1, max_seq_length).bool(), persistent=False)
        self.register_buffer("input_pos", torch.tensor(0).to(torch.int32), persistent=False)
        self.register_buffer("input_pos_", torch.tensor(0).to(torch.int32), persistent=False)

        self.device = torch.device("cpu")
        self.dtype = torch.float32

        # self.h.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length)

        # self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        model_keys = [key for key in state_dict if key.startswith("model.")]
        for key in model_keys:
            new_key = key[len("model.") :]
            state_dict[new_key] = state_dict.pop(key)

    def to(self, *args, **kwds):
        device, dtype, _, __ = torch._C._nn._parse_to(*args, **kwds)
        self.device = device
        self.dtype = dtype
        return super().to(*args, **kwds)

    def cpu(self):
        self.device = torch.device("cpu")
        return super().cpu()

    def cuda(self, device=None):
        if device:
            if isinstance(device, int):
                self.device = torch.device(f"cuda:{device}")
            else:
                self.device = device
        else:
            self.device = torch.device("cuda")
        return super().cuda(device)

    def xpu(self, device=None):
        self.device = torch.device("xpu")
        return super().xpu(device)

    def float(self):
        self.dtype = torch.float16
        return super().float()

    def half(self):
        self.dtype = torch.float16
        return super().half()

    def bfloat16(self):
        self.dtype = torch.bfloat16
        return super().bfloat16()

    def embed(
        self,
        x: List[torch.LongTensor],
        x_lens: torch.LongTensor,
        y: torch.LongTensor,
        bert_feature: List[torch.LongTensor],
    ):
        y_lens = torch.LongTensor([y.shape[1]] * y.shape[0]).to(y.device)
        max_x_len = max(x_lens.tolist())
        max_y_len = max(y_lens.tolist())
        x_pos = torch.zeros((len(x), max_x_len, self.embedding_dim), dtype=bert_feature[0].dtype, device=x[0].device)
        y_pos = torch.zeros((len(x), max_y_len, self.embedding_dim), dtype=bert_feature[0].dtype, device=y[0].device)

        for idx, (x_item, bert_item) in enumerate(zip(x, bert_feature)):
            x_item = self.ar_text_embedding.forward(x_item.unsqueeze(0))  # 1, seq_len, embeding dim
            x_item = x_item + self.bert_proj.forward(bert_item.transpose(0, 1).unsqueeze(0))
            x_item = self.ar_text_position.forward(x_item).squeeze(0)
            x_pos[idx, : x_lens[idx]] = x_item

        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        return x_pos, y_pos

    def embed_left(
        self,
        x: List[torch.LongTensor],
        x_lens: torch.LongTensor,
        y: torch.LongTensor,
        bert_feature: List[torch.LongTensor],
    ):
        y_lens = torch.LongTensor([y.shape[1]] * y.shape[0]).to(y.device)
        max_x_len = int(max(x_lens.tolist()))
        max_y_len = int(max(y_lens.tolist()))
        x_pos = torch.zeros((len(x), max_x_len, self.embedding_dim), dtype=bert_feature[0].dtype, device=x[0].device)
        y_pos = torch.zeros((len(x), max_y_len, self.embedding_dim), dtype=bert_feature[0].dtype, device=y[0].device)

        for idx, (x_item, bert_item) in enumerate(zip(x, bert_feature)):
            x_item = self.ar_text_embedding.forward(x_item.unsqueeze(0))  # 1, seq_len, embeding dim
            x_item = x_item + self.bert_proj.forward(bert_item.transpose(0, 1).unsqueeze(0))
            x_item = self.ar_text_position.forward(x_item).squeeze(0)
            x_pos[idx, -x_lens[idx] :] = x_item

        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        return x_pos, y_pos

    def empty_cache(self):
        with self.device:
            if self.h.layers[0].attention.kv_cache is None:
                self.h.setup_caches(self.max_batch_size, self.max_seq_length, self.dtype)
                return
            for layer in self.h.layers:
                layer.attention.kv_cache.empty()
            self.xy_attn_mask.fill_(True)
            self.h.static_attn_mask.fill_(True)
            self.h.static_xy_pos.zero_()
            self.h.static_out.zero_()
            self.input_pos.zero_()

    def empty_cache_static(self):
        with self.device:
            if self.h.layers[0].attention.kv_cache is None:
                self.h.setup_caches(self.max_batch_size, self.max_seq_length, self.dtype)
                return
            for layer in self.h.layers:
                layer.attention.kv_cache.empty()
            self.xy_attn_mask_.fill_(False)
            self.h.static_attn_mask_.fill_(False)
            self.h.static_xy_pos_.zero_()
            self.h.static_out_.zero_()
            self.input_pos_.zero_()

    def forward(self): ...

    def infer_batch(
        self,
        x: List[torch.LongTensor],  #####全部文本token
        x_lens: torch.LongTensor,
        prompts: torch.LongTensor,  ####参考音频token
        bert_feature: List[torch.LongTensor],
        top_k: int = 5,
        top_p: int = 1,
        early_stop_num: int = -1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        padding_left=True,
        use_cuda_graph=False,
        **kwargs,
    ):
        bsz = len(x)
        max_len = x_lens.max()

        # Empty Cache and Mask
        self.empty_cache()

        y = prompts
        if padding_left:
            x_pos, y_pos = self.embed_left(x, x_lens, y, bert_feature)
        else:
            x_pos, y_pos = self.embed(x, x_lens, y, bert_feature)

        xy_pos = torch.concat([x_pos, y_pos], dim=1)

        x_len = x_pos.shape[1]
        y_len = y_pos.shape[1]
        prefill_len = x_len + y_len

        # Padding Mask
        xy_attn_mask = torch.zeros((self.max_batch_size, max_len + y_len, max_len + y_len), dtype=torch.bool, device=xy_pos.device)
        if padding_left:
            for i in range(bsz):
                xy_attn_mask[i, :, -y_len - x_lens[i] : -y_len] = True
        else:
            for i in range(bsz):
                xy_attn_mask[i, :, : x_lens[i]] = True
        xy_attn_mask[:, -y_len:, -y_len:] = ~torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1)

        xy_attn_mask = xy_attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        self.xy_attn_mask[:, :, :, : x_len + y_len] = xy_attn_mask[:, :, -1].unsqueeze(2)
        self.h.static_attn_mask.copy_(self.xy_attn_mask)

        completed = [False] * bsz
        y_results = [None] * bsz
        self.input_pos.copy_(torch.tensor(prefill_len).to(torch.int32).to(xy_pos.device))

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True
        # ) as prof:

        with contextlib.nullcontext():
            for idx in tqdm(range(1500)):
                # with torch.profiler.record_function("AR"):  # 只追踪 a_operation
                with contextlib.nullcontext():
                    if idx == 0:
                        # input_pos += 1
                        # continue
                        xy_dec = self.h.forward_prefill(xy_pos, xy_attn_mask)
                    else:
                        if idx == 1:
                            import pickle

                            # with open("prefill.pkl", mode="rb") as f:
                            #     xy_pos, k_cache, v_cache, attn_mask, y, y_len = pickle.load(f)
                            # self.h.static_attn_mask[:, :, :, :input_pos] = ~attn_mask
                            # for layer, k, v in zip(self.h.layers, k_cache, v_cache):
                            #     layer.attention.kv_cache.k_cache[:, :, : k.shape[1]] = k.reshape(
                            #         bsz,
                            #         -1,
                            #         self.num_heads,
                            #         self.head_dim,
                            #     ).transpose(
                            #         1,
                            #         2,
                            #     )
                            #     layer.attention.kv_cache.v_cache[:, :, : v.shape[1]] = v.reshape(
                            #         bsz,
                            #         -1,
                            #         self.num_heads,
                            #         self.head_dim,
                            #     ).transpose(
                            #         1,
                            #         2,
                            #     )
                        self.h.static_xy_pos.copy_(xy_pos)

                        if use_cuda_graph and self._CUDA_GRAPH is None:
                            self.CUDAGraphCapture(self.input_pos)
                        if self._CUDA_GRAPH is not None:
                            self._CUDA_GRAPH.replay()
                            xy_dec = self.h.static_out.clone()
                        else:
                            xy_dec = self.h.forward(self.input_pos)
                    logits = self.ar_predict_layer(xy_dec[:, -1])
                self.input_pos += 1

                if idx == 0:
                    logits = logits[:, :-1]

                samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]

                y = torch.concat([y, samples], dim=1)

                tokens = torch.argmax(logits, dim=-1)

                EOS_mask = (samples[:, 0] == self.EOS) | (tokens == self.EOS)
                EOS_indices = torch.where(EOS_mask)[0].tolist()

                for i in EOS_indices:
                    if not completed[i]:
                        y_results[i] = y[i, y_len:-1]
                        completed[i] = True

                if (early_stop_num != -1 and (y.shape[1] - y_len) > early_stop_num) or idx == 1499:
                    tqdm.write(f"Reached early stop limit: {early_stop_num}")
                    for i in range(bsz):
                        if not completed[i]:
                            y_results[i] = y[i, y_len:]
                            completed[i] = True
                    break

                if all(completed):
                    if y.shape[1] == 0:
                        y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                        tqdm.write("bad zero prediction")
                    else:
                        tqdm.write(f"T2S Decoding EOS [{prefill_len} -> {y.shape[1]}]")
                    break

                y_emb = self.ar_audio_embedding(y[:, -1:])
                xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to(
                    dtype=y_emb.dtype, device=y_emb.device
                )
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        return y_results

    def infer_batch_static(
        self,
        x: List[torch.LongTensor],  #####全部文本token
        x_lens: torch.LongTensor,
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
        bsz = len(x)
        max_len = x_lens.max()

        # Empty Cache and Mask
        self.empty_cache_static()

        y = prompts
        x_pos, y_pos = self.embed(x, x_lens, y, bert_feature)
        xy_pos = torch.concat([x_pos, y_pos], dim=1)

        x_len = x_pos.shape[1]
        y_len = y_pos.shape[1]

        # Padding Mask
        xy_attn_mask = torch.zeros((self.max_batch_size, max_len + y_len, max_len + y_len), dtype=torch.bool, device=xy_pos.device)
        for i in range(bsz):
            xy_attn_mask[i, :, : x_lens[i]] = True
        xy_attn_mask[:, -y_len:, -y_len:] = ~torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1)

        xy_attn_mask = xy_attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        self.xy_attn_mask_[:, :, :, : x_len + y_len] = xy_attn_mask[:, :, -1].unsqueeze(2)
        self.h.static_attn_mask_.copy_(self.xy_attn_mask_)

        prefill_len = x_len + y_len

        completed = [False] * bsz
        y_results = [None] * bsz
        self.input_pos_.copy_(torch.tensor(prefill_len).to(torch.int32).to(xy_pos.device))

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=True
        # ) as prof:

        with contextlib.nullcontext():
            for idx in tqdm(range(1500)):
                # with torch.profiler.record_function("AR"):  # 只追踪 a_operation
                with contextlib.nullcontext():
                    if idx == 0:
                        xy_dec = self.h.forward_prefill(xy_pos, xy_attn_mask)
                    else:
                        self.h.static_xy_pos_.copy_(xy_pos)
                        if use_cuda_graph and self._CUDA_GRAPH is None:
                            self.CUDAGraphCaptureStatic(self.input_pos_)
                        if self._CUDA_GRAPH_STATIC is not None:
                            self._CUDA_GRAPH_STATIC.replay()
                            xy_dec = self.h.static_out_.clone()
                        else:
                            xy_dec = self.h.forward_static(self.input_pos_)
                    logits = self.ar_predict_layer(xy_dec[:, -1])
                self.h.static_attn_mask_[:, :, :, self.input_pos_] = True
                self.input_pos_ += 1

                if idx == 0:
                    logits = logits[:, :-1]

                samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]

                y = torch.concat([y, samples], dim=1)

                tokens = torch.argmax(logits, dim=-1)

                EOS_mask = (samples[:, 0] == self.EOS) | (tokens == self.EOS)
                EOS_indices = torch.where(EOS_mask)[0].tolist()

                for i in EOS_indices:
                    if not completed[i]:
                        y_results[i] = y[i, y_len:-1]
                        completed[i] = True

                if (early_stop_num != -1 and (y.shape[1] - y_len) > early_stop_num) or idx == 1499:
                    tqdm.write(f"Reached early stop limit: {early_stop_num}")
                    for i in range(bsz):
                        if not completed[i]:
                            y_results[i] = y[i, y_len:]
                            completed[i] = True
                    break

                if all(completed):  # 所有序列都完成，停止
                    if y.shape[1] == 0:
                        y = torch.concat([y, torch.zeros_like(samples)], dim=1)
                        tqdm.write("bad zero prediction")
                    else:
                        tqdm.write(f"T2S Decoding EOS [{prefill_len} -> {y.shape[1]}]")
                    break

                y_emb = self.ar_audio_embedding(y[:, -1:])
                xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to(
                    dtype=y_emb.dtype, device=y_emb.device
                )
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        return y_results

    def CUDAGraphCapture(self, input_pos):
        assert self._CUDA_GRAPH is None

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(5):
                self.h.forward(input_pos)
        torch.cuda.current_stream().wait_stream(s)

        self._CUDA_GRAPH = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._CUDA_GRAPH):
            self.h.static_out = self.h.forward(input_pos)

        torch.cuda.synchronize()

    def CUDAGraphCaptureStatic(self, input_pos):
        assert self._CUDA_GRAPH_STATIC is None

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for _ in range(5):
                self.h.forward(input_pos)
        torch.cuda.current_stream().wait_stream(s)

        self._CUDA_GRAPH_STATIC = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._CUDA_GRAPH_STATIC):
            self.h.static_out_ = self.h.forward(input_pos)

        torch.cuda.synchronize()
