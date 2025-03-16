from typing import List, Optional, Sequence

import flash_attn
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from GPT_SoVITS.AR.models.utils import sample
from GPT_SoVITS.AR.modules.embedding import SinePositionalEmbedding, TokenEmbedding

Tensor = torch.Tensor
dtype = torch.dtype


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.num_heads = n_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length

        self.register_buffer("k_cache", torch.zeros(size=cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(size=cache_shape), persistent=False)
        self.k_cache: Tensor
        self.v_cache: Tensor

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        # input_pos: [B, 1], k_val: [B, 1, H, D]

        index = (input_pos - 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, self.head_dim)  # (bs, 1, num_head, head_dim)

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

    def prefill_kv(self, bs: int, input_pos: int, k_val: Tensor, v_val: Tensor):
        # input_pos: int, k_val: [B, S, H, D]

        self.k_cache[bs, :input_pos] = k_val
        self.v_cache[bs, :input_pos] = v_val


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

        q = q.contiguous().view(bsz, -1, self.num_heads, self.head_dim)
        k = k.contiguous().view(bsz, -1, self.num_heads, self.head_dim)
        v = v.contiguous().view(bsz, -1, self.num_heads, self.head_dim)

        k, v = self.kv_cache.update(input_pos, k, v)

        attn: Tensor = flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=input_pos)

        attn = attn.view(bsz, seqlen, self.hidden_dim)

        attn = self.out_proj.forward(attn)

        return attn

    def prefill(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:

        bsz = x.shape[0]

        outputs = []

        for bs in range(bsz):

            pos = int(input_pos[bs].item())

            x_b = x[bs, pos]  # (embed_dim,)

            q, k, v = self.in_proj.forward(x_b).chunk(3, dim=-1)

            q = q.contiguous().view(bsz, -1, self.num_heads, self.head_dim)
            k = k.contiguous().view(bsz, -1, self.num_heads, self.head_dim)
            v = v.contiguous().view(bsz, -1, self.num_heads, self.head_dim)

            self.kv_cache.prefill_kv(bs, pos, k, v)

            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

            attn_mask = mask[bs, :pos, :pos]

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            output = self.out_proj(attn_output)

            outputs.append(output)

        return torch.cat(outputs, dim=0)


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

    def prefill(self, x: Tensor, input_pos: Tensor, mask: Tensor) -> Tensor:
        h = self.attention_norm.forward(x + self.attention.prefill(x, input_pos, mask))
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

        self.register_buffer("input_pos", torch.zeros((5,)).to(torch.int32))
        self.register_buffer("xy_pos", torch.zeros((self.max_batch_size, 1, self.hidden_dim)))
        self.register_buffer("xy_dec", torch.zeros((self.max_batch_size, 1, self.hidden_dim)))

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

    def prefill(self, input_pos: Tensor, x: Tensor, mask: Tensor):
        for layer in self.layers:
            x = layer.prefill(x, input_pos, mask)
        return x


class T2SDecoder(nn.Module):
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
        self.ar_text_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
        self.ar_audio_embedding = TokenEmbedding(self.embedding_dim, self.vocab_size, self.p_dropout)
        self.ar_audio_position = SinePositionalEmbedding(self.embedding_dim, dropout=0.1, scale=False, alpha=True)
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
        x_lens: torch.LongTensor,
        y: torch.LongTensor,
        bert_feature: List[torch.LongTensor],
    ):
        y_lens = torch.LongTensor([y.shape[1]] * y.shape[0]).to(y.device)
        max_x_len = max(x_lens.tolist())
        y_emb = self.ar_audio_embedding.forward(y)
        y_pos = self.ar_audio_position.forward(y_emb)
        max_y_len = max(y_lens.tolist())
        xy_pos = torch.zeros((len(x), max_x_len + max_y_len, self.embedding_dim), dtype=bert_feature[0].dtype, device=x[0].device)

        for idx, (x_item, bert_item) in enumerate(zip(x, bert_feature)):
            x_item = self.ar_text_embedding.forward(x_item.unsqueeze(0))  # 1, seq_len, embedding dim
            x_item = x_item + self.bert_proj.forward(bert_item.transpose(0, 1).unsqueeze(0))
            x_item = self.ar_text_position.forward(x_item).squeeze(0)
            xy_pos[idx, : x_lens[idx]] = x_item
            xy_pos[idx, x_lens[idx] : x_lens[idx] + max_y_len] = y_pos[idx]

        return xy_pos

    def forward(self): ...

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
        use_cuda_graph=False,
        **kwargs,
    ):
        self.empty_cache()

        bsz = len(x)
        max_len = x_lens.max()

        y = prompts
        xy_pos = self.embed(x, x_lens, y, bert_feature)

        max_x_len = int(max_len)
        y_len = y.shape[-2]

        prefill_len = x_lens + y_len

        xy_attn_mask = torch.zeros((self.max_batch_size, max_x_len + y_len, max_x_len + y_len), dtype=torch.bool, device=xy_pos.device)
        for bs in range(bsz):
            pos = int(x_lens[bs].item())
            mask = xy_attn_mask[bs, : pos + y_len, : pos + y_len]
            mask[:, :pos].fill_(True)
            mask[-y_len:, -y_len:] = ~torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1)

        completed = [False] * bsz
        y_results = [None] * bsz
        self.h.input_pos += prefill_len
        input_pos = self.h.input_pos

        for idx in tqdm(range(1500)):
            if idx == 0:
                xy_dec = self.h.prefill(input_pos, xy_pos, xy_attn_mask)
            else:
                if torch.cuda.is_available() and use_cuda_graph:
                    self.capture(input_pos, xy_pos)
                if self.__CUDAGraph is not None:
                    self.h.xy_pos.copy_(xy_pos)
                    self.__CUDAGraph.replay()
                    xy_dec = self.h.xy_dec.clone()
                else:
                    xy_dec = self.h.forward(input_pos, xy_pos)

            logits = self.ar_predict_layer(xy_dec[:, -1])
            input_pos += 1

            if idx == 0:
                logits = logits[:, :-1]

            samples = sample(logits, y, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature)[0]

            y = torch.concat([y, samples], dim=1)

            tokens = torch.argmax(logits, dim=-1)

            EOS_mask = (samples[:, 0] == self.EOS) | (tokens == self.EOS)
            EOS_indices: List[int] = torch.where(EOS_mask)[0].tolist()

            for i in EOS_indices:
                if not completed[i]:
                    y_results[i] = y[i, y_len : input_pos[i]]  # type: ignore
                    completed[i] = True

            if (early_stop_num != -1 and (y.shape[1] - y_len) > early_stop_num) or idx == 1499:
                tqdm.write(f"Reached early stop limit: {early_stop_num}")
                for i in range(bsz):
                    if not completed[i]:
                        y_results[i] = y[i, y_len : input_pos[i]]  # type: ignore
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
            xy_pos = y_emb * self.ar_audio_position.x_scale + self.ar_audio_position.alpha * self.ar_audio_position.pe[:, y_len + idx].to(  # type: ignore
                dtype=y_emb.dtype, device=y_emb.device
            )
