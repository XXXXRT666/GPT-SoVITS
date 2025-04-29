import asyncio
import os
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import AsyncGenerator, Generic, List, Literal, Optional, Tuple, TypeVar
from weakref import WeakValueDictionary

import torch
from tqdm import tqdm

from GPT_SoVITS.AR.models.t2s_model_abc import KVCacheNHD as KVCache
from GPT_SoVITS.AR.models.t2s_model_abc import Sampler, T2SDecoderABC, TorchProfiler

Tensor = torch.Tensor
K = TypeVar("K")
V = TypeVar("V")

try:
    import uvloop
except ImportError:
    pass
else:
    asyncio.set_event_loop(uvloop.new_event_loop())


class DQCache(Generic[K, V]):
    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self.cache: WeakValueDictionary[K, V] = WeakValueDictionary()
        self.dq: deque[Tuple[K, V]] = deque(maxlen=max_size)

    def __setitem__(self, request_id: K, session: V):
        self.cache[request_id] = session
        if len(self.dq) == 16:
            evicted = self.dq.popleft()[0]
            self.cache.pop(evicted)
            self.dq.append((request_id, session))

    def __getitem__(self, request_id: K) -> Optional[V]:
        return self.cache[request_id]


@dataclass
class T2SResult:
    result: List[Tensor] | None = None
    status: Literal["Success", "Error", "Cancelled"] = "Success"
    exception: Optional[Exception] = None
    traceback: Optional[str] = None


@dataclass
class T2SRequest:
    x: List[torch.Tensor]
    x_lens: Tensor
    prompts: torch.Tensor
    bert_feature: List[Tensor]
    valid_length: int
    top_k: int = 5
    top_p: int = 1
    early_stop_num: int = -1
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    use_cuda_graph: bool = False
    debug: bool = False
    request_id = str(uuid.uuid1())


class T2SSession:
    def __init__(self, decoder: T2SDecoderABC, request: T2SRequest):
        self.decoder = decoder
        self.request = request

        bsz = len(self.x)
        y_len = self.y.size(-1)
        self.bsz = bsz
        self.y_len = y_len

        # Cache
        self.kv_cache = decoder.init_cache(KVCache, bsz)  # type: ignore
        self.sampler = Sampler(bsz, decoder.vocab_size)

        # Forward args
        self.x = request.x
        self.x_lens = request.x_lens.to(torch.int64)
        self.y = request.prompts
        self.bert_feature = request.bert_feature

        self.prefill_len = self.x_lens + self.y.size(1)

        self.input_pos = decoder.h.input_pos.clone()
        self.input_pos.add_(self.prefill_len)

        # CUDA Graph
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.xy_pos_ = decoder.h.xy_pos.clone()
        self.xy_dec_ = decoder.h.xy_dec.clone()

        # EOS
        self.completed = [False] * len(self.x)
        self.y_results: List[Tensor] = [None] * len(self.x)  # type: ignore

        self.xy_pos = decoder.embed(self.x, self.y, self.bert_feature)

        attn_mask = []
        for bs in range(bsz):
            pos = int(self.x_lens[bs].item())
            mask = torch.zeros(pos + y_len, pos + y_len, device=self.xy_pos.device).bool()
            mask[:, :pos].fill_(True)
            mask[-y_len:, -y_len:] = ~torch.triu(
                torch.ones(y_len, y_len, device=self.xy_pos.device, dtype=torch.bool), diagonal=1
            )
            attn_mask.append(mask)
        self.attn_mask_nested = torch.nested.nested_tensor(attn_mask)


class AsyncT2SEngine:
    def __init__(self, decoder_model: T2SDecoderABC):
        self.decoder_model = decoder_model
        self.background_loop = asyncio.new_event_loop()
        self.sessions: DQCache[str, T2SSession] = DQCache()
        self.futures: WeakValueDictionary[str, Future] = WeakValueDictionary()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(20)

        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.background_loop)
        self.background_loop.run_forever()

    async def _handle_request(self, request: T2SRequest):
        async with self.lock:
            decoder = self.decoder_model
            session = T2SSession(decoder, request)
            self.sessions[request.request_id] = session

        y = session.y
        bsz = y.size(0)
        t1 = 0.0

        torch_profiler = TorchProfiler(request.debug)

        with torch_profiler.profiler():
            for idx in tqdm(range(1500)):
                if idx == 0:
                    xy_dec = decoder.h.prefill(session.xy_pos, session.attn_mask_nested, session.kv_cache)
                    xy_dec = torch.stack([t[[-1]] for t in xy_dec.unbind()])
                else:
                    if request.use_cuda_graph and session.graph is None and torch.cuda.is_available():
                        session.xy_pos_.copy_(session.xy_pos)
                        session.graph = decoder.capture(
                            session.input_pos, session.xy_pos_, session.xy_dec_, kv_caches=session.kv_cache
                        )

                    torch_profiler.start()
                    with torch_profiler.record("AR"):
                        if session.graph:
                            session.xy_pos_.copy_(session.xy_pos)
                            session.graph.replay()
                            xy_dec = session.xy_dec_.clone()
                        else:
                            xy_dec = decoder.h.forward(session.input_pos, session.xy_pos, session.kv_cache)

                logits = decoder.ar_predict_layer(xy_dec[:, -1])
                session.input_pos.add_(1)

                if idx == 0:
                    logits = logits[:, :-1]

                with torch_profiler.record("Sampling"):
                    samples = session.sampler.sample(
                        logits=logits,
                        previous_tokens=session.y,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        repetition_penalty=request.repetition_penalty,
                        temperature=request.temperature,
                        use_cuda_graph=session.graph is not None,
                        idx=idx,
                    )

                    session.y = torch.cat([session.y, samples], dim=1)  # type: ignore

                with torch_profiler.record("EOS"):
                    EOS_mask = (samples[:, 0] == decoder.EOS) | (torch.argmax(logits, dim=-1) == decoder.EOS)
                    EOS_indices: List[int] = torch.where(EOS_mask)[0].tolist()

                    for i in EOS_indices:
                        if not session.completed[i]:
                            session.y_results[i] = session.y[i, session.y_len : -1]
                            session.completed[i] = True

                    if all(session.completed):
                        if session.y.size(1) == 0:
                            session.y = torch.cat([session.y, torch.zeros_like(samples)], dim=1)
                            tqdm.write("Bad Zero Prediction")
                        else:
                            tqdm.write(
                                f"T2S Decoding EOS {session.prefill_len.tolist().__str__().strip('[]')} -> \n{[i.size(0) for i in session.y_results].__str__().strip('[]')}"
                            )
                            tqdm.write(f"Infer Speed: {(idx - 1) / (time.perf_counter() - t1):.2f}")
                        break

                    if request.early_stop_num != -1 and (session.y.size(1) - session.y_len) > request.early_stop_num:
                        for i in range(bsz):
                            if not session.completed[i]:
                                session.y_results[i] = session.y[i, session.y_len :]
                                session.completed[i] = True
                        break

                with torch_profiler.record("NextPos"):
                    y_emb = decoder.ar_audio_embedding(session.y[:, -1:])
                    session.xy_pos = decoder.ar_audio_position.forward(session.input_pos - session.x_lens, y_emb)

                if idx == 2:
                    t1 = time.perf_counter()

                if idx == 51:
                    torch_profiler.end()

                if idx % 10 == 0:
                    await asyncio.sleep(0)

            async with self.lock:
                self.futures.pop(request.request_id, None)

            return session.y_results

    async def generate(self, request: T2SRequest) -> AsyncGenerator[T2SResult]:
        try:
            async with self.semaphore:
                future = asyncio.run_coroutine_threadsafe(self._handle_request(request), self.background_loop)
                async with self.lock:
                    self.futures[request.request_id] = future
                result = future.result()
                t2s_result = T2SResult(result=result, status="Success")
        except asyncio.CancelledError:
            t2s_result = T2SResult(status="Cancelled")
        except Exception as e:
            t2s_result = T2SResult(status="Error", exception=e, traceback=traceback.format_exc())
        yield t2s_result

    async def cancel(self, request_id: str):
        async with self.lock:
            future = self.futures.pop(request_id, None)
        if future and not future.done():
            future.cancel()

    @classmethod
    def from_pretrained(cls, weights_path: os.PathLike, implement: str):
        return cls(cls.load_decoder(weights_path, implement))

    @staticmethod
    def load_decoder(weights_path: os.PathLike, implement: str):
        print(f"Loading Text2Semantic Weights from {weights_path} with {implement.replace('_', ' ').title()} Implement")
        module_path = f"GPT_SoVITS.AR.models.t2s_model_{implement.lower()}"
        cls_name = "T2SDecoder"
        mod = __import__(module_path, fromlist=[cls_name])
        decoder_cls: T2SDecoderABC = getattr(mod, cls_name)
        dict_s1 = torch.load(weights_path, map_location="cpu", weights_only=False, mmap=True)
        config = dict_s1["config"]
        decoder: T2SDecoderABC = decoder_cls(config, max_batch_size=20)
        state_dict = dict_s1["weight"]
        decoder.load_state_dict(state_dict)
        return decoder.eval()
