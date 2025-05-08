import asyncio
import os
import threading

import torch
import torch.nn as nn
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    HubertModel,
    Wav2Vec2FeatureExtractor,
)

from GPT_SoVITS.AsyncTTS.GPT.AsyncT2S import AsyncT2SEngine
from GPT_SoVITS.AsyncTTS.SoVITS.AsyncSoVITS import AsyncSoVITSEngine
from GPT_SoVITS.AsyncTTS.utils import AsyncRLock
from tools.cfg import BERT_DEFAULT, CNHUBERT_DEFAULT, API_Cfg, Inference_WebUI_Cfg, Speaker, Speakers_Cfg

Tensor = torch.Tensor

try:
    import uvloop
except ImportError:
    pass
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

torch.set_grad_enabled(False)


class CNHubert(nn.Module):
    def __init__(
        self,
        base_path: os.PathLike,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        assert device.type in {"cpu", "cuda", "mps"}
        assert dtype in {torch.float16, torch.bfloat16, torch.float32}
        self.device = device
        self.dtype = dtype

        if base_path is None:
            base_path = CNHUBERT_DEFAULT
        if os.path.exists(base_path):
            ...
        else:
            raise FileNotFoundError(base_path)
        self.model = HubertModel.from_pretrained(base_path, local_files_only=True).to(self.device, self.dtype)  # type: ignore
        self.feature_extractor: Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_path, local_files_only=True
        )

    def forward(self, x: Tensor) -> Tensor:
        input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device)  # type: ignore
        feats = self.model(input_values)["last_hidden_state"]
        return feats


class Bert(nn.Module):
    def __init__(
        self,
        base_path: os.PathLike,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        assert device.type in {"cpu", "cuda", "mps"}
        assert dtype in {torch.float16, torch.bfloat16, torch.float32}
        self.device = device
        self.dtype = dtype

        self.bert_tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(base_path)
        self.bert_model = BertForMaskedLM.from_pretrained(base_path)
        self.bert_model = self.bert_model.eval()
        self.bert_model = self.bert_model.to(self.device, self.dtype)  # type: ignore

    def forward(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt")
        for i in inputs.keys():
            inputs[i] = inputs[i].to(self.device)  # type: ignore
        res = self.bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0][1:-1]  # type: ignore
        return res


class AsyncTTSEngine:
    def __init__(
        self,
        cfg: API_Cfg | Inference_WebUI_Cfg,
        speaker_cfg: Speakers_Cfg,
        implement="flash_attn",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert device.type in {"cpu", "cuda", "mps"}
        assert dtype in {torch.float16, torch.bfloat16, torch.float32}
        self.device = device
        self.dtype = dtype

        speaker_name = cfg.speaker_name
        self.speaker_cfg = speaker_cfg
        self.speaker = speaker_cfg.get_speaker(speaker_name)

        self.background_loop = asyncio.new_event_loop()
        self.lock = AsyncRLock()
        self.semaphore = asyncio.Semaphore(30)

        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        self.t2s_engine = AsyncT2SEngine(
            None,  # type: ignore
            background_loop=self.background_loop,
            lock=self.lock,
            semaphore=self.semaphore,
        )
        self.sovits_engine = AsyncSoVITSEngine(
            None,  # type: ignore
            background_loop=self.background_loop,
            lock=self.lock,
            semaphore=self.semaphore,
        )
        self.implement = implement

    def _run_event_loop(self):
        torch.set_grad_enabled(False)
        asyncio.set_event_loop(self.background_loop)
        self.background_loop.run_forever()

    def sync_load_speaker(self, speaker: str):
        pass

    async def load_speaker(self, speaker: Speaker):
        async with self.lock:
            self.load_hubert(CNHUBERT_DEFAULT)
            self.load_bert(BERT_DEFAULT)
            if hasattr(self.t2s_engine, "decoder_path") and speaker.t2s_path != self.t2s_engine.decoder_path:
                await self.load_t2s_model(speaker.t2s_path)
            if hasattr(self.sovits_engine, "sovits_path") and speaker.sovits_path != self.sovits_engine.sovits_path:
                version = self.load_sovits_model(weights_path=speaker.sovits_path)
                self.version = version
                if speaker.name in {"WebUI", "API"}:
                    cur_speaker = self.speaker_cfg.get_speaker(speaker.name)
                    cur_speaker.t2s_path = speaker.t2s_path
                    cur_speaker.sovits_path = speaker.sovits_path
                    self.speaker_cfg.save_as_json()
                return version

    def load_hubert(self, weights_path: os.PathLike):
        if hasattr(self, "cnhuhbert_model"):
            pass
        print(f"Loading CNHuBERT weights from {weights_path}")
        self.cnhuhbert_model = CNHubert(weights_path, device=self.device, dtype=self.dtype)

    def load_bert(self, weights_path: os.PathLike):
        if hasattr(self, "cnhuhbert_model"):
            pass
        print(f"Loading BERT weights from {weights_path}")
        self.bert_model = Bert(weights_path, device=self.device, dtype=self.dtype)

    async def load_t2s_model(self, weights_path: os.PathLike, implement: str = "flash_attn"):
        async with self.lock:
            self.t2s_engine.decoder_model = AsyncT2SEngine.load_decoder(weights_path, implement)
            self.t2s_engine.decoder_path = weights_path
            self.t2s_engine.decoder_model = self.t2s_engine.decoder_model.to(self.device, self.dtype)

    async def load_sovits_model(self, weights_path: os.PathLike):
        async with self.lock:
            version, self.sovits_engine.sovits_model, self.sovits_engine.hps = AsyncSoVITSEngine.load_sovits(
                weights_path=weights_path
            )
            self.sovits_engine.sovits_path = weights_path
            self.sovits_engine.sovits_model.to(self.device, self.dtype)
            self.sovits_engine.load_vocoder(version)
            return version

    def load_t2s_model_sync(self, weights_path: os.PathLike, implement: str = "flash_attn"):
        future = asyncio.run_coroutine_threadsafe(self.load_t2s_model(weights_path, implement), self.background_loop)
        future.result()

    def load_sovits_model_sync(self, weights_path: os.PathLike):
        future = asyncio.run_coroutine_threadsafe(self.load_sovits_model(weights_path), self.background_loop)
        future.result()

    def load_speaker_sync(self, speaker: Speaker):
        future = asyncio.run_coroutine_threadsafe(self.load_speaker(speaker), self.background_loop)
        future.result()
