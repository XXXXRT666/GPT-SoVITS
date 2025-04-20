import asyncio
import os
import pickle
import threading
from typing import Type

import torch
import torch.nn as nn
from peft.mapping import get_peft_model
from peft.tuners.lora.config import LoraConfig
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertForMaskedLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from GPT_SoVITS.AR.models.t2s_model_async import AsyncT2SEngine
from GPT_SoVITS.BigVGAN.bigvgan import BigVGAN
from GPT_SoVITS.feature_extractor.cnhubert import CNHubert
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
from tools.cfg import PRETRAINED_SOVITS_V3, API_Cfg, Inference_WebUI_Cfg, Speakers_Cfg
from tools.my_utils import DictToAttrRecursive

try:
    import uvloop
except ImportError:
    pass
else:
    asyncio.set_event_loop(uvloop.new_event_loop())


class SafePKUnpickler(pickle.Unpickler):
    def __init__(self, file):
        self._file = file
        self._prefix = self._file.read(2)

        if self._prefix == b"PK":
            self._file.seek(0)
            self._file[0:2] = b"PK"
        else:
            pass

        super().__init__(self._file)


class AsyncTTSEngine:
    def __init__(self, cfg: API_Cfg | Inference_WebUI_Cfg, speaker_cfg: Speakers_Cfg, implement="flash_attn") -> None:
        speaker_name = cfg.speaker_name
        self.speaker = speaker_cfg.get_speaker(speaker_name)

        self.t2s_engine = AsyncT2SEngine.from_pretrained(self.speaker.t2s_path, implement)
        self.vits_model: SynthesizerTrn | SynthesizerTrnV3
        self.vocoder: BigVGAN

        self.background_loop = asyncio.new_event_loop()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(20)

        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.background_loop)
        self.background_loop.run_forever()

    async def load_t2s_model(self, weights_path: os.PathLike, implement: str = "flash_attn"):
        async with self.lock:
            self.t2s_engine.decoder_model = AsyncT2SEngine.load_decoder(weights_path, implement)

    def load_hubert(self, weights_path: os.PathLike):
        print(f"Loading CNHuBERT weights from {weights_path}")
        self.cnhuhbert_model = CNHubert(weights_path)
        self.cnhuhbert_model = self.cnhuhbert_model.eval()

    def load_bert(self, weights_path: os.PathLike):
        print(f"Loading BERT weights from {weights_path}")
        self.bert_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(weights_path)
        self.bert_model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(weights_path)
        self.bert_model = self.bert_model.eval()

    def load_vocoder(self, vocoder_cls: Type[BigVGAN]):
        if isinstance(self.vocoder, vocoder_cls):
            return

        match vocoder_cls:
            case BigVGAN:
                self.vocoder = BigVGAN.from_pretrained(self.speaker.bigvgan, use_cuda_kernel=False)
                # remove weight norm in the model and set to eval mode
                self.vocoder.remove_weight_norm()
                self.vocoder = self.vocoder.eval()

    async def load_sovits_model(self, weights_path: os.PathLike):
        async with self.lock:
            print(f"Loading SoVITS weights from {weights_path}")
            dict_s2 = torch.load(
                weights_path, map_location="cpu", weights_only=False, mmap=True, pickle_module=SafePKUnpickler
            )
            hps = dict_s2["config"]
            hps = DictToAttrRecursive(hps)
            hps.model.semantic_frame_rate = "25hz"
            self.hps = hps
            kwds = hps.model

            version = "v0"
            is_lora = False

            if "dec.conv_pre.weight" in dict_s2["weight"].keys():
                if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
                    version = "v1"
                else:
                    version = "v2"
            elif "cfm.estimator.proj_out.bias" in dict_s2["weight"].keys():
                if "cfm.estimator.proj_out.bias" in dict_s2["weight"].keys():
                    version = "v3"

            hps.model.version = version

            if "lora_rank" in dict_s2.keys():
                is_lora = True

            match version:
                case "v1", "v2":
                    self.vits_model = SynthesizerTrn(
                        hps.data.filter_length // 2 + 1,
                        hps.train.segment_size // hps.data.hop_length,
                        n_speakers=hps.data.n_speakers,
                        **kwds,
                    )
                case "v3":
                    self.vits_model = SynthesizerTrnV3(
                        hps.data.filter_length // 2 + 1,
                        hps.train.segment_size // hps.data.hop_length,
                        n_speakers=hps.data.n_speakers,
                        **kwds,
                    )
                    self.load_bigvgan()
                    if "pretrained" not in str(weights_path) and hasattr(self.vits_model, "enc_q"):
                        del self.vits_model.enc_q

            if is_lora:
                print(f"Loading VITS pretrained weights from {weights_path}")
                print(
                    f"{self.vits_model.load_state_dict(torch.load(PRETRAINED_SOVITS_V3, map_location='cpu', pickle_module=SafePKUnpickler)['weight'], strict=False)}"
                )
                lora_rank = dict_s2["lora_rank"]
                lora_config = LoraConfig(
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    init_lora_weights=False,
                )
                self.vits_model.cfm = get_peft_model(self.vits_model.cfm, lora_config, low_cpu_mem_usage=True)  # type: ignore
                print(f"Loading LoRA weights from {weights_path}")
                print(f"{self.vits_model.load_state_dict(dict_s2['weight'], strict=False)}")
                self.vits_model.cfm = self.vits_model.cfm.merge_and_unload()  # type: ignore
            else:
                print(f"Loading VITS weights from {weights_path}")
                print(self.vits_model.load_state_dict(dict_s2["weight"], strict=False))

            self.vits_model.eval()
