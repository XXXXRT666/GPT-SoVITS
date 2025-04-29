import asyncio
import os
import pickle
import threading
import traceback
from collections import defaultdict
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
from GPT_SoVITS.module.models import Generator as HiFiGAN
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
from tools.cfg import PRETRAINED_SOVITS_V3, PRETRAINED_SOVITS_V4, API_Cfg, Inference_WebUI_Cfg, Speakers_Cfg
from tools.my_utils import DictToAttrRecursive

Tensor = torch.Tensor

try:
    import uvloop
except ImportError:
    pass
else:
    asyncio.set_event_loop(uvloop.new_event_loop())


def patch_lora(adapter_state_dict: dict[str, Tensor], adapter_name="default"):
    grouped = defaultdict(dict)

    for full_key in list(adapter_state_dict.keys()):
        if not full_key.endswith(f".{adapter_name}.weight"):
            continue

        # lora_part, adapter_name, weight
        parts = full_key.rsplit(".", 3)
        if len(parts) != 4:
            continue

        module_path, lora_part, adapter_name_check, _ = parts
        if adapter_name_check != adapter_name:
            continue

        if any(x in module_path for x in ["to_q", "to_k", "to_v", "to_out.0"]):
            prefix = module_path.rsplit(".", 1)[0]  # strip to_q / to_k / to_v / to_out.0
            module_name = module_path[len(prefix) + 1 :]
            grouped[prefix][f"{module_name}.{lora_part}"] = full_key

    # Merge
    for prefix, items in grouped.items():
        try:
            prefix = prefix.replace("transformer_blocks", "layers")
            A_q = adapter_state_dict.pop(items["to_q.lora_A"])
            A_k = adapter_state_dict.pop(items["to_k.lora_A"])
            A_v = adapter_state_dict.pop(items["to_v.lora_A"])
            A_qkv = torch.cat([A_q, A_k, A_v], dim=0)

            B_q = adapter_state_dict.pop(items["to_q.lora_B"])
            B_k = adapter_state_dict.pop(items["to_k.lora_B"])
            B_v = adapter_state_dict.pop(items["to_v.lora_B"])
            B_qkv = torch.cat([B_q, B_k, B_v], dim=0)

            adapter_state_dict[f"{prefix}.in_proj.lora_A.{adapter_name}.weight"] = A_qkv
            adapter_state_dict[f"{prefix}.in_proj.lora_B.{adapter_name}.weight"] = B_qkv

            A_out = adapter_state_dict.pop(items["to_out.0.lora_A"])
            B_out = adapter_state_dict.pop(items["to_out.0.lora_B"])
            adapter_state_dict[f"{prefix}.out_proj.lora_A.{adapter_name}.weight"] = A_out
            adapter_state_dict[f"{prefix}.out_proj.lora_B.{adapter_name}.weight"] = B_out

        except KeyError:
            traceback.print_exc()

    return adapter_state_dict


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

    def load_vocoder(self, vocoder_cls: Type[BigVGAN] | Type[HiFiGAN]):
        if isinstance(self.vocoder, vocoder_cls):
            return

        if vocoder_cls == BigVGAN:
            bigvgan_model = BigVGAN.from_pretrained(self.speaker.bigvgan, use_cuda_kernel=False)
            # remove weight norm in the model and set to eval mode
            bigvgan_model.remove_weight_norm()
            bigvgan_model = self.vocoder.eval()

        elif vocoder_cls == HiFiGAN:
            hifigan_model = HiFiGAN(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            hifigan_model.eval()
            hifigan_model.remove_weight_norm()
            state_dict = torch.load(self.speaker.hifigan, map_location="cpu")
            print(f"Loading Vocoder From {self.speaker.hifigan}")
            print(hifigan_model.load_state_dict(state_dict))

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

            version = ""
            is_lora = False

            if "dec.conv_pre.weight" in dict_s2["weight"].keys():
                if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
                    version = "v1"
                else:
                    version = "v2"
            elif "bridge.0.weight" in dict_s2["weight"].keys():
                version = "v3"
                if dict_s2["info"] == "pretrained_s2G_v4":
                    version = "v4"
                if dict_s2["config"].get("version"):
                    version = dict_s2["config"].get("version")

            hps.model.version = version

            if "lora_rank" in dict_s2.keys():
                is_lora = True

            yield version

            match version:
                case "v1", "v2":
                    self.vits_model = SynthesizerTrn(
                        hps.data.filter_length // 2 + 1,
                        hps.train.segment_size // hps.data.hop_length,
                        n_speakers=hps.data.n_speakers,
                        **kwds,
                    )
                case "v3", "v4":
                    self.vits_model = SynthesizerTrnV3(
                        hps.data.filter_length // 2 + 1,
                        hps.train.segment_size // hps.data.hop_length,
                        n_speakers=hps.data.n_speakers,
                        **kwds,
                    )
                    if "pretrained" not in str(weights_path) and hasattr(self.vits_model, "enc_q"):
                        del self.vits_model.enc_q

            if is_lora:
                if version == "v3":
                    print(f"Loading VITS pretrained weights from {PRETRAINED_SOVITS_V3}")
                    print(
                        self.vits_model.load_state_dict(
                            patch_lora(
                                torch.load(
                                    PRETRAINED_SOVITS_V3,
                                    map_location="cpu",
                                    pickle_module=SafePKUnpickler,
                                )["weight"]
                            ),
                            strict=False,
                        )
                    )
                elif version == "v4":
                    print(f"Loading VITS pretrained weights from {PRETRAINED_SOVITS_V4}")
                    print(
                        self.vits_model.load_state_dict(
                            patch_lora(
                                torch.load(
                                    PRETRAINED_SOVITS_V4,
                                    map_location="cpu",
                                    pickle_module=SafePKUnpickler,
                                )["weight"]
                            ),
                            strict=False,
                        )
                    )
                lora_rank = dict_s2["lora_rank"]
                lora_config = LoraConfig(
                    # target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                    target_modules=["in_proj", "out_proj"],
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    init_lora_weights=False,
                )
                self.vits_model.cfm = get_peft_model(self.vits_model.cfm, lora_config, low_cpu_mem_usage=True)  # type: ignore
                print(f"Loading LoRA weights from {weights_path}")
                print(f"{self.vits_model.load_state_dict(dict_s2['weight'], strict=False)}")
                self.vits_model.cfm = self.vits_model.cfm.merge_and_unload(progressbar=True, safe_merge=True)  # type: ignore
            else:
                print(f"Loading VITS weights from {weights_path}")
                print(self.vits_model.load_state_dict(dict_s2["weight"], strict=False))

            self.vits_model.eval()
