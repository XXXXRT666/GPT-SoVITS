import asyncio
import os
from asyncio import AbstractEventLoop, Semaphore
from typing import Literal

import torch
from peft.mapping_func import get_peft_model
from peft.tuners.lora.config import LoraConfig

from GPT_SoVITS.AsyncTTS.SoVITS.utils import SafePKUnpickler, patch_lora
from GPT_SoVITS.AsyncTTS.utils import AsyncRLock
from GPT_SoVITS.BigVGAN.bigvgan import BigVGAN
from GPT_SoVITS.module.models import Generator as HiFiGAN
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
from tools.cfg import BIGVGAN_DEFAULT, HIFIGAN_DEFAULT, PRETRAINED_SOVITS_V3, PRETRAINED_SOVITS_V4
from tools.my_utils import DictToAttrRecursive

Tensor = torch.Tensor

torch.set_grad_enabled(False)

VOCODER_MAPPING = {
    "v3": BigVGAN,
    "v4": HiFiGAN,
}


class AsyncSoVITSEngine:
    def __init__(
        self,
        sovits_model: SynthesizerTrn | SynthesizerTrnV3,
        background_loop: AbstractEventLoop,
        lock: AsyncRLock,
        semaphore: Semaphore,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert device.type in {"cpu", "cuda", "mps"}
        assert dtype in {torch.float16, torch.bfloat16, torch.float32}
        self.device = device
        self.dtype = dtype

        self.sovits_path: os.PathLike
        self.sovits_model: SynthesizerTrn | SynthesizerTrnV3 = sovits_model
        self.vocoder: BigVGAN | HiFiGAN | None = None
        self.hps: DictToAttrRecursive

        self.background_loop = background_loop
        self.lock = lock
        self.semaphore = semaphore

    def load_vocoder(self, version: Literal["v1", "v2", "v3", "v4"]):
        if version in {"v1", "v2"}:
            return

        vocoder_cls = VOCODER_MAPPING["version"]

        if vocoder_cls == BigVGAN:
            print(f"Loading Vocoder From {BIGVGAN_DEFAULT}")
            bigvgan_model = BigVGAN.from_pretrained(BIGVGAN_DEFAULT, use_cuda_kernel=False)
            # remove weight norm in the model and set to eval mode
            bigvgan_model.remove_weight_norm()
            self.vocoder = bigvgan_model.eval()

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
            hifigan_model.remove_weight_norm()
            hifigan_model.eval()
            state_dict = torch.load(HIFIGAN_DEFAULT, map_location="cpu")
            print(f"Loading Vocoder From {HIFIGAN_DEFAULT}")
            print(hifigan_model.load_state_dict(state_dict))
            self.vocoder = hifigan_model
        if self.vocoder is not None:
            self.vocoder = self.vocoder.to(self.device, self.dtype)

    @staticmethod
    def inspect_version(weights_path: os.PathLike):
        print(f"Loading SoVITS weights from {weights_path}")
        dict_s2 = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
            pickle_module=SafePKUnpickler,
        )
        hps = dict_s2["config"]
        # hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"

        version: Literal["v1", "v2", "v3", "v4"] | None = None

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

        if version is None:
            raise RuntimeError("Unknown Version")

        return version, dict_s2, hps

    @staticmethod
    def load_weights(
        weights_path: os.PathLike,
        version: Literal["v1", "v2", "v3", "v4"],
        dict_s2: dict,
        hps: DictToAttrRecursive,
    ):
        sovits_model: SynthesizerTrn | SynthesizerTrnV3
        kwds = hps.model
        match version:
            case "v1" | "v2":
                sovits_model = SynthesizerTrn(
                    hps.data.filter_length // 2 + 1,
                    hps.train.segment_size // hps.data.hop_length,
                    n_speakers=hps.data.n_speakers,
                    **kwds,
                )
                pass
            case "v3" | "v4":
                sovits_model = SynthesizerTrnV3(
                    hps.data.filter_length // 2 + 1,
                    hps.train.segment_size // hps.data.hop_length,
                    n_speakers=hps.data.n_speakers,
                    **kwds,
                )
                if "pretrained" not in str(weights_path) and hasattr(sovits_model, "enc_q"):
                    del sovits_model.enc_q
            case _:
                raise RuntimeError(f"Unknown Version: {version}")

        if "lora_rank" in dict_s2.keys():
            pretrain_dict: dict[str, Tensor] = {}
            if version == "v3":
                print(f"Loading VITS pretrained weights from {PRETRAINED_SOVITS_V3}")
                pretrain_dict.update(
                    torch.load(
                        PRETRAINED_SOVITS_V3,
                        map_location="cpu",
                        pickle_module=SafePKUnpickler,
                    )["weight"]
                )

            elif version == "v4":
                print(f"Loading VITS pretrained weights from {PRETRAINED_SOVITS_V4}")
                pretrain_dict.update(
                    torch.load(
                        PRETRAINED_SOVITS_V4,
                        map_location="cpu",
                        pickle_module=SafePKUnpickler,
                    )["weight"]
                )
            lora_rank = dict_s2["lora_rank"]
            lora_config = LoraConfig(
                target_modules=["in_proj", "out_proj"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=False,
            )
            state_dict: dict[str, Tensor] = pretrain_dict | patch_lora(dict_s2["weight"])
            sovits_model.cfm = get_peft_model(sovits_model.cfm, lora_config, low_cpu_mem_usage=True)  # type: ignore
            print(f"Loading LoRA weights from {weights_path}")
            print(f"{sovits_model.load_state_dict(state_dict, strict=False)}")
            sovits_model.cfm = sovits_model.cfm.merge_and_unload(progressbar=True, safe_merge=True)  # type: ignore
        else:
            print(f"Loading VITS weights from {weights_path}")
            print(sovits_model.load_state_dict(dict_s2["weight"], strict=False))

        return sovits_model.eval(), hps

    @staticmethod
    def load_sovits(weights_path: os.PathLike):
        tmp = AsyncSoVITSEngine.inspect_version(weights_path=weights_path)
        return tmp[0], *AsyncSoVITSEngine.load_weights(weights_path, *tmp)
