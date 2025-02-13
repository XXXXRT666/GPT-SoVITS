import os
import warnings
from functools import partial
from typing import Dict, Optional, Annotated, Literal, TypeVar, Hashable, Self

import torch
from pydantic_core import PydanticCustomError
from pydantic import BaseModel, Field, model_validator, AfterValidator

from tools.i18n.i18n import scan_language_list
from tools.utils.my_utils import check_infer_device

CNHUBERT_DEFAULT = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
BERT_DEFAULT = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

PRETRAINED_T2S_V1 = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
PRETRAINED_T2S_V2 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
PRETRAINED_VITS_V1 = "GPT_SoVITS/pretrained_models/s2G488k.pth"
PRETRAINED_VITS_V2 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"

V1_LANGUAGES = [
    "all_zh",  # All Chinese
    "en",  # All English
    "all_ja",  # All Japanese
    "zh",  # Mixed Chinese & English
    "ja",  # Mixed Japanese & English
    "auto",  # Auto Detect Language
]
V2_LANGUAGES = [
    "all_zh",  # All Chinese
    "en",  # All English
    "all_ja",  # All Japanese
    "all_yue",  # All Cantonese
    "all_ko",  # All Korean
    "zh",  # Mixed Chinese & English
    "ja",  # Mixed Japanese & English
    "yue",  # Mixed Cantonese & English
    "ko",  # Mixed Korean & English
    "auto",  # Auto Detect Language (Without Cantonese)
    "auto_yue",  # Auto Detect Language (with Cantonese)
]

T = TypeVar("T", bound=Hashable)


def _validate_unique_list(v: list[T]) -> list[T]:
    if len(v) != len(set(v)):
        raise PydanticCustomError("unique_list", "list must be unique")
    return v


Uniquelist = Annotated[list[T], AfterValidator(_validate_unique_list)]


class Main_WebUI_Cfg(BaseModel):
    init: bool = Field(default=False, exclude=True)
    cuda: bool = Field(default=False)
    cuda_visible_device: list[int] = Field(default_factory=list, exclude=not torch.cuda.is_available())
    fp16: bool = False
    logging_dir: str = "logs"
    python_exec: str = "python3"
    host: str = "0.0.0.0"
    webui_port: Annotated[int, Field(ge=0, le=65536, strict=True)] = 9874
    uvr5_webui_port: Annotated[int, Field(ge=0, le=65536, strict=True)] = 9873
    subfux_webui_port: Annotated[int, Field(ge=0, le=65536, strict=True)] = 9871
    i18n_language: str = "en_US"
    gradio_share: bool = False

    @model_validator(mode="after")
    @classmethod
    def check_all(cls, vals):
        if vals.init:
            _, vals.fp16 = check_infer_device()
            if torch.cuda.is_available():
                vals.cuda = True
                vals.cuda_visible_device = list(range(0, torch.cuda.device_count()))
        vals.init = False

        if not torch.cuda.is_available():
            vals.cuda = False
        if not vals.cuda:
            vals.cuda_visible_device = []
            vals.fp16 = False

        assert vals.i18n_language in scan_language_list(), ValueError("I18n language not supported")
        return vals


class Inference_WebUI_Cfg(BaseModel):
    init: bool = Field(default=False, exclude=True)
    device: str = "cpu"
    fp16: bool = False
    host: str = "0.0.0.0"
    port: Annotated[int, Field(ge=0, le=65536, strict=True)] = 9872
    i18n_language: str = "en_US"
    gradio_share: bool = False
    speaker_name: str = "WebUI"

    streaming: bool = Field(default=False, exclude=True)
    top_k: Annotated[int, Field(ge=1, le=100, strict=True)] = Field(default=5, exclude=True)
    top_p: Annotated[float, Field(ge=0.01, le=1.0, strict=True)] = Field(default=1.0, exclude=True)
    temperature: Annotated[float, Field(ge=0.01, le=1.0, strict=True)] = Field(default=1.0, exclude=True)
    text_splitting_method: Annotated[str, Field(pattern=r"^cut[0-5]$")] = Field(default="cut1", exclude=True)
    batch_size: int = Field(default=4, exclude=True)
    speed_factor: Annotated[float, Field(ge=0.6, le=1.4, strict=True)] = Field(default=1.0, exclude=True)
    fragment_interval: Annotated[float, Field(ge=0.01)] = Field(default=0.3, exclude=True)
    parallel_inference: bool = Field(default=True, exclude=True)
    reprtition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = Field(default=1.35, exclude=True)
    media_type: Literal["pcm", "wav", "aac", "ogg"] = Field(default="pcm", exclude=True)

    @model_validator(mode="after")
    @classmethod
    def check_all(cls, vals):
        if vals.init:
            vals.device, vals.fp16 = check_infer_device()
        vals.init = False
        device = vals.device
        if device == "cpu":
            pass
        elif device[:-1] == "cuda":
            pass
        else:
            raise ValueError(f"Invalid device: {device}")
        assert 0 < vals.port < 65536, ValueError("Invalid Port")
        assert vals.i18n_language in scan_language_list(), ValueError("I18n language not supported")

        return vals


class API_Batch_Cfg(BaseModel):
    init: bool = Field(default=False, exclude=True)
    device: str = "cpu"
    fp16: bool = False
    host: str = "::"
    port: Annotated[int, Field(ge=0, le=65536, strict=True)] = 9880
    i18n_language: str = "en_US"
    speaker_name: str = "API_Batch"
    streaming: bool = False
    media_type: Literal["pcm", "wav", "aac", "ogg"] = "wav"
    top_k: Annotated[int, Field(ge=1, le=100, strict=True)] = 5
    top_p: Annotated[float, Field(ge=0.01, le=1.0, strict=True)] = 1.0
    temperature: Annotated[float, Field(ge=0.01, le=1.0, strict=True)] = 1.0
    text_splitting_method: Annotated[str, Field(pattern=r"^cut[0-5]$")] = "cut1"
    batch_size: Annotated[int, Field(ge=1, strict=True)] = 5
    speed_factor: Annotated[float, Field(ge=0.6, le=1.4, strict=True)] = 1.0
    fragment_interval: Annotated[float, Field(ge=0.01, le=4.0)] = 0.3
    parallel_inference: bool = True
    reprtition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.35

    @model_validator(mode="after")
    @classmethod
    def check_all(cls, vals):
        if vals.init:
            vals.device, vals.fp16 = check_infer_device()
        vals.init = False
        device = vals.device
        if device == "cpu":
            pass
        elif device[:-1] == "cuda":
            pass
        else:
            raise ValueError(f"Invalid device: {device}")
        assert vals.i18n_language in scan_language_list(), ValueError("I18n language not supported")
        return vals


class Prompt(BaseModel):
    ref_audio_path: Optional[str] = Field(default=None, examples=[None], description="Reference Audio Path for the Speaker")
    prompt_text: Optional[str] = Field(default=None, examples=[None], description="Text for the Ref Audio, Optional")
    prompt_lang: Optional[str] = Field(default=None, examples=[None], description="Language for the Ref Audio, Required iff Prompt Text is not None")
    aux_ref_audio_paths: Annotated[Uniquelist[str], Field(min_items=0, max_items=10)] = Field(default_factory=list, examples=[[]])

    def is_empty(self) -> bool:
        return not self.ref_audio_path

    @model_validator(mode="after")
    @classmethod
    def check_all(cls, vals):
        if vals.ref_audio_path and not os.path.exists(vals.ref_audio_path):
            vals.ref_audio_path = None
            vals.prompt_text = None
            vals.prompt_lang = None
            vals.aux_ref_audio_paths = []
            warnings.warn("Ref audio not found", UserWarning)
        if vals.ref_audio_path:
            if vals.prompt_text and vals.prompt_lang:
                pass
            else:
                vals.prompt_text = None
                vals.prompt_lang = None
        return vals


class Speaker(BaseModel):
    t2s_path: str = PRETRAINED_T2S_V2
    vits_path: str = PRETRAINED_VITS_V2
    version: Literal["v1", "v2"] = "v2"
    prompt: Prompt = Field(default_factory=Prompt)

    cnhubert: str = Field(default=CNHUBERT_DEFAULT, exclude=True)
    bert: str = Field(default=BERT_DEFAULT, exclude=True)

    languages: list = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    @classmethod
    def check_lang(cls, vals):
        version = vals.version
        if version == "v1":
            vals.languages = V1_LANGUAGES
        elif version == "v2":
            vals.languages = V2_LANGUAGES

        prompt_lang = vals.prompt.prompt_lang
        if prompt_lang is not None:
            assert prompt_lang in vals.languages, ValueError("Invalid prompt language")

        if not os.path.exists(vals.t2s_path):
            warnings.warn("File not found, falling back to PRETRAINED_T2S.", UserWarning)
            if version == "v2":
                vals.t2s_path = PRETRAINED_T2S_V2
            elif version == "v1":
                vals.t2s_path = PRETRAINED_T2S_V1

        if not os.path.exists(vals.t2s_path):
            warnings.warn("File not found, falling back to PRETRAINED_VITS.", UserWarning)
            if version == "v2":
                vals.vits_path = PRETRAINED_VITS_V2
            elif version == "v1":
                vals.vits_path = PRETRAINED_VITS_V1

        return vals

    def update_language(self):
        version = self.version
        if version == "v1":
            self.languages = V1_LANGUAGES
        elif version == "v2":
            self.languages = V2_LANGUAGES


class Speakers_Cfg(BaseModel):
    speakers_dict: Dict[str, Speaker] = {
        "WebUI": Speaker(),
        "API": Speaker(),
        "API_Batch": Speaker(),
    }
    file_path: str = Field(default="./tools/cfgs/speakers.json", exclude=True)

    def get_speaker(self, speaker_name: str):
        return self.speakers_dict.get(speaker_name, Speaker())

    def list_speaker(self):
        return self.speakers_dict.keys()

    @model_validator(mode="after")
    @classmethod
    def check_all(cls, vals):
        default_dict = {
            "WebUI": Speaker(),
            "API": Speaker(),
            "API_Batch": Speaker(),
        }
        vals.speakers_dict = {**default_dict, **vals.speakers_dict}

        return vals

    @classmethod
    def from_json(cls, file_path: str) -> Self:
        if not os.path.exists(file_path):
            cls().save_as_json(file_path=file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = f.read()
        model = cls.model_validate_json(json_data)
        model.file_path = file_path
        return model

    def save_as_json(self, file_path: Optional[str] = None):
        if file_path is None:
            file_path = self.file_path
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))


Main_WebUI_Cfg_p = partial(Main_WebUI_Cfg, init=True)
Inference_WebUI_Cfg_p = partial(Inference_WebUI_Cfg, init=True)
API_Batch_Cfg_p = partial(API_Batch_Cfg, init=True)


class Cfg(BaseModel):
    train_webui_cfg: Main_WebUI_Cfg = Field(default_factory=Main_WebUI_Cfg_p)
    inference_webui_cfg: Inference_WebUI_Cfg = Field(default_factory=Inference_WebUI_Cfg_p)
    api_batch_cfg: API_Batch_Cfg = Field(default_factory=API_Batch_Cfg_p)
    file_path: str = Field(default="tools/cfgs/cfg.json", exclude=True)

    @classmethod
    def from_json(cls, file_path: str) -> Self:
        if not os.path.exists(file_path):
            cls().save_as_json(file_path=file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = f.read()
        model = cls.model_validate_json(json_data)
        model.file_path = file_path
        return model

    def save_as_json(self, file_path: Optional[str] = None):
        if file_path is None:
            file_path = self.file_path
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))


if __name__ == "__main__":
    pass
