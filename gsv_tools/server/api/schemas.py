from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


language = Literal["all_zh", "all_yue", "en", "all_ja", "all_ko", "zh", "yue", "ja", "ko", "auto", "auto_yue"]


class TTSRequestAPI(BaseModel):
    text: str = Field(
        examples=["A TEST SENTENCE."],
        description="Text to Synthesize",
    )
    text_language: language = Field(
        default="auto",
        examples=["en"],
        description="The Language of the Text",
    )
    cut_punc: str | None = Field(
        default=None,
        examples=["。."],
        description="Split the sentence using the provided punctuation",
    )
    refer_wav_path: str | None = Field(
        None,
        description="Reference Audio Path for TTS, Optional If Provided in API Settings",
        examples=[None],
    )
    prompt_text: str | None = Field(
        None,
        description="Text of Reference Audio",
        examples=[None],
    )
    prompt_language: language | None = Field(
        default="auto",
        description="Language of Reference Audio",
        examples=[None],
    )
    inp_refs: Annotated[list[str], Field(min_length=0, max_length=10)] | None = Field(
        default=[],
        description="Auxiliary Reference Audio Paths for Synthesis",
        examples=[[]],
    )

    top_k: Annotated[int, Field(ge=1, le=100)] | None = Field(
        default=15,
        examples=[15],
        description="Limits Sampling to the Top-k Most Likely Tokens.",
    )
    top_p: Annotated[float, Field(ge=0.01, le=1.0)] | None = Field(
        default=1.0,
        examples=[1.0],
        description="Samples from the Smallest Set of Tokens with a Cumulative Probability ≥ P",
    )
    temperature: Annotated[float, Field(ge=0, le=1.0)] | None = Field(
        default=1.0,
        examples=[1.0],
        description="Randomness Control, 0.0 For Greedy Decoding",
    )
    speed: Annotated[float, Field(ge=0.6, le=1.4)] | None = Field(
        default=1.0,
        examples=[1.0],
        description="Speech Speed",
    )
    sample_steps: Literal[4, 8, 16, 32, 64, 128] | None = Field(
        default=None,
        examples=[8],
        description="Sample Steps for DiT in V3/V4",
    )
    if_sr: bool | None = Field(
        default=None,
        examples=[False],
        description="Whether to Perform Super-Resolution in V3",
    )

    @field_validator("refer_wav_path")
    def validate_file(cls, v):
        if v is None:
            return v
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File not found: {v}")
        if not path.is_file():
            raise ValueError(f"Not a valid file: {v}")
        return v

    @field_validator("inp_refs")
    def validate_list(cls, v):
        if not v:
            return v
        for item in v:
            path = Path(item)
            if not path.exists():
                raise ValueError(f"File not found: {item}")
            if not path.is_file():
                raise ValueError(f"Not a valid file: {item}")
        return v


class SetModelsAPI(BaseModel):
    gpt_model_path: str | None = Field(
        None,
        description="Path to the GPT Model",
        examples=["GPT_SoVITS/pretrained_models/s1v3.ckpt"],
    )
    sovits_model_path: str | None = Field(
        None,
        description="Path to the SoVITS Model",
        examples=["GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"],
    )

    @field_validator("gpt_model_path", "sovits_model_path")
    def validate_file(cls, v):
        if v is None:
            return v
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File not found: {v}")
        if not path.is_file():
            raise ValueError(f"Not a valid file: {v}")
        return v


class ChangeRefAudioAPI(BaseModel):
    refer_wav_path: str = Field(
        "",
        description="Path to the Reference Audio",
        examples=["./examples.wav"],
    )
    prompt_text: str = Field(
        "",
        description="Text of Reference Audio",
        examples=["Example text."],
    )

    prompt_language: language = Field(
        "auto",
        description="Language of Reference Audio",
        examples=["all_zh", "en"],
    )

    @field_validator("refer_wav_path")
    def validate_file(cls, v):
        if v is None:
            raise ValueError("refer_wav_path cannot be None")
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File not found: {v}")
        if not path.is_file():
            raise ValueError(f"Not a valid file: {v}")
        return v
