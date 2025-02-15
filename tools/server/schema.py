from dataclasses import dataclass
from typing import Annotated, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


@dataclass
class TTSRequest:
    text: str
    text_lang: str

    text_split_method: Optional[str] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    speed_factor: Optional[float] = None
    fragment_interval: Optional[float] = None

    batch_size: Optional[int] = None
    batch_threshold: Optional[float] = None
    split_bucket: Optional[bool] = None
    parallel_infer: Optional[bool] = None
    return_fragment: Optional[bool] = None

    seed: int = -1
    no_prompt_text: bool = False


@dataclass
class TTSResponseSuccess:
    audio: Tuple[int, NDArray[np.int16]]


@dataclass
class TTSResponseSegment:
    audio: Tuple[int, NDArray[np.int16]]


@dataclass
class TTSResponseFailed:
    exception: Exception = Exception()
    tracebacks: str = ""


TTSResponse = Union[TTSResponseSuccess, TTSResponseSegment, TTSResponseFailed]


class TTSRequestAPI(BaseModel):
    text: str = Field(
        examples=["犯大吴疆土者,盛必击而破之。"],
        description="Text to Synthesize",
    )
    text_lang: str = Field(
        examples=["all_zh"],
        description="The Language of the Text",
    )
    speaker_name: str = Field(
        default="API_Batch",
        description="The Speaker for Synthesizing, See cfg.py --> SpeakerCfg and speakers.json for More Information",
    )

    ref_audio_path: Optional[str] = Field(
        None,
        description="Reference Audio Path for TTS, Optional If Provided in Speaker Setting(Prompt)",
        examples=[None],
    )
    prompt_text: Optional[str] = Field(
        None,
        description="Text of Reference Audio",
        examples=[None],
    )
    prompt_lang: Optional[str] = Field(
        None,
        description="Language of Reference Audio",
        examples=[None],
    )
    aux_ref_audio_paths: Optional[Annotated[List[str], Field(min_items=0, max_items=10)]] = Field(
        None,
        description="Auxiliary Reference Audio Paths for Voice Fusion",
        examples=[[]],
    )

    text_split_method: Optional[Literal["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]] = Field(
        default=None,
        examples=["cut1"],
        description="Check text_segmentation_method.py for Details",
    )
    top_k: Optional[Annotated[int, Field(ge=1, le=100)]] = Field(
        default=None,
        examples=[5],
        description="Limits Sampling to the Top-k Most Likely Tokens.",
    )
    top_p: Optional[Annotated[float, Field(ge=0.01, le=1.0)]] = Field(
        default=None,
        examples=[1.0],
        description="Samples from the Smallest Set of Tokens with a Cumulative Probability ≥ P",
    )
    temperature: Optional[Annotated[float, Field(ge=0.01, le=1.0)]] = Field(
        default=None,
        examples=[1.0],
        description="Randomness Control",
    )
    repetition_penalty: Optional[Annotated[float, Field(ge=0.9, le=2.0)]] = Field(
        default=None,
        examples=[1.35],
        description="Penalty for Generating Repetitive Text.",
    )
    fragment_interval: Optional[Annotated[float, Field(ge=0.01, le=4.0)]] = Field(
        default=None,
        examples=[0.3],
        description="Time Intervals between Each Batch",
    )
    speed_factor: Optional[Annotated[float, Field(ge=0.6, le=1.4)]] = Field(
        default=None,
        examples=[1.0],
        description="Speech Speed",
    )

    parallel_infer: Optional[bool] = Field(
        default=None,
        examples=[True],
        description="Batch Inference",
    )
    streaming_mode: Optional[bool] = Field(
        default=None,
        examples=[False],
        description="Return Audio in Streaming Response",
    )
    batch_size: Optional[int] = Field(
        default=None,
        examples=[5],
        description="Numbers of Text Items in Each Batch",
    )
    batch_threshold: Optional[Annotated[float, Field(ge=0.01, le=1.0)]] = Field(
        default=0.75,
        examples=[0.75],
    )
    split_bucket: bool = True

    media_type: Optional[Literal["pcm", "wav", "aac", "ogg"]] = Field(
        default=None,
        examples=["wav"],
        description="Media Format for Audio",
    )
    seed: Optional[Annotated[int, Field(ge=-1)]] = Field(
        -1,
        description="Random Seed",
        examples=[-1],
    )


class TTSRequestAPI_Compiled(BaseModel):
    text: str = Field(examples=["犯大吴疆土者,盛必击而破之。"], description="Text to Synthesize")
    text_lang: str = Field(examples=["all_zh"], description="The Language of the Text")
    speaker_name: str = Field(
        default="API_Batch",
        description="The Speaker for Synthesizing, See cfg.py --> SpeakerCfg and speakers.json for More Information",
    )

    ref_audio_path: Optional[str] = Field(
        None,
        description="Reference Audio Path for TTS, Optional If Provided in Speaker Setting(Prompt)",
        examples=[None],
    )
    prompt_text: Optional[str] = Field(
        None,
        description="Text of Reference Audio",
        examples=[None],
    )
    prompt_lang: Optional[str] = Field(
        None,
        description="Language of Reference Audio",
        examples=[None],
    )
    aux_ref_audio_paths: Optional[Annotated[List[str], Field(min_items=0, max_items=10)]] = Field(
        None,
        description="Auxiliary Reference Audio Paths for Voice Fusion",
        examples=[[]],
    )

    text_split_method: Optional[Literal["cut0", "cut1", "cut2", "cut3", "cut4", "cut5"]] = Field(
        default=None,
        examples=["cut1"],
        description="Check text_segmentation_method.py for Details",
    )
    top_k: Optional[Annotated[int, Field(ge=1, le=100)]] = Field(
        default=None,
        examples=[5],
        description="Limits Sampling to the Top-k Most Likely Tokens.",
    )
    top_p: Optional[Annotated[float, Field(ge=0.01, le=1.0)]] = Field(
        default=None,
        examples=[1.0],
        description="Samples from the Smallest Set of Tokens with a Cumulative Probability ≥ P",
    )
    temperature: Optional[Annotated[float, Field(ge=0.01, le=1.0)]] = Field(
        default=None,
        examples=[1.0],
        description="Randomness Control",
    )
    repetition_penalty: Optional[Annotated[float, Field(ge=0.9, le=2.0)]] = Field(
        default=None,
        examples=[1.35],
        description="Penalty for Generating Repetitive Text.",
    )
    fragment_interval: Optional[Annotated[float, Field(ge=0.01, le=4.0)]] = Field(
        default=None,
        examples=[0.3],
        description="Time Intervals between Each Batch",
    )

    streaming_mode: Optional[bool] = Field(default=None, examples=[False], description="Return Audio in Streaming Response")
    batch_threshold: Optional[Annotated[float, Field(ge=0.01, le=1.0)]] = Field(
        default=0.75,
        examples=[0.75],
    )
    split_bucket: bool = True

    media_type: Optional[Literal["pcm", "wav", "aac", "ogg"]] = Field(
        default=None,
        examples=["wav"],
        description="Media Format for Audio",
    )
    seed: Optional[Annotated[int, Field(ge=-1)]] = Field(
        -1,
        description="Random Seed",
        examples=[-1],
    )


class SpeakerAPI(BaseModel):
    speaker_name: str = Field(
        description="Name of Speaker",
        examples=[""],
    )
    t2s_path: Optional[str] = Field(
        None,
        description="Path of the GPT Model, GPT V2 model if unset",
        examples=[None],
    )
    vits_path: Optional[str] = Field(
        None,
        description="Path of the SoVITS Model, SoVITS V2 model if unset",
        examples=[None],
    )
    version: Literal["v1", "v2"] = Field(
        "v2",
        description="Version of the Model",
        examples=["v2"],
    )
    ref_audio_path: Optional[str] = Field(
        default=None,
        examples=[None],
        description="Reference Audio Path for the Speaker",
    )
    prompt_text: Optional[str] = Field(
        default=None,
        examples=[None],
        description="Text for the Ref Audio, Optional",
    )
    prompt_lang: Optional[str] = Field(
        default=None,
        examples=[None],
        description="Language for the Ref Audio, Required iff Prompt Text is not None",
    )
    aux_ref_audio_paths: Annotated[List[str], Field(min_items=0, max_items=10)] = Field(
        default_factory=list,
        examples=[[]],
    )
