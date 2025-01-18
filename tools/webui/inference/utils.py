import re
import os
from typing import List, Optional
from argparse import ArgumentParser

import gradio as gr
from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine
from tools.server.schema import TTSResponseFailed

GPT_ROOT = ["GPT_weights_v2", "GPT_weights"]
SOVITS_ROOT = ["SoVITS_weights_v2", "SoVITS_weights"]

PRETRAINED_SOVITS = [
    i for i in ["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", "GPT_SoVITS/pretrained_models/s2G488k.pth"] if os.path.exists(i)
]
PRETRAINED_GPT = [
    i
    for i in [
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    ]
    if os.path.exists(i)
]


def parse_args():
    parser = ArgumentParser(description="GPT-SoVITS WebUI")
    parser.add_argument("-c", "--webui-config", type=str, default="tools/cfgs/cfg.json", help="API_Batch Cfg Path")
    parser.add_argument("-s", "--speakers-config", type=str, default="tools/cfgs/speakers.json", help="Speakers Cfg Path")
    parser.add_argument("--compile", action="store_true", help="Compiled the model to accelerate")
    return parser.parse_args()


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split(r"(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def get_sovits_path(SoVITS_weight_root: Optional[List[str]] = None):
    if SoVITS_weight_root is None:
        SoVITS_weight_root = SOVITS_ROOT
    SoVITS_Paths = PRETRAINED_SOVITS
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"):
                SoVITS_Paths.append(f"{path}/{name}")
    return SoVITS_Paths


def get_gpt_paths(GPT_weight_root: Optional[List[str]] = None):
    if GPT_weight_root is None:
        GPT_weight_root = GPT_ROOT
    GPT_Paths = PRETRAINED_GPT
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"):
                GPT_Paths.append(f"{path}/{name}")
    return GPT_Paths


def get_both_paths():
    return get_gpt_paths(GPT_ROOT), get_sovits_path(SOVITS_ROOT)


def build_gradio_exception(tts_response: TTSResponseFailed):
    e = tts_response.exception
    tracebacks = tts_response.tracebacks

    gr.Warning(f"{type(e).__name__}: {e}")
    if tracebacks:
        print(tracebacks)


def get_languages_list(tts_engine: TTSEngine):
    return [tts_engine.i18n(language) for language in tts_engine.speaker.languages]
