import re
import os
import platform
import shutil
from typing import List, Optional
from argparse import ArgumentParser

import gradio as gr
from tools.server.schema import TTSResponseFailed

GPT_ROOT = ["GPT_weights_v2", "GPT_weights"]
SOVITS_ROOT = ["SoVITS_weights_v2", "SoVITS_weights"]

PRETRAINED_SOVITS = [
    i
    for i in [
        "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "GPT_SoVITS/pretrained_models/s2G488k.pth",
    ]
    if os.path.exists(i)
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
    return parser.parse_args()


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split(r"(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def get_sovits_paths(SoVITS_weight_root: Optional[List[str]] = None):
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
    return get_gpt_paths(), get_sovits_paths()


def build_gradio_exception(tts_response: TTSResponseFailed):
    e = tts_response.exception
    tracebacks = tts_response.tracebacks

    gr.Warning(f"{type(e).__name__}: {e}")
    if tracebacks:
        print(tracebacks)


def get_languages_list(tts_engine):
    return [tts_engine.i18n(language) for language in tts_engine.speaker.languages]


def truncate_path(path):
    match = re.search(r"(GPT_SoVITS|GPT_weights|GPT_weights_v2|SoVITS_weights|SoVITS_weights_v2)(?:/|\\).*", path)
    if match:
        return match.group(0)
    else:
        return None


def list_root_directories():
    if platform.system() == "Windows":
        # 在 Windows 上获取所有盘符
        drives = [f"{chr(letter)}:\\" for letter in range(65, 91) if os.path.exists(f"{chr(letter)}:\\")]
        return drives
    else:
        # Linux/macOS
        return ["/"]


def copy_file(src_path: str, target_folder: str, spk: str):

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

    ref_audio_dir = os.path.join(base_dir, "ref_audios", spk, target_folder)

    os.makedirs(ref_audio_dir, exist_ok=True)

    file_name = os.path.basename(src_path)

    dest_path = os.path.join(ref_audio_dir, file_name)

    shutil.copy2(src_path, dest_path)

    return dest_path
