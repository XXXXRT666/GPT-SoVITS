import os
import sys
import traceback
from pathlib import Path
from typing import List

import ffmpeg
import gradio as gr
import numpy as np

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language=os.environ.get("language", "Auto"))


def load_audio(file: str, sr: int):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError("You input a wrong audio path that does not exists, please fix it!")
        out, _ = (
            ffmpeg.input(str(file), threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(i18n("音频加载失败"))

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str: str) -> str:
    if isinstance(path_str, str):
        cleaned_path = str(path_str).strip(" ").strip("'").strip("\n").strip('"').strip("\u202a")
        path_obj = Path(cleaned_path)
    else:
        raise TypeError("Invalid Path")
    path_obj = path_obj.expanduser().resolve(strict=False)
    return str(path_obj)


def check_for_existance(file_list: List[str], is_train=False, is_dataset_processing=False):
    files_status = []
    if is_train == True and file_list:
        file_list.append(os.path.join(file_list[0], "2-name2text.txt"))
        file_list.append(os.path.join(file_list[0], "3-bert"))
        file_list.append(os.path.join(file_list[0], "4-cnhubert"))
        file_list.append(os.path.join(file_list[0], "5-wav32k"))
        file_list.append(os.path.join(file_list[0], "6-name2semantic.tsv"))
    for file in file_list:
        if os.path.exists(file):
            files_status.append(True)
        else:
            files_status.append(False)
    if sum(files_status) != len(files_status):
        if is_train:
            for file, status in zip(file_list, files_status):
                if status:
                    pass
                else:
                    gr.Warning(file)
            gr.Warning(i18n("以下文件或文件夹不存在"))
            return False
        elif is_dataset_processing:
            if files_status[0]:
                return True
            elif not files_status[0]:
                gr.Warning(file_list[0])
            elif not files_status[1] and file_list[1]:
                gr.Warning(file_list[1])
            gr.Warning(i18n("以下文件或文件夹不存在"))
            return False
        else:
            if file_list[0]:
                gr.Warning(file_list[0])
                gr.Warning(i18n("以下文件或文件夹不存在"))
            else:
                gr.Warning(i18n("路径不能为空"))
            return False
    return True


def check_details(path_list=None, is_train=False, is_dataset_processing=False):
    if is_dataset_processing:
        list_path, audio_path = path_list
        if not list_path.endswith(".list"):
            gr.Warning(i18n("请填入正确的List路径"))
            return
        if audio_path:
            if not os.path.isdir(audio_path):
                gr.Warning(i18n("请填入正确的音频文件夹路径"))
                return
        with open(list_path, "r", encoding="utf8") as f:
            line = f.readline().strip("\n").split("\n")
        wav_name, _, __, ___ = line[0].split("|")
        wav_name = clean_path(wav_name)
        if audio_path != "" and audio_path != None:
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s" % (audio_path, wav_name)
        else:
            wav_path = wav_name
        if os.path.exists(wav_path):
            ...
        else:
            gr.Warning(i18n("路径错误"))
        return
    if is_train:
        path_list.append(os.path.join(path_list[0], "2-name2text.txt"))
        path_list.append(os.path.join(path_list[0], "4-cnhubert"))
        path_list.append(os.path.join(path_list[0], "5-wav32k"))
        path_list.append(os.path.join(path_list[0], "6-name2semantic.tsv"))
        phone_path, hubert_path, wav_path, semantic_path = path_list[1:]
        import pandas as pd

        with open(phone_path, "r", encoding="utf-8") as f:
            if f.read(1):
                ...
            else:
                gr.Warning(i18n("缺少音素数据集"))
        if os.listdir(hubert_path):
            ...
        else:
            gr.Warning(i18n("缺少Hubert数据集"))
        if os.listdir(wav_path):
            ...
        else:
            gr.Warning(i18n("缺少音频数据集"))
        df = pd.read_csv(semantic_path, delimiter="\t", encoding="utf-8")
        if len(df) >= 1:
            ...
        else:
            gr.Warning(i18n("缺少语义数据集"))


def check_infer_device():
    import torch

    is_half = True
    if torch.cuda.is_available():
        infer_device = "cuda"
    else:
        infer_device = "cpu"

    if infer_device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
        ):
            is_half = False

    if infer_device == "cpu":
        is_half = False
    return infer_device, is_half


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def update(self, other_dict=None, **kwargs):
        if other_dict:
            if isinstance(other_dict, DictToAttrRecursive):
                other_dict = dict(other_dict)
            for key, value in other_dict.items():
                if isinstance(value, dict):
                    if key in self and isinstance(self[key], DictToAttrRecursive):
                        self[key].update(value)
                    else:
                        self[key] = DictToAttrRecursive(value)
                else:
                    self[key] = value
                setattr(self, key, self[key])
        for key, value in kwargs.items():
            if isinstance(value, dict):
                if key in self and isinstance(self[key], DictToAttrRecursive):
                    self[key].update(value)
                else:
                    self[key] = DictToAttrRecursive(value)
            else:
                self[key] = value
            setattr(self, key, self[key])


class SilentPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout
