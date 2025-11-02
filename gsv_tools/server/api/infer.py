import os
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import regex as re
import torch
import torchaudio
from numpy import int16, int32
from numpy.typing import NDArray
from peft import LoraConfig, get_peft_model
from transformers import BertForMaskedLM, BertTokenizerFast

from config import get_dtype, infer_device, pretrained_sovits_name
from GPT_SoVITS.Accel import MLX, PyTorch, T2SEngineProtocol, T2SRequest, backends
from GPT_SoVITS.BigVGAN import bigvgan
from GPT_SoVITS.feature_extractor import CNHubert
from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch
from GPT_SoVITS.module.models import Generator as HiFiGAN, SynthesizerTrn, SynthesizerTrnV3
from GPT_SoVITS.process_ckpt import inspect_version
from GPT_SoVITS.sv import SV
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from gsv_tools.audio_sr import AP_BWE
from gsv_tools.logger import console, logger
from gsv_tools.my_utils import DictToAttrRecursive


warnings.filterwarnings(
    "ignore", message="MPS: The constant padding of more than 3 dimensions is not currently supported natively."
)
warnings.filterwarnings("ignore", message=".*ComplexHalf support is experimental.*")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

torch.set_grad_enabled(False)

dtype = get_dtype(infer_device.index)

Tensor = torch.Tensor

language = Literal["all_zh", "all_yue", "en", "all_ja", "all_ko", "zh", "yue", "ja", "ko", "auto", "auto_yue"]


class GPT:
    def __init__(self, t2s_engine: T2SEngineProtocol):
        self.t2s_engine = t2s_engine


class SoVITS:
    def __init__(self, vq_model: SynthesizerTrn | SynthesizerTrnV3, hps, model_version: str, lang_version: str):
        self.vq_model = vq_model
        self.hps = hps
        self.model_version = model_version
        self.version = lang_version


class Speaker:
    def __init__(self, gpt: GPT, sovits: SoVITS):
        self.sovits = sovits
        self.gpt = gpt


def is_empty(*items):  # 任意一项不为空返回False
    for item in items:
        if item is not None and item != "":
            return False
    return True


def is_full(*items):  # 任意一项为空返回False
    for item in items:
        if item is None or item == "":
            return False
    return True


class DefaultRefer:
    def __init__(
        self,
        path: str,
        text: str,
        language: language,
    ):
        self.path = path
        self.text = text
        self.language: Literal[
            "all_zh", "all_yue", "en", "all_ja", "all_ko", "zh", "yue", "ja", "ko", "auto", "auto_yue"
        ] = language

    def is_ready(self) -> bool:
        return is_full(self.path)


@dataclass
class APINameSpace:
    backends: str = backends[-1]
    quantization: str | None = None

    speaker = Speaker(gpt=None, sovits=None)  # type: ignore
    default_refer = DefaultRefer("", "", "auto")

    default_cut_punc: str = "。."

    bert_model: BertForMaskedLM = None  # type: ignore
    bert_tokenizer: BertTokenizerFast = None  # type: ignore
    ssl_model: CNHubert = None  # type: ignore

    sv_cn_model: SV | None = None
    hifigan_model: HiFiGAN | None = None
    bigvgan_model: bigvgan.BigVGAN | None = None

    device: torch.device = infer_device
    dtype: torch.dtype = dtype
    is_half: bool = dtype == torch.float16

    media_type: Literal["wav", "ogg", "aac"] = "wav"
    stream: bool = False


api_ns = APINameSpace()


# --------------------------------
# Initialization
# --------------------------------

dict_language = {
    "all_zh": "all_zh",  # All Chinese
    "all_yue": "all_yue",  # All Cantonese
    "en": "en",  # English
    "all_ja": "all_ja",  # All Japanese
    "all_ko": "all_ko",  # All Korean
    "zh": "zh",  # Chinese
    "yue": "yue",  # Cantonese-English Mixed
    "ja": "ja",  # Japanese-English Mixed
    "ko": "ko",  # Korean-English Mixed
    "auto": "auto",  # Auto Detecting, Default Chinese
    "auto_yue": "auto_yue",  # Auto Detecting, Default Cantonese not Chinese
}


def clean_hifigan_model():
    if api_ns.hifigan_model:
        api_ns.hifigan_model = api_ns.hifigan_model.cpu()
        api_ns.hifigan_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def clean_bigvgan_model():
    if api_ns.bigvgan_model:
        api_ns.bigvgan_model = api_ns.bigvgan_model.cpu()
        api_ns.bigvgan_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def clean_sv_cn_model():
    if api_ns.sv_cn_model:
        api_ns.sv_cn_model.embedding_model = api_ns.sv_cn_model.embedding_model.cpu()
        api_ns.sv_cn_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def init_bigvgan():
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        "GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x",
        use_cuda_kernel=False,
    )  # if True, RuntimeError: Ninja is required to load C++ extensions
    # remove weight norm in the model and set to eval mode
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()

    if api_ns.is_half is True:
        bigvgan_model = bigvgan_model.half().to(api_ns.device)
    else:
        bigvgan_model = bigvgan_model.to(api_ns.device)

    api_ns.bigvgan_model = bigvgan_model


def init_hifigan():
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
    state_dict_g = torch.load(
        "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth",
        map_location="cpu",
        weights_only=False,
    )
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    if api_ns.is_half is True:
        hifigan_model = hifigan_model.half().to(api_ns.device)
    else:
        hifigan_model = hifigan_model.to(api_ns.device)
    api_ns.hifigan_model = hifigan_model


def init_sv_cn():
    api_ns.sv_cn_model = SV(api_ns.device, api_ns.is_half)


@lru_cache
def get_resample_transform(sr0: int, sr1: int):
    return torchaudio.transforms.Resample(sr0, sr1)


def resample(audio_tensor: Tensor, sr0: int, sr1: int, device: torch.device):
    resample_transform = get_resample_transform(sr0, sr1).to(device)
    return resample_transform(audio_tensor)


spec_min = -12
spec_max = 2


def norm_spec(x: Tensor):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x: Tensor):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min


def mel_fn(x: Tensor):
    return mel_spectrogram_torch(
        y=x,
        n_fft=1024,
        num_mels=100,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=None,
        center=False,
    )


def mel_fn_v4(x: Tensor):
    return mel_spectrogram_torch(
        y=x,
        n_fft=1280,
        num_mels=100,
        sampling_rate=32000,
        hop_size=320,
        win_size=1280,
        fmin=0,
        fmax=None,
        center=False,
    )


sr_model = None


def audio_sr(audio: Tensor, sr: int):
    global sr_model
    if sr_model is None:
        try:
            sr_model = AP_BWE(infer_device, DictToAttrRecursive)
        except FileNotFoundError:
            logger.info("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载")
            return audio.cpu().detach().numpy(), sr
    return sr_model(audio, sr)


def get_sovits_weights(sovits_path: str):
    path_sovits_v3 = pretrained_sovits_name["v3"]
    path_sovits_v4 = pretrained_sovits_name["v4"]
    is_exist_s2gv3 = os.path.exists(path_sovits_v3)
    is_exist_s2gv4 = os.path.exists(path_sovits_v4)

    model_version, lang_version, is_lora, hps_cfg, dict_s2 = inspect_version(sovits_path)
    print(sovits_path, lang_version, model_version, is_lora)
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4

    if is_lora and not is_exist:
        raise FileNotFoundError(f"SoVITS {model_version} Missing, Can not load LoRA weights ({path_sovits})")

    hps = DictToAttrRecursive(hps_cfg)
    hps.model.semantic_frame_rate = "25hz"
    hps.model.version = model_version

    model_params_dict = vars(hps.model)
    if model_version not in {"v3", "v4"}:
        if "Pro" in model_version and api_ns.sv_cn_model is None:
            init_sv_cn()
        vq_model: SynthesizerTrn | SynthesizerTrnV3 = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict,
        )
    else:
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **model_params_dict,
        ).eval()
        if model_version == "v3":
            init_bigvgan()
        if model_version == "v4":
            init_hifigan()

    if hasattr(vq_model, "enc_q"):
        del vq_model.enc_q

    dict_s2["weight"] = {k: v for k, v in dict_s2["weight"].items() if not k.startswith("enc_q")}

    if not is_lora:
        console.print(f">> loading sovits_{model_version}")
        vq_model.load_state_dict(dict_s2["weight"])
    else:
        console.print(f">> loading sovits_{model_version} pretrained_G")
        dict_pretrain = torch.load(path_sovits, map_location="cpu")["weight"]
        console.print(f">> loading sovits_{model_version}_lora{model_version}")
        dict_pretrain.update(dict_s2["weight"])
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)  # type: ignore
        vq_model.load_state_dict(dict_pretrain, strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()  # type: ignore
        vq_model.eval()

    if api_ns.is_half:
        vq_model = vq_model.half().to(api_ns.device)
    else:
        vq_model = vq_model.to(api_ns.device)
    vq_model.eval()

    return SoVITS(vq_model, hps, model_version, lang_version)


def get_gpt_weights(gpt_path: str, backend: str, quantization):
    if "mlx" in backend.lower():
        t2s_engine: T2SEngineProtocol = MLX.T2SEngineMLX(
            MLX.T2SEngineMLX.load_decoder(Path(gpt_path), backend=backend, quantize_mode=quantization),
            api_ns.device,
            api_ns.dtype,
            cache_size=10,
        )
        total_params = sum(p[-1].size for p in MLX.mxutils.tree_flatten(t2s_engine.decoder_model.parameters()))  # type: ignore
    else:
        t2s_engine = PyTorch.T2SEngineTorch(
            PyTorch.T2SEngineTorch.load_decoder(Path(gpt_path), backend=backend, quantize_mode=quantization),
            api_ns.device,
            api_ns.dtype,
            cache_size=10,
        )
        total_params = sum(p.numel() for p in t2s_engine.decoder_model.parameters())
    console.print(">> Number of parameter: %.2fM" % (total_params / 1e6))
    return GPT(t2s_engine)


def change_gpt_sovits_weights(gpt_path: str | None, sovits_path: str | None):
    if gpt_path:
        gpt = get_gpt_weights(gpt_path, api_ns.backends, api_ns.quantization)
        api_ns.speaker.gpt = gpt
    if sovits_path:
        sovits = get_sovits_weights(sovits_path)
        api_ns.speaker.sovits = sovits
    return {"message": "Success"}


def get_bert_feature(text: str, word2ph: list[int]):
    inputs = api_ns.bert_tokenizer(text, return_tensors="pt")
    for i in inputs:
        inputs[i] = inputs[i].to(api_ns.device)  # type: ignore
    res = api_ns.bert_model(**inputs, output_hidden_states=True)
    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature: list[Tensor] = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature_t = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature_t.T


def clean_text_inf(text: str, language: str, version: str):
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


def get_bert_inf(phones: list[int], word2ph: list[int], norm_text: str, language: str):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(api_ns.device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if api_ns.is_half is True else torch.float32,
        ).to(api_ns.device)

    return bert


def get_phones_and_bert(text: str, language: str, version: str, final: bool = False):
    text = re.sub(r" {2,}", " ", text)
    textlist = []
    langlist = []
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text, "ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text, "ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        for tmp in LangSegmenter.getTexts(text):
            if langlist:
                if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                    textlist[-1] += tmp["text"]
                    continue
            if tmp["lang"] == "en":
                langlist.append(tmp["lang"])
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                langlist.append(language)
            textlist.append(tmp["text"])
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(torch.float16 if api_ns.is_half is True else torch.float32), norm_text


def get_spepc(hps, filename: str, dtype: torch.dtype, device: torch.device, is_v2pro=False):
    sr1 = int(hps.data.sampling_rate)
    audio, sr0 = torchaudio.load_with_torchcodec(filename)
    audio = audio.to(device)

    if sr0 != sr1:
        audio = resample(audio, sr0, sr1, device)
    if audio.shape[0] > 1:
        audio = audio.mean(0).unsqueeze(0)

    maxx = float(audio.abs().max())
    if maxx > 1:
        audio /= min(2, maxx)
    spec = spectrogram_torch(
        audio,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = spec.to(dtype)
    if is_v2pro is True:
        audio = resample(audio, sr1, 16000, device).to(dtype)
    return spec, audio


def cut_text(text, punc):
    if not punc:
        return text
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2], strict=False)]
        # 在句子不存在符号或句尾无符号的时候保证文本完整
        if len(items) % 2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    return text


def only_punc(text):
    return not any(t.isalnum() or t.isalpha() for t in text)


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}

punctuation = {"!", "?", "…", ",", ".", "-", " "}


def process_text(texts: list[str]):
    cleaned = []
    for text in texts:
        if text and text.strip():
            cleaned.append(text)
    if not cleaned:
        raise ValueError("Please provide valid text for synthesis.")
    return cleaned


def merge_short_text_in_array(texts: list[str], threshold: int):
    if len(texts) < 2:
        return texts
    result = []
    buffer = ""
    for ele in texts:
        buffer += ele
        if len(buffer) >= threshold:
            result.append(buffer)
            buffer = ""
    if buffer:
        if not result:
            result.append(buffer)
        else:
            result[-1] += buffer
    return result


def get_tts_wav(
    ref_wav_path: str,
    prompt_text: str,
    prompt_language: str,
    text: str,
    text_language: str,
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    speed: float = 1.0,
    inp_refs: list[str] = None,
    sample_steps: int | None = None,
    if_sr: bool | None = None,
) -> Generator[NDArray[int32] | NDArray[int16], None, None]:
    if inp_refs is None:
        inp_refs = []
    torch.set_grad_enabled(False)
    infer_sovits = api_ns.speaker.sovits
    vq_model = infer_sovits.vq_model
    hps = infer_sovits.hps
    model_version = infer_sovits.model_version
    version = infer_sovits.version

    engine = api_ns.speaker.gpt.t2s_engine

    infer_device = api_ns.device
    dtype = api_ns.dtype

    if model_version == "v3":
        if sample_steps not in [4, 8, 16, 32, 64, 128]:
            sample_steps = 32
    elif model_version == "v4":
        if sample_steps not in [4, 8, 16, 32]:
            sample_steps = 8
    else:
        sample_steps = 32

    if model_version != "v3":
        if_sr = False

    prompt_language = dict_language.get((prompt_language or "zh").lower(), (prompt_language or "zh"))
    text_language = dict_language.get((text_language or "zh").lower(), (text_language or "zh"))
    text = text.strip("\n")
    prompt_text = prompt_text.strip("\n")

    prompt_free = False
    if not text:
        raise ValueError("Missing synthesis text")
    if not prompt_text:
        if model_version in {"v3", "v4"}:
            raise ValueError("Prompt text is required for SoVITS v3/v4 models")
        prompt_free = True

    if not prompt_free and prompt_text[-1] not in splits:
        prompt_text += "。" if prompt_language != "en" else "."

    zero_wav_torch = torch.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=torch.float16 if api_ns.is_half is True else torch.float32,
    ).to(infer_device)

    if not prompt_free:
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        wav16k_t = torch.from_numpy(wav16k).to(infer_device, dtype=dtype)
        wav16k_t = torch.cat([wav16k_t, zero_wav_torch])
        ssl_content = api_ns.ssl_model.model(wav16k_t.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt = codes[0, 0].unsqueeze(0).to(infer_device)

        phones1, bert1, _ = get_phones_and_bert(prompt_text, prompt_language, version)
    else:
        prompt = torch.zeros((1, 0)).to(infer_device, torch.int32)
        phones1, bert1 = [], torch.zeros(1024, 0).to(infer_device, dtype)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    texts = merge_short_text_in_array(process_text(text.split("\n")), 5)

    if model_version in {"v1", "v2", "v2Pro", "v2ProPlus"}:
        sr = 32000
    elif model_version == "v3":
        sr = 24000
    else:
        sr = 48000

    yield np.array(sr).astype(np.int32)

    for sentence in texts:
        if only_punc(sentence):
            continue
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentence[-1] not in splits:
            sentence += "。" if text_language != "en" else "."
        phones2, bert2, _ = get_phones_and_bert(sentence, text_language, version)
        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(infer_device).unsqueeze(0)
        bert = bert.to(infer_device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(infer_device)

        t2s_request = T2SRequest(
            [all_phoneme_ids.squeeze(0)],
            all_phoneme_len,
            prompt,
            [bert.squeeze(0)],
            valid_length=1,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=1024,
            use_cuda_graph=torch.cuda.is_available()
            and torch.version.cuda is not None
            and os.environ.get("CUDAGraph", "1") != "0",
            debug=False,
        )
        t2s_result = engine.generate(t2s_request)
        if t2s_result.exception is not None:
            console.print(t2s_result.traceback)
            raise RuntimeError()
        pred_semantic_list = t2s_result.result
        assert pred_semantic_list
        pred_semantic = pred_semantic_list[0].unsqueeze(0).unsqueeze(0).to(infer_device)

        is_v2pro = model_version in {"v2Pro", "v2ProPlus"}
        if model_version not in {"v3", "v4"}:
            refers = []
            sv_emb = []
            if is_v2pro and api_ns.sv_cn_model is None:
                init_sv_cn()
            if inp_refs:
                for path in inp_refs:
                    refer, audio_tensor = get_spepc(hps, path, dtype, infer_device, is_v2pro)
                    refers.append(refer)
                    if is_v2pro and api_ns.sv_cn_model:
                        sv_emb.append(api_ns.sv_cn_model.compute_embedding(audio_tensor))
            if len(refers) == 0:
                refer_item, audio_tensor = get_spepc(hps, ref_wav_path, dtype, infer_device, is_v2pro)
                refers = [refer_item]

                if is_v2pro and api_ns.sv_cn_model:
                    sv_emb = [api_ns.sv_cn_model.compute_embedding(audio_tensor)]
            phone_tensor = torch.LongTensor(phones2).to(infer_device).unsqueeze(0)
            if is_v2pro and sv_emb:
                audio = vq_model.decode(
                    pred_semantic,
                    phone_tensor,
                    refers,
                    speed=speed,
                    sv_emb=sv_emb,
                )[0][0]  # type: ignore
            else:
                audio = vq_model.decode(
                    pred_semantic,
                    phone_tensor,
                    refers,
                    speed=speed,
                )[0][0]  # type: ignore
        else:
            refer, _ = get_spepc(hps, ref_wav_path, dtype, infer_device)
            phoneme_ids0 = torch.LongTensor(phones1).to(infer_device).unsqueeze(0)
            phoneme_ids1 = torch.LongTensor(phones2).to(infer_device).unsqueeze(0)
            fea_ref, ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)  # type: ignore
            tgt_sr = 24000 if model_version == "v3" else 32000
            ref_audio, sr = torchaudio.load_with_torchcodec(ref_wav_path)
            ref_audio = ref_audio.to(infer_device).float()
            if sr != tgt_sr:
                ref_audio = resample(ref_audio, sr, tgt_sr, infer_device)
            if ref_audio.shape[0] > 1:
                ref_audio = ref_audio.mean(0).unsqueeze(0)
            mel2 = mel_fn(ref_audio) if model_version == "v3" else mel_fn_v4(ref_audio)
            mel2 = norm_spec(mel2)
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            mel2 = mel2[:, :, :T_min]
            fea_ref = fea_ref[:, :, :T_min]
            Tref = 468 if model_version == "v3" else 500
            Tchunk = 934 if model_version == "v3" else 1000
            if T_min > Tref:
                mel2 = mel2[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            chunk_len = Tchunk - T_min
            mel2 = mel2.to(dtype)
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)  # type: ignore
            cfm_resss = []
            idx = 0
            while True:
                fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
                if fea_todo_chunk.shape[-1] == 0:
                    break
                idx += chunk_len
                fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                cfm_res = vq_model.cfm.inference(  # type: ignore
                    fea,
                    torch.LongTensor([fea.size(1)]).to(fea.device),
                    mel2,
                    sample_steps,
                    inference_cfg_rate=0,
                )
                cfm_res = cfm_res[:, :, mel2.shape[2] :]
                mel2 = cfm_res[:, :, -T_min:]
                fea_ref = fea_todo_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
            cfm_res = torch.cat(cfm_resss, 2)
            cfm_res = denorm_spec(cfm_res)
            if model_version == "v3":
                if api_ns.bigvgan_model is None:
                    init_bigvgan()
            else:  # v4
                if api_ns.hifigan_model is None:
                    init_hifigan()
            vocoder_model = api_ns.bigvgan_model if model_version == "v3" else api_ns.hifigan_model
            with torch.inference_mode():
                wav_gen = vocoder_model(cfm_res)  # type: ignore
                audio = wav_gen[0][0]

        max_audio = torch.abs(audio).max()
        if max_audio > 1:
            audio = audio / max_audio

        chunk = torch.cat([audio, zero_wav_torch])
        chunk_np = chunk.detach().cpu().numpy()

        if if_sr and sr == 24000:
            chunk_tensor = torch.from_numpy(chunk_np).float().to(infer_device)
            chunk_np, sr = audio_sr(chunk_tensor.unsqueeze(0), sr)
            chunk_np = np.asarray(chunk_np)
            max_audio_f: float = np.abs(chunk_np).max()
            if max_audio > 1:
                chunk_np /= max_audio_f
            sr = 48000
        chunk_np = (chunk_np * 32767).astype(np.int16)

        yield chunk_np


def change_refer(
    path: str,
    text: str,
    language: language,
):
    if is_empty(path):
        raise ValueError('Something Missing: "path"')

    if path != "":
        api_ns.default_refer.path = path
        api_ns.default_refer.text = text
        api_ns.default_refer.language = language

    logger.info(f"Default Refer Path: {api_ns.default_refer.path}")
    if api_ns.default_refer.text:
        logger.info(f"Default Refer Text: {api_ns.default_refer.text}")
        logger.info(f"Default Refer Language: {api_ns.default_refer.language}")

    return {"message": "Success"}
