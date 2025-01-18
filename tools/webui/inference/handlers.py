import random
from functools import partial
from typing import List, Optional, Dict

import gradio as gr

from tools.webui.inference.utils import get_languages_list
from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine


def inference(
    text: str,
    text_lang: str,
    speaker_name: str,
    ref_audio_path: Optional[str],
    aux_ref_audio_paths: Optional[List[str]],
    prompt_text: Optional[str],
    prompt_lang: Optional[str],
    top_k: int,
    top_p: float,
    temperature: float,
    text_split_method: str,
    batch_size: int,
    speed_factor: float,
    ref_text_free: bool,
    split_bucket: bool,
    fragment_interval: float,
    seed: int,
    keep_random: bool,
    parallel_infer: bool,
    repetition_penalty: float,
    tts_engine: TTSEngine,
    cut_methods_mapping: Dict[str, str],
    language_mapping: Dict[str, str],
    progress_tracker=gr.Progress(),
):
    progress_tracker(0, desc="Starting")
    if keep_random or seed == -1:
        actual_seed = random.randrange(1 << 32)
    else:
        actual_seed = seed

    tts_engine.tts.stop_flag = False
    if ref_text_free:
        prompt_text = None
        prompt_lang = None

    yield None, gr.skip(), gr.Button(interactive=True)

    result = tts_engine.generate(
        text=text,
        text_lang=language_mapping[text_lang],
        speaker_name=speaker_name,
        ref_audio_path=ref_audio_path,
        prompt_text=prompt_text,
        prompt_lang=language_mapping[prompt_lang],
        aux_ref_audio_paths=aux_ref_audio_paths,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        text_split_method=cut_methods_mapping[text_split_method],
        batch_size=batch_size,
        return_fragment=False,
        speed_factor=speed_factor,
        split_bucket=split_bucket,
        fragment_interval=fragment_interval,
        seed=actual_seed,
        parallel_infer=parallel_infer,
        repetition_penalty=repetition_penalty,
        progress_tracker=partial(progress_tracker.tqdm, desc="Gernerating"),
    )
    try:
        audio = next(result)
    except StopIteration:
        return gr.skip(), seed, gr.Button(interactive=False)
    yield audio, actual_seed, gr.Button(interactive=False)


def set_sovits(sovits_path: str, prompt_language: str, text_language: str, tts_engine: TTSEngine):
    tts_engine.set_vits(sovits_path)
    language_list = get_languages_list(tts_engine)
    i18n = tts_engine.i18n
    if prompt_language in language_list:
        prompt_text_update, prompt_language_update = gr.skip(), gr.Dropdown(choices=language_list)
    else:
        prompt_text_update, prompt_language_update = gr.Textbox(value=""), gr.Dropdown(choices=language_list, value=i18n("all_zh"))
    if text_language in language_list:
        text_update, text_language_update = gr.skip(), gr.Dropdown(choices=language_list)
    else:
        text_update, text_language_update = gr.Textbox(value=""), gr.Dropdown(choices=language_list, value=i18n("all_zh"))
    return prompt_text_update, prompt_language_update, text_update, text_language_update


def set_gpt(gpt_path: str, tts_engine: TTSEngine):
    tts_engine.set_t2s(gpt_path)


def compile_func(speaker_name: str, batch_size: int, tts_engine: TTSEngine, progress_tracker=gr.Progress()):
    progress_tracker(0, desc="Starting")
    tts_engine.compile_func(speaker_name, batch_size=batch_size, progress_tracker=partial(progress_tracker.tqdm, desc="Compiling"))
    return gr.Slider(interactive=False), gr.Checkbox(interactive=False)
