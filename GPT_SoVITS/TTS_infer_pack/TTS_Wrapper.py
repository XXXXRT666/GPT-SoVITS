import gc
import os
import traceback
import tempfile
import wave
import sys
from typing import Union, Callable, Optional, Iterable

import numpy as np
from tools.my_utils import SilentPrint
from tools.cfg import Speakers_Cfg, Inference_WebUI_Cfg, API_Batch_Cfg, Speaker, Prompt, Cfg
from tools.i18n.i18n import I18nAuto
from tools.server.schema import TTSRequest, TTSResponseFailed, TTSResponseSuccess, TTSResponseSegment
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits
from GPT_SoVITS.TTS_infer_pack.TTS import TTS


def default_exception_handler(tts_response: TTSResponseFailed):
    print(tts_response.exception, file=sys.stderr)
    if tts_response.tracebacks:
        print(tts_response.tracebacks, file=sys.stderr)
    return


def default_progress_tracker(items: Iterable):
    return items


class TTSEngine:
    def __init__(
        self,
        configs: Union[Inference_WebUI_Cfg, API_Batch_Cfg],
        speakers_cfg: Speakers_Cfg,
        speaker_name: str,
        compile=False,
        exception_handler=default_exception_handler,
    ):
        self.configs = configs
        self.speakers_cfg = speakers_cfg
        self.speaker = self.speakers_cfg.get_speaker(speaker_name)
        self.prompt_cache: dict = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }
        self.i18n = I18nAuto(self.configs.i18n_language)
        self.tts = TTS(self.configs, self.speaker, self.prompt_cache, self.i18n)
        self.compile = compile
        self.exception_handler = exception_handler

    @classmethod
    def get_instance(
        cls,
        cfg_name: str,
        cfg_path: str = "tools/cfgs/cfg.json",
        speakers_cfg_path: str = "tools/cfgs/speakers.json",
        compile=False,  # Not Supported yet
        exception_handler: Callable[[TTSResponseFailed], Optional[Exception]] = default_exception_handler,
    ):
        configs: Union[Inference_WebUI_Cfg, API_Batch_Cfg] = getattr(Cfg.from_json(cfg_path), cfg_name)
        speakers_cfg = Speakers_Cfg.from_json(speakers_cfg_path)
        instance = cls(
            configs=configs, speakers_cfg=speakers_cfg, speaker_name=configs.speaker_name, compile=compile, exception_handler=exception_handler
        )
        if compile:
            instance.compile_func(speaker_name=configs.speaker_name, batch_size=configs.batch_size)
            # To be continued
        instance.warmup(configs.speaker_name)
        return instance

    def clear_prompt_cache(self):
        self.prompt_cache = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }
        self.tts.prompt_cache = self.prompt_cache

    def set_vits(self, vits_path: str):
        self.tts.init_vits_weights(vits_path)
        self.speaker.vits_path = vits_path
        self.speaker.update_language()
        self.speakers_cfg.save_as_json()

    def set_t2s(self, t2s_path: str):
        self.tts.init_t2s_weights(t2s_path)
        self.speaker.t2s_path = t2s_path
        self.speakers_cfg.save_as_json()

    def set_prompt(self, prompt: Prompt):
        if prompt.is_empty():
            self.clear_prompt_cache()
            raise ValueError(self.i18n("Ref audio path can not be empty"))
        self.tts.set_ref_audio(prompt.ref_audio_path)
        self.prompt_cache["prompt_text"] = prompt.prompt_text
        self.prompt_cache["prompt_lang"] = prompt.prompt_lang
        if prompt.prompt_text:
            prompt.prompt_text = prompt.prompt_text.strip("\n")
            if prompt.prompt_text[-1] not in splits:
                prompt.prompt_text += "。" if prompt.prompt_lang != "en" else "."
            phones, bert_features, norm_text = self.tts.text_preprocessor.segment_and_extract_feature_for_text(
                prompt.prompt_text, prompt.prompt_lang, self.speaker.version
            )
            self.prompt_cache["phones"] = phones
            self.prompt_cache["bert_features"] = bert_features
            self.prompt_cache["norm_text"] = norm_text
        else:
            self.prompt_cache["phones"] = None
            self.prompt_cache["bert_features"] = None
            self.prompt_cache["norm_text"] = None
        if set(prompt.aux_ref_audio_paths) != set(self.prompt_cache["aux_ref_audio_paths"]):
            self.prompt_cache["aux_ref_audio_paths"] = prompt.aux_ref_audio_paths
            self.prompt_cache["refer_spec"] = [self.prompt_cache["refer_spec"][0]]
            for path in prompt.aux_ref_audio_paths:
                if path in [None, ""]:
                    continue
                if not os.path.exists(path):
                    print(self.i18n("音频文件不存在，跳过：{}").format(path))
                    continue
                self.prompt_cache["refer_spec"].append(self.tts.get_ref_spec(path))

    def set_speaker(self, speaker_name: str, prompt=True, force=False):
        self._set_speaker(speaker_name, prompt, force)
        gc.collect(1)

    def _set_speaker(self, speaker_name: str, prompt=True, force=False):
        if speaker_name not in self.speakers_cfg.list_speaker():
            print(f"Invalid speaker name, {speaker_name} does not exist")
            return
        if force:
            self.speaker = self.speakers_cfg.get_speaker(speaker_name)
            self.set_vits(self.speaker.vits_path)
            self.set_t2s(self.speaker.t2s_path)
            self.set_prompt(self.speaker.prompt)
            return
        new_spk = self.speakers_cfg.get_speaker(speaker_name)
        if new_spk == self.speaker:
            return
        old_spk = self.speaker
        self.speaker = new_spk
        if new_spk.vits_path != old_spk.vits_path:
            self.set_vits(new_spk.vits_path)
        if new_spk.t2s_path != old_spk.t2s_path:
            self.set_t2s(new_spk.vits_path)
        if prompt:
            self.set_prompt(new_spk.prompt)

    def add_speaker(self, spk_name: str, spk: dict):
        """
        see cfg.Speaker and cfg.Prompt for more information

        spk: {
            t2s_path: str
            vits_path: str
            version: str = "v2"
            prompt: Optional[Prompt]
        }

        prompt: {
            ref_audio_path: Optional[str] = None
            prompt_text: Optional[str] = None
            prompt_lang: Optional[str] = None # prompt_text and prompt_lang must be set at the same time
            aux_ref_audio_paths: Union[list[str], None] = None
        }
        """
        new_spk = Speaker(**spk)
        self.speakers_cfg.speakers_dict.update({spk_name: new_spk})
        self.speakers_cfg.save_as_json()

    def del_speaker(self, spk_name: str):
        self.speakers_cfg.speakers_dict.pop(spk_name, None)

    def get_speaker(self, spk_name: str):
        return self.speakers_cfg.get_speaker(spk_name)

    def list_speaker(self):
        return self.speakers_cfg.list_speaker()

    def warmup(self, speaker_name: str, progress_tracker=default_progress_tracker):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            file_name = temp_file.name
            with wave.open(temp_file, "w") as wav_file:
                channels = 1
                sample_width = 2
                sample_rate = 44100
                duration = 5
                frequency = 440.0

                t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                sine_wave = np.sin(2 * np.pi * frequency * t)  # Sine Wave
                int_wave = (sine_wave * 32767).astype(np.int16)

                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(int_wave.tobytes())
                print("Warm UP")
                print('If "compile" is selected, it may take some time to warm up.')

                with SilentPrint():
                    next(
                        self.generate(
                            progress_tracker,
                            text="犯大吴疆土者,盛必击而破之,犯大吴疆土者,盛必击而破之,犯大吴疆土者,盛必击而破之,犯大吴疆土者,盛必击而破之",
                            text_lang="all_zh",
                            ref_audio_path=file_name,
                            speaker_name=speaker_name,
                            text_split_method="cut5",
                        )
                    )
        self.clear_prompt_cache()

    def precall(self, **kwds) -> Union[TTSRequest, TTSResponseFailed]:
        try:
            speaker_name = kwds.pop("speaker_name", "")
            self.set_speaker(speaker_name, prompt=False)

            ref_audio_path = kwds.pop("ref_audio_path", None)
            aux_ref_audio_paths = kwds.pop("aux_ref_audio_paths", [])

            if ref_audio_path:
                prompt_text = kwds.pop("prompt_text", None)
                prompt_lang = kwds.pop("prompt_lang", None)
            else:
                ref_audio_path = self.speaker.prompt.ref_audio_path
                prompt_text = self.speaker.prompt.prompt_text
                prompt_lang = self.speaker.prompt.prompt_lang
                kwds.pop("prompt_text", None)
                kwds.pop("prompt_lang", None)
            if aux_ref_audio_paths:
                pass
            else:
                aux_ref_audio_paths = self.speaker.prompt.aux_ref_audio_paths

            if isinstance(prompt_lang, str):
                prompt_lang = prompt_lang.lower()

            prompt = Prompt(ref_audio_path=ref_audio_path, prompt_text=prompt_text, prompt_lang=prompt_lang, aux_ref_audio_paths=aux_ref_audio_paths)

            self.set_prompt(prompt)

            ### Other parameters
            kwds["text_lang"] = kwds.get("text_lang", "").lower()
            kwds["top_k"] = kwds.get("top_k") or self.configs.top_k
            kwds["top_p"] = kwds.get("top_p") or self.configs.top_p
            kwds["temperature"] = kwds.get("temperature") or self.configs.temperature
            kwds["repetition_penalty"] = kwds.get("repetition_penalty") or self.configs.reprtition_penalty

            kwds["text_split_method"] = kwds.get("text_split_method") or self.configs.text_splitting_method
            kwds["batch_size"] = kwds.get("batch_size") or self.configs.batch_size
            kwds["batch_threshold"] = kwds.get("batch_threshold", 0.75)
            kwds["speed_factor"] = kwds.get("speed_factor") or self.configs.speed_factor

            kwds["split_bucket"] = kwds.get("split_bucket", True)
            kwds["return_fragment"] = kwds.get("return_fragment") or self.configs.streaming
            kwds["fragment_interval"] = kwds.get("fragment_interval") or self.configs.fragment_interval
            kwds["parallel_infer"] = kwds.get("parallel_infer") or self.configs.parallel_inference
            kwds["seed"] = kwds.get("seed", -1)
            TTS_Request = TTSRequest(**kwds)

            if TTS_Request.parallel_infer:
                print(self.i18n("并行推理模式已开启"))
            else:
                print(self.i18n("并行推理模式已关闭"))

            if TTS_Request.return_fragment:
                print(self.i18n("分段返回模式已开启"))
                if TTS_Request.split_bucket:
                    TTS_Request.split_bucket = False
                    print(self.i18n("分段返回模式不支持分桶处理，已自动关闭分桶处理"))

            if TTS_Request.split_bucket and TTS_Request.speed_factor == 1.0:
                print(self.i18n("分桶处理模式已开启"))
            elif TTS_Request.speed_factor != 1.0:
                print(self.i18n("语速调节不支持分桶处理，已自动关闭分桶处理"))
                TTS_Request.split_bucket = False
            else:
                print(self.i18n("分桶处理模式已关闭"))

            if TTS_Request.fragment_interval < 0.01:
                TTS_Request.fragment_interval = 0.01
                print(self.i18n("分段间隔过小，已自动设置为0.01"))

            TTS_Request.no_prompt_text = not bool(prompt_text)

            assert TTS_Request.text_lang in self.speaker.languages
            if not TTS_Request.no_prompt_text:
                assert prompt_lang in self.speaker.languages

            return TTS_Request

        except Exception as e:
            return TTSResponseFailed(exception=e, tracebacks=traceback.format_exc())

    def __call__(
        self,
        progress_tracker: Callable[[Iterable], Iterable] = default_progress_tracker,
        **kwds,
    ):
        """
        Text to speech inference.

        Some parameters can be set in the config file. If they are not passed in during the call, the values from the config file will be used.

        Args:
            kwds (dict):
                {
                    "text":                     # str.(required) text to be synthesized
                    "text_lang:                 # str.(required) language of the text to be synthesized
                    "speaker_name":             # str.(required) speaker name in the speakers_cfg

                    "ref_audio_path":           # str.(required if not set in speakers_cfg) reference audio path
                    "prompt_text":              # str.(required if not set in speakers_cfg) prompt text for the reference audio
                    "prompt_lang":              # str.(required if not set in speakers_cfg) language of the prompt text for the reference audio

                    The following are optional parameters:

                    "aux_ref_audio_paths"       # list. auxiliary reference audio paths for multi-speaker tone fusion
                    "top_k":                    # int. top-k sampling
                    "top_p":                    # float. top-p sampling
                    "temperature":              # float. temperature for sampling
                    "text_split_method":        # str. text split method, see text_segmentation_method.py for details.
                    "batch_size":               # int. batch size for inference
                    "batch_threshold": 0.75     # float. threshold for batch splitting.
                    "split_bucket: True,        # bool. whether to split the batch into multiple buckets.
                    "return_fragment":          # bool. step by step return the audio fragment.
                    "speed_factor":             # float. control the speed of the synthesized audio.
                    "fragment_interval":        # float. to control the interval of the audio fragment.
                    "seed": -1,                 # int. random seed for reproducibility.
                    "parallel_infer":           # bool. whether to use parallel inference.
                    "repetition_penalty":       # float. repetition penalty for T2S model.
                }
        """

        precall_result = self.precall(**kwds)  # checking
        excptions = None
        if isinstance(precall_result, TTSResponseFailed):
            excptions = self.exception_handler(precall_result)
            if isinstance(excptions, Exception):
                raise excptions
        elif isinstance(precall_result, TTSRequest):
            yield None
            for result in self.tts.run(precall_result, progress_tracker=progress_tracker):
                if isinstance(result, (TTSResponseSuccess, TTSResponseSegment)):
                    yield result.audio
                elif isinstance(result, TTSResponseFailed):
                    excptions = self.exception_handler(result)
        if isinstance(excptions, Exception):
            raise excptions
        return

    def generate(
        self,
        progress_tracker: Callable[[Iterable], Iterable] = default_progress_tracker,
        **kwds,
    ):
        """
        Similar to __call__, but skips the first None.

        Text to speech inference.

        Some parameters can be set in the config file. If they are not passed in during the call, the values from the config file will be used.

        Args:
            kwds (dict):
                {
                    "text":                     # str.(required) text to be synthesized
                    "text_lang:                 # str.(required) language of the text to be synthesized
                    "speaker_name":             # str.(required) speaker name in the speakers_cfg

                    "ref_audio_path":           # str.(required if not set in speakers_cfg) reference audio path
                    "prompt_text":              # str.(required if not set in speakers_cfg) prompt text for the reference audio
                    "prompt_lang":              # str.(required if not set in speakers_cfg) language of the prompt text for the reference audio

                    The following are optional parameters:

                    "aux_ref_audio_paths"       # list. auxiliary reference audio paths for multi-speaker tone fusion
                    "top_k":                    # int. top-k sampling
                    "top_p":                    # float. top-p sampling
                    "temperature":              # float. temperature for sampling
                    "text_split_method":        # str. text split method, see text_segmentation_method.py for details.
                    "batch_size":               # int. batch size for inference
                    "batch_threshold": 0.75     # float. threshold for batch splitting.
                    "split_bucket: True,        # bool. whether to split the batch into multiple buckets.
                    "return_fragment":          # bool. step by step return the audio fragment.
                    "speed_factor":             # float. control the speed of the synthesized audio.
                    "fragment_interval":        # float. to control the interval of the audio fragment.
                    "seed": -1,                 # int. random seed for reproducibility.
                    "parallel_infer":           # bool. whether to use parallel inference.
                    "repetition_penalty":       # float. repetition penalty for T2S model.
                }
        """

        audio_generator = self.__call__(progress_tracker, **kwds)
        next(audio_generator)
        return audio_generator

    def compile_func(self, speaker_name: str, batch_size: int, progress_tracker: Callable[[Iterable], Iterable] = default_progress_tracker):
        ...
        self.warmup(speaker_name, progress_tracker)
