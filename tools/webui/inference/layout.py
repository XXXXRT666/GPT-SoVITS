from functools import partial

import gradio as gr

from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine
from tools.cfg import V2_LANGUAGES
from tools.webui.inference.handlers import inference, set_gpt, set_sovits, compile_func
from tools.webui.inference.utils import get_gpt_paths, get_sovits_path, get_languages_list
from tools.webui.inference.assets import js, css, Seafoam


def build_app(tts_engine: TTSEngine, compile: bool = False):
    i18n = tts_engine.i18n
    configs = tts_engine.configs

    CUT_METHODS_MAPPING = {i18n(f"cut{i}"): f"cut{i}" for i in range(6)}
    LANGUAGE_MAPPING = {i18n(i): i for i in V2_LANGUAGES}

    inference_partial = partial(inference, tts_engine=tts_engine, cut_methods_mapping=CUT_METHODS_MAPPING, language_mapping=LANGUAGE_MAPPING)
    set_gpt_partial = partial(set_gpt, tts_engine=tts_engine)
    set_sovits_partial = partial(set_sovits, tts_engine=tts_engine)
    compile_func_partial = partial(compile_func, tts_engine=tts_engine)

    with gr.Blocks(title=i18n("GPT-SoVITS Inference WebUI"), analytics_enabled=False, fill_width=False, js=js, css=css, theme=Seafoam()) as app:
        gr.Markdown(
            value=i18n(
                "本软件以 MIT 协议开源，作者不对软件具备任何控制力，使用软件者、传播软件导出的声音者自负全责<br>如不认可该条款，则不能使用或引用软件包内任何代码和文件。详见根目录<b> LICENSE</b>."
            ),
            elem_classes="markdown",
        )
        gr.Markdown(value="<center>" + i18n("模型选择") + "</center>")
        with gr.Row(equal_height=True):
            with gr.Column():
                GPT_dropdown = gr.Dropdown(
                    choices=get_gpt_paths(), value=tts_engine.speaker.t2s_path, label=i18n("GPT 模型列表"), interactive=True, scale=1
                )
            with gr.Column():
                SoVITS_dropdown = gr.Dropdown(
                    choices=get_sovits_path(), value=tts_engine.speaker.vits_path, label=i18n("SoVITS 模型列表"), interactive=True, scale=1
                )
            with gr.Column():
                refresh_button = gr.Button(value=i18n("刷新模型和说话人列表"), variant="secondary", scale=1)
        with gr.Row(equal_height=True):
            with gr.Column():
                speakers_dropdown = gr.Dropdown(
                    choices=list(tts_engine.list_speaker()), value=configs.speaker_name, label=i18n("说话人列表"), interactive=True
                )
            with gr.Column():
                new_speaker_name = gr.Textbox(value="WebUI", label=i18n("新说话人名称"))
            with gr.Column():
                add_new_speaker = gr.Button(value=i18n("添加新说话人"), variant="secondary")
        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Row(equal_height=True):
                    ref_audio = gr.Audio(sources="upload", label=i18n("参考音频"), type="filepath", min_length=3, max_length=10, editable=False)
                    aux_ref_audio = gr.File(file_count="multiple", type="filepath", label=i18n("辅助参考音频"), height=200)
                with gr.Row(equal_height=True):
                    prompt_text = gr.Textbox(label=i18n("主参考音频文本"), lines=1, max_lines=1)
                with gr.Row(equal_height=True):
                    with gr.Row(equal_height=True):
                        prompt_lang = gr.Dropdown(
                            choices=get_languages_list(tts_engine),
                            value=i18n("all_zh"),
                            label=i18n("主参考音频语种"),
                            interactive=True,
                        )
                    with gr.Row(equal_height=True):
                        ref_free_mode = gr.Checkbox(label=i18n("无参考文本模式"))
            with gr.Column():
                text = gr.Textbox(label=i18n("目标文本"), lines=13, max_lines=13, elem_id="custom-textbox2")
                text_lang = gr.Dropdown(choices=get_languages_list(tts_engine), value=i18n("all_zh"), label=i18n("目标语种"), interactive=True)
        with gr.Row(equal_height=True):
            with gr.Column():
                batch_size = gr.Slider(minimum=1, maximum=100, value=configs.batch_size, step=1, label=i18n("批处理大小"), interactive=not compile)
                fragment_interval = gr.Slider(
                    minimum=0.1, maximum=1.0, value=configs.fragment_interval, step=0.1, label=i18n("段间间隔"), interactive=True
                )
                speed_factor = gr.Slider(minimum=0.5, maximum=1.5, value=configs.speed_factor, step=0.05, label=i18n("音频速度"), interactive=True)
                top_k = gr.Slider(minimum=1, maximum=100, value=configs.top_k, step=1, label=i18n("Top-k"), interactive=True)
                top_p = gr.Slider(minimum=0, maximum=1, value=configs.top_p, step=0.05, label=i18n("Top-p"), interactive=True)
                temperature = gr.Slider(minimum=0, maximum=1, value=configs.temperature, step=0.05, label=i18n("Temperature"), interactive=True)
                repetition_penalty = gr.Slider(
                    minimum=0, maximum=2, value=configs.reprtition_penalty, step=0.05, label=i18n("reprtition_penalty"), interactive=True
                )
            with gr.Column():
                with gr.Row(equal_height=True):
                    with gr.Row(equal_height=True):
                        parallel_infer = gr.Checkbox(value=False, label=i18n("并行推理"), interactive=True)
                    with gr.Row(equal_height=True):
                        split_bucket = gr.Checkbox(value=False, label=i18n("分桶处理"), interactive=True)
                with gr.Row(equal_height=True):
                    with gr.Row(equal_height=True):
                        keep_random = gr.Checkbox(value=True, label=i18n("保持随机"), interactive=True)
                    with gr.Row(equal_height=True):
                        compile_checkbox = gr.Checkbox(value=compile, label=i18n("编译模型"), interactive=not compile)
                with gr.Row(equal_height=True):
                    with gr.Row(equal_height=True):
                        cut_methods = gr.Dropdown(
                            choices=list(CUT_METHODS_MAPPING.keys()), value=i18n("cut1"), label=i18n("切分方式"), interactive=True
                        )
                    with gr.Row(equal_height=True):
                        seed = gr.Number(value=-1, label=i18n("随机种子"), minimum=-1, interactive=True)
                with gr.Row(equal_height=True):
                    with gr.Row(equal_height=True):
                        output = gr.Audio(label=i18n("生成语音"), editable=False)
                with gr.Row(equal_height=True):
                    with gr.Row(equal_height=True):
                        start_button = gr.Button(value=i18n("合成"), variant="secondary")
                    with gr.Row(equal_height=True):
                        stop_button = gr.Button(value=i18n("中止合成"), variant="stop", interactive=False)

        GPT_dropdown.change(set_gpt_partial, [GPT_dropdown], [])
        SoVITS_dropdown.change(set_sovits_partial, [SoVITS_dropdown], [prompt_text, prompt_lang, text, text_lang])
        compile_checkbox.change(compile_func_partial, [speakers_dropdown, batch_size], [batch_size, compile_checkbox])
        start_button.click(
            inference_partial,
            [
                text,
                text_lang,
                speakers_dropdown,
                ref_audio,
                aux_ref_audio,
                prompt_text,
                prompt_lang,
                top_k,
                top_p,
                temperature,
                cut_methods,
                batch_size,
                speed_factor,
                ref_free_mode,
                split_bucket,
                fragment_interval,
                seed,
                keep_random,
                parallel_infer,
                repetition_penalty,
            ],
            [output, seed, stop_button],
            api_name=False,
            scroll_to_output=True,
            show_api=False,
        )
        stop_button.click(tts_engine.tts.stop, [], [])
    return app
