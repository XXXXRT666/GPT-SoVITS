import os
from functools import partial

import psutil
import gradio as gr

from tools.cfg import Main_WebUI_Cfg
from tools.i18n.i18n import I18nAuto
from tools.webui.assets import seafoam, js, css, top_html, info_html, uvr5_html, subfix_html
from tools.webui.train.subfix import Subfix
from tools.webui.train.handler import open_subfix, close_subfix


def build_app(configs: Main_WebUI_Cfg):
    i18n = I18nAuto(configs.i18n_language)
    # Layout
    with gr.Blocks(theme=seafoam, title="GPT-SoVITS Train WebUI", css=css, js=js, analytics_enabled=False, fill_width=False) as app:
        gr.HTML(
            value=top_html.format(
                i18n(
                    "本软件以 MIT 协议开源，作者不对软件具备任何控制力，使用软件者、传播软件导出的声音者自负全责<br>如不认可该条款，则不能使用或引用软件包内任何代码和文件. 详见根目录<b> LICENSE</b>"
                )
            ),
            elem_classes="markdown",
        )
        with gr.Tab(label=i18n("0-音频预处理")):
            with gr.Tabs(selected=1) as Tab_0:
                with gr.Tab(label=i18n("UVR5分离工具")):
                    gr.HTML(
                        value=uvr5_html.format(
                            *map(
                                i18n,
                                [
                                    "伴奏人声分离&去混响&去回声",
                                    "<br>模型分为三类:<br>&emsp;&emsp;1.伴奏分离(BS_Roformer)<br>&emsp;&emsp;2.和声分离(HP5_Karaoke)<br>&emsp;&emsp;3.混响回声分离(DeEcho, DeReverb)<br><br>·ONNX_DeReverb对于双通道混响是最好的选择，不能去除单通道混响<br>·DeEcho:去除延迟效果. Aggressive比Normal去除得更彻底<br>·DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>·个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive",
                                ],
                            )
                        ),
                        elem_classes="markdown",
                    )
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            with gr.Group():
                                with gr.Row(equal_height=True):
                                    uvr5_inp_path = gr.Textbox(
                                        lines=1,
                                        max_lines=1,
                                        placeholder=r"C:\1.wav or C:\abc" if os.name == "nt" else "~/audio",
                                        label=i18n("UVR5输入音频路径或音频所在文件夹路径"),
                                        interactive=True,
                                    )
                                    slicer_opt_path = gr.Textbox(
                                        value="output/uvr5_opt",
                                        lines=1,
                                        max_lines=1,
                                        placeholder="output/uvr5_opt",
                                        label=i18n("输出文件夹路径"),
                                        interactive=True,
                                    )
                                with gr.Row(equal_height=True):
                                    uvr5_model = gr.Dropdown(
                                        choices=[1, 2, 3, 4, 5],
                                        value=1,
                                        label=i18n("模型"),
                                        interactive=True,
                                    )
                                    vocal_only = gr.Checkbox(
                                        value=True,
                                        label=i18n("仅输出人声"),
                                        elem_id="checkbox_align_train",
                                        interactive=True,
                                    )
                            with gr.Row(equal_height=True):
                                uvr5_output = gr.Textbox(
                                    label=i18n("UVR5输出信息"),
                                    lines=2,
                                    max_lines=2,
                                    interactive=False,
                                )
                        with gr.Column():
                            uvr5_button_start = gr.Button(
                                value=i18n("开始分离"),
                                variant="primary",
                            )
                            uvr5_button_terminate = gr.Button(
                                value=i18n("中止分离"),
                                variant="stop",
                                visible=False,
                            )

                with gr.Tab(label=i18n("音频处理工具"), id=1):
                    gr.Markdown(value=i18n("音频切片工具"))
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            with gr.Group():
                                with gr.Row(equal_height=True):
                                    slicer_inp_path = gr.Textbox(
                                        lines=1,
                                        max_lines=1,
                                        placeholder=r"C:\1.wav or C:\abc" if os.name == "nt" else "~/audio",
                                        label=i18n("待切分音频路径或音频所在文件夹路径"),
                                        interactive=True,
                                    )
                                    slicer_opt_path = gr.Textbox(
                                        value="output/slicer_opt",
                                        lines=1,
                                        max_lines=1,
                                        placeholder="output/slicer_opt",
                                        label=i18n("输出文件夹路径"),
                                        interactive=True,
                                    )
                                with gr.Row(equal_height=True):
                                    threshold = gr.Slider(
                                        minimum=-50,
                                        maximum=-10,
                                        value=-34,
                                        step=1,
                                        label=i18n("静音阈值"),
                                        interactive=True,
                                    )
                                    hop_size = gr.Slider(
                                        minimum=0,
                                        maximum=100,
                                        value=10,
                                        step=10,
                                        label=i18n("步长"),
                                        interactive=True,
                                    )
                                with gr.Row(equal_height=True):
                                    min_length = gr.Slider(
                                        minimum=500,
                                        maximum=10000,
                                        value=4000,
                                        step=500,
                                        label=i18n("切割后音频最短时长"),
                                        interactive=True,
                                    )
                                    min_interval = gr.Slider(
                                        minimum=50,
                                        maximum=500,
                                        value=250,
                                        step=50,
                                        label=i18n("最短切割间隔"),
                                        interactive=True,
                                    )
                                with gr.Row(equal_height=True):
                                    max_silence_kept = gr.Slider(
                                        minimum=100,
                                        maximum=1000,
                                        value=500,
                                        step=100,
                                        label=i18n("最大保留精音时长"),
                                        interactive=True,
                                    )
                                    max_volume = gr.Slider(
                                        minimum=0.1,
                                        maximum=1,
                                        value=0.9,
                                        step=0.05,
                                        label=i18n("音频归一化后最大值"),
                                        interactive=True,
                                    )
                                with gr.Row(equal_height=True):
                                    alpha_mix = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        value=0.25,
                                        step=0.05,
                                        label=i18n("归一化混合比例"),
                                        interactive=True,
                                    )
                                    slicer_process_num = gr.Slider(
                                        minimum=0,
                                        maximum=psutil.cpu_count(logical=False),
                                        value=4,
                                        step=1,
                                        label=i18n("切割进程数"),
                                        interactive=True,
                                    )
                            with gr.Row(equal_height=True):
                                slicer_output = gr.Textbox(
                                    label=i18n("语音切割输出信息"),
                                    interactive=False,
                                )
                        with gr.Column():
                            slicer_button_start = gr.Button(
                                value=i18n("开始语音切割"),
                                variant="primary",
                            )
                            slicer_button_terminate = gr.Button(
                                value=i18n("中止语音切割"),
                                variant="stop",
                                visible=False,
                            )
                    gr.Markdown(value=i18n("语音降噪工具"))
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            with gr.Group():
                                with gr.Row(equal_height=True):
                                    denoise_inp_path = gr.Textbox(
                                        lines=1,
                                        max_lines=1,
                                        placeholder="output/slicer_opt",
                                        label=i18n("待降噪音频所在文件夹路径"),
                                        interactive=True,
                                    )
                                    denoise_opt_path = gr.Textbox(
                                        value="output/denoise_opt",
                                        lines=1,
                                        max_lines=1,
                                        placeholder="output/denoise_opt",
                                        label=i18n("待切分音频路径或音频所在文件夹路径"),
                                        interactive=True,
                                    )
                            with gr.Row(equal_height=True):
                                denoise_output = gr.Textbox(
                                    label=i18n("语音降噪输出信息"),
                                    interactive=False,
                                )
                        with gr.Column():
                            denoise_button_start = gr.Button(
                                value=i18n("开始语音降噪"),
                                variant="primary",
                            )
                            denoise_button_terminate = gr.Button(
                                value=i18n("中止语音降噪"),
                                variant="stop",
                                visible=False,
                            )

                    gr.Markdown(value=i18n("ASR工具"))
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            with gr.Group():
                                with gr.Row(equal_height=True):
                                    asr_inp_path = gr.Textbox(
                                        lines=1,
                                        max_lines=1,
                                        placeholder="output/slicer_opt",
                                        label=i18n("待降噪音频所在文件夹路径"),
                                        interactive=True,
                                    )
                                    asr_opt_path = gr.Textbox(
                                        value="output/asr_opt",
                                        lines=1,
                                        max_lines=1,
                                        placeholder="output/asr_opt",
                                        label=i18n("待切分音频路径或音频所在文件夹路径"),
                                        interactive=True,
                                    )
                                with gr.Row(equal_height=True):
                                    asr_model = gr.Dropdown(
                                        choices=[1, 2, 3, 4, 5],
                                        value=1,
                                        label=i18n("ASR 模型"),
                                        interactive=True,
                                    )
                                    asr_model_size = gr.Dropdown(
                                        choices=[1, 2, 3, 4, 5],
                                        value=1,
                                        label=i18n("模型大小"),
                                        interactive=True,
                                    )
                                with gr.Row(equal_height=True):
                                    asr_language = gr.Dropdown(
                                        choices=[1, 2, 3, 4, 5],
                                        value=1,
                                        label=i18n("ASR 语言"),
                                        interactive=True,
                                    )
                                    asr_precision = gr.Dropdown(
                                        choices=[1, 2, 3, 4, 5],
                                        value=1,
                                        label=i18n("ASR 精度"),
                                        interactive=True,
                                    )
                            with gr.Row():
                                asr_output = gr.Textbox(
                                    label=i18n("ASR输出信息"),
                                    interactive=False,
                                )
                        with gr.Column():
                            asr_button_start = gr.Button(
                                value=i18n("开始ASR"),
                                variant="primary",
                            )
                            asr_button_terminate = gr.Button(
                                value=i18n("中止ASR"),
                                variant="stop",
                                visible=False,
                            )

                    gr.Markdown(value=i18n("文本标注校对工具"))
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):
                            subfix_inp_path = gr.Textbox(
                                placeholder=r"D:\GPT-SoVITS\output\asr_opt\xxx.list" if os.name == "nt" else "~/audio_transcriptions/xxx.list",
                                lines=1,
                                max_lines=1,
                                label=i18n("文本标注文件路径"),
                            )
                        with gr.Column():
                            subfix_button_launch = gr.Button(
                                value=i18n("启动打标WebUI"),
                                variant="primary",
                            )
                            subfix_button_terminate = gr.Button(
                                value=i18n("关闭打标WebUI"),
                                variant="stop",
                                visible=False,
                                elem_id="btn_close",
                            )
                with gr.Tab(label=i18n("文本标注校对工具"), id=2, visible=False) as Tab_0_2:
                    gr.HTML(value=subfix_html.format(*map(i18n, ["语音文本标注校对", "1,2,3"])))
                    subfix = Subfix(i18n)
        with gr.Tab(label=i18n("1-GPT-SoVITS模型训练")):
            with gr.Row(equal_height=True):
                with gr.Column(min_width=160):
                    model_name = gr.Textbox(
                        lines=1,
                        max_lines=1,
                        placeholder="ABC",
                        label=i18n("模型名"),
                        info=i18n("模型文件名前缀"),
                        interactive=True,
                    )
                with gr.Column(min_width=160):
                    gpu_infos = gr.Textbox(
                        value="0 RTX 5090",
                        lines=1,
                        max_lines=1,
                        label=i18n("设备信息"),
                        info=i18n("训练可用设备"),
                        interactive=False,
                    )
                with gr.Column(min_width=160):
                    version = gr.Radio(
                        choices=["v1", "v2"],
                        value="v2",
                        label=i18n("模型版本"),
                        info=i18n("详见`CHANGELOG.md`"),
                        interactive=True,
                    )
            with gr.Row(equal_height=True):
                with gr.Column(min_width=160):
                    pretrained_sovits_g = gr.Textbox(
                        value="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                        lines=2,
                        max_lines=2,
                        placeholder="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                        label="Pretrained SoVITS-G",
                        interactive=True,
                    )
                with gr.Column(min_width=160):
                    pretrained_sovits_d = gr.Textbox(
                        value="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
                        lines=2,
                        max_lines=2,
                        placeholder="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
                        label="Pretrained SoVITS-D",
                        interactive=True,
                    )
                with gr.Column(min_width=160):
                    pretrained_gpt = gr.Textbox(
                        value="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                        lines=2,
                        max_lines=2,
                        label="Pretrained GPT",
                        interactive=True,
                    )
            with gr.Tab(label=i18n("训练集生成")):
                gr.Markdown(value=i18n("`logs/模型名`下应有23456开头的文件和文件夹,**非中文**无`3-bert`"))
                with gr.Row(equal_height=True):
                    with gr.Column():
                        list_inp_path = gr.Textbox(
                            placeholder=r"D:\GPT-SoVITS\output\asr_opt\xxx.list" if os.name == "nt" else "~/audio_transcriptions/xxx.list",
                            lines=1,
                            max_lines=1,
                            label=i18n("文本标注文件路径"),
                            interactive=True,
                        )
                    with gr.Column():
                        audio_inp_path = gr.Textbox(
                            placeholder=r"output\slicer_opt",
                            lines=1,
                            max_lines=1,
                            label=i18n("训练集音频文件夹路径"),
                            interactive=True,
                        )

                gr.Markdown(i18n("1-文本处理"))
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=160):
                        text_process_gpu = gr.Textbox(
                            value="0-0",
                            lines=1,
                            max_lines=1,
                            label=i18n("GPU卡号以'-'分割,每个卡号一个进程"),
                            interactive=True,
                        )
                    with gr.Column(min_width=160):
                        bert_pretrained = gr.Textbox(
                            value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                            placeholder="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                            lines=2,
                            max_lines=2,
                            label=i18n("预训练Bert模型路径"),
                            interactive=False,
                        )
                    with gr.Column(min_width=160):
                        text_process_button_start = gr.Button(
                            value=i18n("开始文本处理"),
                            variant="primary",
                        )
                        text_process_button_terminate = gr.Button(
                            value=i18n("中止文本处理"),
                            visible=False,
                            variant="stop",
                        )
                    with gr.Column(min_width=160):
                        text_process_output = gr.Textbox(label=i18n("文本处理输出信息"), interactive=False)

                gr.Markdown(value=i18n("2-SSL特征提取"))
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=160):
                        ssl_extract_gpu = gr.Textbox(
                            value="0-0",
                            lines=1,
                            max_lines=1,
                            label=i18n("GPU卡号以'-'分割,每个卡号一个进程"),
                            interactive=True,
                        )
                    with gr.Column(min_width=160):
                        ssl_pretrained = gr.Textbox(
                            value="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                            placeholder="GPT_SoVITS/pretrained_models/chinese-hubert-base",
                            lines=2,
                            max_lines=2,
                            label=i18n("预训练SSL模型路径"),
                            interactive=False,
                        )
                    with gr.Column(min_width=160):
                        ssl_extract_button_start = gr.Button(
                            value=i18n("开始SSL提取"),
                            variant="primary",
                        )
                        ssl_extract_button_terminate = gr.Button(
                            value=i18n("中止SSL提取"),
                            visible=False,
                            variant="stop",
                        )
                    with gr.Column(min_width=160):
                        ssl_extract_output = gr.Textbox(
                            label=i18n("SSL提取输出信息"),
                            interactive=False,
                        )
                gr.Markdown(value=i18n("3-语义提取"))
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=160):
                        semantic_extract_gpu = gr.Textbox(
                            value="0-0",
                            lines=1,
                            max_lines=1,
                            label=i18n("GPU卡号以'-'分割,每个卡号一个进程"),
                            interactive=True,
                        )
                    with gr.Column(min_width=160):
                        semantic_pretrained = gr.Textbox(
                            value="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                            placeholder="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                            lines=2,
                            max_lines=2,
                            label=i18n("预训练SoVITS-G模型路径"),
                            interactive=False,
                        )
                    with gr.Column(min_width=160):
                        semantic_extract_button_start = gr.Button(
                            value=i18n("开始语义提取"),
                            variant="primary",
                        )
                        semantic_extract_button_terminate = gr.Button(
                            value=i18n("中止语义提取"),
                            variant="stop",
                            visible=False,
                        )
                    with gr.Column(min_width=160):
                        semantic_extract_output = gr.Textbox(
                            label=i18n("语义提取输出信息"),
                            interactive=False,
                        )
            with gr.Tab(label=i18n("微调训练")):
                gr.Markdown("1-SoVITS训练, 模型保存在`SoVITS_weights_v2`下")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                batch_size_sovits = gr.Slider(
                                    minimum=1,
                                    maximum=40,
                                    value=10,
                                    step=1,
                                    label=i18n("训练批量大小"),
                                    info=i18n("过高可能导致显存不足"),
                                    interactive=True,
                                )
                                total_epoch_sovits = gr.Slider(
                                    minimum=1,
                                    maximum=25,
                                    value=8,
                                    step=1,
                                    label=i18n("总训练轮数"),
                                    info=i18n("不建议过高"),
                                    interactive=True,
                                )
                            with gr.Row(equal_height=True):
                                text_lr = gr.Slider(
                                    minimum=0.2,
                                    maximum=0.6,
                                    value=0.4,
                                    step=0.05,
                                    label=i18n("文本模块学习率系数"),
                                    interactive=True,
                                )
                                save_interval_sovits = gr.Slider(
                                    minimum=1,
                                    maximum=25,
                                    value=4,
                                    step=1,
                                    label=i18n("保存频率"),
                                    interactive=True,
                                )
                    with gr.Column():
                        with gr.Row():
                            with gr.Row():
                                save_checkpoint_sovits = gr.Checkbox(
                                    value=False,
                                    label=i18n("仅保留最新检查点"),
                                    elem_id="checkbox_align_train",
                                )
                            with gr.Row():
                                save_final_model_sovits = gr.Checkbox(
                                    value=False,
                                    label=i18n("输出最终模型"),
                                    elem_id="checkbox_align_train",
                                )
                        with gr.Row():
                            device_ids_sovits = gr.Textbox(
                                value="0",
                                lines=1,
                                max_lines=1,
                                label=i18n("多卡GPU卡号"),
                                interactive=True,
                            )
                with gr.Row(equal_height=True):
                    with gr.Column():
                        sovits_train_button_start = gr.Button(
                            value=i18n("开始SoVITS训练"),
                            interactive=True,
                            variant="primary",
                        )
                        sovits_train_button_terminate = gr.Button(
                            value=i18n("中止SoVITS训练"),
                            variant="stop",
                            visible=False,
                            interactive=True,
                        )
                    with gr.Column():
                        sovits_train_output = gr.Textbox(
                            label=i18n("SoVITS训练输出信息"),
                            lines=2,
                            max_lines=2,
                        )

                gr.Markdown("2-GPT训练, 模型保存在`GPT_weights_v2`下")
                with gr.Row(equal_height=True):
                    with gr.Column():
                        with gr.Group():
                            with gr.Row(equal_height=True):
                                batch_size_gpt = gr.Slider(
                                    minimum=1,
                                    maximum=40,
                                    value=10,
                                    step=1,
                                    label=i18n("训练批量大小"),
                                    info=i18n("过高可能导致显存不足"),
                                    interactive=True,
                                )
                                total_epoch_gpt = gr.Slider(
                                    minimum=1,
                                    maximum=25,
                                    value=8,
                                    step=1,
                                    label=i18n("总训练轮数"),
                                    info=i18n("不建议过高"),
                                    interactive=True,
                                )
                            with gr.Row(equal_height=True):
                                save_interval_gpt = gr.Slider(
                                    minimum=1,
                                    maximum=25,
                                    value=4,
                                    step=1,
                                    label=i18n("保存频率"),
                                    interactive=True,
                                )
                                dpo_enabled = gr.Checkbox(
                                    value=False,
                                    label=i18n("开启DPO训练"),
                                    info=info_html.format(i18n("实验性")),
                                    elem_id="checkbox_train_dpo",
                                    interactive=True,
                                )
                    with gr.Column():
                        with gr.Row():
                            with gr.Row():
                                save_checkpoint_gpt = gr.Checkbox(
                                    value=False,
                                    label=i18n("仅保留最新检查点"),
                                    elem_id="checkbox_align_train",
                                    interactive=True,
                                )
                            with gr.Row():
                                save_final_model_gpt = gr.Checkbox(
                                    value=False,
                                    label=i18n("输出最终模型"),
                                    elem_id="checkbox_align_train",
                                    interactive=True,
                                )
                        with gr.Row():
                            device_ids_gpt = gr.Textbox(
                                value="0",
                                lines=1,
                                max_lines=1,
                                label=i18n("多卡GPU卡号"),
                                interactive=True,
                            )
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gpt_train_button_start = gr.Button(
                            value=i18n("开始GPT训练"),
                            variant="primary",
                        )
                        gpt_train_button_terminate = gr.Button(
                            value=i18n("中止GPT训练"),
                            visible=False,
                            variant="stop",
                        )
                    with gr.Column():
                        gpt_train_output = gr.Textbox(
                            label=i18n("GPT训练输出信息"),
                            lines=2,
                            max_lines=2,
                        )
        with gr.Tab(label=i18n("2-GPT-SoVITS-TTS")):
            gr.Markdown(
                value=i18n("模型存放在对应的weights文件夹下,默认的是预训练模型"),
            )
            with gr.Row(equal_height=True):
                with gr.Column():
                    GPT_dropdown = gr.Dropdown(
                        choices=[1, 2, 3, 4, 5],
                        value=1,
                        label=i18n("GPT 模型列表"),
                        interactive=True,
                        scale=1,
                    )
                with gr.Column():
                    SoVITS_dropdown = gr.Dropdown(
                        choices=[1, 2, 3, 4, 5],
                        value=1,
                        label=i18n("SoVITS 模型列表"),
                        interactive=True,
                        scale=1,
                    )
                with gr.Column():
                    refresh_button = gr.Button(
                        value=i18n("刷新模型列表"),
                        variant="primary",
                        scale=1,
                    )
            with gr.Row(equal_height=True):
                with gr.Column():
                    device_id_infer = gr.Dropdown(
                        choices=[1, 2, 3, 4, 5],
                        value=1,
                        label=i18n("推理GPU卡号"),
                        interactive=True,
                    )
                with gr.Column():
                    infer_webui_output = gr.Textbox(
                        label=i18n("TTS WebUI输出信息"),
                        lines=1,
                        max_lines=1,
                    )
                with gr.Column():
                    infer_webui_button_start = gr.Button(
                        value=i18n("启动TTS WebUI"),
                        variant="primary",
                    )
                    infer_webui_button_terminate = gr.Button(
                        value=i18n("关闭TTS WebUI"),
                        visible=False,
                        variant="stop",
                    )
        with gr.Tab(label=i18n("3-GPT-SoVITS-VC")):
            gr.Markdown(value=i18n("未发布, 施工中, 请静候佳音"), elem_classes="markdown")

        # Event Trigger Binding

        open_subfix_partial = partial(open_subfix, subfix=subfix)

        subfix_button_launch.click(
            open_subfix_partial,
            [subfix_inp_path],
            [
                subfix_button_launch,
                subfix_button_terminate,
                Tab_0,
                Tab_0_2,
                subfix.batch_size_slider,
                subfix.index_slider,
                subfix_inp_path,
            ],
        )

        subfix_button_terminate.click(
            fn=subfix.submit,
            inputs=[
                *subfix.textboxes,
                *subfix.languages,
            ],
            outputs=[],
            show_progress="hidden",
        ).success(
            fn=close_subfix,
            inputs=[],
            outputs=[
                subfix_button_launch,
                subfix_button_terminate,
                Tab_0,
                Tab_0_2,
                subfix_inp_path,
                subfix.batch_size_slider,
            ],
            scroll_to_output=True,
        )
    return app
