import gradio as gr

from tools.cfg import Main_WebUI_Cfg
from tools.i18n.i18n import I18nAuto


def build_app(configs: Main_WebUI_Cfg):
    i18n = I18nAuto(configs.i18n_language)
    with gr.Blocks(title="GPT-SoVITS WebUI", analytics_enabled=False, fill_width=False) as app:
        gr.Markdown(
            value=i18n(
                "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>."
            ),
            elem_classes="markdown",
        )
        gr.Markdown(
            value=i18n("中文教程文档：https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e"),
            elem_classes="markdown",
        )
        with gr.Tab(label=i18n("0-数据集获取")):
            with gr.Row(equal_height=True):
                gr.Markdown(value=i18n("UVR5分离工具"))
                with gr.Row(equal_height=True):
                    with gr.Column():
                        uvr5_output = gr.Textbox(i18n("UVR5 输出信息"), interactive=False)
                        uvr5_button_launch = gr.Button(value=i18n("启动 UVR5 WebUI"), interactive=True)
                        uvr5_button_terminate = gr.Button(value=i18n("关闭 UVR5 WebUI"), visible=False, interactive=True)
            with gr.Row(equal_height=True):
                gr.Markdown(value=i18n("音频切片工具"))
                with gr.Row(equal_height=True):
                    pass
