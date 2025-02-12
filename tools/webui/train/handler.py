import os

import gradio as gr

from tools.webui.train.subfix import Subfix


def open_subfix(list_path: str, subfix: Subfix):
    list_path = list_path.strip("\n").strip("'").strip('"').strip()
    if not os.path.exists(list_path):
        gr.Warning(subfix.i18n("以下文件或文件夹不存在"), title="Subfix Error")
        gr.Warning(list_path, title="List File")
        return (*(gr.skip() for _ in range(7)),)
    else:
        subfix.render(list_path=list_path)
    return (
        gr.Button(visible=False),
        gr.Button(visible=True),
        gr.Tabs(selected=2),
        gr.Tab(visible=True),
        gr.Slider(value=10),
        gr.Slider(value=0, maximum=subfix.max_index),
        gr.Textbox(value=list_path, interactive=False),
    )


def close_subfix():
    return (
        gr.Button(visible=True),
        gr.Button(visible=False),
        gr.Tabs(selected=1),
        gr.Tab(visible=False),
        gr.Textbox(interactive=True),
        gr.Slider(value=1),
    )
