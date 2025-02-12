from typing import Iterable

import gradio.themes.base as ThemeBase
from gradio.themes.utils import colors, fonts, sizes


class Seafoam(ThemeBase.Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.blue,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            "Menlo",
            "Consolas",
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            "Menlo",
            "Consolas",
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            # Light
            body_background_fill="linear-gradient(45deg,rgb(128, 238, 238),rgb(236, 253, 245))",  # 浅色模式背景渐变
            button_secondary_background_fill_hover="rgb(169, 208, 254)",
            button_primary_background_fill="linear-gradient(90deg, *primary_200, *secondary_200)",
            button_primary_background_fill_hover="linear-gradient(90deg,rgb(153, 240, 202),rgb(169, 208, 254))",
            button_primary_text_color="black",
            button_cancel_background_fill="rgb(255, 166, 166)",
            button_cancel_background_fill_hover="rgb(255, 134, 134)",
            slider_color="*secondary_300",
            block_title_text_weight="500",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="32px",
            button_large_text_weight="500",
        )


js = """
function createGradioAnimation() {
    
    const params = new URLSearchParams(window.location.search);
    if (params.get('__theme') !== 'light') { 
        params.set('__theme', 'light'); // 仅当 __theme 不是 'light' 时设置为 'light'
        window.location.search = params.toString(); // 更新 URL，触发页面刷新
    }
    
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = '500';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    container.style.fontFamily = '-apple-system, sans-serif, Arial, Calibri';

    var text = 'Welcome to GPT-SoVITS!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""


css = """
/* CSSStyleRule */

.markdown {
    background-color: lightblue;
    padding: 10px;
}

.checkbox_info {
    color: var(--block-title-text-color) !important;
    font-size: var(--block-title-text-size) !important;
    font-weight: var(--block-title-text-weight) !important;
    height: 22px;
    margin-bottom: 8px !important;
}

::selection {
    background: #ffc078; !important;
}

#checkbox_train_dpo input[type="checkbox"]{
    margin-top: 6px;
}

#checkbox_train_dpo span {
    margin-top: 6px;
}

#checkbox_align_train {
    padding-top: 18px;
    padding-bottom: 18px;
}

#checkbox_align_infer input[type="checkbox"] {
    margin-top: 10px;
}

#checkbox_align_infer span {
    margin-top: 10px;
}

footer {
    height: 50px !important;           /* 设置页脚高度 */
    background-color: transparent !important; /* 背景透明 */
    display: flex;
    justify-content: center;           /* 居中对齐 */
    align-items: center;               /* 垂直居中 */
}

footer * {
    display: none !important;          /* 隐藏所有子元素 */
}

"""

top_html = """
<div align="center">
    <div style="margin-bottom: 10px; font-size: 20px;">{}</div>
    <div style="display: flex; gap: 30px; justify-content: center;">
        <a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-blue?logo=github&style=for-the-badge" style="width: auto; height: 40px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/中文教程-blue?logo=gitbook&style=for-the-badge" style="width: auto; height: 40px;">
        </a>
        <a href="https://rentry.co/GPT-SoVITS-guide#/" target="_blank">
            <img src="https://img.shields.io/badge/English%20Tutorial-blue?logo=gitbook&style=for-the-badge" style="width: auto; height: 40px;">
        </a>
        <a href="https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE" target="_blank">
            <img src="https://img.shields.io/badge/LICENSE-MIT-blue.svg?style=for-the-badge" style="width: auto; height: 40px;">
        </a>
    </div>
</div>
"""

info_html = '<p class="checkbox_info">{}</p>'

uvr5_html = """
<div align="center">
    <div style="margin-bottom: 10px; font-size: 20px;">{}</div>
</div>
<div>
    <p style="font-size: 18px;">{}</p>
</div>
"""

subfix_html = uvr5_html

seafoam = Seafoam()
