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
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
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
            body_background_fill="linear-gradient(45deg, #80eeee, #ecfdf5)",  # 浅色模式背景渐变
            body_background_fill_dark="linear-gradient(45deg, #1c4d4d, #142d2d)",  # 深色模式背景渐变
            # body_background_fill="linear-gradient(45deg, *primary_200, *primary_50)",
            # body_background_fill_dark="linear-gradient(45deg, *primary_800, *primary_900)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_400, *secondary_500)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            # button_secondary_text_color="white",
            button_cancel_background_fill="#ffa6a6",
            button_cancel_background_fill_hover="#ff8686",
            # button_cancel_text_color="white",  # 按钮文字颜色
            # button_cancel_background_fill_dark="linear-gradient(90deg, #b22222, #8b0000)",  # 深色模式按钮渐变
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )


js = """
function createGradioAnimation() {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('__theme')) {
        params.set('__theme', 'light');
        window.location.search = params.toString();
    }
    
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

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
    padding: 20px;
}

/* CSSFontFaceRule */
@font-face {
    font-family: "test-font";
    src: url("https://mdn.github.io/css-examples/web-fonts/VeraSeBd.ttf") format("truetype");
}

/* CSSImportRule */
@import url("https://fonts.googleapis.com/css2?family=Protest+Riot&display=swap");

.markdown {
  font-family: "Protest Riot", sans-serif;
}

::selection {
    background: #ffc078; !important;
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
