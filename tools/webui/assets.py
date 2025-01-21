from typing import Iterable

import gradio.themes.base as ThemeBase
from gradio.themes.utils import colors, fonts, sizes

neutral_50_dark = "#fafafa"
neutral_100_dark = "#f5f5f5"
neutral_200_dark = "#e5e5e5"
neutral_300_dark = "#d4d4d4"
neutral_400_dark = "#a3a3a3"
neutral_500_dark = "#737373"
neutral_600_dark = "#525252"
neutral_700_dark = "#404040"
neutral_800_dark = "#262626"
neutral_900_dark = "#171717"
neutral_950_dark = "#0f0f0f"

primary_50_dark = "#eff6ff"
primary_100_dark = "#dbeafe"
primary_200_dark = "#bfdbfe"
primary_300_dark = "#93c5fd"
primary_400_dark = "#60a5fa"
primary_500_dark = "#3b82f6"
primary_600_dark = "#621dd8"
primary_700_dark = "#1d4ed8"
primary_800_dark = "#1e40af"
primary_900_dark = "#1e3a8a"
primary_950_dark = "#1d3660"

secondary_50_dark = "#ecfdf5"
secondary_100_dark = "#d1fae5"
secondary_200_dark = "#a7f3d0"
secondary_300_dark = "#6ee7b7"
secondary_400_dark = "#34d399"
secondary_500_dark = "#10b981"
secondary_600_dark = "#059669"
secondary_700_dark = "#047857"
secondary_800_dark = "#065f46"
secondary_900_dark = "#064e3b"
secondary_950_dark = "#054436"


border_color_primary_ = neutral_700_dark


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
            # Light
            body_background_fill="linear-gradient(45deg, #80eeee, #ecfdf5)",  # 浅色模式背景渐变
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_400, *secondary_500)",
            button_primary_text_color="black",
            button_cancel_background_fill="#ffa6a6",
            button_cancel_background_fill_hover="#ff8686",
            slider_color="*secondary_300",
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
        params.set('__theme', 'dark');
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

body .markdown {
    background-color: lightblue; /* 适用于 light 主题的颜色 */
    padding: 30px;
}

body.dark .markdown {
    background-color: darkblue; /* 适用于 dark 主题的颜色 */
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

if __name__ == "__main__":
    theme = Seafoam()
    theme.dump("Seafoam.json")
