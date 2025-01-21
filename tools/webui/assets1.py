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
            background_fill_primary_dark="#0f0f0f",
            background_fill_primary="white",
            background_fill_secondary_dark="#171717",
            background_fill_secondary="#eff6ff",
            block_background_fill_dark="#262626",
            block_background_fill="white",
            block_border_color_dark="#404040",
            block_border_color="#bfdbfe",
            block_border_width_dark="3px",
            block_border_width="3px",
            block_info_text_color_dark="#d4d4d4",
            block_info_text_color="#60a5fa",
            block_label_background_fill_dark="#621dd8",
            block_label_background_fill="white",
            block_label_border_color_dark="#404040",
            block_label_border_color="#bfdbfe",
            block_label_border_width_dark="1px",
            block_label_border_width="1px",
            block_label_text_color_dark="*white",
            block_label_text_color="#3b82f6",
            block_shadow_dark="*shadow_drop_lg",
            block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            block_title_background_fill_dark="#621dd8",
            block_title_background_fill="none",
            block_title_border_color_dark="none",
            block_title_border_color="none",
            block_title_border_width_dark="0px",
            block_title_border_width="0px",
            block_title_text_color_dark="*white",
            block_title_text_color="#3b82f6",
            body_background_fill_dark="linear-gradient(to right, #2C5364, #203A43, #0F2027);",  # 深色模式背景渐变
            body_background_fill="linear-gradient(45deg, #80eeee, #ecfdf5)",
            body_text_color_dark="#f5f5f5",
            body_text_color="#1e40af",
            body_text_color_subdued_dark="#d4d4d4",
            body_text_color_subdued="#60a5fa",
            border_color_accent_dark="#525252",
            border_color_accent="#6ee7b7",
            border_color_primary_dark="#404040",
            border_color_primary="#bfdbfe",
            border_color_accent_subdued_dark="#525252",
            border_color_accent_subdued="#6ee7b7",
            button_border_width_dark="0px",
            button_border_width="0px",
            button_cancel_background_fill_dark="#525252",
            button_cancel_background_fill="#ffa6a6",
            button_cancel_background_fill_hover_dark="#3b82f6",
            button_cancel_background_fill_hover="#ff8686",
            button_cancel_border_color_dark="#525252",
            button_cancel_border_color="#bfdbfe",
            button_cancel_border_color_hover_dark="#525252",
            button_cancel_border_color_hover="#bfdbfe",
            button_cancel_text_color_dark="white",
            button_cancel_text_color="black",
            button_cancel_text_color_hover_dark="white",
            button_cancel_text_color_hover="black",
            button_primary_background_fill_dark="#1d4ed8",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover_dark="#3b82f6",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_400, *secondary_500)",
            button_primary_border_color_dark="#621dd8",
            button_primary_border_color="#10b981",
            button_primary_border_color_hover_dark="#621dd8",
            button_primary_border_color_hover="#10b981",
            button_primary_text_color_dark="white",
            button_primary_text_color="black",
            button_primary_text_color_hover_dark="white",
            button_primary_text_color_hover="black",
            button_secondary_background_fill_dark="#525252",
            button_secondary_background_fill="#bfdbfe",
            button_secondary_background_fill_hover_dark="#3b82f6",
            button_secondary_background_fill_hover="#93c5fd",
            button_secondary_border_color_dark="#525252",
            button_secondary_border_color="#bfdbfe",
            button_secondary_border_color_hover_dark="#525252",
            button_secondary_border_color_hover="#bfdbfe",
            button_secondary_text_color_dark="white",
            button_secondary_text_color="black",
            button_secondary_text_color_hover_dark="white",
            button_secondary_text_color_hover="black",
            checkbox_background_color_dark="#262626",
            checkbox_background_color="white",
            checkbox_background_color_focus_dark="#262626",
            checkbox_background_color_focus="white",
            checkbox_background_color_hover_dark="#262626",
            checkbox_background_color_hover="white",
            checkbox_background_color_selected_dark="#1d4ed8",
            checkbox_background_color_selected="#10b981",
            checkbox_border_color_dark="#525252",
            checkbox_border_color="#93c5fd",
            checkbox_border_color_focus_dark="#621dd8",
            checkbox_border_color_focus="#10b981",
            checkbox_border_color_hover_dark="#525252",
            checkbox_border_color_hover="#93c5fd",
            checkbox_border_color_selected_dark="#1d4ed8",
            checkbox_border_color_selected="#10b981",
            checkbox_border_width_dark="0px",
            checkbox_border_width="0px",
            checkbox_label_background_fill_dark="#525252",
            checkbox_label_background_fill="#bfdbfe",
            checkbox_label_background_fill_hover_dark="#3b82f6",
            checkbox_label_background_fill_hover="#93c5fd",
            checkbox_label_background_fill_selected_dark="#621dd8",
            checkbox_label_background_fill_selected="#bfdbfe",
            checkbox_label_border_color_dark="#404040",
            checkbox_label_border_color="#bfdbfe",
            checkbox_label_border_color_hover_dark="#404040",
            checkbox_label_border_color_hover="#bfdbfe",
            checkbox_label_border_width_dark="0px",
            checkbox_label_border_width="0px",
            checkbox_label_text_color_dark="#f5f5f5",
            checkbox_label_text_color="#1e40af",
            checkbox_label_text_color_selected_dark="#f5f5f5",
            checkbox_label_text_color_selected="#1e40af",
            color_accent_soft_dark="#404040",
            color_accent_soft="#ecfdf5",
            error_background_fill_dark="#0f0f0f",
            error_background_fill="#fef2f2",
            error_border_color_dark="#404040",
            error_border_color="#b91c1c",
            error_border_width_dark="1px",
            error_border_width="1px",
            error_icon_color_dark="#ef4444",
            error_icon_color="#b91c1c",
            error_text_color_dark="#ef4444",
            error_text_color="#b91c1c",
            input_background_fill_dark="#262626",
            input_background_fill="#dbeafe",
            input_background_fill_focus_dark="#404040",
            input_background_fill_focus="#dbeafe",
            input_background_fill_hover_dark="#262626",
            input_background_fill_hover="#dbeafe",
            input_border_color_dark="#404040",
            input_border_color="#bfdbfe",
            input_border_color_focus_dark="#404040",
            input_border_color_focus="#93c5fd",
            input_border_color_hover_dark="#404040",
            input_border_color_hover="#bfdbfe",
            input_border_width_dark="0px",
            input_border_width="0px",
            input_placeholder_color_dark="#737373",
            input_placeholder_color="#60a5fa",
            input_shadow_dark="rgba(0,0,0,0.05) 1px 2px 3px 1px",
            input_shadow="none",
            input_shadow_focus_dark="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            input_shadow_focus="none",
            link_text_color_active_dark="#10b981",
            link_text_color_active="#2563eb",
            link_text_color_dark="#10b981",
            link_text_color="#2563eb",
            link_text_color_hover_dark="#34d399",
            link_text_color_hover="#1d4ed8",
            link_text_color_visited_dark="#059669",
            link_text_color_visited="#3b82f6",
            loader_color_dark="#404040",
            loader_color="#10b981",
            panel_background_fill_dark="#171717",
            panel_background_fill="#eff6ff",
            panel_border_color_dark="#404040",
            panel_border_color="#bfdbfe",
            panel_border_width_dark="1px",
            panel_border_width="0",
            shadow_spread_dark="1px",
            shadow_spread="3px",
            slider_color_dark="#621dd8",
            slider_color="#93c5fd",
            stat_background_fill_dark="#3b82f6",
            stat_background_fill="#6ee7b7",
            table_border_color_dark="#404040",
            table_border_color="#93c5fd",
            table_even_background_fill_dark="#0f0f0f",
            table_even_background_fill="white",
            table_odd_background_fill_dark="#171717",
            table_odd_background_fill="#eff6ff",
            table_row_focus_dark="#404040",
            table_row_focus="#ecfdf5",
            accordion_text_color="#1e40af",
            block_info_text_size="14px",
            block_info_text_weight="400",
            block_label_margin="0",
            block_label_padding="2px 4px",
            block_label_radius="calc(*radius_sm - 1px) 0 calc(*radius_sm - 1px) 0",
            block_label_right_radius="0 calc(*radius_sm - 1px) 0 calc(*radius_sm - 1px)",
            block_label_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            block_label_text_size="14px",
            block_label_text_weight="400",
            block_padding="*spacing_xl calc(*spacing_xl + 2px)",
            block_radius="4px",
            block_title_padding="0",
            block_title_radius="4px",
            block_title_text_size="16px",
            block_title_text_weight="600",
            body_text_size="16px",
            body_text_weight="400",
            button_large_padding="32px",
            button_large_radius="6px",
            button_large_text_size="20px",
            button_large_text_weight="600",
            button_medium_padding="*spacing_md calc(2 * *spacing_md)",
            button_medium_radius="6px",
            button_medium_text_size="16px",
            button_medium_text_weight="600",
            button_primary_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            button_primary_shadow_active="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            button_primary_shadow_hover="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            button_secondary_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            button_secondary_shadow_active="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            button_secondary_shadow_hover="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            button_small_padding="*spacing_sm calc(1.5 * *spacing_sm)",
            button_small_radius="6px",
            button_small_text_size="14px",
            button_small_text_weight="400",
            button_transform_active="none",
            button_transform_hover="none",
            button_transition="all 0.2s ease",
            chatbot_text_size="20px",
            checkbox_border_radius="4px",
            checkbox_label_border_color_selected="#bfdbfe",
            checkbox_label_gap="8px",
            checkbox_label_padding="*spacing_md calc(2 * *spacing_md)",
            checkbox_label_shadow="none",
            checkbox_label_text_size="16px",
            checkbox_label_text_weight="400",
            checkbox_shadow="none",
            code_background_fill="#dbeafe",
            color_accent="#10b981",
            container_radius="4px",
            embed_radius="4px",
            form_gap_width="0px",
            input_padding="10px",
            input_radius="4px",
            input_text_size="16px",
            input_text_weight="400",
            layout_gap="16px",
            prose_header_text_weight="600",
            prose_text_size="16px",
            prose_text_weight="400",
            section_header_text_size="16px",
            section_header_text_weight="400",
            shadow_drop="rgba(0,0,0,0.05) 0px 1px 2px 0px",
            shadow_drop_lg="0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
            shadow_inset="rgba(0,0,0,0.05) 0px 2px 4px 0px inset",
            table_radius="4px",
            table_text_color="#1e40af",
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

seafoam = Seafoam(spacing_size=sizes.Size("1px", "1px", "2px", "4px", "6px", "9px", "12px"))

if __name__ == "__main__":
    theme = Seafoam()
    theme.dump("Seafoam.json")
