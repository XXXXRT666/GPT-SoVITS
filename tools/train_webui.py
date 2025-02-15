import logging
import warnings

import click

from tools.cfg import Cfg
from tools.webui.train.layout import build_app
from tools.webui.train.utils import build_gradio_exception, list_root_directories

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)

ROOT_DIR = list_root_directories()
warnings.filterwarnings("ignore")


@click.command(
    name="Train-WebUI",
)
@click.option(
    "-c",
    "webui-cfg",
    default="tools/cfgs/cfg.json",
    type=click.Path(exists=False, dir_okay=False, readable=True, writable=True),
    help="Train WebUI Cfg Path",
    show_default=True,
)
def main(webui_cfg="tools/cfgs/cfg.json"):
    """Train WebUI for GPT-SoVITS

    The command-line arguments accept one OPTIONAL configuration file paths

    The webui-config configures some Gradio parameters and the default settings for training

    If the default config path doesn't exist, create a new config file
    """
    if webui_cfg in [None, ""]:
        webui_cfg = "tools/cfgs/cfg.json"

    configs = Cfg.from_json(webui_cfg).train_webui_cfg
    print(configs)

    app = build_app(configs)

    app.queue().launch(
        server_name=configs.host,
        inbrowser=True,
        share=configs.gradio_share,
        server_port=1111,
        quiet=False,
        allowed_paths=ROOT_DIR,
    )


if __name__ == "__main__":
    main()
