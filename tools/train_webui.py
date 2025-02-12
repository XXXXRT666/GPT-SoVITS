import logging
import warnings

from tools.cfg import Cfg
from tools.webui.train.layout import build_app
from tools.webui.train.utils import parse_args, build_gradio_exception, list_root_directories

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)

ROOT_DIR = list_root_directories()
warnings.filterwarnings("ignore")


def main():
    args = parse_args()
    cfg_path = args.webui_config

    if cfg_path in [None, ""]:
        cfg_path = "tools/cfgs/cfg.json"

    configs = Cfg.from_json(cfg_path).train_webui_cfg
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
