import logging

import click

from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine
from tools.webui.inference.layout import build_app
from tools.webui.inference.utils import parse_args, build_gradio_exception, list_root_directories

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

ROOT_DIR = list_root_directories()


@click.command(
    name="Infer-WebUI",
)
@click.option(
    "-c",
    "--infer-config",
    default="tools/cfgs/cfg.json",
    metavar="<Path>",
    type=click.Path(exists=True, dir_okay=False, readable=True, writable=True),
    help="Path of Infer-WebUI config file",
    show_default=True,
)
@click.option(
    "-s",
    "--speakers-config",
    default="tools/cfgs/speakers.json",
    metavar="<Path>",
    type=click.Path(exists=True, dir_okay=False, readable=True, writable=True),
    help="Path of speakers config file",
    show_default=True,
)
def main(
    infer_config="tools/cfgs/cfg.json",
    speakers_config="tools/cfgs/speakers.json",
    compile=False,
):  # pylint: disable=redefined-builtin
    cfg_path = infer_config
    speakers_cfg_path = speakers_config

    if cfg_path in [None, ""]:
        cfg_path = "tools/cfgs/cfg.json"

    if speakers_cfg_path in [None, ""]:
        speakers_cfg_path = "tools/cfgs/speakers.json"

    tts_engine = TTSEngine.get_instance(
        cfg_name="inference_webui_cfg",
        cfg_path=cfg_path,
        speakers_cfg_path=speakers_cfg_path,
        compile=compile,
        exception_handler=build_gradio_exception,
    )
    configs = tts_engine.configs
    print(tts_engine.configs)

    app = build_app(tts_engine=tts_engine, compile=compile)

    app.queue().launch(
        server_name=configs.host,
        inbrowser=True,
        share=configs.gradio_share,
        server_port=configs.port,
        quiet=True,
        allowed_paths=ROOT_DIR,
        # ssr_mode=True,
    )


if __name__ == "__main__":
    main()
