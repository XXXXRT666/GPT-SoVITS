import logging

from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine
from tools.webui.inference.layout import build_app
from tools.webui.inference.utils import parse_args, build_gradio_exception

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)


def main():
    args = parse_args()
    cfg_path = args.webui_config
    speakers_cfg_path = args.speakers_config
    compile = args.compile

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

    app.queue().launch(server_name=configs.host, inbrowser=True, share=configs.gradio_share, server_port=configs.port, quiet=True)


if __name__ == "__main__":
    main()
