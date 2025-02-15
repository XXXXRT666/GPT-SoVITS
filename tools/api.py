import os
import signal
import sys
import traceback

import click
import uvicorn

from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine
from tools.server.routes import build_APP
from tools.server.utils import build_HTTPException


@click.command(name="API", help="GPT-SoVITS API")
@click.option(
    "-c",
    "--api-config",
    default="tools/cfgs/cfg.json",
    type=str,
    help="Path of api config file",
    show_default=True,
)
@click.option(
    "-s",
    "--speakers-config",
    default="tools/cfgs/speakers.json",
    type=str,
    help="Path of speakers config file",
    show_default=True,
)
@click.option("--compile", is_flag=True, help="Compiled the model to accelerate")
def main(
    api_config: str = "tools/cfgs/cfg.json",
    speakers_config: str = "tools/cfgs/speakers.json",
    compile=False,
):  # pylint: disable=redefined-builtin
    """Batched Inference API for GPT-SoVITS

    The command-line arguments accept two OPTIONAL configuration file paths

    The api-config configures some FastAPI parameters and the default settings for inference, while speakers-config configures speaker information.

    If the default config path doesn't exist, create a new config file

    Compile will accelerate the inference with fixed batch size, but it takes a moment to initialized

    Go to https//127.0.0.1:{api_port} for more information after starting the FastAPI application.
    """

    cfg_path = api_config
    speakers_cfg_path = speakers_config

    if cfg_path in [None, ""]:
        cfg_path = "tools/cfgs/cfg.json"

    if speakers_cfg_path in [None, ""]:
        speakers_cfg_path = "tools/cfgs/speakers.json"

    tts_engine = TTSEngine.get_instance(
        cfg_name="api_batch_cfg", cfg_path=cfg_path, speakers_cfg_path=speakers_cfg_path, compile=compile, exception_handler=build_HTTPException
    )
    print(tts_engine.configs)

    host = tts_engine.configs.host
    port = tts_engine.configs.port

    APP = build_APP(compile=compile)
    APP.state.TTSEngine = tts_engine

    try:
        uvicorn_config = uvicorn.Config(app=APP, host=host, port=port, log_level="info", access_log=False)
        server = uvicorn.Server(uvicorn_config)
        server.run()
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(e)
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        sys.exit(0)


if __name__ == "__main__":
    main()
