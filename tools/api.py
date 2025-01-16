import os
import sys
import traceback
import signal

import uvicorn

from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine
from tools.server.routes import APP
from tools.server.api_utils import parse_args


def main():
    args = parse_args()
    cfg_path = args.api_config
    speakers_cfg_path = args.speakers_config

    if cfg_path in [None, ""]:
        cfg_path = "GPT_SoVITS/configs/Cfg.json"

    if speakers_cfg_path in [None, ""]:
        speakers_cfg_path = "GPT_SoVITS/configs/Speakers.json"

    tts_engine = TTSEngine.get_instance(cfg_name="api_batch_cfg", cfg_path=cfg_path, speakers_cfg_path=speakers_cfg_path)
    print(tts_engine.configs)

    host = tts_engine.configs.host
    port = tts_engine.configs.port

    APP.state.tts_engine = tts_engine

    try:
        uvicorn_config = uvicorn.Config(app=APP, host=host, port=port, log_level="error", access_log=False)
        server = uvicorn.Server(uvicorn_config)
        server.run()
    except Exception as e:
        print(e)
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        sys.exit(0)


if __name__ == "__main__":
    main()
