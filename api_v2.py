"""
# WebAPI文档

` python api_v2.py -c GPT_SoVITS/configs/Cfg.json -s GPT_SoVITS/configs/Speakers_Cfg.json`

## 执行参数:
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/Cfg.yaml"`
    `-s` - `说话人配置文件路径,默认"GPT_SoVITS/configs/Speakers_Cfg.json"`
## 可选执行参数:
    `-a` - `绑定地址, 默认"::"`
    `-p` - `绑定端口, 默认9880`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(optional) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(optional) language of the prompt text for the reference audio
    "top_k": 5,                   # int.(optional) top k sampling
    "top_p": 1,                   # float.(optional) top p sampling
    "temperature": 1,             # float.(optional) temperature for sampling
    "text_split_method": "cut0",  # str.(optional) text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int.(optional) batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": False,      # bool. whether to return a streaming response.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
    "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400
    
"""

import os
import sys
import traceback
import argparse
import subprocess
import wave
import signal
from io import BytesIO
from typing import Generator, Optional, Annotated, Tuple, Literal

import torch
import uvicorn
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from numpy import int16
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, PlainTextResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTS_Engine
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from schema import TTS_Request_API, TTSResponse_Failed, SpeakerAPI
from tools.i18n.i18n import I18nAuto
from cfg import Prompt

i18n = I18nAuto()
cut_method_names = get_cut_method_names()

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="GPT-SoVITS API")
parser.add_argument("-c", "--api-config", type=str, default="GPT_SoVITS/configs/Cfg.json", help="API_Batch配置参数路径")
parser.add_argument("-s", "--speakers-config", type=str, default="GPT_SoVITS/configs/Speakers_Cfg.json", help="说话人配置参数路径")
# parser.add_argument("-a", "--host", type=str, default="127.0.0.1", help="default: 127.0.0.1")
# parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
# parser.add_argument("-d", "--device", type=str, default=device, help="推理设备")
# parser.add_argument("-hp", "--half_precision", action="store_true", default=torch.cuda.is_available(), help="使用半精度")

args = parser.parse_args()
cfg_path = args.api_config
speakers_cfg_path = args.speakers_config

if cfg_path in [None, ""]:
    cfg = "GPT_SoVITS/configs/Cfg.json"

if speakers_cfg_path in [None, ""]:
    cfg = "GPT_SoVITS/configs/Speakers.json"


argv = sys.argv
tts_pipeline = TTS_Engine.get_instance(cfg_name="api_batch_cfg", cfg_path=cfg_path, speakers_cfg_path=speakers_cfg_path)
print(tts_pipeline.configs)

host = tts_pipeline.configs.host
port = tts_pipeline.configs.port

APP = FastAPI()


def build_HTTPException(tts_response: TTSResponse_Failed):
    e = tts_response.exception
    tracebacks = tts_response.tracebacks

    if not tracebacks:
        return HTTPException(status_code=400, detail=f"\n\n{type(e).__name__}: {e}")
    else:
        return HTTPException(status_code=400, detail=f"\n\n{tracebacks}")


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, _: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(channels=1, sample_width=2, sample_rate=32000, frame_input=None):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        if frame_input:
            vfout.writeframes(frame_input)
    wav = wav_buf.getvalue()
    wav_buf.close()
    return wav


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


async def streaming_generator(tts_generator: Generator[Tuple[int, NDArray[int16]], None, None], media_type: str):
    for idx, tts_response in enumerate(tts_generator):
        sr, chunk = tts_response
        if idx == 0 and media_type == "wav":
            yield wave_header_chunk()
            media_type = "raw"
        yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()


async def base_generator(audio):
    yield audio


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (TTS_Request):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "speaker": "API_Batch"        # str.(optional)
                "ref_audio_path": "",         # str.(required if not set in Speakers_Cfg) reference audio path
                "prompt_text": "",            # str.(required if not set in Speakers_Cfg) prompt text for the reference audio
                "prompt_lang": "",            # str.(required if not set in Speakers_Cfg) language of the prompt text for the reference audio
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "top_k": 5,                   # int.(optional) top k sampling
                "top_p": 1,                   # float.(optional) top p sampling
                "temperature": 1,             # float.(optional) temperature for sampling
                "text_split_method": "cut5",  # str.(optional) text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int.(optional) batch size for inference
                "batch_threshold": 0.75,      # float.(optional) threshold for batch splitting.
                "split_bucket: True,          # bool.(optional) whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float.(optional) control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float.(optional) to control the interval of the audio fragment.
                "seed": -1,                   # int.(optional) random seed for reproducibility.
                "media_type": "wav",          # str.(optional) media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool.(optional) whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.pop("streaming_mode", tts_pipeline.configs.streaming)
    media_type = req.pop("media_type", tts_pipeline.configs.media_type)

    req["return_fragment"] = streaming_mode

    if media_type == "ogg" and not streaming_mode:
        raise build_HTTPException(TTSResponse_Failed(RuntimeError("ogg format is not supported in non-streaming mode")))

    tts_generator = tts_pipeline(exception_handler=build_HTTPException, **req)
    next(tts_generator)  # Surface Exceptions Early.
    if streaming_mode:
        return StreamingResponse(
            streaming_generator(
                tts_generator,
                media_type,
            ),
            media_type=f"audio/{media_type}",
        )
    else:
        audio = await anext(streaming_generator(tts_generator, media_type))
        return StreamingResponse(
            base_generator(audio),
            media_type=f"audio/{media_type}",
        )


@APP.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("The client sent invalid data!:")
    for item in exc.errors():
        print(item)
    return await request_validation_exception_handler(request, exc)


@APP.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    print(f"HTTP Error: {str(exc)}")
    return PlainTextResponse(exc.detail, media_type="text/plain; charset=utf-8")


@APP.exception_handler(404)
async def Redirect_404(*args, **kwds):
    return RedirectResponse(url="/docs")


@APP.get("/", include_in_schema=False)
async def redirect_root_to_docs_get():
    return RedirectResponse(url="/docs")


@APP.post("/", include_in_schema=False)
async def redirect_root_to_doc_post():
    return RedirectResponse(url="/docs")


@APP.middleware("http")
async def add_charset_to_json_response(request: Request, call_next):
    response = await call_next(request)
    if isinstance(response, JSONResponse):
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


@APP.get("/tts", tags=["TTS"], summary="TTS_GET_Endpoint")
async def TTS_GET_Endpoint(tts_req: Annotated[TTS_Request_API, Query()]):
    """
    Text To Speech

    If the speaker has a valid prompt in `Speakers.json`, skip Ref Audio Path. Without prompt text or language, inference runs without reference.

    You can set prompt and speaker using other endpoints
    """
    req = tts_req.model_dump(mode="python", exclude_none=True)
    return await tts_handle(req)


@APP.post("/tts", tags=["TTS"], summary="TTS_POST_Endpoint")
async def TTS_POST_Endpoint(request: Annotated[TTS_Request_API, Query()]):
    """
    Text To Speech

    If the speaker has a valid prompt in `Speakers.json`, skip Ref Audio Path. Without prompt text or language, inference runs without reference.

    You can set prompt and speaker using other endpoints
    """
    req = request.model_dump(mode="python", exclude_none=True)
    return await tts_handle(req)


@APP.post("/set_prompt", tags=["Setting Prompt"], summary="Set_Prompt")
async def Set_Prompt(request: Annotated[Prompt, Query()]):
    try:
        tts_pipeline.set_prompt(request)
        tts_pipeline.speaker.prompt = request
        tts_pipeline.speakers_cfg.save_as_json()
    except Exception as e:
        raise build_HTTPException(TTSResponse_Failed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.post("/add_speaker", tags=["Setting Speaker"], summary="Add_Speaker")
async def Add_Speaker(speaker: Annotated[SpeakerAPI, Query()]):
    try:
        tts_pipeline.add_speaker(spk_name=speaker.speaker_name, spk=speaker)
        tts_pipeline.speakers_cfg.save_as_json()
    except Exception as e:
        raise build_HTTPException(TTSResponse_Failed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_speaker", tags=["Setting Speaker"], summary="Set_Speaker_GET")
async def Set_Speaker_GET(speaker_name: str):
    try:
        tts_pipeline.set_speaker(speaker_name)
    except Exception as e:
        raise build_HTTPException(TTSResponse_Failed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.post("/set_speaker", tags=["Setting Speaker"], summary="Set_Speaker_POST")
async def Set_Speaker_POST(speaker_name: str):
    try:
        tts_pipeline.set_speaker(speaker_name)
    except Exception as e:
        raise build_HTTPException(TTSResponse_Failed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights", tags=["Setting Weights"], summary="Set_GPT_Weights_GET")
async def Set_GPT_Weights_GET(weights_path: Optional[str] = None):
    try:
        if weights_path in ["", None]:
            raise build_HTTPException(TTSResponse_Failed(RuntimeError("gpt weight path is required")))
        if isinstance(weights_path, str):
            tts_pipeline.set_t2s(weights_path)
    except Exception as e:
        raise build_HTTPException(TTSResponse_Failed(e, traceback.format_exc()))

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights", tags=["Setting Weights"], summary="Set_SoVITS_Weights_GET")
async def Set_SoVITS_Weights_GET(weights_path: Optional[str] = None):
    try:
        if weights_path in ["", None]:
            raise build_HTTPException(TTSResponse_Failed(RuntimeError("sovits weight path is required")))
        if isinstance(weights_path, str):
            tts_pipeline.set_vits(weights_path)
    except Exception as e:
        raise build_HTTPException(TTSResponse_Failed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/control", tags=["Control"])
async def Control(command: Optional[Literal["reastart", "exit"]] = None):
    if command is None:
        raise build_HTTPException(TTSResponse_Failed(RuntimeError("command is required")))
    handle_control(command)


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = "::"
        uuvicorn_config = uvicorn.Config(app=APP, host=host, port=port, log_level="error", access_log=False)
        server = uvicorn.Server(uuvicorn_config)
        server.run()
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        sys.exit(0)
