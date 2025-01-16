import traceback
from typing import Optional, Annotated, Literal, Any
from contextlib import asynccontextmanager
from functools import partial

from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, Query, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, PlainTextResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine
from tools.server.schema import TTSRequestAPI, TTSResponseFailed, SpeakerAPI
from tools.cfg import Prompt
from tools.server.api_utils import build_HTTPException
from tools.server.service import tts_handle, handle_control

SHARED_DOCS_TTS = "\
# Text To Speech\n\
## If the speaker has a valid prompt in `Speakers.json`, skip Ref Audio Path. Without prompt text or language, inference runs without reference.\n\
#### You can set prompt and speaker using other endpoints"

SHARED_RESPONSE_DICT: dict[int, dict[str, Any]] = {
    200: {
        "description": "Streaming audio content",
        "content": {
            "audio/wav": {"example": "WAV data"},
            "audio/aac": {"example": "AAC data"},
            "audio/ogg": {"example": "OGG data"},
            "audio/raw": {"example": "RAW PCM data"},
        },
    },
    400: {
        "description": "Plain text error response",
        "content": {"text/plain": {"example": "Invalid audio format requested"}},
    },
}

tts_engine: TTSEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_engine, tts_handle
    tts_engine: TTSEngine = app.state.tts_engine
    tts_handle = partial(tts_handle, tts_engine=tts_engine)
    yield
    print("Exited")


APP = FastAPI(title="GPT-SoVITS API", description="GPT-SoVITS API For Batch Inference", lifespan=lifespan)


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


@APP.get(
    "/tts", tags=["TTS"], summary="TTS_GET_Endpoint", description=SHARED_DOCS_TTS, response_class=StreamingResponse, responses=SHARED_RESPONSE_DICT
)
async def TTS_GET_Endpoint(result: Annotated[TTSRequestAPI, Depends(tts_handle)]):
    return result


@APP.post(
    "/tts", tags=["TTS"], summary="TTS_POST_Endpoint", description=SHARED_DOCS_TTS, response_class=StreamingResponse, responses=SHARED_RESPONSE_DICT
)
async def TTS_POST_Endpoint(result: Annotated[TTSRequestAPI, Depends(tts_handle)]):
    return result


@APP.post("/set_prompt", tags=["Setting Prompt"], summary="Set_Prompt")
async def Set_Prompt(prompt: Annotated[Prompt, Query()]):
    try:
        tts_engine.set_prompt(prompt)
        tts_engine.speaker.prompt = prompt
        tts_engine.speakers_cfg.save_as_json()
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.post("/add_speaker", tags=["Setting Speaker"], summary="Add_Speaker")
async def Add_Speaker(speaker: Annotated[SpeakerAPI, Query()]):
    try:
        tts_engine.add_speaker(spk_name=speaker.speaker_name, spk=speaker.model_dump(mode="python", exclude_none=True))
        tts_engine.speakers_cfg.save_as_json()
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_speaker", tags=["Setting Speaker"], summary="Set_Speaker_GET")
async def Set_Speaker_GET(speaker_name: str):
    try:
        tts_engine.set_speaker(speaker_name)
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.post("/set_speaker", tags=["Setting Speaker"], summary="Set_Speaker_POST")
async def Set_Speaker_POST(speaker_name: str):
    try:
        tts_engine.set_speaker(speaker_name)
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights", tags=["Setting Weights"], summary="Set_GPT_Weights_GET")
async def Set_GPT_Weights_GET(weights_path: Optional[str] = None):
    try:
        if weights_path in ["", None]:
            raise build_HTTPException(TTSResponseFailed(RuntimeError("gpt weight path is required")))
        if isinstance(weights_path, str):
            tts_engine.set_t2s(weights_path)
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights", tags=["Setting Weights"], summary="Set_SoVITS_Weights_GET")
async def Set_SoVITS_Weights_GET(weights_path: Optional[str] = None):
    try:
        if weights_path in ["", None]:
            raise build_HTTPException(TTSResponseFailed(RuntimeError("sovits weight path is required")))
        if isinstance(weights_path, str):
            tts_engine.set_vits(weights_path)
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/control", tags=["Control"])
async def Control(command: Literal["reastart", "exit"]):
    handle_control(command)
