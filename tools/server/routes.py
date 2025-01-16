from typing import Annotated, Literal, Any, Union

from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, PlainTextResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from tools.server.schema import TTSRequestAPI, SpeakerAPI
from tools.cfg import Prompt
from tools.server.service import tts_handle, handle_control, set_prompt, add_speaker, set_speaker, set_gpt_weights, set_sovits_weights

SHARED_DOCS_TTS = "\
# Text To Speech\n\
## If the speaker has a valid prompt in `Speakers.json`, skip Ref Audio Path. Without prompt text or language, inference runs without reference.\n\
#### You can set prompt and speaker using other endpoints"

SHARED_RESPONSE_DICT: dict[Union[int, str], dict[str, Any]] = {
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


APP = FastAPI(title="GPT-SoVITS API", description="GPT-SoVITS API For Batch Inference")


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
async def Set_Prompt(result: Annotated[Prompt, Depends(set_prompt)]):
    return result


@APP.post("/add_speaker", tags=["Setting Speaker"], summary="Add_Speaker")
async def Add_Speaker(result: Annotated[SpeakerAPI, Depends(add_speaker)]):
    return result


@APP.get("/set_speaker", tags=["Setting Speaker"], summary="Set_Speaker_GET")
async def Set_Speaker_GET(result: Annotated[str, Depends(set_speaker)]):
    return result


@APP.post("/set_speaker", tags=["Setting Speaker"], summary="Set_Speaker_POST")
async def Set_Speaker_POST(result: Annotated[str, Depends(set_speaker)]):
    return result


@APP.get("/set_gpt_weights", tags=["Setting Weights"], summary="Set_GPT_Weights_GET")
async def Set_GPT_Weights_GET(result: Annotated[str, Depends(set_gpt_weights)]):
    return result


@APP.post("/set_gpt_weights", tags=["Setting Weights"], summary="Set_GPT_Weights_POST")
async def Set_GPT_Weights_POST(result: Annotated[str, Depends(set_gpt_weights)]):
    return result


@APP.get("/set_sovits_weights", tags=["Setting Weights"], summary="Set_SoVITS_Weights_GET")
async def Set_SoVITS_Weights_GET(result: Annotated[str, Depends(set_sovits_weights)]):
    return result


@APP.get("/set_sovits_weights", tags=["Setting Weights"], summary="Set_SoVITS_Weights_POST")
async def Set_SoVITS_Weights_POST(result: Annotated[str, Depends(set_sovits_weights)]):
    return result


@APP.get("/control", tags=["Control"])
async def Control(command: Literal["reastart", "exit"]):
    await handle_control(command)
