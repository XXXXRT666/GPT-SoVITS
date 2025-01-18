from typing import Annotated, Literal, Any, Union

from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse, PlainTextResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from tools.server.schema import SpeakerAPI
from tools.cfg import Prompt
from tools.server.service import (
    tts_handle_query,
    tts_handle_compiled_query,
    handle_control_query,
    set_prompt_query,
    add_speaker_query,
    del_speaker_query,
    list_speaker_query,
    get_speaker_query,
    set_speaker_query,
    set_gpt_weights_query,
    set_sovits_weights_query,
)
from tools.server.service import (
    tts_handle_body,
    tts_handle_compiled_body,
    handle_control_body,
    set_prompt_body,
    add_speaker_body,
    del_speaker_body,
    list_speaker_body,
    get_speaker_body,
    set_speaker_body,
    set_gpt_weights_body,
    set_sovits_weights_body,
)

SHARED_DOCS_API = """
# Text To Speech API Documentation

## API Config File

- **Default:** tools/cfgs/cfg.json

- You can set parameters for API like **host**, **port**, and **device** there

- Also, some defaults value for TTS could be set there, such as **top_k**, **streaming** and **media_type**

## Speaker Config File

- **Default:** tools/cfgs/speakers.json

- You can set **Speaker** or **Prompt** in the file or using API endpoints

- Each Speaker must contains **version** and the path of **GPT** and **SoVITS** model

- The **Prompt** themselves are optional. However, if you want to set the **Prompt**, the **Ref Audio Path** must be specified, while **Prompt Text**, **Prompt Language** and **Aux Ref Audio Paths** are optional.
"""

SHARED_DOCS_TTS = """
# Text To Sppech Endpoint

##  **Behavior Logic**
- If a valid **Prompt** exists in **Speaker** from speaker config file and **Ref Audio Path** is not provided, the **prompt** from the speaker config file will be used.

- If **Ref Audio Path** is provided, it takes precedence, and the **speakers.json** prompt will be ignored.

- If the **Prompt** from **Speaker** or the **Ref Audio Path** from request provides a valid **prompt text** ot **prompt language**, the inference will run with prompt free mode

## Response
- **Success:** Returns the generated speech in stream
- **Error:** Provides clear error messages or traceback
"""

SHARED_DOCS_ADD_SPEAKER = """
## Just For Reminder, This Endpoint **Won't Set** the `Speaker`
"""

SHARED_DOCS_LIST_SPEAKER = """
## Return the list of speaker names, if not exists, return an empty speaker
"""

SHARED_DOCS_GET_SPEAKER = """
## Return the views of this speaker, if not exists, return an empty speaker
"""

SHARED_DOCS_SET_PROMPT = """
## Just For Reminder, This Endpoint **Will Cover** the `Prompt` of Current Speaker in speakers cfg file
"""

SHARED_DOCS_SET_GPT = """
## Just For Reminder, This Endpoint **Will Cover** the `GPT Path` of Current Speaker in speakers cfg file
"""

SHARED_DOCS_SET_SOVITS = """
## Just For Reminder, This Endpoint **Will Cover** the `SoVITS Path` of Current Speaker in speakers cfg file
"""

SHARED_TTS_RESPONSE_DICT: dict[Union[int, str], dict[str, Any]] = {
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
        "content": {"text/plain": {"example": "Error message or tracebacks"}},
    },
}

SHARED_OTHER_RESPONSE_DICT: dict[Union[int, str], dict[str, Any]] = {
    200: {
        "description": "Success message",
        "content": {
            "application/json": {"example": "success"},
        },
    },
    400: {
        "description": "Plain text error response",
        "content": {"text/plain": {"example": "Error message or tracebacks"}},
    },
}

SHARED_LIST_RESPONSE_DICT: dict[Union[int, str], dict[str, Any]] = {
    200: {
        "description": "Success message",
        "content": {
            "application/json": {"example": "[spk1, spk2, ...]"},
        },
    },
    400: {
        "description": "Plain text error response",
        "content": {"text/plain": {"example": "Error message or tracebacks"}},
    },
}

SHARED_GET_RESPONSE_DICT: dict[Union[int, str], dict[str, Any]] = {
    200: {
        "description": "Success message",
        "content": {
            "application/json": {"example": "spk1:{t2s_path: ..., ...}"},
        },
    },
    400: {
        "description": "Plain text error response",
        "content": {"text/plain": {"example": "Error message or tracebacks"}},
    },
}


def build_APP(compile: bool = False):
    APP = FastAPI(
        title="GPT-SoVITS API",
        description=SHARED_DOCS_API,
        version="0.1.0",
        terms_of_service="https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE",
    )

    @APP.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        print("The client sent invalid data!:")
        for item in exc.errors():
            if isinstance(item, dict):
                print("{\n" + "\n".join([str(k) + ":" + str(v) for k, v in item.items()]) + "\n}")
            else:
                print(item)
        return await request_validation_exception_handler(request, exc)

    @APP.exception_handler(StarletteHTTPException)
    async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
        print(f"HTTP Error: {str(exc)}")
        return PlainTextResponse(exc.detail, media_type="text/plain; charset=utf-8")

    @APP.exception_handler(404)
    async def Redirect_404(*args, **kwds):
        return RedirectResponse(url="/docs")

    @APP.middleware("http")
    async def add_charset_to_json_response(request: Request, call_next):
        response = await call_next(request)
        if isinstance(response, JSONResponse):
            response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    @APP.get("/", include_in_schema=False)
    async def redirect_root_to_docs_get():
        return RedirectResponse(url="/docs")

    @APP.post("/", include_in_schema=False)
    async def redirect_root_to_doc_post():
        return RedirectResponse(url="/docs")

    @APP.get(
        "/tts",
        tags=["TTS"],
        summary="TTS_GET_Endpoint",
        description=SHARED_DOCS_TTS,
        response_class=StreamingResponse,
        responses=SHARED_TTS_RESPONSE_DICT,
        include_in_schema=not compile,
    )
    async def TTS_GET_Endpoint(result=Depends(tts_handle_query)):
        return result

    @APP.post(
        "/tts",
        tags=["TTS"],
        summary="TTS_POST_Endpoint",
        description=SHARED_DOCS_TTS,
        response_class=StreamingResponse,
        responses=SHARED_TTS_RESPONSE_DICT,
        include_in_schema=not compile,
    )
    async def TTS_POST_Endpoint(result=Depends(tts_handle_body)):
        return result

    @APP.get(
        "/tts",
        tags=["TTS"],
        summary="TTS_GET_Endpoint_Compile",
        description=SHARED_DOCS_TTS,
        response_class=StreamingResponse,
        responses=SHARED_TTS_RESPONSE_DICT,
        include_in_schema=compile,
    )
    async def TTS_GET_Endpoint_Compile(result=Depends(tts_handle_compiled_query)):
        return result

    @APP.post(
        "/tts",
        tags=["TTS"],
        summary="TTS_POST_Endpoint_Compile",
        description=SHARED_DOCS_TTS,
        response_class=StreamingResponse,
        responses=SHARED_TTS_RESPONSE_DICT,
        include_in_schema=compile,
    )
    async def TTS_POST_Endpoint_Compile(result=Depends(tts_handle_compiled_body)):
        return result

    @APP.get(
        "/set_prompt", tags=["Setting Prompt"], summary="Set_Prompt_GET", description=SHARED_DOCS_SET_PROMPT, responses=SHARED_OTHER_RESPONSE_DICT
    )
    async def Set_Prompt_GET(result: Annotated[Prompt, Depends(set_prompt_query)]):
        return result

    @APP.post(
        "/set_prompt", tags=["Setting Prompt"], summary="Set_Prompt_POST", description=SHARED_DOCS_SET_PROMPT, responses=SHARED_OTHER_RESPONSE_DICT
    )
    async def Set_Prompt_POST(result: Annotated[Prompt, Depends(set_prompt_body)]):
        return result

    @APP.get(
        "/add_speaker", tags=["Setting Speaker"], summary="Add_Speaker_GET", description=SHARED_DOCS_ADD_SPEAKER, responses=SHARED_OTHER_RESPONSE_DICT
    )
    async def Add_Speaker_GET(result: Annotated[SpeakerAPI, Depends(add_speaker_query)]):
        return result

    @APP.post(
        "/add_speaker",
        tags=["Setting Speaker"],
        summary="Add_Speaker_POST",
        description=SHARED_DOCS_ADD_SPEAKER,
        responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Add_Speaker_POST(result: Annotated[SpeakerAPI, Depends(add_speaker_body)]):
        return result

    @APP.get("/del_speaker", tags=["Setting Speaker"], summary="Del_Speaker_GET", responses=SHARED_OTHER_RESPONSE_DICT)
    async def Del_Speaker_GET(result: Annotated[SpeakerAPI, Depends(del_speaker_query)]):
        return result

    @APP.post("/del_speaker", tags=["Setting Speaker"], summary="Del_Speaker_POST", responses=SHARED_OTHER_RESPONSE_DICT)
    async def Del_Speaker_POST(result: Annotated[SpeakerAPI, Depends(del_speaker_body)]):
        return result

    @APP.get(
        "/list_speaker",
        tags=["Setting Speaker"],
        summary="List_Speaker_GET",
        description=SHARED_DOCS_LIST_SPEAKER,
        responses=SHARED_LIST_RESPONSE_DICT,
    )
    async def List_Speaker_GET(result: Annotated[None, Depends(list_speaker_query)]):
        return result

    @APP.post(
        "/list_speaker",
        tags=["Setting Speaker"],
        summary="List_Speaker_POST",
        description=SHARED_DOCS_LIST_SPEAKER,
        responses=SHARED_LIST_RESPONSE_DICT,
    )
    async def List_Speaker_POST(result: Annotated[None, Depends(list_speaker_body)]):
        return result

    @APP.get(
        "/get_speaker", tags=["Setting Speaker"], summary="Get_Speaker_GET", description=SHARED_DOCS_GET_SPEAKER, responses=SHARED_GET_RESPONSE_DICT
    )
    async def Get_Speaker_GET(result: Annotated[str, Depends(get_speaker_query)]):
        return result

    @APP.post(
        "/get_speaker", tags=["Setting Speaker"], summary="Get_Speaker_POST", description=SHARED_DOCS_GET_SPEAKER, responses=SHARED_GET_RESPONSE_DICT
    )
    async def Get_Speaker_POST(result: Annotated[str, Depends(get_speaker_body)]):
        return result

    @APP.get("/set_speaker", tags=["Setting Speaker"], summary="Set_Speaker_POST", responses=SHARED_OTHER_RESPONSE_DICT)
    async def Set_Speaker_GET(result: Annotated[str, Depends(set_speaker_query)]):
        return result

    @APP.post("/set_speaker", tags=["Setting Speaker"], summary="Set_Speaker_POST", responses=SHARED_OTHER_RESPONSE_DICT)
    async def Set_Speaker_POST(result: Annotated[str, Depends(set_speaker_body)]):
        return result

    @APP.get(
        "/set_gpt_weights",
        tags=["Setting Weights"],
        summary="Set_GPT_Weights_GET",
        description=SHARED_DOCS_SET_GPT,
        responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Set_GPT_Weights_GET(result: Annotated[str, Depends(set_gpt_weights_query)]):
        return result

    @APP.post(
        "/set_gpt_weights",
        tags=["Setting Weights"],
        summary="Set_GPT_Weights_POST",
        description=SHARED_DOCS_SET_GPT,
        responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Set_GPT_Weights_POST(result: Annotated[str, Depends(set_gpt_weights_body)]):
        return result

    @APP.get(
        "/set_sovits_weights",
        tags=["Setting Weights"],
        summary="Set_SoVITS_Weights_GET",
        description=SHARED_DOCS_SET_SOVITS,
        responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Set_SoVITS_Weights_GET(result: Annotated[str, Depends(set_sovits_weights_query)]):
        return result

    @APP.post(
        "/set_sovits_weights",
        tags=["Setting Weights"],
        summary="Set_SoVITS_Weights_POST",
        description=SHARED_DOCS_SET_SOVITS,
        responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Set_SoVITS_Weights_POST(result: Annotated[str, Depends(set_sovits_weights_body)]):
        return result

    @APP.get("/control", tags=["Control"], summary="Control_GET")
    async def Control_GET(command: Literal["restart", "exit"]):
        await handle_control_query(command)

    @APP.post("/control", tags=["Control"], summary="Control_POST")
    async def Control_POST(command: Literal["restart", "exit"]):
        await handle_control_body(command)

    return APP
