from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from gsv_tools.logger import console

from ..shared_docs import SHARED_DOCS_API, SHARED_DOCS_SET_PROMPT, SHARED_DOCS_TTS, SHARED_TTS_RESPONSE_DICT
from .service import (
    set_model_body,
    set_model_query,
    set_refer_body,
    set_refer_query,
    tts_handle_body,
    tts_handle_query,
)


def build_APP():
    APP = FastAPI(
        title="GPT-SoVITS API",
        description=SHARED_DOCS_API,
        version="0.1.0",
        terms_of_service="https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE",
    )

    @APP.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        console.print("The client sent invalid data!:")
        for item in exc.errors():
            if isinstance(item, dict):
                console.print("{\n" + "\n".join([str(k) + ":" + str(v) for k, v in item.items()]) + "\n}")
            else:
                console.print(item)
        return await request_validation_exception_handler(request, exc)

    @APP.exception_handler(StarletteHTTPException)
    async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):  # pylint: disable=unused-argument
        console.print(f"HTTP Error: {str(exc)}")
        return PlainTextResponse(exc.detail, media_type="text/plain; charset=utf-8")

    @APP.exception_handler(404)
    async def Redirect_404(*args, **kwds):  # pylint: disable=unused-argument
        return RedirectResponse(url="/docs")

    @APP.middleware("http")
    async def add_charset_to_json_response(request: Request, call_next):
        response = await call_next(request)
        if isinstance(response, JSONResponse):
            response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    @APP.get(
        "/",
        include_in_schema=False,
    )
    async def redirect_root_to_docs_get():
        return RedirectResponse(url="/docs")

    @APP.post(
        "/",
        include_in_schema=False,
    )
    async def redirect_root_to_doc_post():
        return RedirectResponse(url="/docs")

    @APP.get(
        "/tts",
        tags=["TTS"],
        summary="TTS_GET_Endpoint",
        description=SHARED_DOCS_TTS,
        response_class=StreamingResponse,
        responses=SHARED_TTS_RESPONSE_DICT,
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
    )
    async def TTS_POST_Endpoint(result=Depends(tts_handle_body)):
        return result

    @APP.get(
        "/change_refer",
        tags=["Setting Prompt"],
        summary="Set_Prompt_GET",
        description=SHARED_DOCS_SET_PROMPT,
        # responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Set_Prompt_GET(result=Depends(set_refer_query)):
        return result

    @APP.post(
        "/change_refer",
        tags=["Setting Prompt"],
        summary="Set_Prompt_POST",
        description=SHARED_DOCS_SET_PROMPT,
        # responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Set_Prompt_POST(result=Depends(set_refer_body)):
        return result

    @APP.get(
        "/set_model",
        tags=["Setting Weights"],
        summary="Set_Model_GET",
        # responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Set_Model_GET(result: Annotated[str, Depends(set_model_query)]):
        return result

    @APP.post(
        "/set_model",
        tags=["Setting Weights"],
        summary="Set_Model_POST",
        # responses=SHARED_OTHER_RESPONSE_DICT,
    )
    async def Set_Model_POST(result: Annotated[str, Depends(set_model_body)]):
        return result

    return APP
