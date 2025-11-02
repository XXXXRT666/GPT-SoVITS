import datetime
from typing import Annotated

from fastapi import Body, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from gsv_tools.logger import logger
from gsv_tools.my_utils import encode_stream

from .infer import api_ns, change_gpt_sovits_weights, change_refer, cut_text, get_tts_wav
from .schemas import ChangeRefAudioAPI, SetModelsAPI, TTSRequestAPI


async def tts_handle_query(tts_req: Annotated[TTSRequestAPI, Query()], request: Request):
    refer_wav_path = tts_req.refer_wav_path
    prompt_text = tts_req.prompt_text or ""
    prompt_language = tts_req.prompt_language or "auto"
    text = tts_req.text
    text_language = tts_req.text_language
    top_k = tts_req.top_k or 15
    top_p = tts_req.top_p or 1.0
    temperature = tts_req.temperature or 1.0
    speed = tts_req.speed or 1.0
    inp_refs = tts_req.inp_refs or []
    sample_steps = tts_req.sample_steps
    if_sr = tts_req.if_sr
    if refer_wav_path == "" or refer_wav_path is None:
        refer_wav_path, prompt_text, prompt_language = (
            api_ns.default_refer.path,
            api_ns.default_refer.text,
            api_ns.default_refer.language,
        )
        if not api_ns.default_refer.is_ready():
            raise HTTPException(status_code=400, detail="Ref Audio Missing")

    try:
        if tts_req.cut_punc is None:
            text = cut_text(text, api_ns.default_cut_punc)
        else:
            text = cut_text(text, tts_req.cut_punc)

        assert refer_wav_path

        cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        return StreamingResponse(
            encode_stream(
                get_tts_wav(
                    refer_wav_path,
                    prompt_text,
                    prompt_language,
                    text,
                    text_language,
                    top_k,
                    top_p,
                    temperature,
                    speed,
                    inp_refs,
                    sample_steps,
                    if_sr,
                ),
                api_ns.media_type,
            ),
            headers={
                "Content-Disposition": f"attachment; filename=audio_{cur_time}.{api_ns.media_type}",
            },
            media_type="audio/" + api_ns.media_type,
        )

    except Exception as e:
        logger.bind(show_locals=False).exception("")
        raise HTTPException(status_code=500, detail=str(e))


async def tts_handle_body(tts_req: Annotated[TTSRequestAPI, Body()], request: Request):
    return await tts_handle_query(tts_req, request)


async def set_refer_query(prompt: Annotated[ChangeRefAudioAPI, Query()], request: Request):
    try:
        return change_refer(
            prompt.refer_wav_path,
            prompt.prompt_text,
            prompt.prompt_language,
        )
    except Exception as e:
        logger.bind(show_locals=False).exception("")
        raise HTTPException(status_code=400, detail=str(e))


async def set_refer_body(prompt: Annotated[ChangeRefAudioAPI, Body()], request: Request):
    return await set_refer_query(prompt, request)


async def set_model_query(models_path: Annotated[SetModelsAPI, Body()], request: Request):
    try:
        return change_gpt_sovits_weights(models_path.gpt_model_path, models_path.sovits_model_path)
    except Exception as e:
        logger.bind(show_locals=False).exception("")
        raise HTTPException(status_code=400, detail=str(e))


async def set_model_body(models_path: Annotated[SetModelsAPI, Body()], request: Request):
    return await set_model_query(models_path, request)
