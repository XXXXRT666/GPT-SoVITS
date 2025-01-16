import sys
import os
import signal
import traceback
from typing import Annotated

from fastapi import Query, Request
from fastapi.responses import StreamingResponse

from GPT_SoVITS.TTS_infer_pack.TTS_Wrapper import TTSEngine
from tools.server.schema import TTSRequestAPI, TTSResponseFailed, SpeakerAPI
from tools.server.api_utils import build_HTTPException, streaming_generator, base_generator
from tools.cfg import Prompt


async def tts_handle(tts_req: Annotated[TTSRequestAPI, Query()], request: Request):
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
                "media_type": "wav",          # str.(optional) media type of the output audio, support "wav", "pcm", "ogg", "aac".
                "streaming_mode": False,      # bool.(optional) whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
            }
    returns:
        StreamingResponse: audio stream response.
    """
    tts_engine: TTSEngine = request.app.state.TTSEngine
    req = tts_req.model_dump(mode="python", exclude_none=True)
    streaming_mode = req.pop("streaming_mode", tts_engine.configs.streaming)
    media_type = req.pop("media_type", tts_engine.configs.media_type)

    req["return_fragment"] = streaming_mode

    if media_type == "ogg" and not streaming_mode:
        raise build_HTTPException(TTSResponseFailed(RuntimeError("ogg format is not supported in non-streaming mode")))

    tts_generator = tts_engine(exception_handler=build_HTTPException, **req)
    next(tts_generator)  # Surface Exceptions Early.
    if streaming_mode:
        return StreamingResponse(
            streaming_generator(
                tts_generator,
                media_type,
            ),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{media_type}",
            },
            media_type=f"audio/{media_type if media_type != 'pcm' else 'raw'}",
        )
    else:
        audio = await anext(streaming_generator(tts_generator, media_type))
        return StreamingResponse(
            base_generator(audio),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{media_type}",
            },
            media_type=f"audio/{media_type if media_type != 'pcm' else 'raw'}",
        )


async def set_prompt(prompt: Annotated[Prompt, Query()], request: Request):
    try:
        tts_engine: TTSEngine = request.app.state.TTSEngine
        tts_engine.set_prompt(prompt)
        tts_engine.speaker.prompt = prompt
        tts_engine.speakers_cfg.save_as_json()
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))
    return {"message": "success"}


async def add_speaker(speaker: Annotated[SpeakerAPI, Query()], request: Request):
    try:
        tts_engine: TTSEngine = request.app.state.TTSEngine
        tts_engine.add_speaker(spk_name=speaker.speaker_name, spk=speaker.model_dump(mode="python", exclude_none=True))
        tts_engine.speakers_cfg.save_as_json()
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))
    return {"message": "success"}


async def set_speaker(speaker_name: str, request: Request):
    try:
        tts_engine: TTSEngine = request.app.state.TTSEngine
        tts_engine.set_speaker(speaker_name)
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))
    return {"message": "success"}


async def set_gpt_weights(weights_path: str, request: Request):
    try:
        if isinstance(weights_path, str):
            tts_engine: TTSEngine = request.app.state.TTSEngine
            tts_engine.set_t2s(weights_path)
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))

    return {"message": "success"}


async def set_sovits_weights(weights_path: str, request: Request):
    try:
        if isinstance(weights_path, str):
            tts_engine: TTSEngine = request.app.state.TTSEngine
            tts_engine.set_vits(weights_path)
    except Exception as e:
        raise build_HTTPException(TTSResponseFailed(e, traceback.format_exc()))

    return {"message": "success"}


async def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
