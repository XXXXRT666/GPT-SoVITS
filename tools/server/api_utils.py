import os
import sys
import subprocess
import wave
import signal
from argparse import ArgumentParser
from io import BytesIO
from typing import Generator, Tuple

import soundfile as sf
from numpy.typing import NDArray
from numpy import int16
from fastapi import HTTPException

from tools.server.schema import TTSResponseFailed


def parse_args():
    parser = ArgumentParser(description="GPT-SoVITS API")
    parser.add_argument("-c", "--api-config", type=str, default="GPT_SoVITS/configs/Cfg.json", help="API_Batch配置参数路径")
    parser.add_argument("-s", "--speakers-config", type=str, default="GPT_SoVITS/configs/Speakers.json", help="说话人配置参数路径")
    return parser.parse_args()


def build_HTTPException(tts_response: TTSResponseFailed):
    e = tts_response.exception
    tracebacks = tts_response.tracebacks

    if not tracebacks:
        return HTTPException(status_code=400, detail=f"\n\n{type(e).__name__}: {e}")
    else:
        return HTTPException(status_code=400, detail=f"\n\n{tracebacks}")


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: NDArray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_pcm(io_buffer: BytesIO, data: NDArray, _: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: NDArray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: NDArray, rate: int):
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


def pack_audio(io_buffer: BytesIO, data: NDArray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_pcm(io_buffer, data, rate)
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


async def streaming_generator(tts_generator: Generator[Tuple[int, NDArray[int16]], None, None], media_type: str):
    for idx, tts_response in enumerate(tts_generator):
        sr, chunk = tts_response
        if idx == 0 and media_type == "wav":
            yield wave_header_chunk()
            media_type = "pcm"
        yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()


async def base_generator(audio):
    yield audio
