import os
import sys
import shutil
import traceback
import multiprocessing
from pathlib import Path

from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
import click

from tools.utils.my_utils import load_audio
from tools.utils.slicer import Slicer

AUDIO_EXTENSIONS = [
    "wav",
    "flac",
    "alac",
    "aiff",
    "mp3",
    "m4a",
    "aac",
    "ogg",
    "wma",
    "opus",
    "amr",
    "dts",
    "aif",
    "aifc",
]
AUDIO_EXTENSIONS = {"." + ext for i in AUDIO_EXTENSIONS for ext in (i, i.upper())}


def worker(task):
    sr: int
    threshold: float
    min_length: int
    min_interval: int
    hop_size: int
    max_sil_kept: int
    max_volume: float
    alpha_mix: float
    opt: str

    idx, inp_path, params = task
    sr, threshold, min_length, min_interval, hop_size, max_sil_kept, max_volume, alpha_mix, opt = params
    slicer = Slicer(
        sr=sr,
        threshold=threshold,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_sil_kept,
    )
    try:
        name = os.path.basename(inp_path)
        audio = load_audio(inp_path, sr)
        for chunk, start, end in slicer.slice(audio):
            tmp_max = np.abs(chunk).max()
            if tmp_max > 1:
                chunk /= tmp_max
            chunk = (chunk / tmp_max * (max_volume * alpha_mix)) + (1 - alpha_mix) * chunk
            wavfile.write(
                "%s/%s_%010d_%010d.wav" % (opt, name, start, end),
                32000,
                (chunk * 32767).astype(np.int16),
            )
        return idx + 1
    except Exception as e:
        tqdm.write("** Error Msg", file=sys.stderr)
        tqdm.write(type(e).__name__ + ": " + str(e), file=sys.stderr)
        tqdm.write("**tracebacks", file=sys.stderr)
        tqdm.write(inp_path + " ->fail->\n" + traceback.format_exc(), file=sys.stderr)
        return None


@click.command(name="slice_audio", help="Audio Slicer")
@click.argument("input-path")
@click.argument("output-path")
@click.option(
    "-n",
    "--num-workers",
    default=4,
    type=int,
    help="Multiprocessing workers number",
    show_default=True,
)
@click.option(
    "--threshold",
    default=-34,
    type=int,
    help="Threshold for silence part",
    show_default=True,
)
@click.option(
    "--hop-size",
    default=10,
    type=int,
    help="Frame step size",
    show_default=True,
)
@click.option(
    "--min-length",
    default=4000,
    type=int,
    help="Minimum length of output audio in ms",
    show_default=True,
)
@click.option(
    "--min-interval",
    default=250,
    type=int,
    help="Minimum silence duration for cutting in ms",
    show_default=True,
)
@click.option(
    "--max-sil-kept",
    default=300,
    type=int,
    help="Maxium silence duration after slicing in ms",
    show_default=True,
)
@click.option(
    "--max-volume",
    default=0.9,
    type=float,
    help="Maximum volume after audio normalization",
    show_default=True,
)
@click.option(
    "--alpha-mix",
    default=0.25,
    type=float,
    help="Normalized mixing ratio",
    show_default=True,
)
@click.option(
    "-r",
    default=0.25,
    type=float,
    help="Normalized mixing ratio",
    show_default=True,
)
@click.option(
    "--alpha-mix",
    default=0.25,
    type=float,
    help="Normalized mixing ratio",
    show_default=True,
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    default=False,
    show_default=True,
    help="Search the path recursively",
)
@click.option(
    "-c",
    "--clear-opt",
    is_flag=True,
    default=False,
    show_default=True,
    help="Clear the output path before processing",
)
def main(
    input_path: str = "",
    output_path: str = "",
    threshold: int = -34,
    min_length: int = 4000,
    min_interval: int = 250,
    hop_size: int = 10,
    max_sil_kept: float = 300,
    max_volume: float = 0.9,
    alpha_mix: float = 0.25,
    num_workers: int = 4,
    recursive: bool = False,
    clear_opt: bool = False,
):
    """A command-line tool for slicing audio files based on silence detection.

    This tool processes an input audio file and slices it into smaller segments based on silence thresholds, allowing for more manageable audio chunks.

    It supports multiprocessing to improve processing speed and provides options to customize silence detection and slicing behavior.
    """
    if clear_opt:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    if os.path.isfile(input_path):
        inputs = list(Path(input_path))
    elif os.path.isdir(input_path):
        if recursive:
            inputs = [file for file in Path(input_path).rglob("*") if file.is_file()]
        else:
            inputs = [file for file in Path(input_path).glob("*") if file.is_file()]
    else:
        raise RuntimeError("Unknown Input Type")

    inputs = [f for f in inputs if f.suffix in AUDIO_EXTENSIONS]

    inputs = sorted(inputs)
    total_tasks = len(inputs)
    params = (32000, threshold, min_length, min_interval, hop_size, max_sil_kept, max_volume, alpha_mix, output_path)
    tasks = [(idx, path, params) for idx, path in enumerate(inputs)]

    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(worker, tasks), total=total_tasks):
            pass


if __name__ == "__main__":
    main()
