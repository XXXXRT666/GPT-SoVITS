import os
import click

CONTEXT_SETTINGS = dict(
    help_option_names=["-h", "--help"],
    max_content_width=120,
)


@click.group(
    name="GPT-SoVITS",
    epilog="Check README at https://github.com/RVC-Boss/GPT-SoVITS for more details",
    context_settings=CONTEXT_SETTINGS,
)
def cli():
    """GPT-SoVITS Command Line Tools"""
    os.chdir(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../",
        )
    )


# Subfix WebUI
@click.command(
    name="Subfix",
    context_settings=CONTEXT_SETTINGS,
)
@click.option(
    "--i18n-lang",
    type=str,
    default="Auto",
    help="Languages for internationalisation",
    show_default=True,
)
@click.argument(
    "list-path",
    metavar="<Path>",
    type=click.Path(exists=True, dir_okay=False, readable=True, writable=True),
    required=True,
)
def subfix(*args, **kwds):
    """Web-Based audio subtitle editing and multilingual annotation Tool

    Accept a transcription list path to launch a Gradio WebUI for text editing

    \b
    Examples:
      GPT-SoVITS Subfix --i18n-lang zh_CN output/asr_opt/ABC.list
    """
    from tools.webui.train.subfix import main as subfix_main

    subfix_main.callback(*args, **kwds)


# API
@click.command(
    name="API",
    context_settings=CONTEXT_SETTINGS,
)
@click.option(
    "-c",
    "--api-config",
    default="tools/cfgs/cfg.json",
    metavar="<Path>",
    type=click.Path(exists=False, dir_okay=False, readable=True, writable=True),
    help="Path of api config file",
    show_default=True,
)
@click.option(
    "-s",
    "--speakers-config",
    default="tools/cfgs/speakers.json",
    metavar="<Path>",
    type=click.Path(exists=False, dir_okay=False, readable=True, writable=True),
    help="Path of speakers config file",
    show_default=True,
)
@click.option("--compile", is_flag=True, help="Compiled the model to accelerate")
def api(*args, **kwds):
    """Batched Inference API for GPT-SoVITS

    The command-line arguments accept two OPTIONAL configuration file paths

    The api-config configures some FastAPI parameters and the default settings for inference, while speakers-config configures speaker information.

    If the default config path doesn't exist, create a new config file

    Compile will accelerate the inference with fixed batch size, but it takes a moment to initialized

    \b
    Examples:
      GPT-SoVITS API -c tools/cfgs/cfg.json -s tools/cfgs/speakers.json --compile

    Go to https//127.0.0.1:{api_port} for more information after starting the FastAPI application.
    """
    from tools.api import main as api_main

    api_main.callback(*args, **kwds)


# Audio Slicer
@click.command(
    name="Audio-Slicer",
    context_settings=CONTEXT_SETTINGS,
)
@click.option(
    "-n",
    "--num-workers",
    default=click.IntRange(min=1, clamp=True),
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
    type=click.FloatRange(min=0.1, max=1.0, clamp=True),
    help="Maximum volume after audio normalization",
    show_default=True,
)
@click.option(
    "--alpha-mix",
    default=0.25,
    type=click.FloatRange(min=0.1, max=1.0, clamp=True),
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
@click.argument(
    "input-path",
    type=click.Path(exists=True, readable=True, writable=True),
)
@click.argument(
    "output-path",
    type=click.Path(exists=False, file_okay=False, readable=True, writable=True),
)
def slice_audio(*args, **kwds):
    """A command-line tool for slicing audio files based on silence detection.

    This tool processes an input audio file and slices it into smaller segments based on silence thresholds, allowing for more manageable audio chunks.

    It supports multiprocessing to improve processing speed and provides options to customize silence detection and slicing behavior.

    \b
    Examples:
      GPT-SoVITS Audio-Slicer -n 8 --min-interval 250 -cr ~/datasets/wav_folder output/slicer_opt
    """
    from tools.utils.slice_audio import main as audio_slicer

    audio_slicer.callback(*args, **kwds)


# Train WebUI
@click.command(
    name="Train-WebUI",
    context_settings=CONTEXT_SETTINGS,
)
@click.option(
    "-c",
    "webui-cfg",
    default="tools/cfgs/cfg.json",
    type=click.Path(exists=False, dir_okay=False, readable=True, writable=True),
    help="Train WebUI Cfg Path",
    show_default=True,
)
def open_train_webui(*args, **kwds):
    """Train WebUI for GPT-SoVITS

    The command-line arguments accept one OPTIONAL configuration file paths

    The webui-config configures some Gradio parameters and the default settings for training

    If the default config path doesn't exist, create a new config file

    \b
    Examples:
      GPT-SoVITS Train-WebUI -c tools/cfgs/cfg.json
    """
    from tools.train_webui import main as train_webui_main

    train_webui_main.callback(*args, **kwds)


# Infer WebUI
@click.command(
    name="Infer-WebUI",
    context_settings=CONTEXT_SETTINGS,
)
@click.option(
    "-c",
    "--infer-config",
    default="tools/cfgs/cfg.json",
    metavar="<Path>",
    type=click.Path(exists=False, dir_okay=False, readable=True, writable=True),
    help="Path of Infer-WebUI config file",
    show_default=True,
)
@click.option(
    "-s",
    "--speakers-config",
    default="tools/cfgs/speakers.json",
    metavar="<Path>",
    type=click.Path(exists=False, dir_okay=False, readable=True, writable=True),
    help="Path of speakers config file",
    show_default=True,
)
@click.option("--compile", is_flag=True, help="Compiled the model to accelerate")
def open_infer_webui(*args, **kwds):
    """Train WebUI for GPT-SoVITS

    The command-line arguments accept one OPTIONAL configuration file paths

    The webui-config configures some Gradio parameters and the default settings for training

    If the default config path doesn't exist, create a new config file

    \b
    Examples:
      GPT-SoVITS Train-WebUI -c tools/cfgs/cfg.json
    """
    from tools.inference_webui import main as infer_webui_main

    infer_webui_main.callback(*args, **kwds)


cli.add_command(api)
cli.add_command(slice_audio)
cli.add_command(subfix)
cli.add_command(open_train_webui)
cli.add_command(open_infer_webui)

if __name__ == "__main__":
    cli()
