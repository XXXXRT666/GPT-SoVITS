import argparse
from typing import Literal

import torch
import uvicorn
from rich_argparse import RawDescriptionRichHelpFormatter
from transformers import BertForMaskedLM, BertTokenizerFast

import GPT_SoVITS.text.g2pw.converter as g2pw_converter
from config import get_dtype, infer_device, pretrained_gpt_name, pretrained_sovits_name
from GPT_SoVITS.Accel import MLX, PyTorch, backends
from GPT_SoVITS.feature_extractor import cnhubert
from gsv_tools.logger import logger
from gsv_tools.server.api.infer import DefaultRefer, api_ns, change_gpt_sovits_weights
from gsv_tools.server.api.routes import build_APP


def none_or_str(value: str):
    if value == "None":
        return None
    return value


RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"

RawDescriptionRichHelpFormatter.styles["argparse.args"] = "blue"

# Arguments Parser
parser = argparse.ArgumentParser(
    description=f"{BLUE}GPT-SoVITS API Server{RESET}\n\t{RED}Docs URL: http://BIND_ADDRESS:PORT{RESET}",
    formatter_class=RawDescriptionRichHelpFormatter,
)
parser.add_argument(
    "-b",
    "--backends",
    choices=backends,
    default=backends[-1],
    help="AR Inference Backend",
)
parser.add_argument(
    "--quantization",
    "-q",
    default="None",
    choices=MLX.quantization_methods_mlx + PyTorch.quantization_methods_torch,
    type=none_or_str,
    help="Quantization Method",
    required=False,
)

parser.add_argument(
    "-d",
    "--device",
    type=str,
    metavar="device_str",
    default=str(infer_device),
    help="Infer Device, Such as 'cpu:0', 'cuda:0', 'mps:0'",
)
parser.add_argument(
    "-a",
    "--bind-addr",
    type=str,
    metavar="IP Address",
    default="0.0.0.0",
    help="API Bind Address",
)
parser.add_argument(
    "-p",
    "--port",
    type=int,
    metavar="Int",
    default=9880,
    help="API Port",
)
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    metavar="Int",
    default=1,
    help="Uvicorn Workers",
)

parser.add_argument(
    "--gpt",
    metavar="File Path",
    default=pretrained_gpt_name["v2Pro"],
    help="GPT Model",
    required=False,
)
parser.add_argument(
    "--sovits",
    metavar="File Path",
    default=pretrained_sovits_name["v2Pro"],
    help="SoVITS Model",
    required=False,
)

parser.add_argument(
    "-dr",
    "--default-refer-path",
    metavar="File Path",
    type=str,
    default="",
    help="Default Reference Audio Path",
)
parser.add_argument(
    "-dt",
    "--default-refer-text",
    metavar="Str",
    type=str,
    default="",
    help="Default Reference Audio Text",
)
parser.add_argument(
    "-dl",
    "--default-refer-language",
    metavar="Language",
    type=str,
    default="",
    help="Default Reference Audio Language",
)


parser.add_argument(
    "-s",
    "--stream",
    action="store_true",
    default=False,
    help="Stream Audio Response",
)
parser.add_argument(
    "-mt",
    "--media-type",
    type=str,
    choices=["wav", "ogg", "aac"],
    default="wav",
    help="Audio Format",
)
parser.add_argument(
    "-cp",
    "--cut-punc",
    metavar="Str",
    type=str,
    default=".。",
    help=r"Cut Punc in {,.;?!、，。？！；：…}",
)
parser.add_argument(
    "--cnhubert",
    metavar="Folder Path",
    default="GPT_SoVITS/pretrained_models/chinese-hubert-base",
    help="CNHuBERT Pretrain",
    required=False,
)
parser.add_argument(
    "--bert",
    metavar="Folder Path",
    default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    help="BERT Pretrain",
    required=False,
)

args = parser.parse_args()

host = str(args.bind_addr)
port = int(args.port)
workers = int(args.workers)


def create_app():
    sovits_path = args.sovits
    gpt_path = args.gpt

    cnhubert_base_path = args.cnhubert
    bert_path = args.bert
    default_cut_punc = args.cut_punc

    default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)

    if default_refer.path == "" or default_refer.text == "" or default_refer.language == "":
        default_refer.path, default_refer.text, default_refer.language = "", "", "auto"
        logger.info("Missing Default Reference Audio")
    else:
        logger.info(f"Default Reference Audio Path: {default_refer.path}")
        logger.info(f"Default Reference Audio Text: {default_refer.text}")
        logger.info(f"Default Reference Audio Language: {default_refer.language}")

    infer_device = torch.device(str(args.device))
    dtype = get_dtype(infer_device.index)
    is_half = dtype == torch.float16

    logger.info(f"Infer Device: {infer_device}, Dtype: {dtype}")

    stream = args.stream
    logger.info(f"Stream Response: {stream}")

    media_type: Literal["wav", "ogg", "aac"] = args.media_type

    assert media_type in {"wav", "ogg", "aac"}

    logger.info(f"Audio Format: {media_type}")

    # Pretrained Models Initialization
    cnhubert.cnhubert_base_path = cnhubert_base_path
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(bert_path)
    bert_model = BertForMaskedLM.from_pretrained(bert_path).to(infer_device, dtype=dtype)  # type: ignore
    ssl_model = cnhubert.get_model().to(infer_device, dtype=dtype)

    g2pw_converter.device = infer_device
    g2pw_converter.dtype = dtype

    api_ns.backends = args.backends
    api_ns.quantization = args.quantization

    api_ns.bert_model = bert_model
    api_ns.bert_tokenizer = tokenizer
    api_ns.ssl_model = ssl_model
    api_ns.default_refer = default_refer
    api_ns.device = infer_device
    api_ns.dtype = dtype
    api_ns.is_half = is_half

    api_ns.default_cut_punc = default_cut_punc

    api_ns.media_type = media_type
    api_ns.stream = stream

    change_gpt_sovits_weights(gpt_path, sovits_path)

    app = build_APP()

    return app


if __name__ == "__main__":
    uvicorn.run("api:create_app", host=host, port=port, workers=workers, factory=True)
