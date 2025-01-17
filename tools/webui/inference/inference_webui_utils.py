from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="GPT-SoVITS API")
    parser.add_argument("-c", "--api-config", type=str, default="tools/cfgs/cfg.json", help="API_Batch Cfg Path")
    parser.add_argument("-s", "--speakers-config", type=str, default="tools/cfgs/speakers.json", help="Speakers Cfg Path")
    parser.add_argument("--compile", action="store_true", help="Compiled the model to accelerate")
    return parser.parse_args()
