import os
import shutil
import traceback
from collections import OrderedDict
from time import time as ttime

import torch

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()


def my_save(fea, path):  # fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


def savee(ckpt, name, epoch, steps, hps, model_version=None, lora_rank=None):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = hps
        opt["info"] = "%sepoch_%siteration" % (epoch, steps)
        if lora_rank:
            opt["lora_rank"] = lora_rank
        my_save(opt, f"{hps.save_weight_dir}/{name}.pth")
        return "Success."
    except Exception:
        return traceback.format_exc()


def inspect_version(f: str):
    dict_s2 = torch.load(f, map_location="cpu", mmap=True)
    hps = dict_s2["config"]
    version = None
    if "version" in hps:
        version = hps.version
    is_lora = "lora_rank" in dict_s2.keys()

    if version is not None:
        lang_version = "v2"
        model_version = version
    else:
        if "dec.conv_pre.weight" in dict_s2["weight"].keys():
            if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
                lang_version = model_version = "v1"
            else:
                lang_version = model_version = "v2"
        else:
            lang_version = "v2"
            model_version = "v3"
            if dict_s2["info"] == "pretrained_s2G_v4":
                model_version = "v4"

    return model_version, lang_version, is_lora, hps, dict_s2
