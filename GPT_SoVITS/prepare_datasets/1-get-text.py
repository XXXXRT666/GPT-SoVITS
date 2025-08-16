import argparse
import os
import os.path
import traceback
from multiprocessing import Process, Queue, set_start_method

import torch
from rich.progress import track
from transformers import AutoModelForMaskedLM, AutoTokenizer

from GPT_SoVITS.Accelerate import logger, tb
from GPT_SoVITS.text.cleaner import clean_text
from tools.my_utils import clean_path

torch.set_grad_enabled(False)

set_start_method("spawn", force=True)


def lang_map(lang: str) -> str:
    m = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "KO": "ko",
        "Ko": "ko",
        "ko": "ko",
        "yue": "yue",
        "YUE": "yue",
        "Yue": "yue",
    }
    return m.get(lang, "")


def parse_inp_text_line(line: str) -> tuple[str, str, str]:
    wav_name, _, language, text = line.split("|", 3)
    return wav_name, language, text


def build_device_strings(device_type: str, device_ids: list[int], procs_per_device: int) -> list[str]:
    devices = []
    for device_id in device_ids:
        dstr = f"{device_type}:{device_id}"
        devices.extend([dstr] * procs_per_device)
    return devices


def worker_run(
    wid: int,
    device_str: str,
    tasks_q: Queue[tuple[int, str, str, str]],
    results_q: Queue[tuple[int, tuple[str, str, list[int] | None, str]]],
    bert_pretrained_dir: str,
    opt_dir: str,
    fp16: bool,
    version: str,
):
    device = torch.device(device_str)

    if device.type == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_device(device.index)
    elif device.type == "mps":
        assert torch.backends.mps.is_available()

    bert_dir = os.path.join(opt_dir, "3-bert")
    os.makedirs(bert_dir, exist_ok=True)

    if not os.path.exists(bert_pretrained_dir):
        raise FileNotFoundError(bert_pretrained_dir)

    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)

    if fp16:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text: str, word2ph: list[int]) -> torch.Tensor:
        inputs = tokenizer(text, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        out = bert_model(**inputs, output_hidden_states=True)
        layer = out.hidden_states[-3][0].cpu()[1:-1]  # [seq-2, hid]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            phone_level_feature.append(layer[i].repeat(word2ph[i], 1))
        feats = torch.cat(phone_level_feature, dim=0)  # [phones, hid]
        return feats.T  # [hid, phones]

    while True:
        item = tasks_q.get()
        if item is None:
            break

        idx, wav_name, language, text = item
        try:
            name = clean_path(os.path.basename(wav_name))
            mapped_lang = lang_map(language)
            if not mapped_lang:
                logger.warning(f"[W{wid}] Unsupported language: {language} of {wav_name}")
                results_q.put((idx, ("", "", [], "")))
                continue

            phones, word2ph, norm_text = clean_text(
                text.replace("%", "-").replace("￥", ","),
                mapped_lang,
                version,
            )

            if mapped_lang == "zh":
                path_bert = os.path.join(bert_dir, f"{name}.pt")
                if not os.path.exists(path_bert):
                    assert word2ph
                    bert_feature = get_bert_feature(norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    torch.save(bert_feature, path_bert)

            phones_str = " ".join(phones)
            results_q.put((idx, (name, phones_str, word2ph, norm_text)))
        except Exception:
            logger.error(f"[W{wid}] Failed: {wav_name} | {text}\n{tb()}")
            results_q.put((idx, ("", "", [], "")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True, help="list File：wav|spk|lang|text")
    parser.add_argument("--opt", type=str, required=True)
    parser.add_argument("--bert", type=str, required=True)
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--device-id", type=str, default="0", help="CUDA_VISIBLE_DEVICE")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    device_ids = [int(x) for x in args.devices.split(",") if x.strip() != ""]
    if args.device in {"cpu", "mps"} and device_ids != [0]:
        raise ValueError(f"Invalid Device ID {device_ids}")
    if args.nproc < 1:
        raise ValueError(f"Invalid Num Process {args.nproc}")

    os.makedirs(args.opt, exist_ok=True)
    merged_path = os.path.join(args.opt, "2-name2text.txt")

    with open(args.inp, "r", encoding="utf8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]

    tasks_all: list[tuple[int, str, str, str]] = []
    for idx, line in enumerate(lines):
        try:
            wav_name, language, text = parse_inp_text_line(line)
            tasks_all.append((idx, wav_name, language, text))
        except Exception:
            logger.error(f"Skip line {idx}: {line}\n{traceback.format_exc()}")

    n_tasks = len(tasks_all)
    if n_tasks == 0:
        logger.warning("Empty list")
        with open(merged_path, "w", encoding="utf8") as fout:
            pass
        return

    device_strs = build_device_strings(args.device, device_ids, args.nproc)
    total_workers = len(device_strs)

    tasks_q: Queue[tuple[int, str, str, str] | None] = Queue(maxsize=total_workers * 2)
    results_q: Queue = Queue()

    for task in tasks_all:
        tasks_q.put(task)
    for _ in range(total_workers):
        tasks_q.put(None)

    procs: list[Process] = []
    for wid, dstr in enumerate(device_strs):
        p = Process(
            target=worker_run,
            args=(wid, dstr, tasks_q, results_q, args.bert, args.opt, bool(args.fp16), args.version),
            daemon=False,
        )
        p.start()
        procs.append(p)

    ordered: list[tuple[str, str, list[int], str]] = [("", "", [], "")] * n_tasks
    for _ in track(range(n_tasks)):
        idx, tup = results_q.get()  # (idx, (name, phones_str, word2ph, norm_text))
        ordered[idx] = tup

    for p in procs:
        p.join()

    with open(merged_path, "w", encoding="utf8") as fout:
        for name, phones_str, word2ph, norm_text in ordered:
            if name == "":
                pass
            else:
                fout.write(f"{name}\t{phones_str}\t{word2ph}\t{norm_text}\n")

    logger.info(f"Done: {merged_path}")


if __name__ == "__main__":
    main()
