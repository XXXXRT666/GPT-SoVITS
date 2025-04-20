import contextlib
import os
import re
import sys
import time
from functools import wraps

import click
import soundfile as sf
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def get_dynamic_implement_list() -> list[str]:
    model_dir = "GPT_SoVITS/AR/models"
    pattern = re.compile(r"t2s_model_(\w+)\.py$")
    try:
        files = os.listdir(model_dir)
    except FileNotFoundError:
        return []

    implements = [match.group(1) for f in files if (match := pattern.match(f))]
    return [i for i in implements if i not in {"abc", "onnx"}]


class DynamicChoice(click.ParamType):
    name = "dynamic choice"

    def __init__(self, choices_fn) -> None:
        self.choices_fn = choices_fn

    def convert(self, value, param, ctx):
        choices = self.choices_fn()
        if value in choices:
            return value
        self.fail(
            f"{value!r} is not a valid choice. (choose from {', '.join(choices)})",
            param,
            ctx,
        )


@contextlib.contextmanager
def suppress_stdout():
    with open("/dev/null", "w") as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def with_sdpa_kernel_math(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return func(*args, **kwargs)

    return wrapper


def with_sdpa_kernel_eff(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return func(*args, **kwargs)

    return wrapper


def with_sdpa_kernel_cudnn(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return func(*args, **kwargs)

    return wrapper


def with_sdpa_kernel_flash(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return func(*args, **kwargs)

    return wrapper


@click.command()
@click.option("--cuda-graph", is_flag=True, help="Enable CUDA Graph.")
@click.option("--compile", is_flag=True, help="Enable compilation mode.")
@click.option("--bs", type=int, default=20, help="Batch Size")
@click.option(
    "--implement", type=DynamicChoice(get_dynamic_implement_list), default="flash_attn", help="T2S Decoder Implement"
)
def main(cuda_graph=False, compile=False, bs=20, implement="naive_static"):
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    if cuda_graph and compile:
        raise click.UsageError("Options --cuda-graph and --compile cannot be used together.")

    click.echo(f"CUDA Graph: {cuda_graph}")
    click.echo(f"Compile: {compile}")

    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")

    GPT = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

    SoVITS = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"

    tts_config.t2s_weights_path = GPT

    tts_config.vits_weights_path = SoVITS

    tts_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    tts_config.is_half = torch.cuda.is_available()

    print(tts_config)

    Prompt_Audio = "不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊.wav"

    Prompt_Text = "不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊"

    Prompt_Lang = "all_zh"

    text = "我在我青春韶华的时候遇到了你，还记得刚刚开学的时候，那是第一次见你，我和我朋友在楼道间打闹的时候无意间瞟到了你正在学习时的侧颜，微风吹过你的脸庞，吹起了你的头发，从这天开始，我开始变得有点心不在焉，一心只幻想着以后与你的点滴，想你那百媚生的回眸，我也曾对着空白的纸试着写下你的美，不曾想思念随墨水溢出，浸满整张宣纸，你的味道如墨香挥之不去。抬头看向夕阳余晖，心里装的依然还是你的美。后来啊记得我们当时一起在网上聊天，聊你喜欢的动漫电影，聊一些有趣的人和事，一起分享日常，聊着聊着我发现我们两个人可以聊的非常投机，也发现我们在看待事物时的态度也很相像，从这一刻开始我彻底迷上了你。记得我们第一次近距离相处还是我为了想与你多点相处时间故意坐公交车，在车站等车时我一直在旁偷偷的看你，看那路灯打在你的脸上，显得你是那么的灵动可爱，我盯着你看了半天甚至都有点失了神，一直想啊想竟然忘记了登上公交。在回家的路上我一个人静悄悄的走着，一个人偷偷的回想，我喜欢这种感觉，甚至可以说是痴迷上这种感觉了。还记得我们第一次面对面说话，当时我和你一起坐公交车回家，我坐在你的前面，你轻轻的拍了一下我的肩，我有点愕然，刚从失神中缓过神来看向后面的时候，就看到你冲着我微笑，你跟我说着学校里的日常，我在一旁静静地听着，当时心里有千言万语想跟你说但最终还是开不了口，我当时真的很胆小，我面对你的时候也十分紧张，我不知道该怎么开口，最后也只是断断续续的几个哈哈哈哈。回家的路上，我十分后悔，我也十分生气-我后悔是因为我没有敢吧想说的话告诉你，我生气是因为我痛恨我自己的软弱……我们感情的开始要从暑假开始说，当时在玩面玩完回来的时候，跟你分享玩的时候的乐趣，结果说着说着我就不小心把我喜欢你的事情给说漏了嘴，我当时很激动也很紧张，我激动是因为我终于把这句话说出了口，我紧张是因为我怕你对我并没有感觉，以后甚至做不了朋友，但是上天还是善良的，你同意了我的表白，我当时的心情真的是无法言语的快乐，我一直幻想着能与你开开心心的在一起，不知不觉就从晚上想到了第二天早上…..我当时也认识到了这段感情的来之不易，我也暗暗发誓以后一定要珍惜这段感情可是到后来聊着聊着我发现我对你的感情自己变了，我没有像原先一样对你那么的着迷了，我甚至感觉我没有那么喜欢你了，我不知道怎么用言语来描述这种感觉，仅仅记得当时我不怎么理你，也经常不回你消息，好几次还把你惹生气了，我也有过反思，我发现我真的很对不起你，你那么在意我来找我说话我却对你爱答不理，搞得你不理我了，还记得那次回家路上心情不好加上你因为我不回你消息而不理我，我当时心里五味杂陈，我为什么要这样呢？我让一个除了我父母以外最关心我的人如此失望，我把一个眼里只有我一个男孩子的女生的心给伤了，我当时真的很想和你说上几句，对你反思我的过错，可是做错事就要付出代价，我一个人走完了回家的路，我在路上哭的稀里哗啦的，我哭是因为我着急，也是因为我认识到了我的错误而后悔的哭。现在我明白了世界这么大遇见你不容易，能够认识你是我的荣幸 虽然我们认识不长时间 我希望我们可以一直走到老。因为我最近一系列的下头事情弄的我们两个人存在了很多隔阂，但是我还是真的特别想和你走到最后 特别特别想和你在一起不会分开、不会离开彼此我真的好希望好希望。我以后做什么事情也会考虑你的感受，我以后一定好好珍惜这段来之不易的爱情，保证不让你伤心。"

    BATCH_SIZE = bs

    tts_pipeline = TTS(tts_config, BATCH_SIZE, implement)

    tts_pipeline.t2s_model.model = (
        tts_pipeline.t2s_model.model.cuda().half().requires_grad_(False)
        if torch.cuda.is_available()
        else tts_pipeline.t2s_model.model.requires_grad_(False)
    )

    if compile:
        tts_pipeline.t2s_model.model.compile()

    tts_req = {
        "text": "我在我青春韶华的时候遇到了你",
        # "text": text,
        "text_lang": "all_zh",
        "ref_audio_path": Prompt_Audio,
        "prompt_lang": Prompt_Lang,
        "prompt_text": Prompt_Text,
        "batch_size": BATCH_SIZE,
        "text_split_method": "cut1",
        "seed": 6666666,
        "split_bucket": False,
        "parallel_infer": True,
        "use_cuda_graph": cuda_graph,
    }

    if compile:
        tts_pipeline.run = with_sdpa_kernel_math(tts_pipeline.run)

    t1 = time.time()

    # with suppress_stdout():
    sr, audio = next(tts_pipeline.run(tts_req))

    t1 = time.time() - t1

    if compile:
        # with suppress_stdout():
        sr, audio = next(tts_pipeline.run(tts_req))

    sf.write("output.wav", audio, sr)

    t2 = time.time()

    tts_req["text"] = text

    sr, audio = next(tts_pipeline.run(tts_req))

    t2 = time.time() - t2

    sf.write("output1.wav", audio, sr)

    print(t1, t2)

    print(f"Exec Time: {t2:.2f}")


if __name__ == "__main__":
    main()
