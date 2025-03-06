from functools import partial

import click
import soundfile as sf
import torch
from TTS_infer_pack.TTS import TTS, TTS_Config


@click.command()
@click.option("--cuda-graph", is_flag=True, help="Enable CUDA Graph.")
@click.option("--compile", is_flag=True, help="Enable compilation mode.")
@click.option("--static", is_flag=True, help="Enable static mode.")
def main(cuda_graph=False, compile=False, static=False):
    if cuda_graph and compile:
        raise click.UsageError("Options --cuda-graph and --compile cannot be used together.")

    click.echo(f"CUDA Graph: {cuda_graph}")
    click.echo(f"Compile: {compile}")
    click.echo(f"Static: {static}")

    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")

    print(tts_config)

    GPT = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

    SoVITS = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"

    tts_config.t2s_weights_path = GPT

    tts_config.vits_weights_path = SoVITS

    tts_config.device = "cpu"

    tts_config.is_half = False

    Prompt_Audio = "不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊.wav"

    Prompt_Text = "不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊"

    Prompt_Lang = "all_zh"

    text = "我在我青春韶华的时候遇到了你，还记得刚刚开学的时候，那是第一次见你，我和我朋友在楼道间打闹的时候无意间瞟到了你正在学习时的侧颜，微风吹过你的脸庞，吹起了你的头发，从这天开始，我开始变得有点心不在焉，一心只幻想着以后与你的点滴，想你那百媚生的回眸，我也曾对着空白的纸试着写下你的美，不曾想思念随墨水溢出，浸满整张宣纸，你的味道如墨香挥之不去。抬头看向夕阳余晖，心里装的依然还是你的美。后来啊记得我们当时一起在网上聊天，聊你喜欢的动漫电影，聊一些有趣的人和事，一起分享日常，聊着聊着我发现我们两个人可以聊的非常投机，也发现我们在看待事物时的态度也很相像，从这一刻开始我彻底迷上了你。记得我们第一次近距离相处还是我为了想与你多点相处时间故意坐公交车，在车站等车时我一直在旁偷偷的看你，看那路灯打在你的脸上，显得你是那么的灵动可爱，我盯着你看了半天甚至都有点失了神，一直想啊想竟然忘记了登上公交。在回家的路上我一个人静悄悄的走着，一个人偷偷的回想，我喜欢这种感觉，甚至可以说是痴迷上这种感觉了。"

    tts_pipeline = TTS(tts_config)

    if cuda_graph:
        if not static:
            tts_pipeline.t2s_model.model.infer_batch = partial(tts_pipeline.t2s_model.model.infer_batch, use_cuda_graph=True)
        else:
            tts_pipeline.t2s_model.model.infer_batch_static = partial(tts_pipeline.t2s_model.model.infer_batch_static, use_cuda_graph=True)

    if compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        # Experimental features to reduce compilation times, will be on by default in future
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.triton.cudagraph_trees = True

        if not static:
            tts_pipeline.t2s_model.model.h.forward = torch.compile(
                tts_pipeline.t2s_model.model.h.forward,
                mode="reduce-overhead",
                fullgraph=True,
            )
        else:
            tts_pipeline.t2s_model.model.h.forward_static = torch.compile(
                tts_pipeline.t2s_model.model.h.forward_static,
                mode="reduce-overhead",
                fullgraph=True,
            )

    tts_req = {
        "text": text,
        "text_lang": "all_zh",
        "ref_audio_path": Prompt_Audio,
        "prompt_lang": Prompt_Lang,
        "prompt_text": Prompt_Text,
        "batch_size": 5,
        "text_split_method": "cut1",
        "seed": -1,
        "split_bucket": False,
        "parallel_infer": not static,
    }

    sr, audio = next(tts_pipeline.run(tts_req))

    sf.write("output.wav", audio, sr)

    sr, audio = next(tts_pipeline.run(tts_req))

    sf.write("output1.wav", audio, sr)


if __name__ == "__main__":
    main()
