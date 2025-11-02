import GPT_SoVITS.inference_webui
import GPT_SoVITS.text.g2pw.converter
from GPT_SoVITS.inference_webui import get_tts_wav, i18n
from gsv_tools.logger import timer


GPT_SoVITS.text.g2pw.converter.device = GPT_SoVITS.inference_webui.device
GPT_SoVITS.text.g2pw.converter.dtype = GPT_SoVITS.inference_webui.dtype

ref_wav = "/Users/XXXXRT/Desktop/参考/不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊.wav"

prompt_text = "不过呢因为有些特殊情况，所以我在一年半之前并没有这个，退网啊"

prompt_language = i18n("中文")

ref_wav = "/Users/XXXXRT/Desktop/参考/ほら、ドドコマ、ママにバイバイって言って。.wav"

prompt_text = "ほら、ドドコマ、ママにバイバイって言って。"

prompt_language = i18n("日文")

text = "显得你是那么的灵动可爱，我盯着你看了半天甚至都有点失了神."

text_language = i18n("中文")

how_to_cut = i18n("凑四句一切")

text = "静かな雨が街を包み、遠くの灯りが揺れながら夜の空気に溶けていく。"
text_language = i18n("日文")

a = get_tts_wav(ref_wav, prompt_text, prompt_language, text, text_language, how_to_cut, temperature=0)

next(a)

timer.clear()

text = """我在我青春韶华的时候遇到了你，还记得刚刚开学的时候，那是第一次见你，\
我和我朋友在楼道间打闹的时候无意间瞟到了你正在学习时的侧颜，微风吹过你的脸庞，\
吹起了你的头发，从这天开始，我开始变得有点心不在焉，一心只幻想着以后与你的点滴，\
想你那百媚生的回眸，我也曾对着空白的纸试着写下你的美，不曾想思念随墨水溢出，\
浸满整张宣纸，你的味道如墨香挥之不去。抬头看向夕阳余晖，心里装的依然还是你的美。\
后来啊记得我们当时一起在网上聊天，聊你喜欢的动漫电影，聊一些有趣的人和事，\
一起分享日常，聊着聊着我发现我们两个人可以聊的非常投机，也发现我们在看待事物时的态度也很相像，\
从这一刻开始我彻底迷上了你。记得我们第一次近距离相处还是我为了想与你多点相处时间故意坐公交车，\
在车站等车时我一直在旁偷偷的看你，看那路灯打在你的脸上，显得你是那么的灵动可爱，\
我盯着你看了半天甚至都有点失了神."""

text = "静かな雨が街を包み、遠くの灯りが揺れながら夜の空気に溶けていく。"
text_language = i18n("日文")

a = get_tts_wav(ref_wav, prompt_text, prompt_language, text, text_language, how_to_cut, temperature=0)

next(a)

timer.summary()

timer.clear()

a = get_tts_wav(ref_wav, prompt_text, prompt_language, text, text_language, how_to_cut, temperature=0)

next(a)

timer.summary()
