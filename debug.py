import torch

a = torch.load("/Users/XXXXRT/GPT-SoVITS/SoVITS_weights/艾琳_奇遇典章_e4_s32.pth", map_location="cpu", weights_only=False)
b = torch.load("/Users/XXXXRT/GPT-SoVITS/GPT_weights/YCY-e10.ckpt", map_location="cpu", weights_only=False)

print(a["config"], type(a["config"]))

for key, value in a.items():
    if not isinstance(value, (torch.Tensor, int, float, str, list, dict)):
        print(f"Key: {key}, Type: {type(value)}")
