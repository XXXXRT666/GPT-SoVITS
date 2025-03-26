import os
import pickle
import random

import numpy as np
import torch
from AR.models.t2s_model_xformers import T2SDecoder
from torch.nn import functional as F
from tqdm import tqdm

from GPT_SoVITS.AR.models.t2s_model import Text2SemanticDecoder
from GPT_SoVITS.AR.models.t2s_model_flash_attn import T2SDecoder as T2SDecoder_flash
from GPT_SoVITS.AR.models.utils import sample


def set_seed(seed: int):
    seed = int(seed)
    seed = seed if seed != -1 else random.randrange(1 << 32)
    print(f"Set seed to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = True
            # 开启后会影响精度
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed


path = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
config: dict = {
    "model": {
        "hidden_dim": 512,
        "embedding_dim": 512,
        "head": 16,
        "n_layer": 24,
        "vocab_size": 1025,
        "phoneme_vocab_size": 732,
        "dropout": 0.0,
        "EOS": 1024,
    },
}

A = Text2SemanticDecoder(config=config)
B = T2SDecoder_flash(config=config, max_batch_size=2)


ckpt = torch.load(path, map_location="cuda")["weight"]
print(A.load_state_dict(ckpt))
print(B.load_state_dict(ckpt))

A.eval()
B.eval()

A = A.cuda().half()
B = B.cuda().half()

with open("infer.pkl", mode="rb") as f:
    all_phoneme_ids, all_phoneme_lens, prompt, all_bert_features = pickle.load(f)

all_phoneme_ids = [i.cuda() for i in all_phoneme_ids]
all_phoneme_lens = all_phoneme_lens.cuda()
prompt = prompt.cuda()
all_bert_features = [i.cuda().half() for i in all_bert_features]

I = 1

all_phoneme_ids_ = all_phoneme_ids[I]
prompt_ = prompt[I]
all_bert_features_ = all_bert_features[I]

# with open("prefill.pkl", mode="rb") as f:
#     xy_pos, k_cache, v_cache, attn_mask, y, y_len = pickle.load(f)

# print(xy_attn_mask[1, 0, 0])

with torch.no_grad():
    # xy_pos_A = A.infer_panel_batch_infer(all_phoneme_ids, all_phoneme_lens, prompt, all_bert_features)

    # xy_pos_B = B.infer_batch(all_phoneme_ids, all_phoneme_lens, prompt, all_bert_features)

    # a = xy_pos_A

    # b = xy_pos_B

    # print(a.shape)

    # print(b.shape)

    # print(torch.mean((a - b) ** 2))

    # A_xy_dec, _, __ = A.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)

    # A_logits = A.ar_predict_layer(A_xy_dec[:, -1])

    # B.h.setup_caches()

    # for layer, k, v in zip(B.h.layers, k_cache, v_cache):
    #     layer.attention.kv_cache.k_cache[:, :, : k.shape[1]] = k.reshape(5, -1, 16, 32).permute(0, 2, 1, 3)
    #     layer.attention.kv_cache.v_cache[:, :, : v.shape[1]] = v.reshape(5, -1, 16, 32).permute(0, 2, 1, 3)

    # B.h.static_xy_pos.copy_(xy_pos)
    # input_pos = torch.tensor(k.shape[1]).to(torch.int32)
    # B.h.static_attn_mask[:, :, :, : input_pos + 1] = attn_mask

    # a = A.infer_panel_batch_infer(all_phoneme_ids, all_phoneme_lens, prompt, all_bert_features, 5, 1)

    a = A.infer_panel_naive(all_phoneme_ids_.unsqueeze(0), all_phoneme_lens, prompt_.unsqueeze(0), all_bert_features_.unsqueeze(0))

    b = B.infer_batch(all_phoneme_ids[: I + 1], all_phoneme_lens[: I + 1], prompt[: I + 1], all_bert_features[: I + 1])

    print(a.shape)

    print(b.shape)

    # print([(x != y).sum() for x, y in zip(a, b)])

    # a = F.scaled_dot_product_attention(*a)

    # b = F.scaled_dot_product_attention(*b)

    print(torch.mean((a.float() - b.float()) ** 2))
