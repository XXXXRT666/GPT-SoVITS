import os
import pickle
import random

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from GPT_SoVITS.AR.models.t2s_model import Text2SemanticDecoder
from GPT_SoVITS.AR.models.t2s_model_compile import T2SDecoder
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


path = "/Users/XXXXRT/Desktop/GPT-SoVITS Main/GPT_weights_v2/BXY_V2-e10.ckpt"
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

# A = Text2SemanticDecoder(config=config)
B = T2SDecoder(config=config)


ckpt = torch.load(path, map_location="cpu")["weight"]
# print(A.load_state_dict(ckpt))
print(B.load_state_dict(ckpt))

# A.eval()
B.eval()

with open("infer.pkl", mode="rb") as f:
    all_phoneme_ids, all_phoneme_lens, prompt, all_bert_features = pickle.load(f)

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

    a = B.infer_batch(all_phoneme_ids, all_phoneme_lens, prompt, all_bert_features)

    b = B.infer_batch(all_phoneme_ids, all_phoneme_lens, prompt, all_bert_features)

    # print(a.shape)

    # print(b.shape)

    # print([(x != y).sum() for x, y in zip(a, b)])

    # a = F.scaled_dot_product_attention(*a)

    # b = F.scaled_dot_product_attention(*b)

    print(torch.mean((a.float() - b.cpu().float()) ** 2))
