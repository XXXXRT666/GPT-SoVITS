import pickle
import traceback
from collections import defaultdict

import torch

from tools.my_utils import DictToAttrRecursive

Tensor = torch.Tensor


def patch_lora(adapter_state_dict: dict[str, Tensor], adapter_name="default"):
    grouped: defaultdict[str, dict[str, str]] = defaultdict(dict)

    for full_key in list(adapter_state_dict.keys()):
        if not full_key.endswith(f".{adapter_name}.weight"):
            continue

        # lora_part, adapter_name, weight
        parts = full_key.rsplit(".", 3)
        if len(parts) != 4:
            continue

        module_path, lora_part, adapter_name_check, _ = parts
        if adapter_name_check != adapter_name:
            raise RuntimeError("Adapter Name Mismatch")

        if any(x in module_path for x in ["to_q", "to_k", "to_v", "to_out.0"]):
            prefix = module_path.rsplit(".attn.", 1)[0] + ".attn"  # strip to_q / to_k / to_v / to_out.0
            module_name = module_path[len(prefix) + 1 :]
            grouped[prefix][f"{module_name}.{lora_part}"] = full_key

    # Merge
    for prefix, items in grouped.items():
        try:
            prefix = prefix.replace("transformer_blocks", "layers")
            A_q = adapter_state_dict.pop(items["to_q.lora_A"])
            A_k = adapter_state_dict.pop(items["to_k.lora_A"])
            A_v = adapter_state_dict.pop(items["to_v.lora_A"])
            A_qkv = torch.cat([A_q, A_k, A_v], dim=0)

            B_q = adapter_state_dict.pop(items["to_q.lora_B"])
            B_k = adapter_state_dict.pop(items["to_k.lora_B"])
            B_v = adapter_state_dict.pop(items["to_v.lora_B"])
            B_qkv = torch.cat([B_q, B_k, B_v], dim=0)

            adapter_state_dict[f"{prefix}.in_proj.lora_A.{adapter_name}.weight"] = A_qkv
            adapter_state_dict[f"{prefix}.in_proj.lora_B.{adapter_name}.weight"] = B_qkv

            A_out = adapter_state_dict.pop(items["to_out.0.lora_A"])
            B_out = adapter_state_dict.pop(items["to_out.0.lora_B"])
            adapter_state_dict[f"{prefix}.out_proj.lora_A.{adapter_name}.weight"] = A_out
            adapter_state_dict[f"{prefix}.out_proj.lora_B.{adapter_name}.weight"] = B_out

        except KeyError:
            traceback.print_exc()

    return adapter_state_dict


class SafePKUnpickler(pickle.Unpickler):
    def __init__(self, file):
        self._file = file
        self._prefix = self._file.read(2)

        if self._prefix == b"PK":
            self._file.seek(0)
            self._file[0:2] = b"PK"
        else:
            pass

        super().__init__(self._file)

    def find_class(self, module_name: str, global_name: str):
        if global_name == "utils.HParams":
            return DictToAttrRecursive
        else:
            return super().find_class(module_name, global_name)
