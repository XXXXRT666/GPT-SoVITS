import os
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

from tools.utils.my_utils import load_audio
from tools.webui.uvr5.lib.lib_v5 import nets_61968KB as Nets
from tools.webui.uvr5.lib.lib_v5 import spec_utils
from tools.webui.uvr5.lib.lib_v5.model_param_init import ModelParameters
from tools.webui.uvr5.lib.lib_v5.nets_new import CascadedNet
from tools.webui.uvr5.lib.utils import inference_torch as ModelInference

warnings.simplefilter("ignore", UserWarning)


class VRModel:
    def __init__(self, agg, model_path, device, is_half, tta=False):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": tta,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }

        self.is_half = is_half
        self.mp: ModelParameters
        self.model: nn.Module
        self.from_pretrained(model_path)

    def from_pretrained(self, model_path):
        cpk = torch.load(model_path, map_location="cpu")
        print(self.model.load_state_dict(cpk))
        self.model.eval()
        if self.is_half:
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)

    def preprocess(self, music_file):
        X_spec_s: list[torch.Tensor] = []
        bands_n = len(self.mp.param["band"])
        input_high_end_h, input_high_end = None, None
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                X_wave = load_audio(music_file, sr=bp["sr"], channel=2)
                X_wave = torch.from_numpy(X_wave).to(self.device)
            else:  # lower bands
                if self.device != torch.device("cpu"):
                    X_wave = torchaudio.functional.resample(
                        X_wave,
                        orig_freq=self.mp.param["band"][d + 1]["sr"],
                        new_freq=bp["sr"],
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method="sinc_interp_kaiser",
                        beta=14.769656459379492,
                    )
                else:
                    X_wave = torchaudio.functional.resample(
                        X_wave,
                        orig_freq=self.mp.param["band"][d + 1]["sr"],
                        new_freq=bp["sr"],
                        lowpass_filter_width=12,
                    )
            # Stft of wave source
            X_spec_s.append(
                spec_utils.wave_to_spectrogram_torch(
                    X_wave,
                    bp["hl"],
                    bp["n_fft"],
                    self.mp.param["mid_side"],
                    self.mp.param["mid_side_b2"],
                    self.mp.param["reverse"],
                )
            )
            # pdb.set_trace()
            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"])
                input_high_end = X_spec_s[0][:, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :]
        X_spec_s.reverse()
        X_spec_m = spec_utils.combine_spectrograms_torch(X_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        return X_spec_m, aggressiveness, input_high_end_h, input_high_end

    def postprocess(self, pred, X_mag, X_phase, X_spec_m):
        if self.data["postprocess"]:
            pred_inv = torch.clamp(X_mag - pred, min=0)
            pred = spec_utils.mask_silence_torch(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m
        return y_spec_m, v_spec_m

    def spec_to_audio(self, spec_m, input_high_end, input_high_end_h, prefix: str, audio_format: str, opt_folder, name: str):
        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring_torch(self.data["high_end_process"], spec_m, input_high_end, self.mp)
            wav_tensor = spec_utils.cmb_spectrogram_to_wave_torch(spec_m, self.mp, input_high_end_h, input_high_end_.cpu())
        else:
            wav_tensor = spec_utils.cmb_spectrogram_to_wave_torch(spec_m, self.mp)
        if audio_format in ["wav", "flac"]:
            sf.write(
                os.path.join(
                    opt_folder,
                    f"{prefix}_{name}_{self.data['agg']}.{audio_format}",
                ),
                (np.array(wav_tensor) * 32768).astype(np.int16),
                self.mp.param["sr"],
            )  #
        else:
            path = os.path.join(
                opt_folder,
                f"{prefix}_{name}_{self.data['agg']}.wav",
            )
            sf.write(
                path,
                (np.array(wav_tensor) * 32768).astype(np.int16),
                self.mp.param["sr"],
            )
            if os.path.exists(path):
                opt_audio_format_path = f"{os.path.splitext(path)[0]}.{audio_format}"
                os.system(f"ffmpeg -i {path} -vn {opt_audio_format_path} -q:a 2 -y")
                if os.path.exists(opt_audio_format_path):
                    try:
                        os.remove(path)
                    finally:
                        pass


class VRModel_HP(VRModel):
    def __init__(self, agg, model_path, device, is_half, tta=False):
        self.mp = ModelParameters("tools/webui/uvr5/lib/lib_v5/modelparams/4band_v2.json")
        self.model = Nets.CascadedASPPNet(self.mp.param["bins"] * 2)
        super().__init__(agg, model_path, device, is_half, tta)

    @torch.no_grad()
    def inference(self, music_file, ins_root=None, vocal_root=None, audio_format="wav", is_hp3=False):

        if ins_root is None and vocal_root is None:
            raise RuntimeError("No save root.")
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)
        X_spec_m, aggressiveness, input_high_end_h, input_high_end = self.preprocess(music_file)
        pred, X_mag, X_phase = ModelInference(X_spec_m, self.device, self.model, aggressiveness, self.data)
        y_spec_m, v_spec_m = self.postprocess(pred, X_mag, X_phase, X_spec_m)
        if is_hp3 is True:
            ins_root, vocal_root = vocal_root, ins_root
        if ins_root is not None:
            if is_hp3 is True:
                prefix = "vocal"
            else:
                prefix = "instrument"
            self.spec_to_audio(
                y_spec_m,
                input_high_end,
                input_high_end_h,
                prefix,
                audio_format,
                ins_root,
                name,
            )
            print(f"{name} instruments done")

        if vocal_root is not None:
            if is_hp3 == True:
                prefix = "instrument"
            else:
                prefix = "vocal"
            self.spec_to_audio(
                v_spec_m,
                input_high_end,
                input_high_end_h,
                "vocal",
                audio_format,
                vocal_root,
                name,
            )
            print(f"{name} vocals done")


class VRModel_DeEcho(VRModel):
    def __init__(self, agg, model_path, device, is_half, tta=False):
        self.mp = ModelParameters("tools/webui/uvr5/lib/lib_v5/modelparams/4band_v3.json")
        nout = 64 if "DeReverb" in model_path else 48
        self.model = CascadedNet(self.mp.param["bins"] * 2, nout)
        super().__init__(agg, model_path, device, is_half, tta)

    @torch.no_grad()
    def inference(self, music_file, ins_root=None, vocal_root=None, audio_format="wav", is_hp3=False):

        if ins_root is None and vocal_root is None:
            raise RuntimeError("No save root.")
        name = os.path.basename(music_file)
        if ins_root is not None:
            os.makedirs(ins_root, exist_ok=True)
        if vocal_root is not None:
            os.makedirs(vocal_root, exist_ok=True)

        X_spec_m, aggressiveness, input_high_end_h, input_high_end = self.preprocess(music_file)
        pred, X_mag, X_phase = ModelInference(X_spec_m, self.device, self.model, aggressiveness, self.data)
        y_spec_m, v_spec_m = self.postprocess(pred, X_mag, X_phase, X_spec_m)

        if is_hp3 is True:
            ins_root, vocal_root = vocal_root, ins_root
        if ins_root is not None:
            if is_hp3 is True:
                prefix = "vocal"
            else:
                prefix = "instrument"
            self.spec_to_audio(
                v_spec_m,
                input_high_end,
                input_high_end_h,
                prefix,
                audio_format,
                ins_root,
                name,
            )
            print(f"{name} instruments done")

        if vocal_root is not None:
            if is_hp3 is True:
                prefix = "instrument"
            else:
                prefix = "vocal"
            self.spec_to_audio(
                y_spec_m,
                input_high_end,
                input_high_end_h,
                "vocal",
                audio_format,
                vocal_root,
                name,
            )
            print(f"{name} vocals done")
