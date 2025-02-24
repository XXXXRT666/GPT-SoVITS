import math
import os
import threading

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    # s_freq = (h2_shape[2] - h1_shape[2]) // 2
    # e_freq = s_freq + h1_shape[2]
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


def wave_to_spectrogram(wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False):
    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])

    return spec


def wave_to_spectrogram_mt(wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False):

    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = None
    spec_right = None

    def run_thread(**kwargs):
        nonlocal spec_left
        spec_left = librosa.stft(**kwargs)

    thread = threading.Thread(
        target=run_thread,
        kwargs={"y": wave_left, "n_fft": n_fft, "hop_length": hop_length},
    )
    thread.start()
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)
    thread.join()

    spec = np.asfortranarray([spec_left, spec_right])

    return spec


def wave_to_spectrogram_torch(wave: torch.Tensor, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False) -> torch.Tensor:
    """
    Args:
        wave: Tensor shape [2, T]

    Returns:
        torch.Tensor [2, F, T]
    """
    # 处理输入
    window = torch.hann_window(n_fft, device=wave.device)
    if reverse:
        wave_left = torch.flip(wave[0], dims=[0])
        wave_right = torch.flip(wave[1], dims=[0])
    elif mid_side:
        wave_left = (wave[0] + wave[1]) / 2
        wave_right = wave[0] - wave[1]
    elif mid_side_b2:
        wave_left = wave[1] + wave[0] * 0.5
        wave_right = wave[0] - wave[1] * 0.5
    else:
        wave_left = wave[0]
        wave_right = wave[1]

    # 计算 STFT
    spec_left = torchaudio.functional.spectrogram(
        wave_left.unsqueeze(0), pad=0, window=window, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, power=None, normalized=False
    )
    spec_right = torchaudio.functional.spectrogram(
        wave_right.unsqueeze(0), pad=0, window=window, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, power=None, normalized=False
    )
    # 组合左右声道
    spec = torch.cat([spec_left, spec_right], dim=0)  # shape [2, F, T]
    return spec


def combine_spectrograms_torch(specs: list[torch.Tensor], mp):
    """
    Args:
        specs (list[torch.Tensor]): [2, F, T]
        mp (dict): params dict

    Returns:
        torch.Tensor: [2, bins+1, T], torch.complex64。
    """

    l = min([spec.shape[2] for spec in specs])
    spec_c = torch.zeros((2, mp.param["bins"] + 1, l), device=specs[0].device, dtype=torch.complex64)

    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        band = mp.param["band"][d]
        h = band["crop_stop"] - band["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d - 1][:, band["crop_start"] : band["crop_stop"], :l].to(torch.complex64)
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    if mp.param["pre_filter_start"] > 0:
        if bands_n == 1:
            spec_c = fft_lp_filter_torch(spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
        else:
            gp = 1
            for b in range(mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]):
                g = math.pow(10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0)
                gp = g
                spec_c[:, b, :] *= g  # 滤波系数

    return spec_c


def combine_spectrograms(specs, mp):
    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][:, mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"], :l]
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    # lowpass fiter
    if mp.param["pre_filter_start"] > 0:  # and mp.param['band'][bands_n]['res_type'] in ['scipy', 'polyphase']:
        if bands_n == 1:
            spec_c = fft_lp_filter(spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"])
        else:
            gp = 1
            for b in range(mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]):
                g = math.pow(10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0)
                gp = g
                spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)


def spectrogram_to_image(spec, mode="magnitude"):
    if mode == "magnitude":
        if np.iscomplexobj(spec):
            y = np.abs(spec)
        else:
            y = spec
        y = np.log10(y**2 + 1e-8)
    elif mode == "phase":
        if np.iscomplexobj(spec):
            y = np.angle(spec)
        else:
            y = spec

    y -= y.min()
    y *= 255 / y.max()
    img = np.uint8(y)

    if y.ndim == 3:
        img = img.transpose(1, 2, 0)
        img = np.concatenate([np.max(img, axis=2, keepdims=True), img], axis=2)

    return img


def reduce_vocal_aggressively(X, y, softmask):
    v = X - y
    y_mag_tmp = np.abs(y)
    v_mag_tmp = np.abs(v)

    v_mask = v_mag_tmp > y_mag_tmp
    y_mag = np.clip(y_mag_tmp - v_mag_tmp * v_mask * softmask, 0, np.inf)

    return y_mag * np.exp(1.0j * np.angle(y))


def mask_silence_torch(mag, ref, thres=0.2, min_range=64, fade_size=32):
    """
    Args:
        mag (torch.Tensor): [2, F, T]
        ref (torch.Tensor): [2, F, T]
        thres (float, optional): threshold
        min_range (int, optional): min step
        fade_size (int, optional): fade step

    Returns:
        torch.Tensor: [2, F, T]
    """
    if min_range < fade_size * 2:
        raise ValueError("min_range must be >= fade_size * 2")

    mag = mag.clone()

    ref_mean = ref.mean(dim=(0, 1))  # shape: (T,)

    idx = torch.where(ref_mean < thres)[0]

    diffs = torch.diff(idx)
    start_mask = torch.cat([torch.tensor([True], device=idx.device), diffs != 1])
    end_mask = torch.cat([diffs != 1, torch.tensor([True], device=idx.device)])

    starts = idx[start_mask]
    ends = idx[end_mask]

    valid = (ends - starts) > min_range
    starts, ends = starts[valid], ends[valid]

    old_e = None
    for s, e in zip(starts.tolist(), ends.tolist()):
        if old_e is not None and s - old_e < fade_size:
            s = old_e - fade_size * 2

        if s != 0:
            weight = torch.linspace(0, 1, fade_size, device=mag.device).view(1, 1, -1)
            mag[:, :, s : s + fade_size] += weight * ref[:, :, s : s + fade_size]
        else:
            s -= fade_size

        if e != mag.shape[2]:
            weight = torch.linspace(1, 0, fade_size, device=mag.device).view(1, 1, -1)
            mag[:, :, e - fade_size : e] += weight * ref[:, :, e - fade_size : e]
        else:
            e += fade_size

        mag[:, :, s + fade_size : e - fade_size] += ref[:, :, s + fade_size : e - fade_size]
        old_e = e

    return mag


def mask_silence(mag, ref, thres=0.2, min_range=64, fade_size=32):
    if min_range < fade_size * 2:
        raise ValueError("min_range must be >= fade_area * 2")

    mag = mag.copy()

    idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
    starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
    uninformative = np.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2

            if s != 0:
                weight = np.linspace(0, 1, fade_size)
                mag[:, :, s : s + fade_size] += weight * ref[:, :, s : s + fade_size]
            else:
                s -= fade_size

            if e != mag.shape[2]:
                weight = np.linspace(1, 0, fade_size)
                mag[:, :, e - fade_size : e] += weight * ref[:, :, e - fade_size : e]
            else:
                e += fade_size

            mag[:, :, s + fade_size : e - fade_size] += ref[:, :, s + fade_size : e - fade_size]
            old_e = e

    return mag


def align_wave_head_and_tail(a, b):
    l = min([a[0].size, b[0].size])

    return a[:l, :l], b[:l, :l]


def spectrogram_to_wave_torch(spec, hop_length, mid_side=False, mid_side_b2=False, reverse=False):
    """
    Args:
        spec (torch.Tensor): [2, F, T] (torch.complex64 / complex128)。
        hop_length (int): Windows size
        mid_side (bool, optional): mid_side
        mid_side_b2 (bool, optional): mid_side
        reverse (bool, optional): flip audios

    Returns:
        torch.Tensor: [2, T]
    """
    n_fft = 2 * (spec.shape[-2] - 1)
    win_length = n_fft

    window = torch.hann_window(win_length, device=spec.device)
    frames = spec.shape[-1]
    length = (frames - 1) * hop_length

    wave_left = torchaudio.functional.inverse_spectrogram(
        spec[0],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        length=length,
        pad=0,
        window=window,
        normalized=False,
    ).cpu()
    wave_right = torchaudio.functional.inverse_spectrogram(
        spec[1],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        length=length,
        pad=0,
        window=window,
        normalized=False,
    ).cpu()

    if reverse:
        return torch.stack([torch.flip(wave_left, dims=[0]), torch.flip(wave_right, dims=[0])])

    elif mid_side:
        return torch.stack([wave_left + wave_right / 2, wave_left - wave_right / 2])

    elif mid_side_b2:
        return torch.stack([wave_right / 1.25 + 0.4 * wave_left, wave_left / 1.25 - 0.4 * wave_right])

    else:
        return torch.stack([wave_left, wave_right])


def spectrogram_to_wave(spec, hop_length, mid_side, mid_side_b2, reverse):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
    elif mid_side_b2:
        return np.asfortranarray(
            [
                np.add(wave_right / 1.25, 0.4 * wave_left),
                np.subtract(wave_left / 1.25, 0.4 * wave_right),
            ]
        )
    else:
        return np.asfortranarray([wave_left, wave_right])


def spectrogram_to_wave_mt(spec, hop_length, mid_side, reverse, mid_side_b2):

    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    def run_thread(**kwargs):
        global wave_left
        wave_left = librosa.istft(**kwargs)

    thread = threading.Thread(target=run_thread, kwargs={"stft_matrix": spec_left, "hop_length": hop_length})
    thread.start()
    wave_right = librosa.istft(spec_right, hop_length=hop_length)
    thread.join()

    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray([np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)])
    elif mid_side_b2:
        return np.asfortranarray(
            [
                np.add(wave_right / 1.25, 0.4 * wave_left),
                np.subtract(wave_left / 1.25, 0.4 * wave_right),
            ]
        )
    else:
        return np.asfortranarray([wave_left, wave_right])


def cmb_spectrogram_to_wave_torch(spec_m, mp, extra_bins_h=None, extra_bins=None):
    """
    Args:
        spec_m (torch.Tensor): [2, F, T]
        mp (dict): params
        extra_bins_h (int, optional): High freq bins
        extra_bins (torch.Tensor, optional): High freq extra

    Returns:
        torch.Tensor: [2, T]
    """

    wave = None
    bands_n = len(mp.param["band"])
    offset = 0
    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        n_fft = bp["n_fft"]
        crop_start, crop_stop = bp["crop_start"], bp["crop_stop"]

        spec_s = torch.zeros((2, n_fft // 2 + 1, spec_m.shape[2]), dtype=spec_m.dtype, device=spec_m.device)
        h = crop_stop - crop_start
        spec_s[:, crop_start:crop_stop, :] = spec_m[:, offset : offset + h, :]
        offset += h
        if d == bands_n:
            if extra_bins_h:
                max_bin = n_fft // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp["hpf_start"] > 0:
                spec_s = fft_hp_filter_torch(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave_torch(spec_s, bp["hl"], mp.param["mid_side"], mp.param["mid_side_b2"], mp.param["reverse"]).cpu()
            else:
                wave += spectrogram_to_wave_torch(spec_s, bp["hl"], mp.param["mid_side"], mp.param["mid_side_b2"], mp.param["reverse"]).cpu()
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:
                spec_s = fft_lp_filter_torch(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = torchaudio.functional.resample(
                    spectrogram_to_wave_torch(spec_s, bp["hl"], mp.param["mid_side"], mp.param["mid_side_b2"], mp.param["reverse"]),
                    orig_freq=bp["sr"],
                    new_freq=sr,
                ).cpu()
            else:
                spec_s = fft_hp_filter_torch(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                spec_s = fft_lp_filter_torch(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = wave + spectrogram_to_wave_torch(spec_s, bp["hl"], mp.param["mid_side"], mp.param["mid_side_b2"], mp.param["reverse"]).cpu()
                wave = torchaudio.functional.resample(
                    wave2,
                    orig_freq=bp["sr"],
                    new_freq=sr,
                ).cpu()

    return wave.T


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None):
    wave_band = {}
    bands_n = len(mp.param["band"])
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.ndarray(shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex)
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[:, offset : offset + h, :]

        offset += h
        if d == bands_n:  # higher
            if extra_bins_h:  # if --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[:, :extra_bins_h, :]
            if bp["hpf_start"] > 0:
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    mp.param["mid_side"],
                    mp.param["mid_side_b2"],
                    mp.param["reverse"],
                )
            else:
                wave = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lower
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = librosa.resample(
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                    orig_sr=bp["sr"],
                    target_sr=sr,
                    res_type="sinc_fastest",
                )
            else:  # mid
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
                # wave = librosa.core.resample(wave2, orig_sr=bp['sr'], target_sr=sr, res_type="sinc_fastest")
                wave = librosa.core.resample(wave2, orig_sr=bp["sr"], target_sr=sr, res_type="scipy")

    return wave.T


def fft_lp_filter_torch(spec: torch.Tensor, bin_start: int, bin_stop: int) -> torch.Tensor:
    g = 1.0
    step = 1 / (bin_stop - bin_start)

    for b in range(bin_start, bin_stop):
        g -= step
        spec[:, b, :] *= g

    spec[:, bin_stop:, :] *= 0

    return spec


def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0

    return spec


def fft_hp_filter_torch(spec: torch.Tensor, bin_start: int, bin_stop: int) -> torch.Tensor:

    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0

    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0

    return spec


def mirroring_torch(a, spec_m, input_high_end, mp):
    """
    Args:
        a (str): "mirroring" or "mirroring2"
        spec_m (torch.Tensor): [2, F, T]
        input_high_end (torch.Tensor): [2, H, T]
        mp (dict): params

    Returns:
        torch.Tensor: [2, H, T]
    """
    pre_filter_start = mp.param["pre_filter_start"]

    mirror_idx_start = pre_filter_start - 10 - input_high_end.shape[1]
    mirror_idx_end = pre_filter_start - 10

    mirror = torch.flip(torch.abs(spec_m[:, mirror_idx_start:mirror_idx_end, :]), dims=[1])

    if a == "mirroring":
        mirror = mirror * torch.exp(1.0j * torch.angle(input_high_end))
        return torch.where(torch.abs(input_high_end) <= torch.abs(mirror), input_high_end, mirror)

    elif a == "mirroring2":
        mi = mirror * (input_high_end * 1.7)
        return torch.where(torch.abs(input_high_end) <= torch.abs(mi), input_high_end, mi)

    else:
        raise ValueError("Invalid Choice")


def mirroring(a, spec_m, input_high_end, mp):
    if "mirroring" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10,
                    :,
                ]
            ),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))

        return np.where(np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror)

    if "mirroring2" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"] - 10 - input_high_end.shape[1] : mp.param["pre_filter_start"] - 10,
                    :,
                ]
            ),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)


def ensembling(a, specs):
    for i in range(1, len(specs)):
        if i == 1:
            spec = specs[0]

        ln = min([spec.shape[2], specs[i].shape[2]])
        spec = spec[:, :, :ln]
        specs[i] = specs[i][:, :, :ln]

        if "min_mag" == a:
            spec = np.where(np.abs(specs[i]) <= np.abs(spec), specs[i], spec)
        if "max_mag" == a:
            spec = np.where(np.abs(specs[i]) >= np.abs(spec), specs[i], spec)

    return spec


def stft(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])

    return spec


def istft(spec, hl):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hl)
    wave_right = librosa.istft(spec_right, hop_length=hl)
    wave = np.asfortranarray([wave_left, wave_right])


if __name__ == "__main__":
    import argparse
    import time

    import cv2

    from tools.webui.uvr5.lib.lib_v5.model_param_init import ModelParameters

    p = argparse.ArgumentParser()
    p.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=["invert", "invert_p", "min_mag", "max_mag", "deep", "align"],
        default="min_mag",
    )
    p.add_argument(
        "--model_params",
        "-m",
        type=str,
        default=os.path.join("modelparams", "1band_sr44100_hl512.json"),
    )
    p.add_argument("--output_name", "-o", type=str, default="output")
    p.add_argument("--vocals_only", "-v", action="store_true")
    p.add_argument("input", nargs="+")
    args = p.parse_args()

    start_time = time.time()

    if args.algorithm.startswith("invert") and len(args.input) != 2:
        raise ValueError("There should be two input files.")

    if not args.algorithm.startswith("invert") and len(args.input) < 2:
        raise ValueError("There must be at least two input files.")

    wave, specs = {}, {}
    mp = ModelParameters(args.model_params)

    for idx, audio in enumerate(args.input):
        spec = {}

        for d in range(len(mp.param["band"]), 0, -1):
            bp = mp.param["band"][d]

            if d == len(mp.param["band"]):  # high-end band
                wave[d], _ = librosa.load(
                    audio,
                    sr=bp["sr"],
                    mono=False,
                    dtype=np.float32,
                    res_type=bp["res_type"],
                )

                if len(wave[d].shape) == 1:  # mono to stereo
                    wave[d] = np.array([wave[d], wave[d]])
            else:  # lower bands
                wave[d] = librosa.resample(
                    wave[d + 1],
                    orig_sr=mp.param["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=bp["res_type"],
                )

            spec[d] = wave_to_spectrogram(
                wave[d],
                bp["hl"],
                bp["n_fft"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )

        specs[idx] = combine_spectrograms(spec, mp)

    del wave

    if args.algorithm == "deep":
        d_spec = np.where(np.abs(specs[0]) <= np.abs(spec[1]), specs[0], spec[1])
        v_spec = d_spec - specs[1]
        sf.write(
            os.path.join("{}.wav".format(args.output_name)),
            cmb_spectrogram_to_wave(v_spec, mp),
            mp.param["sr"],
        )

    if args.algorithm.startswith("invert"):
        ln = min([specs[0].shape[2], specs[1].shape[2]])
        specs[0] = specs[0][:, :, :ln]
        specs[1] = specs[1][:, :, :ln]

        if "invert_p" == args.algorithm:
            X_mag = np.abs(specs[0])
            y_mag = np.abs(specs[1])
            max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)
            v_spec = specs[1] - max_mag * np.exp(1.0j * np.angle(specs[0]))
        else:
            specs[1] = reduce_vocal_aggressively(specs[0], specs[1], 0.2)
            v_spec = specs[0] - specs[1]

            if not args.vocals_only:
                X_mag = np.abs(specs[0])
                y_mag = np.abs(specs[1])
                v_mag = np.abs(v_spec)

                X_image = spectrogram_to_image(X_mag)
                y_image = spectrogram_to_image(y_mag)
                v_image = spectrogram_to_image(v_mag)

                cv2.imwrite("{}_X.png".format(args.output_name), X_image)
                cv2.imwrite("{}_y.png".format(args.output_name), y_image)
                cv2.imwrite("{}_v.png".format(args.output_name), v_image)

                sf.write(
                    "{}_X.wav".format(args.output_name),
                    cmb_spectrogram_to_wave(specs[0], mp),
                    mp.param["sr"],
                )
                sf.write(
                    "{}_y.wav".format(args.output_name),
                    cmb_spectrogram_to_wave(specs[1], mp),
                    mp.param["sr"],
                )

        sf.write(
            "{}_v.wav".format(args.output_name),
            cmb_spectrogram_to_wave(v_spec, mp),
            mp.param["sr"],
        )
    else:
        if not args.algorithm == "deep":
            sf.write(
                os.path.join("ensembled", "{}.wav".format(args.output_name)),
                cmb_spectrogram_to_wave(ensembling(args.algorithm, specs), mp),
                mp.param["sr"],
            )

    if args.algorithm == "align":
        trackalignment = [
            {
                "file1": '"{}"'.format(args.input[0]),
                "file2": '"{}"'.format(args.input[1]),
            }
        ]

        for i, e in tqdm(enumerate(trackalignment), desc="Performing Alignment..."):
            os.system(f"python lib/align_tracks.py {e['file1']} {e['file2']}")

    # print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))
