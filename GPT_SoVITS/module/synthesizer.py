import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

from GPT_SoVITS.module import commons
from GPT_SoVITS.module.dit import DiT
from GPT_SoVITS.module.models import Encoder, ResidualVectorQuantizer, TextEncoder
from GPT_SoVITS.module.modules import MelStyleEncoder

Tensor = torch.Tensor


class CFM(nn.Module):
    def __init__(self, in_channels: int, dit: DiT):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator: DiT = dit

        self.in_channels = in_channels

    @torch.no_grad()
    def inference(self, mu: Tensor, x_lens: Tensor, prompt: Tensor, n_timesteps: int, temperature=1.0):
        """Forward diffusion"""
        B, T = mu.size(0), mu.size(1)
        prompt = prompt.transpose(1, 2)
        mu = mu.transpose(2, 1)
        prompt_len = prompt.size(1)

        x = torch.randn([B, T, self.in_channels], device=mu.device, dtype=mu.dtype) * temperature
        prompt_x = torch.zeros_like(x, dtype=mu.dtype)
        prompt_x[:, :prompt_len] = prompt[:, :prompt_len]
        x[:, :prompt_len] = 0

        t = 0
        d = 1 / n_timesteps
        for _ in range(n_timesteps):
            t_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * t
            d_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * d
            v_pred = self.estimator.forward(x, prompt_x, x_lens, t_tensor, d_tensor, mu)
            x = x + d * v_pred
            t = t + d
            x[:, :prompt_len] = 0
        return x


class CFMSynthesizer(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v3",
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.model_dim = 512
        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.ref_enc = MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer
        inter_channels2 = 512
        self.bridge = nn.Sequential(nn.Conv1d(inter_channels, inter_channels2, 1, stride=1), nn.LeakyReLU())
        self.wns1 = Encoder(inter_channels2, inter_channels2, inter_channels2, 5, 1, 8, gin_channels=gin_channels)
        self.linear_mel = nn.Conv1d(inter_channels2, 100, 1, stride=1)
        self.cfm = CFM(
            100,
            DiT(
                **dict(
                    dim=1024,
                    depth=22,
                    heads=16,
                    ff_mult=2,
                    text_dim=inter_channels2,
                    conv_layers=4,
                )
            ),
        )  # text_dim is condition feature dim

    def forward(
        self, ssl, y, mel, ssl_lengths, y_lengths, text, text_lengths, mel_lengths, use_grad_ckpt
    ):  # ssl_lengths no need now
        with autocast("cuda", enabled=False):
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()  #
                    self.quantizer.eval()
                    self.enc_p.eval()
                ssl = self.ssl_proj(ssl)
                quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])
                quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")  ##BCT
                x, m_p, logs_p, y_mask = self.enc_p(quantized, y_lengths, text, text_lengths, ge)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")  ##BCT
        fea, y_mask_ = self.wns1(
            fea, mel_lengths, ge
        )  ##If the 1-minute fine-tuning works fine, no need to manually adjust the learning rate.
        B = ssl.shape[0]
        prompt_len_max = mel_lengths * 2 / 3
        prompt_len = (torch.rand([B], device=fea.device) * prompt_len_max).floor().to(dtype=torch.long)
        minn = min(mel.shape[-1], fea.shape[-1])
        mel = mel[:, :, :minn]
        fea = fea[:, :, :minn]
        cfm_loss = self.cfm(mel, mel_lengths, prompt_len, fea, use_grad_ckpt)
        return cfm_loss

    @torch.no_grad()
    def decode_encp(self, codes, text, refer, ge=None, speed=1):
        # print(2333333,refer.shape)
        # ge=None
        if ge == None:
            refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
            refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        y_lengths = torch.LongTensor([int(codes.size(2) * 2)]).to(codes.device)
        if speed == 1:
            sizee = int(codes.size(2) * 2.5 * 1.5)
        else:
            sizee = int(codes.size(2) * 2.5 * 1.5 / speed) + 1
        y_lengths1 = torch.LongTensor([sizee]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")  ##BCT
        x, m_p, logs_p, y_mask = self.enc_p(quantized, y_lengths, text, text_lengths, ge, speed)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=1.875, mode="nearest")  ##BCT
        ####more wn paramter to learn mel
        fea, y_mask_ = self.wns1(fea, y_lengths1, ge)
        return fea, ge

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)
