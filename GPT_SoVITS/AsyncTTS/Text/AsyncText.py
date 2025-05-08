import asyncio
from asyncio import AbstractEventLoop, Semaphore

import torch

from GPT_SoVITS.AsyncTTS.AsyncTTSEngine import Bert
from GPT_SoVITS.AsyncTTS.utils import AsyncRLock

torch.set_grad_enabled(False)

Tensor = torch.Tensor


class AsyncTextEngine:
    def __init__(
        self,
        bert_model: Bert,
        background_loop: AbstractEventLoop,
        lock: AsyncRLock,
        semaphore: Semaphore,
    ) -> None:
        self.bert = bert_model

        self.device = self.bert.device
        self.dtype = self.bert.dtype

        self.background_loop = background_loop
        self.lock = lock
        self.semaphore = semaphore

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        language = language.replace("all_", "")
        if language == "zh":
            feature = self.get_bert_feature(
                norm_text,
                word2ph,
            )
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=self.dtype,
                device=self.device,
            )

        return feature

    def get_bert_feature(self, text: str, word2ph: list) -> torch.Tensor:
        res = self.bert.forward(text=text)
        assert len(word2ph) == len(text)
        phone_level_feature: list[Tensor] = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        bert_feature = torch.cat(phone_level_feature, dim=0)
        return bert_feature.T
