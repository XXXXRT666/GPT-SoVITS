import asyncio
import threading

from GPT_SoVITS.BigVGAN.bigvgan import BigVGAN
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3

try:
    import uvloop
except ImportError:
    pass
else:
    asyncio.set_event_loop(uvloop.new_event_loop())


class AsyncSoVITSEngine:
    def __init__(self, sovits_model: SynthesizerTrn | SynthesizerTrnV3) -> None:
        self.sovits_model = sovits_model

        self.background_loop = asyncio.new_event_loop()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(20)

        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.background_loop)
        self.background_loop.run_forever()

    def load_bigvgan(self, bigvgan_path: str):
        if self.bigvgan_model is not None:
            return
        self.bigvgan_model = BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=False)
        self.bigvgan_model.remove_weight_norm()
        self.bigvgan_model = self.bigvgan_model.eval()
