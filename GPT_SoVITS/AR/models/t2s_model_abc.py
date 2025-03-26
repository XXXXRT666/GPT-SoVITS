from abc import ABC, abstractmethod
from typing import List

import torch
import torch._inductor.config
import torch.nn as nn

Tensor = torch.Tensor


class T2SDecoderABC(ABC, nn.Module):
    @abstractmethod
    def empty_cache(self): ...

    @abstractmethod
    def embed(self, x: List[torch.LongTensor], y: torch.LongTensor, bert_features: List[torch.LongTensor]) -> Tensor: ...

    @abstractmethod
    def forward(
        self,
        x: List[torch.LongTensor],
        x_lens: torch.Tensor,
        prompts: torch.LongTensor,
        bert_feature: List[torch.LongTensor],
        top_k: int,
        top_p: int,
        early_stop_num: int,
        temperature: float,
        repetition_penalty: float,
        **kwargs,
    ) -> List[Tensor]: ...

    def compile(self, *args, **kwargs):
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        # Experimental features to reduce compilation times, will be on by default in future
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.triton.cudagraph_trees = True
        torch._inductor.config.triton.cudagraph_support_input_mutation = True

        self.forward = torch.compile(
            self.forward,
            fullgraph=True,
            dynamic=True,
            mode="reduce-overhead",
        )
