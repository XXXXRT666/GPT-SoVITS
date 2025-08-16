from . import MLX, PyTorch
from .logger import logger, tb
from .PyTorch import T2SEngineTorch, T2SRequest, T2SResult

backends = PyTorch.backends + MLX.backends

backends = [
    b.replace("_", "-").title().replace("Mlx", "MLX").replace("Mps", "MPS").replace("Cuda", "CUDA") for b in backends
]


__all__ = ["T2SEngineTorch", "T2SRequest", "T2SResult", "backends", "MLX", "PyTorch", "logger", "tb"]
