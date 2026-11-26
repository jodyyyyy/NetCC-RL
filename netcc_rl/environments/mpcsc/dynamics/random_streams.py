"""Instance-owned RNGs keep environments independent of the learner's RNG."""

import random

import torch


class RandomStreams:
    def reseed(self, seed: int) -> None:
        self.py_random = random.Random(int(seed))
        self.torch_generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            tensor.shape, dtype=tensor.dtype, device="cpu", generator=self.torch_generator
        )
