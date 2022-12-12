from typing import Iterator, List, Union

import numpy as np
import torch
from torch.utils.data import Sampler


class OrderedSampler(Sampler[int]):

    def __init__(self, data_source: Union[np.ndarray, torch.Tensor, List]) -> None:
        self.data_source = data_source
        super().__init__(data_source)

    def __iter__(self) -> Iterator[int]:
        return iter(self.data_source)

    def __len__(self) -> int:
        return len(self.data_source)


class ConsecutiveSampler(Sampler[int]):

    def __init__(self, batch_kl_div: np.ndarray) -> None:
        data_source = np.arange(batch_kl_div.shape[0])
        self.data_source = data_source
        self.batch_kl_div = batch_kl_div
        self.last_train_batch_id = 0
        super().__init__(data_source)

    def __iter__(self) -> Iterator[int]:
        probs = self.batch_kl_div.copy()

        last = self.last_train_batch_id
        num_batches = probs.shape[0]

        fetch_idx = []

        next_id = 0
        while np.any(probs):
            next_id = np.random.choice(num_batches, size=None, replace=False, p=probs[last] / probs[last].sum())
            last = next_id
            fetch_idx.append(next_id)
            probs[:, next_id] = 0.

        self.last_train_batch_id = next_id

        return iter(fetch_idx)

    def __len__(self) -> int:
        return len(self.data_source)
