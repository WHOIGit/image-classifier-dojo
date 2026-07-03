"""Dataset samplers for P2 supervised data loading."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sized

from torch.utils.data import Sampler


def class_weight_by_index(
    class_counts: dict[int, int],
    *,
    scheme: str = "inverse_frequency",
    beta: float = 0.9999,
) -> dict[int, float]:
    """Return per-class sampling weights from frozen train-split counts."""

    weights: dict[int, float] = {}
    for class_index, count in class_counts.items():
        if count <= 0:
            continue
        if scheme == "inverse_frequency":
            weights[int(class_index)] = 1.0 / float(count)
        elif scheme == "effective_number":
            weights[int(class_index)] = (1.0 - beta) / (1.0 - beta**count)
        else:
            raise ValueError(
                "class_weight_scheme must be 'inverse_frequency' or 'effective_number'"
            )
    return weights


def sample_weights_for_dataset(
    dataset: Sized,
    class_counts: dict[int, int],
    *,
    scheme: str = "inverse_frequency",
    beta: float = 0.9999,
    target_name: str | None = None,
) -> list[float]:
    """Map a dataset's target labels to per-sample weights."""

    if not hasattr(dataset, "target_for_index"):
        raise TypeError("dataset does not expose target_for_index")
    class_weights = class_weight_by_index(class_counts, scheme=scheme, beta=beta)
    return [
        class_weights.get(int(dataset.target_for_index(index, target_name)), 0.0)
        for index in range(len(dataset))
    ]


def _weighted_epoch_indices(
    *,
    indices: list[int],
    weights_by_index: list[float] | None,
    rng: random.Random,
    shuffle: bool,
) -> list[int]:
    if weights_by_index is None:
        sampled = list(indices)
        if shuffle:
            rng.shuffle(sampled)
        return sampled

    weights = [weights_by_index[index] for index in indices]
    if not any(weight > 0 for weight in weights):
        raise ValueError("weighted sampler received no positive sample weights")
    return rng.choices(indices, weights=weights, k=len(indices))


class AspectBucketBatchSampler(Sampler[list[int]]):
    """Yield batches whose samples share one deterministic aspect bucket.

    Optional ``weights_by_index`` samples with replacement inside each bucket,
    which composes class balancing with size-homogeneous batches.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 0,
        weights_by_index: list[float] | None = None,
    ) -> None:
        if not hasattr(dataset, "aspect_bucket_for_index"):
            raise TypeError("dataset does not expose aspect_bucket_for_index")
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._seed = seed
        self._weights_by_index = weights_by_index

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed)
        buckets: dict[str, list[int]] = defaultdict(list)
        for index in range(len(self._dataset)):
            bucket = self._dataset.aspect_bucket_for_index(index) or "__default__"
            buckets[bucket].append(index)

        bucket_names = sorted(buckets)
        if self._shuffle:
            rng.shuffle(bucket_names)

        for bucket_name in bucket_names:
            indices = _weighted_epoch_indices(
                indices=buckets[bucket_name],
                weights_by_index=self._weights_by_index,
                rng=rng,
                shuffle=self._shuffle,
            )
            for start in range(0, len(indices), self._batch_size):
                batch = indices[start : start + self._batch_size]
                if len(batch) == self._batch_size or not self._drop_last:
                    yield batch

    def __len__(self) -> int:
        total = 0
        buckets: dict[str, int] = defaultdict(int)
        for index in range(len(self._dataset)):
            bucket = self._dataset.aspect_bucket_for_index(index) or "__default__"
            buckets[bucket] += 1
        for count in buckets.values():
            total += count // self._batch_size
            if count % self._batch_size and not self._drop_last:
                total += 1
        return total


class WeightedBatchSampler(Sampler[list[int]]):
    """Yield weighted batches over a single fixed-shape dataset split."""

    def __init__(
        self,
        dataset: Sized,
        *,
        batch_size: int,
        weights_by_index: list[float],
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if len(weights_by_index) != len(dataset):
            raise ValueError("weights_by_index length must equal dataset length")
        self._dataset = dataset
        self._batch_size = batch_size
        self._weights_by_index = weights_by_index
        self._drop_last = drop_last
        self._seed = seed

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed)
        indices = _weighted_epoch_indices(
            indices=list(range(len(self._dataset))),
            weights_by_index=self._weights_by_index,
            rng=rng,
            shuffle=True,
        )
        for start in range(0, len(indices), self._batch_size):
            batch = indices[start : start + self._batch_size]
            if len(batch) == self._batch_size or not self._drop_last:
                yield batch

    def __len__(self) -> int:
        count = len(self._dataset)
        total = count // self._batch_size
        if count % self._batch_size and not self._drop_last:
            total += 1
        return total
