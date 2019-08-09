from typing import Callable, Iterable

from prefetch_generator import background
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def iterable_len(iterable: Iterable) -> int:
    return sum(1 for _ in iterable)


def sample_generator(dataset: tf.data.Dataset, n_workers: int = 4):
    """TF dataset -> python generator

    Args:
        dataset:
        n_workers:

    Thanks to @velikodniy

    """
    iterator = dataset.make_one_shot_iterator()
    handlers = iterator.get_next()
    with tf.Session() as sess:
        @background(n_workers)
        def load():
            try:
                while True:
                    yield sess.run(handlers)
            except tf.errors.OutOfRangeError:
                return

        yield from load()


class BaseAutoDLDataset(Dataset):
    def __init__(self, sample_generator, n_samples: int, transform: Callable = None):
        self.generator = sample_generator
        self.n_samples = n_samples
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        features, targets = next(self.generator)
        if self.transform is not None:
            features = self.transform(features)

        return {'features': features, 'targets': targets}


def get_dataloader(
    dataset: tf.data.Dataset,
    transform: Callable,
    train: bool = False,
    n_samples: int = None,
    epoch_frac: float = None,
    batch_size: int = 4,
    n_workers: int = 4,
    random_seed: int = 82
):
    n_samples = n_samples or iterable_len(sample_generator(dataset, n_workers))
    n_samples = int(n_samples * epoch_frac) if epoch_frac is not None else n_samples

    if train:
        # shuffle data
        dataset = dataset.shuffle(
            buffer_size=n_samples,
            seed=random_seed,
            reshuffle_each_iteration=True
        )

    sample_generator_ = sample_generator(dataset, n_workers)
    dataset = BaseAutoDLDataset(sample_generator=sample_generator_, n_samples=n_samples, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=train
    )

    return loader
