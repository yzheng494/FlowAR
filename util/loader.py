import os
import numpy as np

import torch
import torchvision.datasets as datasets


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
            allow_empty=True,
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        with np.load(path) as data:
            if torch.rand(1) < 0.5:  # randomly hflip
                moments = data['moments'].copy()
            else:
                moments = data['moments_flip'].copy()

        return moments, target


class CachedEvalFolder(datasets.DatasetFolder):
    """Like CachedFolder but no random flip — deterministic for evaluation."""
    def __init__(self, root: str):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
            allow_empty=True,
        )

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with np.load(path) as data:
            moments = data['moments'].copy()
        return moments, target
