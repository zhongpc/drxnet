import gc
import json
import shutil
from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from tqdm.autonotebook import tqdm

# These functions are adpated from the implementation from roost by Rhys E. A. Goodall & Alpha A. Lee
# Source: https://github.com/CompRhys/roost

class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.tensor(0)
        self.std = torch.tensor(1)

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @classmethod
    def from_state_dict(cls, state_dict):
        instance = cls()
        instance.mean = state_dict["mean"].cpu()
        instance.std = state_dict["std"].cpu()

        return instance


class Featurizer:
    """Base class for featurizing nodes and edges."""

    def __init__(self, allowed_types):
        self.allowed_types = allowed_types
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = self._embedding.keys()

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(list(self._embedding.values())[0])

    @classmethod
    def from_json(cls, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = embedding.keys()
        instance = cls(allowed_types)
        for key, value in embedding.items():
            instance._embedding[key] = np.array(value, dtype=float)
        return instance


def save_checkpoint(state, is_best, model_name, run_id):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """
    checkpoint = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
    best = f"models/{model_name}/best-r{run_id}.pth.tar"

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)
