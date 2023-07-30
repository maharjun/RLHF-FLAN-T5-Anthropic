from abc import abstractmethod
import torch
from collections import Sequence

class PretrainedEmbeddingsModel(torch.nn.Module):

    def __init__(self, PTE_keys, use_pretrained_output):
        super().__init__()

        assert isinstance(PTE_keys, Sequence) and all(isinstance(x, str) for x in PTE_keys), "PTE_keys must be a sequence of strings"
        self.PTE_keys = list(PTE_keys)
        self.use_pretrained_output = use_pretrained_output
        self.calculate_only_pretrained = False

    def full_forward(self, *args):
        out = self.pretrained_embedding(*args)
        self.forward_with_PTE(self.PTE_tensors_from_dict(out))
        return self.forward_with_PTE(out)

    def forward(self, *args, **kwargs):
        if self.calculate_only_pretrained:
            return self.pretrained_embedding(*args, **kwargs)
        if self.use_pretrained_output:
            return self.forward_with_PTE(*args, **kwargs)
        else:
            return self.full_forward(*args, **kwargs)

    def pretrained_embedding(self, *args, key_prefix=''):
        old_training = self.training
        self.eval()
        try:
            if not hasattr(self, 'PTE_keys'):
                raise AttributeError("You need to initialize the PretrainedEmbeddingsModel before calling pretrained_embedding")

            PTE = self.get_PTE(*args)
            if not hasattr(PTE, 'keys'):
                raise TypeError(f"the return value of get_PTE must be a dict, got {type(PTE)}")
            if set(PTE.keys()) != set(self.PTE_keys):
                raise ValueError(f"The Keys in the mapping returned by self.get_PTE ({', '.join(PTE.keys())}) must match self.PTE_keys ({', '.join(self.PTE_keys)})")
        finally:
            self.train(old_training)
        
        return {key_prefix+k:v for k, v in PTE.items()}

    def PTE_tensors_from_dict(self, PTE_dict, key_prefix=''):
        if not hasattr(PTE_dict, 'keys'):
            raise TypeError(f"PTE_dict must be a dict, got {type(PTE_dict)}")
        effective_PTE_keys = [key_prefix+x for x in self.PTE_keys]

        if set(PTE_dict.keys()) != set(effective_PTE_keys):
            raise ValueError(f"The Keys in PTE_dict ({', '.join(PTE_dict.keys())}) must match '{key_prefix}'+self.PTE_keys ({', '.join(self.PTE_keys)})")

        tensor_list = [PTE_dict[key_prefix+key] for key in self.PTE_keys]
        return tensor_list

    @abstractmethod
    def get_PTE(self, *args):
        ...

    @abstractmethod
    def forward_with_PTE(self, *PTE_input_tensors):
        ...

