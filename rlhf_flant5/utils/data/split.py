from typing import Union, Tuple
from collections.abc import Iterable

import torch


def train_val_eval_data_split(binary_labels: torch.Tensor, train_fraction: Union[float, Tuple[float, float]], val_fraction: Union[float, Tuple[float, float]], generator: torch.Generator):

    def read_fraction(fraction):
        if isinstance(train_fraction, Iterable):
            fraction_0, fraction_1 = fraction
        else:
            fraction_0 = fraction
            fraction_1 = fraction
        return fraction_0, fraction_1

    train_fraction_0, train_fraction_1 = read_fraction(train_fraction)
    val_fraction_0, val_fraction_1 = read_fraction(val_fraction)

    target_0_inds = torch.where(binary_labels == 0)[0]
    target_1_inds = torch.where(binary_labels == 1)[0]

    n_points_0 = torch.as_tensor(len(target_0_inds), dtype=torch.int64, device=target_0_inds.device)
    n_points_1 = torch.as_tensor(len(target_1_inds), dtype=torch.int64, device=target_1_inds.device)

    assert n_points_0 + n_points_1 == len(binary_labels), "The target labels 'binary_labels' must be either 0 or 1"

    # random_permutation_0 = torch.randperm(n_points_0, dtype=torch.int64, device=device, generator=generator)
    # random_permutation_1 = torch.randperm(n_points_1, dtype=torch.int64, device=device, generator=generator)
    device = binary_labels.device
    random_permutation_0 = torch.argsort(torch.rand(n_points_0, device=device, generator=generator))
    random_permutation_1 = torch.argsort(torch.rand(n_points_1, device=device, generator=generator))

    n_train_0 = torch.floor(train_fraction_0*n_points_0).to(dtype=torch.int64)
    n_train_1 = torch.floor(train_fraction_1*n_points_1).to(dtype=torch.int64)

    n_val_0 = torch.floor(val_fraction_0*n_points_0).to(dtype=torch.int64)
    n_val_1 = torch.floor(val_fraction_1*n_points_1).to(dtype=torch.int64)

    train_0_inds = target_0_inds[random_permutation_0[:n_train_0]]
    val_0_inds = target_0_inds[random_permutation_0[n_train_0:n_train_0+n_val_0]]
    eval_0_inds = target_0_inds[random_permutation_0[n_train_0+n_val_0:]]

    train_1_inds = target_1_inds[random_permutation_1[:n_train_1]]
    val_1_inds = target_1_inds[random_permutation_1[n_train_1:n_train_1+n_val_1]]
    eval_1_inds = target_1_inds[random_permutation_1[n_train_1+n_val_1:]]

    # dataset_target_0 = initial_dataset.get_subset(target_0_inds)
    # dataset_target_1 = initial_dataset.get_subset(target_1_inds)
    # train_0_data = dataset_target_0.get_subset(train_inds_0)
    # eval_0_data = dataset_target_0.get_subset(eval_inds_0)
    # train_1_data = dataset_target_1.get_subset(train_inds_1)
    # eval_1_data = dataset_target_1.get_subset(eval_inds_1)

    return train_0_inds, train_1_inds, val_0_inds, val_1_inds, eval_0_inds, eval_1_inds


