import torch


def random_batch_indices(num_data_per_epoch, batch_size: int, generator: torch.Generator, n_repeat = 1):
    """
    This is a generator that does something that torch can't namely shuffle each
    epoch while ensuring samples don't overlap over epochs while maintaining a constant batch size
    """
    assert num_data_per_epoch > 0, "There must be at-least one data-point per epoch to batch"
    num_data = num_data_per_epoch

    assert batch_size <= num_data, "The batch size must be less than or equal to the size of the dataset"

    def randperm_old(ndata):
        """
        randperm function in torch 1.7 gives segfault
        """
        return torch.argsort(torch.rand(ndata, device=generator.device, generator=generator))

    current_cursor = 0
    shuffle_inds = randperm_old(num_data)
    epoch_ended = False
    while True:
        end_cursor = current_cursor + batch_size
        end_cursor_first = min(end_cursor, num_data)

        batch_indices = shuffle_inds[current_cursor:end_cursor_first]
        if end_cursor >= num_data:
            # A piece of logic that ensures that the same elements in x_data are not taken again
            perm1 = randperm_old(current_cursor)
            new_shuffle_inds = shuffle_inds.detach().clone()
            new_shuffle_inds[:current_cursor] = new_shuffle_inds[:current_cursor][perm1]

            perm2 = randperm_old(2*num_data - end_cursor)
            new_shuffle_inds[end_cursor-num_data:] = new_shuffle_inds[end_cursor-num_data:][perm2]

            batch_indices_second = new_shuffle_inds[:end_cursor-num_data]
            batch_indices = torch.cat([batch_indices, batch_indices_second], dim=0)

            shuffle_inds = new_shuffle_inds
            epoch_ended = True

        yield torch.repeat_interleave(batch_indices, n_repeat), epoch_ended

        epoch_ended = False
        current_cursor = end_cursor % num_data


def batch_indices_for_one_epoch(num_data_per_epoch, batch_size: int, generator: torch.Generator = None, n_repeat: int = 1, device: torch.device = None):
    """
    This is a generator that does batching in a vectorized way. if `generator` is
    not specified, there is no shuffling of data
    """
    assert num_data_per_epoch > 0, "There must be at-least one data-point per epoch to batch"
    num_data = num_data_per_epoch

    current_cursor = 0
    if generator is not None:
        if not generator.device == device:
            raise TypeError(f"batch_input_for_one_epoch: The device of the dataset {device}"
                            f" does not match the device of the random generator {generator.device}")
        shuffle_inds = torch.randperm(num_data, device=generator.device, dtype=torch.int64, generator=generator)
    else:
        shuffle_inds = torch.arange(num_data, device=device, dtype=torch.int64)

    end_cursor = 0
    while end_cursor < num_data:
        end_cursor = min(current_cursor + batch_size, num_data)

        batch_indices = shuffle_inds[current_cursor:end_cursor]
        current_cursor = end_cursor

        yield torch.repeat_interleave(batch_indices, n_repeat)


def random_batch_dataset(dataset: torch.utils.data.Dataset, batch_size: int, generator: torch.Generator, n_repeat: int = 1):
    for batch_indices, epoch_ended in random_batch_indices(len(dataset), batch_size, generator, n_repeat):
        data_batch = dataset[batch_indices]
        yield data_batch, epoch_ended


def batch_dataset_for_one_epoch(dataset: torch.utils.data.Dataset, batch_size: int, generator: torch.Generator = None, n_repeat: int = 1):
    if not hasattr(dataset, 'device'):
        raise TypeError("batch_input_for_one_epoch: The dataset object must implement a device property")
    device = dataset.device

    for batch_indices in batch_indices_for_one_epoch(len(dataset), batch_size, generator, n_repeat, device):
        data_batch = dataset[batch_indices]
        yield data_batch