import torch
import numpy as np
from synth_orig_disc.utils.ragged.ragged_tensor import RaggedTensor
from synth_orig_disc.utils.ragged.ragged_array import RaggedArray
from synth_orig_disc.utils.nested.nestedtree import map_structure

##################################################################################
# Nested dataset manipulation
##################################################################################
def _generic_concat(*list_of_tensor_or_ragged, ignore_none=False):
    assert len(list_of_tensor_or_ragged) > 0, "list_of_tensor_or_ragged must contain at-least one element"
    if isinstance(list_of_tensor_or_ragged[0], RaggedTensor):
        return RaggedTensor.concat(list_of_tensor_or_ragged, dim=0)
    elif isinstance(list_of_tensor_or_ragged[0], RaggedArray):
        return RaggedArray.concat(list_of_tensor_or_ragged, axis=0)
    elif torch.is_tensor(list_of_tensor_or_ragged[0]):
        return torch.cat(list_of_tensor_or_ragged, dim=0)
    elif isinstance(list_of_tensor_or_ragged[0], np.ndarray):
        return np.concatenate(list_of_tensor_or_ragged, axis=0)
    elif list_of_tensor_or_ragged[0] is None:
        if ignore_none:
            if not all(x is None for x in list_of_tensor_or_ragged):
                raise TypeError("All elements of the list should be None if ignore_none=True")
            else:
                return None

    raise TypeError("Unrecognised type to concatenate, need torch.Tensor, RaggedTensor, np.array, or RaggedArray")


def _generic_index(tensor_or_ragged, index):
    if torch.is_tensor(tensor_or_ragged) or isinstance(tensor_or_ragged, RaggedTensor):
        return tensor_or_ragged[index]
    else:
        if torch.is_tensor(index):
            index = index.detach().cpu().numpy()

        return tensor_or_ragged[index]


def _detach_cpu_if_tensor(element):
    if isinstance(element, torch.Tensor):
        # assumes all tensors are on the same device
        return element.detach().cpu()
    else:
        return element

# With map_structure, This takes a list of nested structures, and returns a
# nested structure where each atom is a RaggedTensor containing the
# corresponding tensors in the nested structures. If the list is a list of
# scalars, it returns a tensor containing these scalars. If the list is a
# string, it returns a numpy chararray
def _collate_list_of_tensors(*tensors_or_arrays):
    assert len(set(type(x) for x in tensors_or_arrays)) == 1
    if isinstance(tensors_or_arrays[0], np.ndarray):
        return RaggedArray(tensors_or_arrays)
    elif torch.is_tensor(tensors_or_arrays[0]):
        if tensors_or_arrays[0].ndim >= 1:
            return RaggedTensor(tensors_or_arrays)
        else:
            return torch.as_tensor(tensors_or_arrays)
    elif isinstance(tensors_or_arrays[0], str):
        return np.asarray(tensors_or_arrays)
    else:
        return torch.as_tensor(tensors_or_arrays)


def collate_datapoints_into_nested_struct(datapoint_struct_list):
    return map_structure(_collate_list_of_tensors, *datapoint_struct_list)


def concat_nested_structs(nested_struct_list, ignore_none=False):
    return map_structure(_generic_concat, *nested_struct_list, ignore_none=ignore_none)


def index_nested_struct(nested_struct, index):
    return map_structure(_generic_index, nested_struct, index=index)

def detach_cpu_nested_struct(nested_struct):
    return map_structure(_detach_cpu_if_tensor, nested_struct)
