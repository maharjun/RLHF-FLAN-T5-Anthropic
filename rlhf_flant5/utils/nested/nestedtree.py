import collections.abc
import torch
import numpy as np

from rlhf_flant5.utils.ragged.ragged_tensor import RaggedTensor
from rlhf_flant5.utils.ragged.ragged_array import RaggedArray
from rlhf_flant5.utils.numpyutils import get_consistent_type


def is_atom(item):
    return (not isinstance(item, (collections.abc.Sequence, collections.abc.Mapping))
            or isinstance(item, (str, list))
            or torch.is_tensor(item)
            or isinstance(item, np.ndarray)
            or isinstance(item, RaggedArray)
            or isinstance(item, RaggedTensor))

is_atom.message = ("An atomic type is a type that is either a list, string, torch.Tensor,"
                   " np.ndarray or any other type that is not a Sequence or a Mapping")

def flatten(nested_structure):
    if is_atom(nested_structure):
        return [nested_structure]

    flat_list = []
    if isinstance(nested_structure, collections.abc.Mapping):
        items = sorted(nested_structure.items(), key=lambda x: x[0])
        items = [x[1] for x in items]  # only get values
    else:
        items = nested_structure

    for item in items:
        flat_list.extend(flatten(item))
    return flat_list


def _validate_enough_availble(flat_list, required_n_items, start_index):
    if start_index + required_n_items > len(flat_list):
        raise ValueError(f"There are not enough elements in the flat list to pack, required"
                         f" {required_n_items}, available {len(flat_list) - start_index}"
                         f" (starting from index {start_index})")

def _pack_dict(structure, flat_list, start_index):
    required_n_items = len(structure)
    _validate_enough_availble(flat_list, required_n_items, start_index)

    keys = sorted(structure.keys())
    packed_vals = []
    for key in keys:
        substructure = structure[key]
        packed_item, start_index = _pack_sequence_as(substructure, flat_list, start_index)
        packed_vals.append(packed_item)

    return {key: val for key, val in zip(keys, packed_vals)}, start_index


def _pack_list_tuple(structure, flat_list, start_index):
    required_n_items = len(structure)
    _validate_enough_availble(flat_list, required_n_items, start_index)

    packed = []
    for substructure in structure:
        packed_item, start_index = _pack_sequence_as(substructure, flat_list, start_index)
        packed.append(packed_item)

    return structure.__class__(packed), start_index


def _pack_sequence_as(structure, flat_list, start_index):
    if is_atom(structure):
        _validate_enough_availble(flat_list, 1, start_index)
        return flat_list[start_index], start_index + 1

    if isinstance(structure, collections.abc.Sequence):
        return _pack_list_tuple(structure, flat_list, start_index)

    if isinstance(structure, collections.abc.Mapping):
        return _pack_dict(structure, flat_list, start_index)

    assert False, "This should not happen"


def _check_structures_identical(*structures, path):
    if len(structures) == 0:
        return

    if all(is_atom(structure) for structure in structures):
        types = [type(structure) for structure in structures]
        if len(set(types)) != 1:
            raise TypeError(f"Inconsistent types at path '{path}': {types}")
        if all(torch.is_tensor(structure) for structure in structures):
            dtypes = [structure.dtype for structure in structures]
            if len(set(dtypes)) != 1:
                raise TypeError(f"Inconsistent tensor dtypes at path '{path}': {dtypes}")
        if all(isinstance(structure, np.ndarray) for structure in structures):
            dtypes = [structure.dtype for structure in structures]
            if get_consistent_type(dtypes) is None:
                raise TypeError(f"Inconsistent np.ndarray dtypes at path '{path}': {dtypes}")
        elif all(isinstance(structure, RaggedTensor) for structure in structures):
            structure: RaggedTensor
            dtypes = [structure.dtype for structure in structures]
            shapes = [structure.element_shape for structure in structures]
            if len(set(dtypes)) != 1:
                raise TypeError(f"Inconsistent ragged tensor dtypes at path '{path}': {dtypes}")
            if len(set(shapes)) != 1:
                raise TypeError(f"Inconsistent ragged tensor element shapes at path '{path}': {shapes}")
        return
    elif any(is_atom(structure) for structure in structures):
        raise TypeError(f"Inconsistent node types at path '{path}': {[type(structure) for structure in structures]}")

    if all(isinstance(structure, collections.abc.Mapping) for structure in structures):
        unique_keys = set(frozenset(structure.keys()) for structure in structures)
        if len(unique_keys) != 1:
            raise TypeError(f"Inconsistent keys at path '{path}': {unique_keys}")
        unique_keys = next(iter(unique_keys))

        for key in unique_keys:
            _check_structures_identical(*[structure[key] for structure in structures], path=f"{path}.{key}" if path else str(key))
        return

    if all(isinstance(structure, collections.abc.Sequence) for structure in structures):
        unique_dtypes = set(type(structure) for structure in structures)
        if len(unique_dtypes) != 1:
            raise TypeError(f"Inconsistent Sequence types found at path '{path}' {unique_dtypes}")

        unique_list_lengths = set(len(structure) for structure in structures)
        if len(unique_list_lengths) != 1:
            raise TypeError(f"Inconsistent list sizes found at path '{path}' {unique_list_lengths}")

        list_length = next(iter(unique_list_lengths))
        for i in range(list_length):
            _check_structures_identical(*[structure[i] for structure in structures], path=f"{path}.{i}" if path else str(i))
        return

    raise TypeError(f"Inconsistent node types at path '{path}': {[type(structure) for structure in structures]}")



def pack_sequence_as(structure, flat_list):
    packed_sequence, final_start_index = _pack_sequence_as(structure, flat_list, 0)
    if final_start_index < len(flat_list):
        raise ValueError("It appears that not all elements in flat_list could be packed")

    return packed_sequence


def check_structures_identical(*structures):
    _check_structures_identical(*structures, path='')


def map_structure(func, *structures, check_output_type=False, _null_return=None, **kwargs):

    if len(structures) == 0:
        return _null_return

    check_structures_identical(*structures)
    flattened_structures = [flatten(structure) for structure in structures]
    output_flat_struct = [func(*x, **kwargs) for x in zip(*flattened_structures)]

    non_atom_output_types = set(type(it) for it in output_flat_struct if not is_atom(it))
    if non_atom_output_types:
        raise TypeError(f"The function output should be an atom. {is_atom.message}. Received non-atomic outputs of type {non_atom_output_types}")

    output_structure = pack_sequence_as(structures[0], output_flat_struct)

    if check_output_type:
        check_structures_identical(structures[0], output_structure)

    return output_structure

def test_case_1():
    # Example nested dicts
    nested_dict1 = {
        'a': torch.tensor([1, 2], dtype=torch.float32),
        'b': {
            'c': torch.tensor([3], dtype=torch.float32),
            'd': torch.tensor([4, 5], dtype=torch.float32)
        }
    }

    nested_dict2 = {
        'a': torch.tensor([6, 7], dtype=torch.float32),
        'b': {
            'c': torch.tensor([8], dtype=torch.float32),
            'd': torch.tensor([9, 10], dtype=torch.float32)
        }
    }

    # Function to concatenate tensors
    def concat_tensors(*tensors):
        return torch.cat(tensors, dim=0)

    # Concatenate nested dicts using map_structure
    concatenated_dict = map_structure(concat_tensors, nested_dict1, nested_dict2, check_output_type=1)

    print("Concatenated dict:", concatenated_dict)
    # {
    #   'a': tensor([ 1.,  2.,  6.,  7.]),
    #   'b': {
    #     'c': tensor([3., 8.]),
    #     'd': tensor([ 4.,  5.,  9., 10.])
    #   }
    # }

def test_case_2():
    # Example nested dicts
    nested_dict1 = {
        'a': torch.tensor([1, 2], dtype=torch.float32),
        'b': {
            'c': torch.tensor([3], dtype=torch.float32),
            'd': torch.tensor([4, 5], dtype=torch.float32)
        }
    }

    nested_dict2 = {
        'a': torch.tensor([6, 7], dtype=torch.float32),
        'b': {
            'c': torch.tensor([8], dtype=torch.float32),
            'd': torch.tensor([9, 10], dtype=torch.float32)
        }
    }

    # Function to collate tensors into lists (wrong, returns non-atomic tuple)
    def listified_tensors_wrong(*tensors):
        return tensors

    def listified_tensors(*tensors):
        return list(tensors)

    # Function to concatenate tensors
    def concat_tensors(list_of_tensors):
        return torch.cat(list_of_tensors, dim=0)

    # Concatenate nested dicts using map_structure
    try:
        listified_dict = map_structure(listified_tensors_wrong, nested_dict1, nested_dict2)
    except TypeError as E:
        if 'should be an atom' in E.args[0]:
            print(f"ERROR: {' '.join(E.args)}")
        else:
            raise E
    else:
        raise RuntimeError("Could not generate expected TypeError")

    listified_dict = map_structure(listified_tensors, nested_dict1, nested_dict2)
    concatenated_dict = map_structure(concat_tensors, listified_dict)

    print("test_case_2: Concatenated dict:", concatenated_dict)
    # {
    #   'a': tensor([ 1.,  2.,  6.,  7.]),
    #   'b': {
    #     'c': tensor([3., 8.]),
    #     'd': tensor([ 4.,  5.,  9., 10.])
    #   }
    # }

def test_case_3():
    structure1 = {
        'a': torch.tensor([1, 2], dtype=torch.float32),
        'b': (
            torch.tensor([3], dtype=torch.float32),
            torch.tensor([4, 5], dtype=torch.float32)
        )
    }

    structure2 = {
        'a': torch.tensor([1, 2], dtype=torch.float32),
        'b': (
            torch.tensor([3], dtype=torch.float32),
            torch.tensor([4, 5], dtype=torch.int32)
        )
    }

    structure3 = {
        'a': torch.tensor([1, 2], dtype=torch.float32),
        'b': (
            torch.tensor([3], dtype=torch.float32),
            torch.tensor([4, 5], dtype=torch.float32),
            torch.tensor([6], dtype=torch.float32)
        )
    }

    structure4 = {
        'a': torch.tensor([1, 2], dtype=torch.float32),
        'c': (
            torch.tensor([3], dtype=torch.float32),
            torch.tensor([4, 5], dtype=torch.float32),
            torch.tensor([6], dtype=torch.float32)
        )
    }

    def print_error_check(s1, s2):
        try:
            check_structures_identical(s1, s2)  # Inconsistent tensor dtypes at path 'b.1': [torch.float32, torch.int32]
        except TypeError as E:
            print(f"ERROR: {''.join(E.args)}")
        else:
            raise RuntimeError("Did not generate expected TypeError")

    print_error_check(structure1, structure2)  # ERROR: Inconsistent tensor dtypes at path 'b.1': [torch.float32, torch.int32]
    print_error_check(structure1, structure3)  # ERROR: Inconsistent list sizes found at path 'b' {2, 3}
    print_error_check(structure1, structure4)  # ERROR: Inconsistent keys at path '': {frozenset({'b', 'a'}), frozenset({'c', 'a'})}


if __name__ == '__main__':
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        print("Test Case 1")
        test_case_1()
        print("Test Case 2")
        test_case_2()
        print("Test Case 3")
        test_case_3()
