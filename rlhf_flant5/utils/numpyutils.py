from typing import List
import numpy as np
import logging

logger = logging.getLogger('synth_orig_disc.utils.numpyutils')

_sizeof_numpy_unicode_char = np.dtype('U1').itemsize

def get_consistent_type(list_of_dtypes: List[np.dtype]) -> bool:
    sample_type = list_of_dtypes[0]
    is_valid = False
    if sample_type.kind == 'U':
        is_valid = all(x.kind == 'U' for x in list_of_dtypes)
    else:
        is_valid = all(x == sample_type for x in list_of_dtypes)

    if is_valid:
        if sample_type.kind == 'U':
            return np.dtype(f'=U{max(x.itemsize // _sizeof_numpy_unicode_char for x in list_of_dtypes)}')
        else:
            return sample_type
    else:
        return None


def verify_chararray(name, arrayobj, expected_shape, error_prefix):
    expected_msg = f"{error_prefix}: Expected {name} to be a numpy chararray of unicode strings of shape {expected_shape}, but got"

    if not isinstance(arrayobj, np.ndarray):
        raise TypeError(f"{expected_msg} object of type {type(arrayobj).__name__}.")
    if arrayobj.dtype.kind != 'U':
        raise TypeError(f"{expected_msg} array of type {arrayobj.dtype}.")
    if arrayobj.shape != expected_shape:
        raise ValueError(f"{expected_msg} array of shape {arrayobj.shape}.")


def validate_index(index, prefix, allow_scalar=True):
    if allow_scalar:
        type_error_msg = f"{prefix}: index must be either an integer (int, torch.int[32|64]) or a slice, or convertible to a Tensor"
    else:
        type_error_msg = f"{prefix}: index must be either a slice, or convertible to a Tensor"

    index_is_integer = False
    if isinstance(index, int) or (isinstance(index, np.ndarray) and index.ndim == 0 and index.dtype in {np.int64, np.int32}):
        index_is_integer = True
    elif isinstance(index, slice):
        pass
    else:
        try:
            index = np.as_array(index)
        except Exception:
            raise TypeError(type_error_msg)

        if index.ndim != 1:
            raise ValueError("Tensors provided as dataset indices must be 1D")

    if allow_scalar:
        return index, index_is_integer
    else:
        return index


def zero_size_like(reference_array: np.ndarray):
    return np.zeros((0,)+reference_array.shape[1:],
                    dtype=reference_array.dtype)


def prepend_zero(array: np.ndarray):
    if not isinstance(array, np.ndarray) or array.ndim != 1:
        raise ValueError(f"prepend_zero expects a one-dimensional array")

    zero_1D = np.zeros((1,), dtype=array.dtype)
    return np.concatenate([zero_1D, array])


def map_tokens(token_map: np.ndarray, tokens: np.ndarray, forgive_missing=False):
    """
    Maps a numpy array of tokens to their corresponding indices based on a token map.

    The function first ensures that the token map contains unique entries. It then checks 
    whether each token in the input array is present in the token map. If a token is not found, 
    it raises a ValueError. If all tokens are found, it maps the tokens to their indices in the 
    token map, and returns a new numpy array with the same shape as the input array, but with 
    tokens replaced by their corresponding indices.

    Parameters
    ----------
    token_map : np.ndarray
        A numpy character array that serves as a one-to-one mapping between tokens and
        numbers. The tokens are the values stored in token_map, while the corresponding
        value is the index of the token in the map. The token_map must contain unique
        entries.
        
    tokens : np.ndarray
        A numpy array of tokens to be mapped to their corresponding indices. All tokens in this 
        array must be present in the token_map.

    forgive_missing: bool (default: True)
        If True, any missing tokens will be mapped to 0. Additionally, an
        additional boolean array (of the same shape as token_map) is returned which
        returns for each token whether it was successfully mapped or not

    Returns
    -------
    np.ndarray
        A numpy array with the same shape as the input 'tokens' array, but with each token 
        replaced by its corresponding index from the 'token_map'.

    Raises
    ------
    ValueError
        If 'token_map' contains non-unique entries, or if any token in the 'tokens' array 
        is not found in the 'token_map' and forgive_missing=False.
    """

    # token_map[sort_index] = sorted_token_map
    sorted_token_map, sort_index = np.lib.arraysetops.unique(token_map, return_index=True)

    if len(token_map) != len(sorted_token_map):
        raise ValueError("It appears that token_map contains non-unique entries")

    # tokens = unique_tokens[reverse_index]
    unique_tokens, reverse_index = np.lib.arraysetops.unique(tokens, return_inverse=True)
    reverse_index = np.reshape(reverse_index, tokens.shape)

    if not np.all(np.lib.arraysetops.in1d(unique_tokens, sorted_token_map)):
        error_message = (f"The following tokens are not present in map: "
                         f"{{{', '.join(str(x) for x in np.lib.arraysetops.setdiff1d(unique_tokens, sorted_token_map))}}}")

        if not forgive_missing:
            raise ValueError(error_message)
        else:
            logger.warning(error_message)
            # unique_tokens_existing = unique_tokens[existing_inds]
            unique_tokens_existing, existing_inds, _ = np.lib.arraysetops.intersect1d(unique_tokens, sorted_token_map, return_indices=True)
    else:
        unique_tokens_existing = unique_tokens
        existing_inds = np.arange(len(unique_tokens_existing), dtype=np.int64)

    # unique_tokens_existing = sorted_token_map[np.searchsorted(sorted_token_map, unique_tokens_existing, 'left')]
    # => unique_tokens_existing = token_map[sort_index][np.searchsorted(sorted_token_map, unique_tokens, 'left')]
    # => unique_tokens_existing = token_map[sort_index[np.searchsorted(sorted_token_map, unique_tokens, 'left')]]
    # => unique_tokens_existing = token_map[unique_tokens_existing_mapped]
    unique_tokens_existing_mapped = sort_index[np.searchsorted(sorted_token_map, unique_tokens_existing, 'left')]

    # tokens = unique_tokens[reverse_index]
    # => mapped_tokens = unique_tokens_mapped[reverse_index]
    if not forgive_missing:
        unique_tokens_mapped = unique_tokens_existing_mapped
        mapped_tokens = unique_tokens_mapped[reverse_index]
        return mapped_tokens
    else:
        # unique_tokens_existing = unique_tokens[existing_inds]
        # => unique_tokens_mapped[existing_inds] = unique_tokens_existing_mapped
        #    Any non-existing tokens are mapped to zero
        unique_tokens_mapped = np.zeros(len(unique_tokens), dtype=np.int64)
        unique_tokens_mapped[existing_inds] = unique_tokens_existing_mapped

        is_unique_token_mapped = np.zeros(len(unique_tokens), dtype=bool)
        is_unique_token_mapped[existing_inds] = True

        mapped_tokens = unique_tokens_mapped[reverse_index]
        is_token_mapped = is_unique_token_mapped[reverse_index]

        return mapped_tokens, is_token_mapped