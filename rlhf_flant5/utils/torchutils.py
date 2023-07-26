import torch


def random_seed(rng: torch.Generator):
    device = rng.device
    return torch.randint(torch.as_tensor(0, device=device),
                         torch.as_tensor(2**32, device=device),
                         (),
                         dtype=torch.int64, device=device, generator=rng)


def _print_module_params_leaf(prefix, logger, module: torch.nn.Module, excluded_modules=None):

    if excluded_modules is None:
        excluded_modules = []

    for name, param in module.named_parameters(recurse=False):
        if isinstance(module, tuple(excluded_modules)):
            continue

        if logger is None:
            printfunc = print
        else:
            printfunc = logger.info

        if name.startswith('raw_'):
            printfunc(f'{prefix}{name[4:]}: {getattr(module, name[4:])}')
        else:
            printfunc(f'{prefix}{name}: {getattr(module, name)}')


def print_module_params(module: torch.nn.Module, logger=None, recurse=True, excluded_modules=None, prefix=''):

    if recurse:
        for mname, mod in module.named_children():
            print_module_params(mod, logger, recurse=recurse, excluded_modules=excluded_modules, prefix=f'{prefix}{mname}.')
    _print_module_params_leaf(prefix, logger, module, excluded_modules)


def get_device_name(device: torch.device):
    """
    Get name of specified device object.
    """
    assert isinstance(device, torch.device), "device must be a torch.device"
    if device.index:
        return f'{device.type}:{device.index}'
    else:
        return device.type


def get_gpu_name_if_available(gpu_index=None):
    use_cuda = torch.cuda.is_available()
    if use_cuda and gpu_index is not None:
        device_name = 'cuda:{}'.format(gpu_index)
    elif use_cuda:
        device_name = 'cuda'
    else:
        device_name = 'cpu'
    return device_name


def validate_index(index, prefix, device=None, allow_scalar=True):

    if allow_scalar:
        type_error_msg = f"{prefix}: index must be either an integer (int, torch.int[32|64]) or a slice, or convertible to a Tensor"
    else:
        type_error_msg = f"{prefix}: index must be either a slice, or convertible to a Tensor"

    index_is_integer = False
    if isinstance(index, int) or (torch.is_tensor(index) and index.ndim == 0 and index.dtype in {torch.int64, torch.int32}):
        index_is_integer = True
    elif isinstance(index, slice):
        pass
    else:
        try:
            index = torch.as_tensor(index, device=device)
        except Exception:
            raise TypeError(type_error_msg)

        if index.ndim != 1:
            raise ValueError("Tensors provided as dataset indices must be 1D")

    if allow_scalar:
        return index, index_is_integer
    else:
        return index


def zero_size_like(reference_tensor: torch.Tensor):
    return torch.zeros((0,)+reference_tensor.shape[1:],
                       dtype=reference_tensor.dtype,
                       device=reference_tensor.device)


def prepend_zero(tensor: torch.Tensor):
    if not torch.is_tensor(tensor) or tensor.ndim != 1:
        raise ValueError(f"prepend_zero expects a one-dimensional tensor")

    zero_1D = torch.zeros((1,), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([zero_1D, tensor])