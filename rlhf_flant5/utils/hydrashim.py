import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# =============================================
# Setting up hydra related config reading stuff
# =============================================
def _prod(numbers):
    res = 1
    for n in numbers:
        res *= n
    return res


def get_gpu_name_if_available(*args):
    """
    Wrapper to avoid circular imports
    """
    from synth_orig_disc.utils.torchutils import get_gpu_name_if_available
    return get_gpu_name_if_available(*args)


OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
OmegaConf.register_new_resolver(
    "mul", lambda *numbers: _prod(numbers)
)
OmegaConf.register_new_resolver(
    "inv", lambda x: 1.0/float(x)
)
OmegaConf.register_new_resolver(
    "join", lambda *listtojoin: listtojoin[0].join(str(x) for x in listtojoin[1:] if x is not None)
)
OmegaConf.register_new_resolver(
    "joinlist", lambda sep, listtojoin: sep.join(str(x) for x in listtojoin if x is not None)
)
OmegaConf.register_new_resolver(
    "gpu_if_available", get_gpu_name_if_available
)
