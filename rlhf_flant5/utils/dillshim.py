"""
This is a shim for dill to be used with torch (namely that when used in a project
that pickles torch objects, dill should be imported from this module).
for example::

    from rlhf_flant5.utils.dillshim import dill

The purpose of this shim is register the pickling and unpickling logic
for certain native pytorch types such as torch random generators that
otherwise cannot be pickled by dill, as well as to be able to unpickle
objects that were created in different devices

The above is modified from the following under the BSD 3-Clause License

    https://gist.github.com/maharjun/48b4c583572ecaf655a52fd56b420f9b

Additionally we register additional picklers for GP Models that are memory
efficient. (They don't seem to work as intended. long standing bugfix)
"""

from rlhf_flant5.utils.torchutils import get_device_name

import torch
import io
import dill

def _recreate_generator(gen_state: torch.Tensor, gen_device):
    return_gen: torch.Generator = torch.Generator(device=gen_device)
    return_gen.set_state(gen_state)
    return return_gen

class device_unpickler(dill.Unpickler):
    """
    This is an extension of the dill unpickler that unpickles tensors onto the device specified in the member variable device.

    Examples
    --------

    One can set the device in the class member `device` and unpickle a file as below::

        from rlhf_flant5.utils.generic.dillshim import device_unpickler

        device_unpickler.device = torch.device('cpu')
        with open('pickle_file.p', 'rb') as fin:
            values = device_unpickler(fin).load()

    One may also set the device for each instance of the device_unpickler as follows::

        from rlhf_flant5.utils.generic.dillshim import device_unpickler

        with open('pickle_file.p', 'rb') as fin:
            unpickler = device_unpickler(fin)
            unpickler.device = torch.device('cpu')
            values = unpickler.load()
    """
    device = None
    def find_class(self, module, name):
        if self.device is not None and module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=get_device_name(self.device))
        else: return super().find_class(module, name)
