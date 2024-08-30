from torch.autograd import Function

from sam2.csrc.backend import _backend

__all__ = ['connect']


def connect(mask):
    return _backend.get_connected_componnets(mask)
