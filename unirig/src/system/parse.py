import torch
from torch.optim import Optimizer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter

from .ar import ARSystem, ARWriter
from .skin import SkinSystem, SkinWriter

def get_system(**kwargs) -> LightningModule:
    MAP = {
        'ar': ARSystem,
        'skin': SkinSystem,
    }
    __target__ = kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    del kwargs['__target__']
    return MAP[__target__](**kwargs)

def get_writer(**kwargs) -> BasePredictionWriter:
    MAP = {
        'ar': ARWriter,
        'skin': SkinWriter,
    }
    __target__ = kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    del kwargs['__target__']
    return MAP[__target__](**kwargs)