from .unirig_ar import UniRigAR
from .unirig_skin import UniRigSkin

from .spec import ModelSpec

def get_model(**kwargs) -> ModelSpec:
    MAP = {
        'unirig_ar': UniRigAR,
        'unirig_skin': UniRigSkin,
    }
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](**kwargs)