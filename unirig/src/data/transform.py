from collections import defaultdict
from dataclasses import dataclass
from typing import Union, List, Tuple
from copy import deepcopy

from .asset import Asset
from .augment import AugmentConfig, Augment, get_augments
from .order import OrderConfig, Order, get_order
from .sampler import SamplerConfig, get_sampler
from .vertex_group import VertexGroupConfig, get_vertex_groups
from .tail import TailConfig, get_tail
from .spec import ConfigSpec

@dataclass
class TransformConfig(ConfigSpec):
    
    tail_config: Union[TailConfig, None]=None,
    
    order_config: Union[OrderConfig, None]=None,
    
    vertex_group_config: Union[VertexGroupConfig, None]=None,
    
    augment_config: Union[AugmentConfig, None]=None,
    
    sampler_config: Union[SamplerConfig, None]=None,
    
    @classmethod
    def parse(cls, config) -> 'TransformConfig':
        cls.check_keys(config)
        tail_config = config.get('tail_config', None)
        order_config = config.get('order_config', None)
        vertex_group_config = config.get('vertex_group_config', None)
        augment_config = config.get('augment_config', None)
        sampler_config = config.get('sampler_config', None)
        
        if tail_config is not None:
            tail_config = TailConfig.parse(config=tail_config)
        if order_config is not None:
            order_config = OrderConfig.parse(config=order_config)
        if vertex_group_config is not None:
            vertex_group_config = VertexGroupConfig.parse(config=vertex_group_config)
        if augment_config is not None:
            augment_config = AugmentConfig.parse(config=augment_config)
        if sampler_config is not None:
            sampler_config = SamplerConfig.parse(config=sampler_config)
        
        return TransformConfig(
            tail_config=tail_config,
            order_config=order_config,
            vertex_group_config=vertex_group_config,
            augment_config=augment_config,
            sampler_config=sampler_config,
        )

def transform_asset(
    asset: Asset,
    transform_config: TransformConfig,
) -> Tuple[List[Augment], List[Augment]]:
    assert isinstance(transform_config, TransformConfig), f"found {type(transform_config)}"
    # 1. try processing tails
    # TODO: use a better method
    if transform_config.tail_config is not None:
        tail = get_tail(config=transform_config.tail_config)
        tail.process_tail(asset=asset)
    
    # 2. arrange bones
    if transform_config.order_config is not None:
        order = get_order(config=transform_config.order_config)
        asset.set_order(order=order)
    
    # 3. collapse must perform first
    if transform_config.augment_config:
        first_augments, second_augments = get_augments(config=transform_config.augment_config)
    else:
        first_augments = []
        second_augments = []
        
    kwargs = {}
    for augment in first_augments:
        augment.transform(asset=asset, **kwargs)
    
    # 4. get vertex groups
    if transform_config.vertex_group_config is not None:
        vertex_groups = get_vertex_groups(config=transform_config.vertex_group_config)
        d = {}
        for v in vertex_groups:
            d.update(v.get_vertex_group(asset=asset))
        asset.vertex_groups = d
    else:
        asset.vertex_groups = {}
    
    # 5. regular augments
    for augment in second_augments:
        augment.transform(asset=asset, **kwargs)
    
    # 6. sample
    if transform_config.sampler_config is not None:
        sampler = get_sampler(config=transform_config.sampler_config)
        res = sampler.sample(asset=asset)
        asset.sampled_vertices      = res.vertices
        asset.sampled_normals       = res.normals
        asset.sampled_vertex_groups = res.vertex_groups
    else:
        asset.sampled_vertices      = asset.vertices.copy()
        asset.sampled_normals       = asset.vertex_normals.copy()
        asset.sampled_vertex_groups = deepcopy(asset.vertex_groups)
    return first_augments, second_augments