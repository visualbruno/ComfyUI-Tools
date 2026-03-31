from typing import List
from heapq import heappush, heappop, heapify
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray

from typing import Dict, Tuple

from .asset import Asset
from .spec import ConfigSpec

@dataclass
class SamplerConfig(ConfigSpec):
    '''
    Config to handle bones re-ordering.
    '''
    # which sampler to use
    method: str
    
    # how many samples in total
    num_samples: int
    
    # how many vertex samples
    vertex_samples: int
    
    # kwargs
    kwargs: Dict[str, Dict]
    
    @classmethod
    def parse(cls, config) -> 'SamplerConfig':
        cls.check_keys(config)
        return SamplerConfig(
            method=config.method,
            num_samples=config.get('num_samples', 0),
            vertex_samples=config.get('vertex_samples', 0),
            kwargs=config.get('kwargs', {}),
        )

@dataclass
class SamplerResult():
    # sampled vertices
    vertices: ndarray
    
    # sampled normals
    normals: ndarray
    
    # sampled vertex groups
    vertex_groups: Dict[str, ndarray]

class Sampler(ABC):
    '''
    Abstract class for samplers.
    '''
    
    def _sample_barycentric(
        self,
        vertex_group: ndarray,
        faces: ndarray,
        face_index: ndarray,
        random_lengths: ndarray,
    ):
        v_origins = vertex_group[faces[face_index, 0]]
        v_vectors = vertex_group[faces[face_index, 1:]]
        v_vectors -= v_origins[:, np.newaxis, :]
        
        sample_vector = (v_vectors * random_lengths).sum(axis=1)
        v_samples = sample_vector + v_origins
        return v_samples
    
    @abstractmethod
    def __init__(self, config: SamplerConfig):
        pass
    
    @abstractmethod
    def sample(
        self,
        asset: Asset,
    ) -> SamplerResult:
        '''
        Return sampled vertices, sampled normals and vertex groups.
        '''
        pass

class SamplerOrigin(Sampler):
    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        self.num_samples    = config.num_samples
        self.vertex_samples = config.vertex_samples
        
    def sample(
        self,
        asset: Asset,
    ) -> SamplerResult:
        perm = np.random.permutation(asset.vertices.shape[0])
        if asset.vertices.shape[0] < self.num_samples:
            m = self.num_samples - asset.vertices.shape[0]
            perm = np.concatenate([perm, np.random.randint(0, asset.vertices.shape[0], (m,))])
        perm = perm[:self.num_samples]
        n_v = asset.vertices[perm]
        n_n = asset.vertex_normals[perm]
        n_vg = {name: v[perm] for name, v in asset.vertex_groups.items()}
        return SamplerResult(
            vertices=n_v,
            normals=n_n,
            vertex_groups=n_vg,
        )

class SamplerMix(Sampler):
    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        self.num_samples    = config.num_samples
        self.vertex_samples = config.vertex_samples
        assert self.num_samples >= self.vertex_samples, 'num_samples should >= vertex_samples'
    
    @property
    def mesh_preserve(self):
        return self.num_samples==-1
    
    def sample(
        self,
        asset: Asset,
    ) -> SamplerResult:
        # 1. sample vertices
        num_samples = self.num_samples
        perm = np.random.permutation(asset.vertices.shape[0])
        vertex_samples = min(self.vertex_samples, asset.vertices.shape[0])
        num_samples -= vertex_samples
        perm = perm[:vertex_samples]
        n_vertex = asset.vertices[perm]
        n_normal = asset.vertex_normals[perm]
        n_v = {name: v[perm] for name, v in asset.vertex_groups.items()}
        
        # 2. sample surface
        perm = np.random.permutation(num_samples)
        vertex_samples, face_index, random_lengths = sample_surface(
            num_samples=num_samples,
            vertices=asset.vertices,
            faces=asset.faces,
            return_weight=True,
        )
        vertex_samples = np.concatenate([n_vertex, vertex_samples], axis=0)
        normal_samples = np.concatenate([n_normal, asset.face_normals[face_index]], axis=0)
        vertex_group_samples = {}
        for n, v in asset.vertex_groups.items():
            g = self._sample_barycentric(
                vertex_group=v,
                faces=asset.faces,
                face_index=face_index,
                random_lengths=random_lengths,
            )
            vertex_group_samples[n] = np.concatenate([n_v[n], g], axis=0)
        return SamplerResult(
            vertices=vertex_samples,
            normals=normal_samples,
            vertex_groups=vertex_group_samples,
        )

def sample_surface(
    num_samples: int,
    vertices: ndarray,
    faces: ndarray,
    return_weight: bool=False,
):
    '''
    Randomly pick samples according to face area.
    
    See sample_surface: https://github.com/mikedh/trimesh/blob/main/trimesh/sample.py
    '''
    # get face area
    offset_0 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    offset_1 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    face_weight = np.cross(offset_0, offset_1, axis=-1)
    face_weight = (face_weight * face_weight).sum(axis=1)
    
    weight_cum = np.cumsum(face_weight, axis=0)
    face_pick = np.random.rand(num_samples) * weight_cum[-1]
    face_index = np.searchsorted(weight_cum, face_pick)
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]]
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]
    
    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = np.random.rand(len(tri_vectors), 2, 1)
    
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)
    vertex_samples = sample_vector + tri_origins
    if not return_weight:
        return vertex_samples
    return vertex_samples, face_index, random_lengths

def get_sampler(config: SamplerConfig) -> Sampler:
    method = config.method
    if method=='origin':
        sampler = SamplerOrigin(config)
    elif method=='mix':
        sampler = SamplerMix(config)
    else:
        raise ValueError(f"sampler method {method} not supported")
    return sampler