from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from typing import Dict, Union, List, final
import lightning.pytorch as pl

from ..data.asset import Asset
from ..data.augment import Augment

@dataclass
class ModelInput():
    # tokens for ar input
    tokens: Union[ndarray, None]=None
    
    # pad token
    pad: Union[int, None]=None
    
    # vertices(usually sampled), (N, 3)
    vertices: Union[ndarray, None]=None
    
    # normals(usually sampled), (N, 3)
    normals: Union[ndarray, None]=None
    
    # joints
    joints: Union[ndarray, None]=None
    
    # tails
    tails: Union[ndarray, None]=None
    
    # assets for debug usage
    asset: Union[Asset, None]=None
    
    # augments asset used
    augments: Union[Augment, None]=None

class ModelSpec(pl.LightningModule, ABC):
    
    @abstractmethod
    def __init__(self):
        super().__init__()
    
    @final
    def _process_fn(self, batch: List[ModelInput]) -> List[Dict]:
        '''
        Returns
            cls: List[str]

            path: List[str]
            
            data_name: List[str]
            
            joints: shape (B, J, 3), J==max_bones
            
            tails: shape (B, J, 3)
            
            parents: shape (B, J), -1 represents no parent(should always appear at 0-th position)
            
            num_bones: shape (B), the true number of bones
            
            skin: shape (B, J), padding value==0.
            
            vertices: (B, N, 3)
            
            normals: (B, N, 3)
            
            matrix_local: (B, J, 4, 4), current matrix_local
            
            pose_matrix: (B, J, 4, 4), for motion loss calculation
        '''
        n_batch = self.process_fn(batch)
        BAN = ['cls', 'path', 'data_name', 'joints', 'tails', 'parents', 'num_bones', 'vertices',
               'normals', 'matrix_local', 'pose_matrix', 'num_points', 'origin_vertices',
               'origin_vertex_normals', 'origin_face_normals', 'num_faces', 'faces']
        # skin should be in vertex group
        max_bones = 0
        max_points = 0
        max_faces = 0
        for b in batch:
            if b.joints is not None:
                max_bones = max(max_bones, b.asset.J)
            max_faces = max(max_faces, b.asset.F)
            max_points = max(max_points, b.asset.N)
        self._augments = []
        self._assets = []
        for (id, b) in enumerate(batch):
            for ban in BAN:
                assert ban not in n_batch[id], f"cannot override `{ban}` in process_fn"
            n_batch[id]['cls'] = b.asset.cls
            n_batch[id]['path'] = b.asset.path
            n_batch[id]['data_name'] = b.asset.data_name
            if b.asset.joints is not None:
                n_batch[id]['joints'] = np.pad(b.asset.joints, ((0, max_bones-b.asset.J), (0, 0)), mode='constant', constant_values=0.)
                n_batch[id]['num_bones'] = b.asset.J
            if b.asset.tails is not None:
                n_batch[id]['tails'] = np.pad(b.asset.tails, ((0, max_bones-b.asset.J), (0, 0)), mode='constant', constant_values=0.)
            if b.asset.parents is not None:
                parents = b.asset.parents.copy() # cannot put None into dict
                parents[0] = -1
                parents = np.pad(parents, (0, max_bones-b.asset.J), 'constant', constant_values=-1)
                n_batch[id]['parents'] = parents
            if b.asset.matrix_local is not None:
                J = b.asset.J
                matrix_local = np.pad(b.asset.matrix_local, ((0, max_bones-J), (0, 0), (0, 0)), 'constant', constant_values=0.)
                # set identity to prevent singular matrix in lbs
                matrix_local[J:, 0, 0] = 1.
                matrix_local[J:, 1, 1] = 1.
                matrix_local[J:, 2, 2] = 1.
                matrix_local[J:, 3, 3] = 1.
                n_batch[id]['matrix_local'] = matrix_local
            if b.asset.pose_matrix is not None:
                J = b.asset.J
                pose_matrix = np.pad(b.asset.pose_matrix, ((0, max_bones-J), (0, 0), (0, 0)), 'constant', constant_values=0.)
                pose_matrix[J:, 0, 0] = 1.
                pose_matrix[J:, 1, 1] = 1.
                pose_matrix[J:, 2, 2] = 1.
                pose_matrix[J:, 3, 3] = 1.
                n_batch[id]['pose_matrix'] = pose_matrix
            n_batch[id]['vertices'] = b.vertices
            n_batch[id]['normals'] = b.normals
            n_batch[id]['num_points'] = b.asset.N
            n_batch[id]['origin_vertices'] = np.pad(b.asset.vertices, ((0, max_points-b.asset.N), (0, 0)))
            n_batch[id]['origin_vertex_normals'] = np.pad(b.asset.vertex_normals, ((0, max_points-b.asset.N), (0, 0)))
            n_batch[id]['num_faces'] = b.asset.F
            n_batch[id]['origin_faces'] = np.pad(b.asset.faces, ((0, max_faces-b.asset.F), (0, 0)))
            n_batch[id]['origin_face_normals'] = np.pad(b.asset.face_normals, ((0, max_faces-b.asset.F), (0, 0)))
        return n_batch
    
    @abstractmethod
    def process_fn(self, batch: List[ModelInput]) -> Dict:
        '''
        Fetch data from dataloader and turn it into Tensor objects.
        '''
        pass