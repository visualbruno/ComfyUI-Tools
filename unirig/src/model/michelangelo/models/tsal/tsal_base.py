# -*- coding: utf-8 -*-
#
# This file is part of UniRig.
# 
# This file is derived from https://github.com/NeuralCarver/Michelangelo
#
# Copyright (c) https://github.com/NeuralCarver/Michelangelo original authors
# Copyright (c) 2025 VAST-AI-Research and contributors.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch.nn as nn
from typing import Tuple, List, Optional
import lightning.pytorch as pl


class Point2MeshOutput(object):
    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None
        self.center = None
        self.pc = None


class Latent2MeshOutput(object):

    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None


class AlignedMeshOutput(object):

    def __init__(self):
        self.mesh_v = None
        self.mesh_f = None
        self.surface = None
        self.image = None
        self.text: Optional[str] = None
        self.shape_text_similarity: Optional[float] = None
        self.shape_image_similarity: Optional[float] = None


class ShapeAsLatentPLModule(pl.LightningModule):
    latent_shape: Tuple[int]

    def encode(self, surface, *args, **kwargs):
        raise NotImplementedError

    def decode(self, z_q, *args, **kwargs):
        raise NotImplementedError

    def latent2mesh(self, latents, *args, **kwargs) -> List[Latent2MeshOutput]:
        raise NotImplementedError

    def point2mesh(self, *args, **kwargs) -> List[Point2MeshOutput]:
        raise NotImplementedError


class ShapeAsLatentModule(nn.Module):
    latent_shape: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError


class AlignedShapeAsLatentPLModule(pl.LightningModule):
    latent_shape: Tuple[int]

    def set_shape_model_only(self):
        raise NotImplementedError

    def encode(self, surface, *args, **kwargs):
        raise NotImplementedError

    def decode(self, z_q, *args, **kwargs):
        raise NotImplementedError

    def latent2mesh(self, latents, *args, **kwargs) -> List[Latent2MeshOutput]:
        raise NotImplementedError

    def point2mesh(self, *args, **kwargs) -> List[Point2MeshOutput]:
        raise NotImplementedError


class AlignedShapeAsLatentModule(nn.Module):
    shape_model: ShapeAsLatentModule
    latent_shape: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_shape_model_only(self):
        raise NotImplementedError

    def encode_image_embed(self, *args, **kwargs):
        raise NotImplementedError

    def encode_text_embed(self, *args, **kwargs):
        raise NotImplementedError

    def encode_shape_embed(self, *args, **kwargs):
        raise NotImplementedError


class TexturedShapeAsLatentModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError

    def query_color(self, *args, **kwargs):
        raise NotImplementedError
