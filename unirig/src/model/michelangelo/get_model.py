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

import torch

from .models.tsal.sal_perceiver import AlignedShapeLatentPerceiver, ShapeAsLatentPerceiverEncoder

def get_encoder(
    pretrained_path: str=None,
    freeze_decoder: bool=False,
    **kwargs
) -> AlignedShapeLatentPerceiver:
    model = AlignedShapeLatentPerceiver(**kwargs)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, weights_only=True)
        model.load_state_dict(state_dict)
    if freeze_decoder:
        model.geo_decoder.requires_grad_(False)
        model.encoder.query.requires_grad_(False)
        model.pre_kl.requires_grad_(False)
        model.post_kl.requires_grad_(False)
        model.transformer.requires_grad_(False)
    return model

def get_encoder_simplified(
    pretrained_path: str=None,
    **kwargs
) -> ShapeAsLatentPerceiverEncoder:
    model = ShapeAsLatentPerceiverEncoder(**kwargs)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, weights_only=True)
        model.load_state_dict(state_dict)
    return model