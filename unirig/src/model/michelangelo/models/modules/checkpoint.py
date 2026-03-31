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
"""
Adapted from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L124
"""

import torch
from typing import Callable, Iterable, Sequence, Union
from packaging import version


def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
    use_deepspeed: bool = False
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    :param use_deepspeed: if True, use deepspeed
    """
    if flag:
        if use_deepspeed:
            import deepspeed
            return deepspeed.checkpointing.checkpoint(func, *inputs)

        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def _get_fwd_decorator():
        if version.parse(torch.__version__) >= version.parse('2.5.0'):
            return torch.amp.custom_fwd(device_type='cuda')
        else:
            return torch.cuda.amp.custom_fwd()

    @staticmethod
    def _get_bwd_decorator():
        if version.parse(torch.__version__) >= version.parse('2.5.0'):
            return torch.amp.custom_bwd(device_type='cuda')
        else:
            def custom_bwd(bwd):
                return torch.cuda.amp.custom_bwd(bwd=bwd)
            return custom_bwd

    @staticmethod
    @_get_fwd_decorator()
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    @_get_bwd_decorator()
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
