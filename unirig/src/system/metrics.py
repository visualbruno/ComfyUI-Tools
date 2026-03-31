'''
use metrics from rignet
'''
import torch
from torch import FloatTensor
from typing import Tuple

def J2J(
    joints_a: FloatTensor,
    joints_b: FloatTensor,
    continuous_range: Tuple[float, float]
) -> FloatTensor:
    '''
    joints_a: (J1, 3) joint

    joints_b: (J2, 3) joint
    '''
    dis1 = torch.cdist(joints_a, joints_b)
    loss1, _ = dis1.min(dim=-1)
    dis2 = torch.cdist(joints_b, joints_a)
    loss2, _ = dis2.min(dim=-1)
    s = continuous_range[1] - continuous_range[0]
    return (loss1.mean() + loss2.mean()) / 2 / s

def sample_bones(bones: FloatTensor, num: int=100) -> FloatTensor:
    sample_coord = []
    for i in range(num):
        l = i / num
        sample_coord.append(bones[:, :3] * l + bones[:, 3:] * (1-l))
    return torch.cat(sample_coord)

def points_to_segments_distance(
    coord: FloatTensor,
    bones_head: FloatTensor,
    bones_tail: FloatTensor,
    keepdims: bool,
    eps: float=1e-6
) -> FloatTensor:
    # (J, 3)
    offset = bones_tail - bones_head
    inv = (1./(offset * offset + eps).sum(dim=-1)).unsqueeze(0)
    # head
    g0 = bones_tail.unsqueeze(0) - coord.unsqueeze(1)
    c0 = (g0 * offset.unsqueeze(0)).sum(dim=-1) * inv
    # tail
    g1 = coord.unsqueeze(1) - bones_head.unsqueeze(0)
    c1 = (g1 * offset.unsqueeze(0)).sum(dim=-1) * inv
    # (N, J)
    scale0 = (c0.clamp(min=0., max=1.) + eps) / (c0.clamp(min=0., max=1.) + c1.clamp(min=0., max=1.) + eps * 2)
    scale1 = -scale0 + 1
    # (N, J, 3)
    nearest = scale0.unsqueeze(2) * bones_head.unsqueeze(0) + scale1.unsqueeze(2) * bones_tail.unsqueeze(0)
    # (N, J)
    dis = (coord.unsqueeze(1) - nearest).norm(dim=-1)
    dis, _ = dis.min(dim=-1)
    if keepdims:
        return dis
    return dis.mean()

def J2B(
    joints_a: FloatTensor,
    joints_b: FloatTensor,
    bones_a: FloatTensor,
    bones_b: FloatTensor,
    continuous_range: Tuple[float, float]
) -> FloatTensor:
    '''
    joints_a: (J1, 3) joint
    
    joints_b: (J2, 3) joint

    bones_a: (J1, 6) (position parent, position)

    bones_b: (J2, 6) (position parent, position)
    '''
    s = continuous_range[1] - continuous_range[0]
    
    def one_way_chamfer_dist(joints_a: FloatTensor, joints_b: FloatTensor) -> FloatTensor:
        # for all points in joints_a, calc distance to the nearest point in joints_b and return average distance
        dist = torch.cdist(joints_a, joints_b)
        min_dist, _ = dist.min(dim=1)
        return min_dist.mean()

    sample_a = sample_bones(bones_a)
    sample_b = sample_bones(bones_b)
    return (one_way_chamfer_dist(joints_a, sample_b) + one_way_chamfer_dist(joints_b, sample_a)) / 2 / s

def B2B(
    bones_a: FloatTensor,
    bones_b: FloatTensor,
    continuous_range: Tuple[float, float]
) -> FloatTensor:
    '''
    bones_a: (J1, 6) (position parent, position)

    bones_b: (J2, 6) (position parent, position)
    '''
    s = continuous_range[1] - continuous_range[0]
    sampled_a = sample_bones(bones=bones_a)
    sampled_b = sample_bones(bones=bones_b)
    return J2J(joints_a=sampled_a, joints_b=sampled_b, continuous_range=continuous_range)