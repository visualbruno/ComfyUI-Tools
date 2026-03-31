import torch
import numpy as np
from numpy import ndarray
from torch import Tensor, FloatTensor
from typing import Tuple, Union

from scipy.spatial.transform import Rotation as R
from scipy.sparse import csc_matrix
import numpy as np

def quaternion_to_matrix(x, use_4x4=True) -> FloatTensor:
    """
    Ref: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_matrix
    """
    if not isinstance(x, Tensor):
        quaternions = torch.tensor(x, dtype=torch.float32)
    else:
        quaternions = x
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    device = quaternions.device
    
    if use_4x4:
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                torch.zeros(quaternions.shape[:-1], device=device, dtype=torch.float32),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                torch.zeros(quaternions.shape[:-1], device=device, dtype=torch.float32),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
                torch.zeros(quaternions.shape[:-1], device=device, dtype=torch.float32),
                torch.zeros(quaternions.shape[:-1], device=device, dtype=torch.float32),
                torch.zeros(quaternions.shape[:-1], device=device, dtype=torch.float32),
                torch.zeros(quaternions.shape[:-1], device=device, dtype=torch.float32),
                torch.ones(quaternions.shape[:-1], device=device, dtype=torch.float32),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (4, 4))
    else:
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle: FloatTensor) -> FloatTensor:
    """
    Ref: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#axis_angle_to_quaternion
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def axis_angle_to_matrix(axis_angle: Union[FloatTensor, ndarray]) -> Union[FloatTensor, ndarray]:
    """
    Ref: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#axis_angle_to_matrix
    """
    if isinstance(axis_angle, FloatTensor):
        return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
    else:
        res = np.pad(R.from_rotvec(axis_angle).as_matrix(), ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
        assert res.ndim == 3
        res[:, -1, -1] = 1
        return res

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def linear_blend_skinning(
    vertex: Union[FloatTensor, ndarray],
    matrix_local: Union[FloatTensor, ndarray],
    matrix: Union[FloatTensor, ndarray],
    skin: Union[FloatTensor, ndarray],
    pad: int=0,
    value: float=0.,
) -> Union[FloatTensor, ndarray]:
    '''
    Args:
        vertex: (B, N, 4-pad) or (N, 4-pad)
        matrix_local: (B, J, 4, 4) or (J, 4, 4)
        matrix: (B, J, 4, 4) or (J, 4, 4)
        skin: (B, N, J) or (N, J), value of pseudo bones should be 0
    Returns:
        (B, N, 3) or (N, 3)
    '''
    assert vertex.shape[-1] + pad == 4
    if isinstance(vertex, Tensor):
        dims = vertex.dim()
    elif isinstance(vertex, ndarray):
        dims = vertex.ndim
    else:
        raise NotImplementedError()
    if dims == 3:  # Case: (B, N, 3+pad)
        assert isinstance(vertex, Tensor)
        J = matrix_local.shape[1]
        # (B, J, 3+pad, N)
        offset = (
            matrix_local.inverse() @
            torch.nn.functional.pad(vertex, (0, pad, 0, 0, 0, 0), value=value).unsqueeze(1).transpose(2, 3).repeat(1, J, 1, 1)
        )
        # (B, J, 4, N)
        per_bone_matrix = matrix @ offset
        # (B, J, 4, N)
        weighted_per_bone_matrix = skin.transpose(1, 2).unsqueeze(2) * per_bone_matrix
        # (B, 3, N)
        g = weighted_per_bone_matrix.sum(dim=1)
        # (B, 3, N)
        final = g[:, 0:3, :] / (skin.transpose(1, 2).sum(dim=1) + 1e-8).unsqueeze(1)
        return final.permute(0, 2, 1)
    
    elif dims == 2:  # Case: (N, 3+pad)
        if isinstance(vertex, Tensor):
            J = matrix_local.shape[0]
            offset = (
                matrix_local.inverse() @
                torch.nn.functional.pad(vertex, (0, pad, 0, 0), value=value).unsqueeze(0).transpose(1, 2).repeat(J, 1, 1)
            )
            per_bone_matrix = matrix @ offset
            weighted_per_bone_matrix = skin.transpose(0, 1).unsqueeze(1) * per_bone_matrix
            g = weighted_per_bone_matrix.sum(dim=0)
            final = g[0:3, :] / (skin.transpose(0, 1).sum(dim=0) + 1e-8).unsqueeze(0)
            return final.permute(1, 0)  # Output shape (N, 3)
        else:
            J = matrix_local.shape[0]
            N = vertex.shape[0]
            # (4, N)
            padded = np.pad(vertex, ((0, 0), (0, pad)), 'constant', constant_values=(0, value)).T
            # (J, 4, 4)
            trans = matrix @ np.linalg.inv(matrix_local)
            weighted_per_bone_matrix = []
            # (J, N)
            mask = (skin > 0).T
            for i in range(J):
                offset = np.zeros((4, N), dtype=np.float32)
                offset[:, mask[i]] = (trans[i] @ padded[:, mask[i]]) * skin.T[i, mask[i]]
                weighted_per_bone_matrix.append(offset)
            weighted_per_bone_matrix = np.stack(weighted_per_bone_matrix)
            g = np.sum(weighted_per_bone_matrix, axis=0)
            final = g[:3, :] / (np.sum(skin, axis=1) + 1e-8)
            return final.T
    else:
        assert 0, f'unsupported shape: {vertex.shape}'
