import torch
from typing import Optional
from torch.types import Device
import numpy as np

def pc_normalization(pc:torch.Tensor) -> torch.Tensor:
        """ pc: NxC, return NxC """
        centroid = torch.mean(pc, dim=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
        pc = pc / m
        return pc

def random_shuffle(tensor:torch.Tensor) -> torch.Tensor:
    """ pc: NxC, return NxC """
    idx = torch.randperm(tensor.size(0))
    return tensor[idx]

def pc_random_rotation(pc:torch.Tensor) -> torch.Tensor:
    """Apply a random SO(3) rotation to the point cloud pc: Nx3"""
    rand_quat = random_quaternions(1)
    return quaternion_apply(rand_quat, pc)

def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o

def _copysign( a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def quaternion_raw_multiply( a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_invert( quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def pcd_scale_and_translate(pc, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
    # Apply uniform scale and translation per sample
    xyz1 = np.random.uniform(low=scale_low, high=scale_high, size=[3])
    xyz2 = np.random.uniform(low=-translate_range, high=translate_range, size=[3])
    pc[ :, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + \
                    torch.from_numpy(xyz2).float()
    return pc 
if __name__ == "__main__":
# Test the transformations
    pc = torch.tensor([[10.0, 0.0, 0.0],
                        [0.0, 20.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [-10.0, 0.0, 0.0],
                        [0.0, -15.0, 0.0],
                        [0.0, 0.0, -12.0]])
    print("Original Point Cloud:\n", pc)

    # Test Normalization
    pc_normalized = pc_normalization(pc)
    print("Normalized Point Cloud:\n", pc_normalized)

    # Test Random Rotation

    pc_rotated = pc_random_rotation(pc_normalized[:, :3])
    print("Rotated Point Cloud:\n", pc_rotated)

    # Test Random Shuffle

    pc_shuffled = random_shuffle(pc_normalized)
    print("Shuffled Point Cloud:\n", pc_shuffled)
