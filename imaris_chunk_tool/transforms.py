"""Coordinate transforms between extracted chunks and original Imaris volumes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("This command requires numpy. Install it with: pip install numpy") from exc
    return np


@dataclass(frozen=True)
class ChunkTransform:
    """Mapping metadata for one extracted chunk.

    Coordinates are stored in XYZ order. Image arrays are still ZYX.
    `chunk_to_original_affine_xyz` maps chunk voxel coordinates into the full
    original dataset coordinate frame.
    """

    source_shape_zyx: tuple[int, int, int]
    chunk_shape_zyx: tuple[int, int, int]
    center_xyz: tuple[int, int, int]
    origin_xyz: tuple[int, int, int]
    rotation_degrees_xyz: tuple[float, float, float]
    chunk_to_original_affine_xyz: list[list[float]]

    def chunk_point_to_original_xyz(self, point_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        np = require_numpy()
        point = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
        affine = np.asarray(self.chunk_to_original_affine_xyz, dtype=np.float64)
        mapped = affine @ point
        return float(mapped[0]), float(mapped[1]), float(mapped[2])


def rotation_matrix_xyz(angle_x: float, angle_y: float, angle_z: float) -> Any:
    np = require_numpy()
    rx = math.radians(angle_x)
    ry = math.radians(angle_y)
    rz = math.radians(angle_z)
    cos_x, sin_x = math.cos(rx), math.sin(rx)
    cos_y, sin_y = math.cos(ry), math.sin(ry)
    cos_z, sin_z = math.cos(rz), math.sin(rz)

    matrix_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]],
        dtype=np.float64,
    )
    matrix_y = np.array(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
        dtype=np.float64,
    )
    matrix_z = np.array(
        [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matrix_z @ matrix_y @ matrix_x


def clamped_origin_from_center(
    source_shape_zyx: tuple[int, int, int],
    chunk_size_xyz: tuple[int, int, int],
    center_xyz: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    max_z, max_y, max_x = source_shape_zyx
    size_x = min(chunk_size_xyz[0], max_x)
    size_y = min(chunk_size_xyz[1], max_y)
    size_z = min(chunk_size_xyz[2], max_z)
    center_x, center_y, center_z = center_xyz
    origin_x = max(0, min(max_x - size_x, center_x - size_x // 2))
    origin_y = max(0, min(max_y - size_y, center_y - size_y // 2))
    origin_z = max(0, min(max_z - size_z, center_z - size_z // 2))
    return origin_x, origin_y, origin_z, size_x, size_y, size_z


def build_chunk_transform(
    source_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    center_xyz: tuple[int, int, int],
    origin_xyz: tuple[int, int, int],
    rotation_degrees_xyz: tuple[float, float, float],
) -> ChunkTransform:
    np = require_numpy()
    size_z, size_y, size_x = chunk_shape_zyx
    angle_x, angle_y, angle_z = rotation_degrees_xyz
    rotation = rotation_matrix_xyz(angle_x, angle_y, angle_z)

    if angle_x or angle_y or angle_z:
        output_center_xyz = np.array(
            [(size_x - 1) / 2.0, (size_y - 1) / 2.0, (size_z - 1) / 2.0],
            dtype=np.float64,
        )
        center = np.asarray(center_xyz, dtype=np.float64)
        translation = center - rotation @ output_center_xyz
    else:
        translation = np.asarray(origin_xyz, dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation
    affine[:3, 3] = translation
    return ChunkTransform(
        source_shape_zyx=source_shape_zyx,
        chunk_shape_zyx=chunk_shape_zyx,
        center_xyz=center_xyz,
        origin_xyz=origin_xyz,
        rotation_degrees_xyz=rotation_degrees_xyz,
        chunk_to_original_affine_xyz=affine.tolist(),
    )

