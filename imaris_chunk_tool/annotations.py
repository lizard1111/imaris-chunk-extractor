"""Annotation export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from .transforms import ChunkTransform


def save_points_csv(
    path: Path,
    points_xyz: list[tuple[int, int, int]],
    transform: ChunkTransform | None = None,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        if transform is None:
            writer.writerow(["x", "y", "z"])
            writer.writerows(points_xyz)
            return

        writer.writerow(
            ["chunk_x", "chunk_y", "chunk_z", "original_x", "original_y", "original_z"]
        )
        for point in points_xyz:
            original = transform.chunk_point_to_original_xyz(point)
            writer.writerow([point[0], point[1], point[2], *original])
