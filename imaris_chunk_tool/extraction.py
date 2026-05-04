"""Chunk extraction backend."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .imaris_io import data_path, require_h5py
from .transforms import (
    ChunkTransform,
    build_chunk_transform,
    clamped_origin_from_center,
    require_numpy,
    rotation_matrix_xyz,
)


def require_scipy_ndimage() -> Any:
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError(
            "Rotated extraction requires scipy. Install it with: pip install scipy"
        ) from exc
    return ndimage


@dataclass(frozen=True)
class ChunkRequest:
    x: int
    y: int
    z: int
    size_x: int
    size_y: int
    size_z: int
    label: str
    channel: int
    timepoint: int
    resolution: int


@dataclass(frozen=True)
class ExtractionResult:
    output_path: Path
    metadata_path: Path
    chunk: ChunkRequest
    dataset_shape_zyx: tuple[int, int, int]
    transform: ChunkTransform


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "chunk"


def center_chunk_request(
    dataset_shape_zyx: tuple[int, int, int],
    size_x: int,
    size_y: int,
    size_z: int,
    channel: int,
    timepoint: int,
    resolution: int,
    label: str = "middle",
    center_x: int | None = None,
    center_y: int | None = None,
    center_z: int | None = None,
) -> ChunkRequest:
    max_z, max_y, max_x = dataset_shape_zyx
    center = (
        max_x // 2 if center_x is None else center_x,
        max_y // 2 if center_y is None else center_y,
        max_z // 2 if center_z is None else center_z,
    )
    origin_x, origin_y, origin_z, clipped_x, clipped_y, clipped_z = clamped_origin_from_center(
        dataset_shape_zyx,
        (size_x, size_y, size_z),
        center,
    )
    return ChunkRequest(
        x=origin_x,
        y=origin_y,
        z=origin_z,
        size_x=clipped_x,
        size_y=clipped_y,
        size_z=clipped_z,
        label=safe_label(label),
        channel=channel,
        timepoint=timepoint,
        resolution=resolution,
    )


def extract_chunk(ims: Any, chunk: ChunkRequest) -> Any:
    np = require_numpy()
    path = data_path(chunk.resolution, chunk.timepoint, chunk.channel)
    if path not in ims:
        raise KeyError(f"Could not find dataset {path}")

    dataset = ims[path]
    z_end = chunk.z + chunk.size_z
    y_end = chunk.y + chunk.size_y
    x_end = chunk.x + chunk.size_x
    max_z, max_y, max_x = dataset.shape

    if x_end > max_x or y_end > max_y or z_end > max_z:
        raise ValueError(
            f"{chunk.label} exceeds dataset bounds: requested "
            f"x={chunk.x}:{x_end}, y={chunk.y}:{y_end}, z={chunk.z}:{z_end}; "
            f"available x=0:{max_x}, y=0:{max_y}, z=0:{max_z}"
        )

    return np.asarray(dataset[chunk.z:z_end, chunk.y:y_end, chunk.x:x_end])


def extract_rotated_chunk(
    ims: Any,
    dataset_shape_zyx: tuple[int, int, int],
    center_x: int,
    center_y: int,
    center_z: int,
    size_x: int,
    size_y: int,
    size_z: int,
    channel: int,
    timepoint: int,
    resolution: int,
    angle_x: float,
    angle_y: float,
    angle_z: float,
) -> Any:
    np = require_numpy()
    ndimage = require_scipy_ndimage()
    path = data_path(resolution, timepoint, channel)
    if path not in ims:
        raise KeyError(f"Could not find dataset {path}")

    max_z, max_y, max_x = dataset_shape_zyx
    size_x = min(size_x, max_x)
    size_y = min(size_y, max_y)
    size_z = min(size_z, max_z)
    center_xyz = np.array([center_x, center_y, center_z], dtype=np.float64)
    output_shape_zyx = (size_z, size_y, size_x)
    output_center_zyx = np.array(
        [(size_z - 1) / 2.0, (size_y - 1) / 2.0, (size_x - 1) / 2.0],
        dtype=np.float64,
    )
    output_center_xyz = np.array(
        [(size_x - 1) / 2.0, (size_y - 1) / 2.0, (size_z - 1) / 2.0],
        dtype=np.float64,
    )

    rotation_xyz = rotation_matrix_xyz(angle_x, angle_y, angle_z)
    corners = np.array(
        [
            [x, y, z]
            for x in (0.0, size_x - 1.0)
            for y in (0.0, size_y - 1.0)
            for z in (0.0, size_z - 1.0)
        ],
        dtype=np.float64,
    )
    input_corners = center_xyz + (corners - output_center_xyz) @ rotation_xyz.T
    min_xyz = np.floor(np.min(input_corners, axis=0) - 2.0).astype(int)
    max_xyz = np.ceil(np.max(input_corners, axis=0) + 3.0).astype(int)
    min_xyz = np.maximum(min_xyz, np.array([0, 0, 0]))
    max_xyz = np.minimum(max_xyz, np.array([max_x, max_y, max_z]))
    if np.any(max_xyz <= min_xyz):
        raise ValueError("Rotated chunk falls outside the dataset bounds")

    dataset = ims[path]
    bbox = np.asarray(
        dataset[
            min_xyz[2] : max_xyz[2],
            min_xyz[1] : max_xyz[1],
            min_xyz[0] : max_xyz[0],
        ]
    )

    xyz_from_zyx = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    zyx_from_xyz = xyz_from_zyx.T
    matrix = zyx_from_xyz @ rotation_xyz @ xyz_from_zyx
    offset = zyx_from_xyz @ (center_xyz - min_xyz) - matrix @ output_center_zyx

    return ndimage.affine_transform(
        bbox,
        matrix=matrix,
        offset=offset,
        output_shape=output_shape_zyx,
        order=1,
        mode="constant",
        cval=0,
    )


def write_chunk(array: Any, output_base: Path, output_format: str) -> Path:
    np = require_numpy()
    if output_format == "npy":
        output_path = output_base.with_suffix(".npy")
        np.save(output_path, array)
        return output_path

    if output_format in {"tif", "tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError(
                "TIFF export requires tifffile. Install it with: pip install tifffile"
            ) from exc
        output_path = output_base.with_suffix(".tif")
        tifffile.imwrite(output_path, array, photometric="minisblack")
        return output_path

    raise ValueError(f"Unsupported output format: {output_format}")


def write_metadata(
    output_path: Path,
    ims_path: Path,
    resolution: int,
    timepoint: int,
    channel: int,
    transform: ChunkTransform,
) -> Path:
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    payload = {
        "source_ims": str(ims_path),
        "dataset_path": data_path(resolution, timepoint, channel),
        "resolution": resolution,
        "timepoint": timepoint,
        "channel": channel,
        "transform": asdict(transform),
    }
    metadata_path.write_text(json.dumps(payload, indent=2))
    return metadata_path


def load_transform_metadata(chunk_path: Path) -> ChunkTransform | None:
    metadata_path = chunk_path.with_suffix(chunk_path.suffix + ".json")
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text())
    transform_payload = payload.get("transform")
    if not transform_payload:
        return None
    return ChunkTransform(
        source_shape_zyx=tuple(transform_payload["source_shape_zyx"]),
        chunk_shape_zyx=tuple(transform_payload["chunk_shape_zyx"]),
        center_xyz=tuple(transform_payload["center_xyz"]),
        origin_xyz=tuple(transform_payload["origin_xyz"]),
        rotation_degrees_xyz=tuple(transform_payload["rotation_degrees_xyz"]),
        chunk_to_original_affine_xyz=transform_payload["chunk_to_original_affine_xyz"],
    )


def extract_center_chunk_to_file(
    ims_path: Path,
    output_dir: Path,
    size_x: int,
    size_y: int,
    size_z: int,
    channel: int,
    timepoint: int,
    resolution: int,
    output_format: str,
    label: str = "middle",
    center_x: int | None = None,
    center_y: int | None = None,
    center_z: int | None = None,
    angle_x: float = 0.0,
    angle_y: float = 0.0,
    angle_z: float = 0.0,
) -> ExtractionResult:
    h5py = require_h5py()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(ims_path, "r") as ims:
        path = data_path(resolution, timepoint, channel)
        if path not in ims:
            raise KeyError(f"Could not find dataset {path}")

        dataset_shape = tuple(ims[path].shape)
        max_z, max_y, max_x = dataset_shape
        center = (
            max_x // 2 if center_x is None else center_x,
            max_y // 2 if center_y is None else center_y,
            max_z // 2 if center_z is None else center_z,
        )
        chunk = center_chunk_request(
            dataset_shape,
            size_x=size_x,
            size_y=size_y,
            size_z=size_z,
            channel=channel,
            timepoint=timepoint,
            resolution=resolution,
            label=label,
            center_x=center[0],
            center_y=center[1],
            center_z=center[2],
        )
        transform = build_chunk_transform(
            source_shape_zyx=dataset_shape,
            chunk_shape_zyx=(chunk.size_z, chunk.size_y, chunk.size_x),
            center_xyz=center,
            origin_xyz=(chunk.x, chunk.y, chunk.z),
            rotation_degrees_xyz=(angle_x, angle_y, angle_z),
        )
        if angle_x or angle_y or angle_z:
            array = extract_rotated_chunk(
                ims,
                dataset_shape_zyx=dataset_shape,
                center_x=center[0],
                center_y=center[1],
                center_z=center[2],
                size_x=chunk.size_x,
                size_y=chunk.size_y,
                size_z=chunk.size_z,
                channel=channel,
                timepoint=timepoint,
                resolution=resolution,
                angle_x=angle_x,
                angle_y=angle_y,
                angle_z=angle_z,
            )
        else:
            array = extract_chunk(ims, chunk)

    stem = (
        f"{chunk.label}_r{chunk.resolution}_t{chunk.timepoint}_c{chunk.channel}"
        f"_cx{center[0]}_cy{center[1]}_cz{center[2]}"
        f"_{chunk.size_x}x{chunk.size_y}x{chunk.size_z}"
    )
    if angle_x or angle_y or angle_z:
        stem += f"_rx{angle_x:g}_ry{angle_y:g}_rz{angle_z:g}"
    output_path = write_chunk(array, output_dir / stem, output_format)
    metadata_path = write_metadata(output_path, ims_path, resolution, timepoint, channel, transform)
    return ExtractionResult(
        output_path=output_path,
        metadata_path=metadata_path,
        chunk=chunk,
        dataset_shape_zyx=dataset_shape,
        transform=transform,
    )
