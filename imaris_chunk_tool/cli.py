"""Command-line interface for the Imaris chunk tool."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .extraction import ChunkRequest, extract_center_chunk_to_file, extract_chunk, safe_label, write_chunk
from .imaris_io import iter_data_paths, require_h5py


MIDDLE_CHUNK_SIZE_X = 512
MIDDLE_CHUNK_SIZE_Y = 512
MIDDLE_CHUNK_SIZE_Z = 512
MIDDLE_CHUNK_CHANNEL = 1
MIDDLE_CHUNK_TIMEPOINT = 0
MIDDLE_CHUNK_RESOLUTION = 0


def positive_int(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be >= 0, got {parsed}")
    return parsed


def read_manifest(
    path: Path, default_channel: int, default_timepoint: int, default_resolution: int
) -> list[ChunkRequest]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"x", "y", "z", "size_x", "size_y", "size_z"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Manifest is missing required columns: {missing_text}")

        chunks: list[ChunkRequest] = []
        for row_number, row in enumerate(reader, start=2):
            label = row.get("label") or f"chunk_{row_number - 1:04d}"
            try:
                chunk = ChunkRequest(
                    x=positive_int(row["x"], "x"),
                    y=positive_int(row["y"], "y"),
                    z=positive_int(row["z"], "z"),
                    size_x=positive_int(row["size_x"], "size_x"),
                    size_y=positive_int(row["size_y"], "size_y"),
                    size_z=positive_int(row["size_z"], "size_z"),
                    label=safe_label(label),
                    channel=positive_int(row.get("channel") or str(default_channel), "channel"),
                    timepoint=positive_int(
                        row.get("timepoint") or str(default_timepoint), "timepoint"
                    ),
                    resolution=positive_int(
                        row.get("resolution") or str(default_resolution), "resolution"
                    ),
                )
            except ValueError as exc:
                raise ValueError(f"Manifest row {row_number}: {exc}") from exc
            if chunk.size_x == 0 or chunk.size_y == 0 or chunk.size_z == 0:
                raise ValueError(f"Manifest row {row_number}: chunk sizes must be > 0")
            chunks.append(chunk)

    if not chunks:
        raise ValueError("Manifest did not contain any chunk rows")
    return chunks


def list_datasets(ims_path: Path) -> None:
    h5py = require_h5py()
    with h5py.File(ims_path, "r") as ims:
        rows = list(iter_data_paths(ims))

    if not rows:
        print("No Imaris image datasets were found.")
        return

    print("resolution,timepoint,channel,shape_z_y_x,path")
    for resolution, timepoint, channel, path, shape in rows:
        shape_text = "x".join(str(item) for item in shape)
        print(f"{resolution},{timepoint},{channel},{shape_text},{path}")


def run_extract(args: argparse.Namespace) -> None:
    h5py = require_h5py()
    chunks = read_manifest(
        args.manifest,
        default_channel=args.channel,
        default_timepoint=args.timepoint,
        default_resolution=args.resolution,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    with h5py.File(args.ims_file, "r") as ims:
        for index, chunk in enumerate(chunks, start=1):
            array = extract_chunk(ims, chunk)
            stem = (
                f"{index:04d}_{chunk.label}_r{chunk.resolution}"
                f"_t{chunk.timepoint}_c{chunk.channel}"
                f"_x{chunk.x}_y{chunk.y}_z{chunk.z}"
            )
            output_path = write_chunk(array, args.output_dir / stem, args.format)
            written.append(output_path)

    print(f"Extracted {len(written)} chunk(s) to {args.output_dir}")
    for path in written:
        print(path)


def run_middle(args: argparse.Namespace) -> None:
    result = extract_center_chunk_to_file(
        ims_path=args.ims_file,
        output_dir=args.output_dir,
        size_x=MIDDLE_CHUNK_SIZE_X,
        size_y=MIDDLE_CHUNK_SIZE_Y,
        size_z=MIDDLE_CHUNK_SIZE_Z,
        channel=MIDDLE_CHUNK_CHANNEL,
        timepoint=MIDDLE_CHUNK_TIMEPOINT,
        resolution=MIDDLE_CHUNK_RESOLUTION,
        output_format=args.format,
    )
    chunk = result.chunk
    max_z, max_y, max_x = result.dataset_shape_zyx
    print(f"Dataset shape z,y,x: {max_z},{max_y},{max_x}")
    print(
        "Extracted middle chunk "
        f"x={chunk.x}:{chunk.x + chunk.size_x}, "
        f"y={chunk.y}:{chunk.y + chunk.size_y}, "
        f"z={chunk.z}:{chunk.z + chunk.size_z}"
    )
    print(result.output_path)
    print(result.metadata_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a group of voxel chunks from an Imaris .ims file."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List image datasets in an .ims file")
    list_parser.add_argument("ims_file", type=Path)
    list_parser.set_defaults(func=lambda args: list_datasets(args.ims_file))

    extract_parser = subparsers.add_parser("extract", help="Extract chunks from a manifest")
    extract_parser.add_argument("ims_file", type=Path)
    extract_parser.add_argument("manifest", type=Path)
    extract_parser.add_argument("output_dir", type=Path)
    extract_parser.add_argument("--channel", type=int, default=0)
    extract_parser.add_argument("--timepoint", type=int, default=0)
    extract_parser.add_argument("--resolution", type=int, default=0)
    extract_parser.add_argument("--format", choices=("npy", "tif", "tiff"), default="npy")
    extract_parser.set_defaults(func=run_extract)

    middle_parser = subparsers.add_parser(
        "middle", help="Extract one hardwired-size chunk from the center of the volume"
    )
    middle_parser.add_argument("ims_file", type=Path)
    middle_parser.add_argument("output_dir", type=Path)
    middle_parser.add_argument("--format", choices=("npy", "tif", "tiff"), default="tif")
    middle_parser.set_defaults(func=run_middle)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
