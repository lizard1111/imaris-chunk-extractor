"""Discovery helpers for raw tiled PNG acquisition folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


Z_RELATIVE_RAW_TO_UM = 0.1
DEFAULT_TILE_OVERLAP_FRACTION = 0.10


@dataclass(frozen=True)
class RawTilePlane:
    channel: str
    stage_x_raw: int
    stage_y_raw: int
    z_rel_raw: int
    z_rel_um: float
    path: Path
    col_index: int
    row_index: int
    z_index: int


@dataclass(frozen=True)
class RawTileStack:
    channel: str
    stage_x_raw: int
    stage_y_raw: int
    stack_dir: Path
    col_index: int
    row_index: int
    planes: tuple[RawTilePlane, ...]

    @property
    def middle_plane(self) -> RawTilePlane | None:
        if not self.planes:
            return None
        return self.planes[len(self.planes) // 2]


@dataclass(frozen=True)
class RawChannelGrid:
    channel: str
    channel_dir: Path
    stacks: tuple[RawTileStack, ...]
    stage_x_values: tuple[int, ...]
    stage_y_values: tuple[int, ...]

    @property
    def n_cols(self) -> int:
        return len(self.stage_x_values)

    @property
    def n_rows(self) -> int:
        return len(self.stage_y_values)


def is_channel_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("Ex_") and "_Ch" in path.name


def discover_channel_dirs(raw_root: Path) -> list[Path]:
    return sorted([path for path in raw_root.iterdir() if is_channel_dir(path)], key=lambda p: p.name)


def parse_tile_stack_dir(tile_dir: Path) -> tuple[int, int]:
    parts = tile_dir.name.split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected tile folder name like X_Y, got {tile_dir.name!r}")
    return int(parts[0]), int(parts[1])


def discover_channel_grid(channel_dir: Path) -> RawChannelGrid:
    column_dirs = sorted(
        [path for path in channel_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    stage_x_values = tuple(int(path.name) for path in column_dirs)
    discovered: list[tuple[int, int, Path]] = []

    for column_dir in column_dirs:
        for tile_dir in sorted([path for path in column_dir.iterdir() if path.is_dir()]):
            stage_x_raw, stage_y_raw = parse_tile_stack_dir(tile_dir)
            discovered.append((stage_x_raw, stage_y_raw, tile_dir))

    stage_y_values = tuple(sorted({stage_y for _stage_x, stage_y, _path in discovered}, reverse=True))
    col_lookup = {stage_x: index for index, stage_x in enumerate(stage_x_values)}
    row_lookup = {stage_y: index for index, stage_y in enumerate(stage_y_values)}
    stacks: list[RawTileStack] = []

    for stage_x_raw, stage_y_raw, tile_dir in discovered:
        png_paths = sorted(tile_dir.glob("*.png"), key=lambda path: int(path.stem))
        col_index = col_lookup[stage_x_raw]
        row_index = row_lookup[stage_y_raw]
        planes = tuple(
            RawTilePlane(
                channel=channel_dir.name,
                stage_x_raw=stage_x_raw,
                stage_y_raw=stage_y_raw,
                z_rel_raw=int(path.stem),
                z_rel_um=int(path.stem) * Z_RELATIVE_RAW_TO_UM,
                path=path,
                col_index=col_index,
                row_index=row_index,
                z_index=z_index,
            )
            for z_index, path in enumerate(png_paths)
        )
        stacks.append(
            RawTileStack(
                channel=channel_dir.name,
                stage_x_raw=stage_x_raw,
                stage_y_raw=stage_y_raw,
                stack_dir=tile_dir,
                col_index=col_index,
                row_index=row_index,
                planes=planes,
            )
        )

    return RawChannelGrid(
        channel=channel_dir.name,
        channel_dir=channel_dir,
        stacks=tuple(sorted(stacks, key=lambda stack: (stack.col_index, stack.row_index))),
        stage_x_values=stage_x_values,
        stage_y_values=stage_y_values,
    )
