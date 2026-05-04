"""Imaris .ims HDF5 access helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable


DATA_PATH_RE = re.compile(
    r"^/DataSet/ResolutionLevel (?P<resolution>\d+)/"
    r"TimePoint (?P<timepoint>\d+)/Channel (?P<channel>\d+)/Data$"
)


def require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("This command requires h5py. Install it with: pip install h5py") from exc
    return h5py


def data_path(resolution: int, timepoint: int, channel: int) -> str:
    return (
        f"/DataSet/ResolutionLevel {resolution}/"
        f"TimePoint {timepoint}/Channel {channel}/Data"
    )


def iter_data_paths(ims: Any) -> Iterable[tuple[int, int, int, str, tuple[int, ...]]]:
    h5py = require_h5py()
    found: list[tuple[int, int, int, str, tuple[int, ...]]] = []

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        path = f"/{name}"
        match = DATA_PATH_RE.match(path)
        if not match:
            return
        found.append(
            (
                int(match.group("resolution")),
                int(match.group("timepoint")),
                int(match.group("channel")),
                path,
                tuple(obj.shape),
            )
        )

    ims.visititems(visitor)
    yield from sorted(found)


def list_dataset_rows(ims_path: Path) -> list[tuple[int, int, int, str, tuple[int, ...]]]:
    h5py = require_h5py()
    with h5py.File(ims_path, "r") as ims:
        return list(iter_data_paths(ims))

