#!/usr/bin/env python3
"""Compatibility wrapper for the Imaris chunk extraction CLI."""

from imaris_chunk_tool.cli import (
    MIDDLE_CHUNK_CHANNEL,
    MIDDLE_CHUNK_RESOLUTION,
    MIDDLE_CHUNK_SIZE_X,
    MIDDLE_CHUNK_SIZE_Y,
    MIDDLE_CHUNK_SIZE_Z,
    MIDDLE_CHUNK_TIMEPOINT,
    main,
)
from imaris_chunk_tool.extraction import (
    ChunkRequest,
    ExtractionResult,
    center_chunk_request,
    extract_center_chunk_to_file,
    extract_chunk,
    safe_label,
    write_chunk,
)
from imaris_chunk_tool.imaris_io import data_path, iter_data_paths, require_h5py
from imaris_chunk_tool.transforms import require_numpy


if __name__ == "__main__":
    main()
