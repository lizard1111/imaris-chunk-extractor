# Imaris Chunk Extractor

This small command-line program extracts a group of voxel chunks from an Imaris
`.ims` file. It reads Imaris image datasets directly from the HDF5 structure and
exports each requested region as `.npy` or `.tif`.

The code is split into reusable modules:

```text
imaris_chunk_tool/
  imaris_io.py      # .ims dataset discovery and HDF5 access
  transforms.py     # chunk-to-original affine coordinate transforms
  extraction.py     # crop/rotated extraction and metadata sidecars
  annotations.py    # point CSV export
  raw_tiles.py      # raw PNG tile-folder discovery
  cli.py            # command-line interface
```

Every GUI extraction writes a metadata sidecar next to the output image, for
example `chunk.tif.json`. That sidecar stores the affine transform from chunk
coordinates back to the original full-volume Imaris coordinates. When you save
points from a chunk that has metadata, the CSV includes both chunk coordinates
and original-volume coordinates.

## Install Dependencies

```bash
pip install h5py numpy
```

For TIFF export:

```bash
pip install tifffile
```

For the desktop GUI:

```bash
pip install PyQt5
```

## Open The GUI

```bash
python imaris_chunk_gui.py
```

The GUI lets you choose an `.ims` file, scan available datasets, pick the
channel/timepoint/resolution, choose the extraction center, optionally apply
X/Y/Z rotations in degrees, and extract a test chunk. It can also open the
extracted `.tif` or `.npy` stack, browse orthogonal XY/XZ/YZ views, record
clicked `x,y,z` points, and save those points as CSV.

## Open The Raw PNG Tile Grid GUI

```bash
python raw_tile_grid_gui.py
```

The raw PNG grid viewer expects acquisition folders like:

```text
RawDataset/
  Ex_488_Ch0/
    453370/
      453370_578470/
        000000.png
        000020.png
```

It treats the channel folder as the channel, the column folder as raw stage X,
the tile folder as raw stage X/Y, and the PNG filename as relative Z movement.
For example, `000020.png` is interpreted as `2.0 um` relative Z. The overview
uses the middle Z plane from each tile stack and lays adjacent tiles out with a
default 10% overlap.

## Check What Is In The File

```bash
python extract_imaris_chunks.py list your_file.ims
```

This prints available resolution levels, timepoints, channels, and dataset
shapes in `z,y,x` order.

## Create A Chunk Manifest

Use a CSV with voxel coordinates in `x,y,z` order:

```csv
label,x,y,z,size_x,size_y,size_z,channel,timepoint,resolution
nucleus_region_1,0,0,0,64,64,16,0,0,0
nucleus_region_2,128,96,20,64,64,16,0,0,0
```

The columns `channel`, `timepoint`, and `resolution` are optional. If omitted,
the command-line defaults are used.

## Extract Chunks

```bash
python extract_imaris_chunks.py extract your_file.ims example_chunks.csv extracted_chunks
```

Export TIFF stacks instead of NumPy arrays:

```bash
python extract_imaris_chunks.py extract your_file.ims example_chunks.csv extracted_chunks --format tif
```

## Notes

- Coordinates are voxel indices, not microns.
- Input coordinates are `x,y,z`, but Imaris stores image arrays as `z,y,x`.
- This extracts rectangular cuboid regions. If you mean Imaris Surpass objects
  such as Spots, Surfaces, or a grouped folder in the scene tree, the best route
  is an Imaris XTension instead of direct `.ims` reading.
