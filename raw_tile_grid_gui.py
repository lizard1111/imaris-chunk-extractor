#!/usr/bin/env python3
"""PyQt5 GUI for viewing raw tiled PNG acquisition grids."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imaris_chunk_tool.raw_tiles import (
    DEFAULT_TILE_OVERLAP_FRACTION,
    RawChannelGrid,
    RawTilePlane,
    discover_channel_dirs,
    discover_channel_grid,
)


def load_pystripe_filter() -> Any:
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "bool"):
        np.bool = bool
    if not hasattr(np, "complex"):
        np.complex = complex
    try:
        from pystripe.core import filter_streaks
    except ImportError as exc:
        raise RuntimeError(
            "PyStripe is not installed in this Python environment. "
            "Install it in the preprocessing conda env to use this option."
        ) from exc
    return filter_streaks


def generate_illumination_correction_map(
    image_shape: tuple[int, int],
    params: dict[str, Any],
) -> np.ndarray | None:
    if not params.get("enable_illumination", False):
        return None

    height, width = image_shape
    tile_cols = max(1, int(params["illumination_tile_cols"]))
    left_factor = float(params["illumination_left_factor"])
    right_factor = float(params["illumination_right_factor"])
    blend_width = max(0, int(params["illumination_blend_width"]))
    correction_map = np.ones((height, width), dtype=np.float32)
    tile_width = max(1, width // tile_cols)
    left_cols = min(2, tile_cols // 2)
    right_cols = min(2, tile_cols // 2)

    for col in range(tile_cols):
        x_start = col * tile_width
        x_end = (col + 1) * tile_width if col < tile_cols - 1 else width
        if col < left_cols:
            correction_map[:, x_start:x_end] *= left_factor
        elif col >= tile_cols - right_cols:
            correction_map[:, x_start:x_end] *= right_factor

    for col in range(1, tile_cols):
        x_center = min(width - 1, col * tile_width)
        blend_start = max(0, x_center - blend_width)
        blend_end = min(width, x_center + blend_width)
        if blend_width > 0 and blend_start < blend_end:
            left_val = correction_map[height // 2, max(0, x_center - 1)]
            right_val = correction_map[height // 2, x_center]
            weights = np.linspace(0.0, 1.0, blend_end - blend_start, dtype=np.float32)
            correction_map[:, blend_start:blend_end] = (
                left_val * (1.0 - weights) + right_val * weights
            )

    if blend_width > 0:
        try:
            from scipy import ndimage

            correction_map = ndimage.gaussian_filter(correction_map, sigma=(0, blend_width / 4.0))
        except ImportError:
            pass
    return correction_map


def apply_pystripe_pipeline(
    array: np.ndarray,
    filter_streaks: Any,
    params: dict[str, Any],
) -> np.ndarray:
    if array.ndim == 3:
        array = array[..., 0]
    original_dtype = array.dtype
    working = np.asarray(array)
    correction_map = generate_illumination_correction_map(working.shape, params)
    if correction_map is not None:
        working = working.astype(np.float32) * correction_map

    corrected = filter_streaks(
        working,
        sigma=[params["sigma1"], params["sigma2"]],
        level=params["level"],
        wavelet=params["wavelet"],
        crossover=params["crossover"],
        threshold=params["threshold"],
        flat=None,
        dark=params["dark"],
    )
    corrected = np.asarray(corrected)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        return np.clip(corrected, info.min, info.max).astype(original_dtype)
    return corrected.astype(original_dtype, copy=False)


class TilePixmapItem(QGraphicsPixmapItem):
    def __init__(self, window: "RawTileGridWindow", plane: RawTilePlane, array: np.ndarray) -> None:
        super().__init__()
        self.window = window
        self.plane = plane
        self.array = array
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.LeftButton)

    def mousePressEvent(self, event: Any) -> None:
        self.window.select_tile(self)
        super().mousePressEvent(event)


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.is_panning = False
        self.last_pan_pos = QPoint()

    def wheelEvent(self, event: Any) -> None:
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.RightButton or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.MetaModifier
        ):
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self.is_panning:
            delta = event.pos() - self.last_pan_pos
            self.last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self.is_panning and event.button() in {Qt.RightButton, Qt.LeftButton}:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ScanChannelsWorker(QThread):
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, raw_root: Path) -> None:
        super().__init__()
        self.raw_root = raw_root

    def run(self) -> None:
        try:
            self.finished_ok.emit(discover_channel_dirs(self.raw_root))
        except Exception:
            self.failed.emit(traceback.format_exc())


class BuildMosaicWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object, list, int, int)
    failed = pyqtSignal(str)

    def __init__(
        self,
        channel_dir: Path,
        thumbnail_size: int,
        overlap_fraction: float,
        flip_vertical: bool,
    ) -> None:
        super().__init__()
        self.channel_dir = channel_dir
        self.thumbnail_size = thumbnail_size
        self.overlap_fraction = overlap_fraction
        self.flip_vertical = flip_vertical

    def run(self) -> None:
        try:
            grid = discover_channel_grid(self.channel_dir)
            loaded: list[tuple[RawTilePlane, np.ndarray]] = []
            total = len(grid.stacks)
            for index, stack in enumerate(grid.stacks, start=1):
                plane = stack.middle_plane
                if plane is None:
                    continue
                self.progress.emit(index, total, f"Loading {plane.path.name}")
                loaded.append((plane, self.load_array(plane.path)))

            tile_width = max((array.shape[1] for _plane, array in loaded), default=self.thumbnail_size)
            tile_height = max((array.shape[0] for _plane, array in loaded), default=self.thumbnail_size)
            step_x = max(1, int(round(tile_width * (1.0 - self.overlap_fraction))))
            step_y = max(1, int(round(tile_height * (1.0 - self.overlap_fraction))))
            items: list[tuple[RawTilePlane, np.ndarray, int, int]] = []
            for plane, array in loaded:
                row_index = grid.n_rows - 1 - plane.row_index if self.flip_vertical else plane.row_index
                x = plane.col_index * step_x
                y = row_index * step_y
                items.append((plane, array, x, y))
            self.finished_ok.emit(grid, items, step_x, step_y)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def load_array(self, path: Path) -> np.ndarray:
        array = iio.imread(path)
        if array.ndim == 3:
            array = array[..., 0]
        array = np.asarray(array)
        if self.thumbnail_size <= 0:
            return array

        height, width = array.shape
        scale = min(self.thumbnail_size / width, self.thumbnail_size / height)
        if scale >= 1.0:
            return array

        try:
            from scipy import ndimage

            return ndimage.zoom(array, zoom=scale, order=1)
        except ImportError:
            stride = max(1, int(round(1.0 / scale)))
            return array[::stride, ::stride]


class TileDiagnosticsWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        tile_dir: Path,
        display_min: int,
        display_max: int,
        slice_step: int,
        use_pystripe: bool,
        pystripe_params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.tile_dir = tile_dir
        self.display_min = display_min
        self.display_max = display_max
        self.slice_step = max(1, slice_step)
        self.use_pystripe = use_pystripe
        self.pystripe_params = pystripe_params

    def run(self) -> None:
        try:
            png_paths = sorted(self.tile_dir.glob("*.png"), key=lambda path: int(path.stem))
            if not png_paths:
                raise ValueError(f"No PNG files found in {self.tile_dir}")

            selected_paths = png_paths[:: self.slice_step]
            if png_paths[-1] not in selected_paths:
                selected_paths.append(png_paths[-1])

            filter_streaks = self.load_pystripe() if self.use_pystripe else None
            planes: list[np.ndarray] = []
            original_planes: list[np.ndarray] = []
            total = len(selected_paths)
            for index, path in enumerate(selected_paths, start=1):
                self.progress.emit(index, total, f"Loading {path.name}")
                array = iio.imread(path)
                if array.ndim == 3:
                    array = array[..., 0]
                array = np.asarray(array, dtype=np.float32)
                original_planes.append(array)
                if filter_streaks is not None:
                    self.progress.emit(index, total, f"PyStripe {path.name}")
                    array = np.asarray(
                        apply_pystripe_pipeline(array, filter_streaks, self.pystripe_params),
                        dtype=np.float32,
                    )
                planes.append(array)
            self.progress.emit(0, 0, "Stacking sampled slices...")
            stack = np.stack(planes, axis=0)
            original_stack = np.stack(original_planes, axis=0)
            original_middle = original_stack[original_stack.shape[0] // 2]
            middle = stack[stack.shape[0] // 2]
            self.progress.emit(1, 6, "Calculating mean projection...")
            mean_projection = np.mean(stack, axis=0)
            self.progress.emit(2, 6, "Calculating median projection...")
            median_projection = np.median(stack, axis=0)
            self.progress.emit(3, 6, "Calculating standard deviation projection...")
            std_projection = np.std(stack, axis=0)
            self.progress.emit(4, 6, "Calculating low-percentile projection...")
            low_projection = np.percentile(stack, 10, axis=0)

            try:
                from scipy import ndimage

                self.progress.emit(5, 6, "Estimating blurred flatfield...")
                flatfield = ndimage.gaussian_filter(low_projection, sigma=120)
            except ImportError:
                flatfield = low_projection
            flatfield = np.maximum(flatfield, np.percentile(flatfield, 1))
            self.progress.emit(6, 6, "Calculating corrected preview...")
            corrected = middle / flatfield * np.median(flatfield)

            self.finished_ok.emit(
                {
                    "tile_dir": self.tile_dir,
                    "n_planes": stack.shape[0],
                    "total_planes": len(png_paths),
                    "slice_step": self.slice_step,
                    "shape": stack.shape,
                    "original_middle": original_middle,
                    "middle": middle,
                    "mean": mean_projection,
                    "median": median_projection,
                    "std": std_projection,
                    "low": low_projection,
                    "flatfield": flatfield,
                    "corrected": corrected,
                    "use_pystripe": self.use_pystripe,
                    "pystripe_params": self.pystripe_params,
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())

    def load_pystripe(self) -> Any:
        return load_pystripe_filter()


class PyStripeProcessWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        stacks: list[Any],
        output_root: Path,
        pystripe_params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.stacks = stacks
        self.output_root = output_root
        self.pystripe_params = pystripe_params

    def run(self) -> None:
        try:
            filter_streaks = load_pystripe_filter()
            planes = [plane for stack in self.stacks for plane in stack.planes]
            if not planes:
                raise ValueError("No PNG planes found to process.")

            total = len(planes)
            written: list[Path] = []
            for index, plane in enumerate(planes, start=1):
                self.progress.emit(index, total, f"PyStripe {plane.path.name}")
                array = iio.imread(plane.path)
                corrected = apply_pystripe_pipeline(array, filter_streaks, self.pystripe_params)
                output_path = (
                    self.output_root
                    / plane.channel
                    / str(plane.stage_x_raw)
                    / f"{plane.stage_x_raw}_{plane.stage_y_raw}"
                    / plane.path.name
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(output_path, corrected)
                written.append(output_path)

            self.finished_ok.emit(
                {
                    "planes": total,
                    "stacks": len(self.stacks),
                    "output_root": self.output_root,
                    "first_output": written[0],
                    "last_output": written[-1],
                    "pystripe_params": self.pystripe_params,
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class PyStripeMosaicPreviewWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        items: list[tuple[RawTilePlane, np.ndarray]],
        pystripe_params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.items = items
        self.pystripe_params = pystripe_params

    def run(self) -> None:
        try:
            filter_streaks = load_pystripe_filter()
            processed: list[tuple[RawTilePlane, np.ndarray]] = []
            total = len(self.items)
            for index, (plane, array) in enumerate(self.items, start=1):
                self.progress.emit(index, total, f"Preview PyStripe {plane.path.name}")
                processed.append(
                    (plane, apply_pystripe_pipeline(array, filter_streaks, self.pystripe_params))
                )
            self.finished_ok.emit(processed)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ImagePreview(QLabel):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220, 220)
        self.setStyleSheet("background: #111; border: 1px solid #444;")

    def set_array(self, array: np.ndarray, display_min: float, display_max: float) -> None:
        finite = np.asarray(array, dtype=np.float32)
        high = display_max if display_max > display_min else display_min + 1.0
        scaled = ((finite - display_min) * 255.0 / (high - display_min)).clip(0, 255).astype(np.uint8)
        height, width = scaled.shape
        image = QImage(scaled.data, width, height, width, QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)
        self.setToolTip(self.title)


class ZoomableArrayPreview(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title_label = QLabel(title)
        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.view, stretch=1)

    def set_array(self, array: np.ndarray, display_min: float, display_max: float) -> None:
        image = array_to_qimage(array, display_min, display_max)
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.scene.setSceneRect(QRectF(0, 0, image.width(), image.height()))
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


def array_to_qimage(array: np.ndarray, display_min: float, display_max: float) -> QImage:
    finite = np.asarray(array, dtype=np.float32)
    high = display_max if display_max > display_min else display_min + 1.0
    scaled = ((finite - display_min) * 255.0 / (high - display_min)).clip(0, 255).astype(np.uint8)
    height, width = scaled.shape
    return QImage(scaled.data, width, height, width, QImage.Format_Grayscale8).copy()


class TileDiagnosticsWindow(QMainWindow):
    def __init__(
        self,
        plane: RawTilePlane,
        display_min: int,
        display_max: int,
        slice_step: int,
        use_pystripe: bool,
        pystripe_params: dict[str, Any],
    ) -> None:
        super().__init__()
        self.plane = plane
        self.display_min = display_min
        self.display_max = display_max
        self.slice_step = slice_step
        self.use_pystripe = use_pystripe
        self.pystripe_params = pystripe_params
        self.worker: TileDiagnosticsWorker | None = None
        self.results: dict[str, Any] = {}
        self.setWindowTitle(f"Tile Diagnostics - {plane.channel} {plane.stage_x_raw}_{plane.stage_y_raw}")
        self.resize(1100, 780)

        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(90)
        self.original_view = ZoomableArrayPreview("Original Middle Slice")
        self.processed_view = ZoomableArrayPreview("Processed Middle Slice")
        self.previews = {
            "mean": ImagePreview("Mean projection"),
            "median": ImagePreview("Median projection"),
            "low": ImagePreview("10th percentile projection"),
            "flatfield": ImagePreview("Estimated flatfield"),
            "corrected": ImagePreview("Flatfield corrected preview"),
        }
        self.setCentralWidget(self.build_ui())
        self.run_diagnostics()

    def build_ui(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        compare = QSplitter(Qt.Horizontal)
        compare.addWidget(self.original_view)
        compare.addWidget(self.processed_view)
        compare.setSizes([1, 1])
        grid = QGridLayout()
        labels = {
            "mean": "Mean Z",
            "median": "Median Z",
            "low": "Low Percentile",
            "flatfield": "Flatfield",
            "corrected": "Flatfield Corrected",
        }
        for index, key in enumerate(labels):
            row = index // 3
            col = index % 3
            cell = QVBoxLayout()
            cell.addWidget(QLabel(labels[key]))
            cell.addWidget(self.previews[key])
            grid.addLayout(cell, row, col)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(self.summary)
        layout.addWidget(compare, stretch=3)
        layout.addLayout(grid, stretch=1)
        layout.addLayout(progress_layout)
        return root

    def run_diagnostics(self) -> None:
        self.status_label.setText("Loading stack...")
        self.progress_bar.setRange(0, 0)
        self.worker = TileDiagnosticsWorker(
            self.plane.path.parent,
            display_min=self.display_min,
            display_max=self.display_max,
            slice_step=self.slice_step,
            use_pystripe=self.use_pystripe,
            pystripe_params=self.pystripe_params,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_progress(self, current: int, total: int, message: str) -> None:
        if total <= 0:
            self.status_label.setText(message)
            self.progress_bar.setRange(0, 0)
            return
        self.status_label.setText(f"{message} ({current}/{total})")
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)

    def on_finished(self, results: dict[str, Any]) -> None:
        self.results = results
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.original_view.set_array(results["original_middle"], self.display_min, self.display_max)
        self.processed_view.set_array(results["middle"], self.display_min, self.display_max)
        for key, preview in self.previews.items():
            preview.set_array(results[key], self.display_min, self.display_max)
        self.summary.setPlainText(
            f"Tile: {results['tile_dir']}\n"
            f"Sampled stack shape z,y,x: {results['shape']}\n"
            f"Sampled planes: {results['n_planes']} of {results['total_planes']} "
            f"(every {results['slice_step']} slice(s))\n"
            f"PyStripe: {results['use_pystripe']} {results['pystripe_params'] if results['use_pystripe'] else ''}\n"
            "Flatfield preview: 10th-percentile Z projection with heavy Gaussian blur."
        )

    def on_failed(self, details: str) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")
        self.summary.setPlainText(details)


class RawTileGridWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Raw PNG Tile Grid Viewer")
        self.resize(1100, 800)
        self.scan_worker: ScanChannelsWorker | None = None
        self.mosaic_worker: BuildMosaicWorker | None = None
        self.pystripe_worker: PyStripeProcessWorker | None = None
        self.pystripe_preview_worker: PyStripeMosaicPreviewWorker | None = None
        self.channel_dirs: list[Path] = []
        self.current_grid: RawChannelGrid | None = None
        self.mosaic_items: list[tuple[RawTilePlane, np.ndarray, int, int]] = []
        self.tile_graphics: list[tuple[TilePixmapItem, np.ndarray]] = []
        self.processed_mosaic_arrays: list[np.ndarray] | None = None
        self.grid_rects: list[QGraphicsRectItem] = []
        self.selected_tile: TilePixmapItem | None = None
        self.selection_rect: QGraphicsRectItem | None = None
        self.diagnostics_windows: list[TileDiagnosticsWindow] = []

        self.root_path = QLineEdit()
        self.channel_combo = QComboBox()
        self.thumbnail_size = QSpinBox()
        self.thumbnail_size.setRange(0, 8192)
        self.thumbnail_size.setValue(0)
        self.thumbnail_size.setSpecialValueText("Original")
        self.overlap_percent = QSpinBox()
        self.overlap_percent.setRange(0, 50)
        self.overlap_percent.setValue(int(DEFAULT_TILE_OVERLAP_FRACTION * 100))
        self.flip_vertical = QCheckBox()
        self.flip_vertical.setChecked(True)
        self.show_grid = QCheckBox()
        self.show_grid.setChecked(True)
        self.show_grid.stateChanged.connect(self.update_grid_visibility)
        self.display_min = QSpinBox()
        self.display_min.setRange(0, 65535)
        self.display_min.setValue(0)
        self.display_min.valueChanged.connect(self.update_live_contrast)
        self.display_max = QSpinBox()
        self.display_max.setRange(1, 65535)
        self.display_max.setValue(500)
        self.display_max.valueChanged.connect(self.update_live_contrast)
        self.diagnostic_slice_step = QSpinBox()
        self.diagnostic_slice_step.setRange(1, 1000)
        self.diagnostic_slice_step.setValue(10)
        self.use_pystripe = QCheckBox()
        self.use_pystripe.setChecked(False)
        self.preview_pystripe_grid = QCheckBox()
        self.preview_pystripe_grid.setChecked(False)
        self.preview_pystripe_grid.stateChanged.connect(self.update_pystripe_grid_preview)
        self.pystripe_wavelet = QComboBox()
        self.pystripe_wavelet.addItems(["db3", "db5", "sym3", "coif1", "bior2.2"])
        self.pystripe_sigma1 = QDoubleSpinBox()
        self.pystripe_sigma1.setRange(0.0, 512.0)
        self.pystripe_sigma1.setSingleStep(0.5)
        self.pystripe_sigma1.setValue(25.6)
        self.pystripe_sigma2 = QDoubleSpinBox()
        self.pystripe_sigma2.setRange(0.0, 512.0)
        self.pystripe_sigma2.setSingleStep(0.5)
        self.pystripe_sigma2.setValue(25.6)
        self.pystripe_level = QSpinBox()
        self.pystripe_level.setRange(0, 8)
        self.pystripe_level.setValue(0)
        self.pystripe_level.setSpecialValueText("auto")
        self.pystripe_crossover = QDoubleSpinBox()
        self.pystripe_crossover.setRange(0.0, 100.0)
        self.pystripe_crossover.setSingleStep(1.0)
        self.pystripe_crossover.setValue(10.0)
        self.pystripe_threshold = QDoubleSpinBox()
        self.pystripe_threshold.setRange(-100.0, 100.0)
        self.pystripe_threshold.setSingleStep(0.5)
        self.pystripe_threshold.setValue(-1.0)
        self.pystripe_dark = QDoubleSpinBox()
        self.pystripe_dark.setRange(0.0, 1000.0)
        self.pystripe_dark.setSingleStep(1.0)
        self.pystripe_dark.setValue(0.0)
        self.output_root = QLineEdit()
        self.pystripe_scope = QComboBox()
        self.pystripe_scope.addItems(["Selected tile stack", "Current channel"])
        self.enable_illumination = QCheckBox()
        self.enable_illumination.setChecked(False)
        self.illumination_tile_cols = QSpinBox()
        self.illumination_tile_cols.setRange(1, 20)
        self.illumination_tile_cols.setValue(4)
        self.illumination_tile_rows = QSpinBox()
        self.illumination_tile_rows.setRange(1, 20)
        self.illumination_tile_rows.setValue(1)
        self.illumination_left_factor = QDoubleSpinBox()
        self.illumination_left_factor.setRange(0.5, 2.0)
        self.illumination_left_factor.setSingleStep(0.05)
        self.illumination_left_factor.setValue(1.0)
        self.illumination_right_factor = QDoubleSpinBox()
        self.illumination_right_factor.setRange(0.5, 2.0)
        self.illumination_right_factor.setSingleStep(0.05)
        self.illumination_right_factor.setValue(1.0)
        self.illumination_blend_width = QSpinBox()
        self.illumination_blend_width.setRange(0, 200)
        self.illumination_blend_width.setValue(50)
        self.connect_pystripe_preview_refresh()
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(96)
        self.selected_tile_info = QTextEdit()
        self.selected_tile_info.setReadOnly(True)
        self.selected_tile_info.setMaximumHeight(76)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        self.setCentralWidget(self.build_ui())

    def connect_pystripe_preview_refresh(self) -> None:
        self.pystripe_wavelet.currentTextChanged.connect(self.invalidate_pystripe_preview)
        for spin_box in (
            self.pystripe_sigma1,
            self.pystripe_sigma2,
            self.pystripe_level,
            self.pystripe_crossover,
            self.pystripe_threshold,
            self.pystripe_dark,
            self.illumination_tile_cols,
            self.illumination_tile_rows,
            self.illumination_left_factor,
            self.illumination_right_factor,
            self.illumination_blend_width,
        ):
            spin_box.valueChanged.connect(self.invalidate_pystripe_preview)
        self.enable_illumination.stateChanged.connect(self.invalidate_pystripe_preview)

    def invalidate_pystripe_preview(self, *_args: Any) -> None:
        self.processed_mosaic_arrays = None
        if self.preview_pystripe_grid.isChecked() and self.tile_graphics:
            self.update_pystripe_grid_preview()

    def build_ui(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Horizontal)

        controls = QGroupBox("Raw Dataset")
        controls_layout = QGridLayout(controls)
        browse = QPushButton("Browse")
        browse.clicked.connect(self.choose_root)
        scan = QPushButton("Scan Channels")
        scan.clicked.connect(self.scan_channels)
        build = QPushButton("Build Middle-Z Grid")
        build.clicked.connect(self.build_mosaic)
        fit = QPushButton("Fit")
        fit.clicked.connect(self.fit_mosaic)
        actual = QPushButton("100%")
        actual.clicked.connect(self.actual_size)
        diagnostics = QPushButton("Tile Diagnostics")
        diagnostics.clicked.connect(self.open_tile_diagnostics)
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self.choose_output_root)
        process_pystripe = QPushButton("Export PNGs")
        process_pystripe.clicked.connect(self.process_pystripe)

        controls_layout.addWidget(QLabel("Root"), 0, 0)
        controls_layout.addWidget(self.root_path, 0, 1)
        controls_layout.addWidget(browse, 0, 2)
        controls_layout.addWidget(QLabel("Channel"), 1, 0)
        controls_layout.addWidget(self.channel_combo, 1, 1)
        controls_layout.addWidget(scan, 1, 2)
        controls_layout.addWidget(QLabel("Thumbnail px"), 2, 0)
        controls_layout.addWidget(self.thumbnail_size, 2, 1)
        controls_layout.addWidget(build, 2, 2)
        controls_layout.addWidget(QLabel("Tile overlap %"), 3, 0)
        controls_layout.addWidget(self.overlap_percent, 3, 1)
        controls_layout.addWidget(QLabel("Flip vertical"), 4, 0)
        controls_layout.addWidget(self.flip_vertical, 4, 1)
        controls_layout.addWidget(QLabel("Grid lines"), 4, 2)
        controls_layout.addWidget(self.show_grid, 4, 3)
        controls_layout.addWidget(QLabel("Display min"), 5, 0)
        controls_layout.addWidget(self.display_min, 5, 1)
        controls_layout.addWidget(QLabel("Display max"), 6, 0)
        controls_layout.addWidget(self.display_max, 6, 1)
        controls_layout.addWidget(fit, 5, 2)
        controls_layout.addWidget(actual, 6, 2)
        controls_layout.addWidget(QLabel("Diag slice step"), 7, 0)
        controls_layout.addWidget(self.diagnostic_slice_step, 7, 1)
        controls_layout.addWidget(diagnostics, 7, 2)

        pystripe_group = QGroupBox("PyStripe")
        pystripe_layout = QFormLayout(pystripe_group)
        pystripe_layout.addRow("Use in diagnostics", self.use_pystripe)
        pystripe_layout.addRow("Preview grid", self.preview_pystripe_grid)
        pystripe_layout.addRow("Wavelet", self.pystripe_wavelet)
        pystripe_layout.addRow("Sigma 1", self.pystripe_sigma1)
        pystripe_layout.addRow("Sigma 2", self.pystripe_sigma2)
        pystripe_layout.addRow("Level", self.pystripe_level)
        pystripe_layout.addRow("Crossover", self.pystripe_crossover)
        pystripe_layout.addRow("Threshold", self.pystripe_threshold)
        pystripe_layout.addRow("Dark offset", self.pystripe_dark)

        processing_group = QGroupBox("Export PyStripe PNGs")
        processing_layout = QGridLayout(processing_group)
        processing_layout.addWidget(QLabel("Output root"), 0, 0)
        processing_layout.addWidget(self.output_root, 0, 1)
        processing_layout.addWidget(browse_output, 0, 2)
        processing_layout.addWidget(QLabel("Scope"), 1, 0)
        processing_layout.addWidget(self.pystripe_scope, 1, 1)
        processing_layout.addWidget(process_pystripe, 1, 2)

        illumination_group = QGroupBox("Illumination Correction")
        illumination_layout = QFormLayout(illumination_group)
        illumination_layout.addRow("Enable", self.enable_illumination)
        illumination_layout.addRow("Tile columns", self.illumination_tile_cols)
        illumination_layout.addRow("Tile rows", self.illumination_tile_rows)
        illumination_layout.addRow("Left columns factor", self.illumination_left_factor)
        illumination_layout.addRow("Right columns factor", self.illumination_right_factor)
        illumination_layout.addRow("Edge blend width", self.illumination_blend_width)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.addWidget(controls)
        sidebar_layout.addWidget(pystripe_group)
        sidebar_layout.addWidget(illumination_group)
        sidebar_layout.addWidget(processing_group)
        sidebar_layout.addStretch()

        sidebar = QScrollArea()
        sidebar.setWidget(sidebar_content)
        sidebar.setWidgetResizable(True)
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(460)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.addWidget(self.view, stretch=1)
        viewer_layout.addWidget(QLabel("Selected Tile"))
        viewer_layout.addWidget(self.selected_tile_info)
        viewer_layout.addLayout(progress_layout)
        viewer_layout.addWidget(QLabel("Log"))
        viewer_layout.addWidget(self.log)

        splitter.addWidget(sidebar)
        splitter.addWidget(viewer_panel)
        splitter.setSizes([380, 1200])
        layout.addWidget(splitter)
        return root

    def choose_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose raw PNG dataset root")
        if path:
            self.root_path.setText(path)
            if not self.output_root.text().strip():
                raw_root = Path(path)
                self.output_root.setText(str(raw_root.parent / f"{raw_root.name}_pystripe"))
            self.scan_channels()

    def choose_output_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose PyStripe output root")
        if path:
            self.output_root.setText(path)

    def scan_channels(self) -> None:
        raw_root = Path(self.root_path.text()).expanduser()
        if not raw_root.exists():
            self.show_error("Choose an existing raw dataset root first.")
            return
        self.set_busy("Scanning channels...", True)
        self.scan_worker = ScanChannelsWorker(raw_root)
        self.scan_worker.finished_ok.connect(self.on_channels_scanned)
        self.scan_worker.failed.connect(self.on_worker_failed)
        self.scan_worker.start()

    def on_channels_scanned(self, channel_dirs: list[Path]) -> None:
        self.set_busy("Ready", False)
        self.channel_dirs = channel_dirs
        self.current_grid = None
        self.selected_tile = None
        self.selected_tile_info.clear()
        self.channel_combo.clear()
        for channel_dir in channel_dirs:
            self.channel_combo.addItem(channel_dir.name, userData=channel_dir)
        self.log_message(f"Found {len(channel_dirs)} channel folder(s).")

    def build_mosaic(self) -> None:
        channel_dir: Path | None = self.channel_combo.currentData()
        if channel_dir is None:
            self.show_error("Scan and choose a channel first.")
            return
        self.scene.clear()
        self.set_busy("Building middle-Z grid...", True)
        self.mosaic_worker = BuildMosaicWorker(
            channel_dir=channel_dir,
            thumbnail_size=self.thumbnail_size.value(),
            overlap_fraction=self.overlap_percent.value() / 100.0,
            flip_vertical=self.flip_vertical.isChecked(),
        )
        self.mosaic_worker.progress.connect(self.on_mosaic_progress)
        self.mosaic_worker.finished_ok.connect(self.on_mosaic_finished)
        self.mosaic_worker.failed.connect(self.on_worker_failed)
        self.mosaic_worker.start()

    def on_mosaic_progress(self, current: int, total: int, message: str) -> None:
        self.status_label.setText(f"{message} ({current}/{total})")
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)

    def on_mosaic_finished(
        self,
        grid: RawChannelGrid,
        items: list[tuple[RawTilePlane, np.ndarray, int, int]],
        step_x: int,
        step_y: int,
    ) -> None:
        self.current_grid = grid
        self.mosaic_items = items
        self.tile_graphics = []
        self.processed_mosaic_arrays = None
        self.preview_pystripe_grid.setChecked(False)
        self.grid_rects = []
        self.selected_tile = None
        self.selection_rect = None
        self.selected_tile_info.clear()
        self.scene.clear()
        pen = QPen(QColor(0, 180, 255), 1)
        for plane, array, x, y in items:
            image = self.array_to_qimage(array)
            pixmap_item = TilePixmapItem(self, plane, array)
            pixmap_item.setPixmap(QPixmap.fromImage(image))
            pixmap_item.setPos(x, y)
            pixmap_item.setToolTip(
                f"{plane.channel}\n"
                f"stage x,y: {plane.stage_x_raw}, {plane.stage_y_raw}\n"
                f"z rel: {plane.z_rel_raw} ({plane.z_rel_um:.1f} um)\n"
                f"{plane.path}"
            )
            self.scene.addItem(pixmap_item)
            rect = self.scene.addRect(QRectF(x, y, image.width(), image.height()), pen)
            rect.setVisible(self.show_grid.isChecked())
            self.grid_rects.append(rect)
            self.tile_graphics.append((pixmap_item, array))

        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.fit_mosaic()
        self.set_busy("Ready", False)
        self.log_message(
            f"Rendered {len(items)} tile(s) for {grid.channel}: "
            f"{grid.n_cols} columns x {grid.n_rows} rows, "
            f"thumbnail={self.thumbnail_label()}, step_x={step_x}px, step_y={step_y}px, "
            f"display={self.display_min.value()}-{self.display_max.value()}, "
            f"flip_vertical={self.flip_vertical.isChecked()}."
        )

    def thumbnail_label(self) -> str:
        value = self.thumbnail_size.value()
        return "Original" if value == 0 else f"{value}px"

    def array_to_qimage(self, array: np.ndarray) -> QImage:
        return array_to_qimage(array, float(self.display_min.value()), float(self.display_max.value()))

    def update_live_contrast(self) -> None:
        if not self.tile_graphics:
            return
        arrays = self.displayed_mosaic_arrays()
        for (pixmap_item, _array), display_array in zip(self.tile_graphics, arrays):
            pixmap_item.setPixmap(QPixmap.fromImage(self.array_to_qimage(display_array)))

    def displayed_mosaic_arrays(self) -> list[np.ndarray]:
        if self.preview_pystripe_grid.isChecked() and self.processed_mosaic_arrays is not None:
            return self.processed_mosaic_arrays
        return [array for _pixmap_item, array in self.tile_graphics]

    def update_pystripe_grid_preview(self) -> None:
        if not self.tile_graphics:
            self.preview_pystripe_grid.setChecked(False)
            return
        if not self.preview_pystripe_grid.isChecked():
            self.update_live_contrast()
            self.log_message("Showing original middle-Z grid.")
            return
        if self.processed_mosaic_arrays is not None:
            self.update_live_contrast()
            self.log_message("Showing PyStripe preview on the middle-Z grid.")
            return
        if self.pystripe_preview_worker is not None and self.pystripe_preview_worker.isRunning():
            return

        items = [(pixmap_item.plane, array) for pixmap_item, array in self.tile_graphics]
        self.set_busy("Building PyStripe grid preview...", True)
        self.pystripe_preview_worker = PyStripeMosaicPreviewWorker(items, self.pystripe_params())
        self.pystripe_preview_worker.progress.connect(self.on_pystripe_progress)
        self.pystripe_preview_worker.finished_ok.connect(self.on_pystripe_preview_finished)
        self.pystripe_preview_worker.failed.connect(self.on_pystripe_preview_failed)
        self.pystripe_preview_worker.start()

    def on_pystripe_preview_finished(self, processed: list[tuple[RawTilePlane, np.ndarray]]) -> None:
        self.processed_mosaic_arrays = [array for _plane, array in processed]
        self.set_busy("Ready", False)
        self.update_live_contrast()
        self.log_message("Showing PyStripe preview on the middle-Z grid.")

    def on_pystripe_preview_failed(self, details: str) -> None:
        self.preview_pystripe_grid.blockSignals(True)
        self.preview_pystripe_grid.setChecked(False)
        self.preview_pystripe_grid.blockSignals(False)
        self.on_worker_failed(details)

    def update_grid_visibility(self) -> None:
        visible = self.show_grid.isChecked()
        for rect in self.grid_rects:
            rect.setVisible(visible)

    def select_tile(self, tile_item: TilePixmapItem) -> None:
        self.selected_tile = tile_item
        scene_rect = tile_item.sceneBoundingRect()
        if self.selection_rect is None:
            pen = QPen(QColor(255, 220, 0), 4)
            self.selection_rect = self.scene.addRect(scene_rect, pen)
            self.selection_rect.setZValue(10)
        else:
            self.selection_rect.setRect(scene_rect)
        plane = tile_item.plane
        self.selected_tile_info.setPlainText(
            f"Channel: {plane.channel}\n"
            f"Stage X/Y: {plane.stage_x_raw}, {plane.stage_y_raw}\n"
            f"Grid col/row: {plane.col_index}, {plane.row_index}\n"
            f"Middle Z: {plane.z_rel_raw} ({plane.z_rel_um:.1f} um), z index {plane.z_index}\n"
            f"Path: {plane.path}"
        )
        self.log_message(
            f"Selected tile {plane.channel} x={plane.stage_x_raw} y={plane.stage_y_raw} "
            f"z={plane.z_rel_raw}"
        )

    def open_tile_diagnostics(self) -> None:
        if self.selected_tile is None:
            self.show_error("Select a tile first.")
            return
        window = TileDiagnosticsWindow(
            self.selected_tile.plane,
            display_min=self.display_min.value(),
            display_max=self.display_max.value(),
            slice_step=self.diagnostic_slice_step.value(),
            use_pystripe=self.use_pystripe.isChecked(),
            pystripe_params=self.pystripe_params(),
        )
        self.diagnostics_windows.append(window)
        window.show()

    def process_pystripe(self) -> None:
        channel_dir: Path | None = self.channel_combo.currentData()
        if channel_dir is None:
            self.show_error("Scan and choose a channel first.")
            return

        output_root_text = self.output_root.text().strip()
        if not output_root_text:
            raw_root = Path(self.root_path.text()).expanduser()
            output_root = raw_root.parent / f"{raw_root.name}_pystripe"
            self.output_root.setText(str(output_root))
        else:
            output_root = Path(output_root_text).expanduser()

        try:
            if self.current_grid is not None and self.current_grid.channel_dir == channel_dir:
                grid = self.current_grid
            else:
                grid = discover_channel_grid(channel_dir)
            if self.pystripe_scope.currentIndex() == 0:
                if self.selected_tile is None:
                    self.show_error("Select a tile first, or change the scope to Current channel.")
                    return
                stacks = [
                    stack
                    for stack in grid.stacks
                    if stack.stack_dir == self.selected_tile.plane.path.parent
                ]
                if not stacks:
                    self.show_error("Could not match the selected tile to the current channel grid.")
                    return
            else:
                stacks = list(grid.stacks)
        except Exception:
            self.on_worker_failed(traceback.format_exc())
            return

        self.set_busy("Running PyStripe...", True)
        self.pystripe_worker = PyStripeProcessWorker(
            stacks=stacks,
            output_root=output_root,
            pystripe_params=self.pystripe_params(),
        )
        self.pystripe_worker.progress.connect(self.on_pystripe_progress)
        self.pystripe_worker.finished_ok.connect(self.on_pystripe_finished)
        self.pystripe_worker.failed.connect(self.on_worker_failed)
        self.pystripe_worker.start()

    def on_pystripe_progress(self, current: int, total: int, message: str) -> None:
        self.status_label.setText(f"{message} ({current}/{total})")
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)

    def on_pystripe_finished(self, results: dict[str, Any]) -> None:
        self.set_busy("Ready", False)
        self.log_message(
            f"PyStripe processed {results['planes']} plane(s) from {results['stacks']} tile stack(s).\n"
            f"Output root: {results['output_root']}\n"
            f"First output: {results['first_output']}\n"
            f"Last output: {results['last_output']}\n"
            f"Parameters: {results['pystripe_params']}"
        )

    def pystripe_params(self) -> dict[str, Any]:
        return {
            "wavelet": self.pystripe_wavelet.currentText(),
            "sigma1": self.pystripe_sigma1.value(),
            "sigma2": self.pystripe_sigma2.value(),
            "level": self.pystripe_level.value(),
            "crossover": self.pystripe_crossover.value(),
            "threshold": self.pystripe_threshold.value(),
            "dark": self.pystripe_dark.value(),
            "enable_illumination": self.enable_illumination.isChecked(),
            "illumination_tile_cols": self.illumination_tile_cols.value(),
            "illumination_tile_rows": self.illumination_tile_rows.value(),
            "illumination_left_factor": self.illumination_left_factor.value(),
            "illumination_right_factor": self.illumination_right_factor.value(),
            "illumination_blend_width": self.illumination_blend_width.value(),
        }

    def fit_mosaic(self) -> None:
        if self.scene.items():
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def actual_size(self) -> None:
        self.view.resetTransform()

    def set_busy(self, message: str, busy: bool) -> None:
        self.status_label.setText(message)
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def on_worker_failed(self, details: str) -> None:
        self.set_busy("Ready", False)
        self.log_message(details)
        self.show_error("Operation failed. See the log for details.")

    def log_message(self, message: str) -> None:
        self.log.append(message)

    def show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Raw PNG Tile Grid Viewer", message)


def main() -> int:
    app = QApplication(sys.argv)
    window = RawTileGridWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
