#!/usr/bin/env python3
"""PyQt5 GUI for viewing raw tiled PNG acquisition grids."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QFileDialog,
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
    QSpinBox,
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

    def wheelEvent(self, event: Any) -> None:
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)


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


class RawTileGridWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Raw PNG Tile Grid Viewer")
        self.resize(1100, 800)
        self.scan_worker: ScanChannelsWorker | None = None
        self.mosaic_worker: BuildMosaicWorker | None = None
        self.channel_dirs: list[Path] = []
        self.current_grid: RawChannelGrid | None = None
        self.mosaic_items: list[tuple[RawTilePlane, np.ndarray, int, int]] = []
        self.tile_graphics: list[tuple[TilePixmapItem, np.ndarray]] = []
        self.grid_rects: list[QGraphicsRectItem] = []
        self.selected_tile: TilePixmapItem | None = None
        self.selection_rect: QGraphicsRectItem | None = None

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
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.selected_tile_info = QTextEdit()
        self.selected_tile_info.setReadOnly(True)
        self.selected_tile_info.setMaximumHeight(96)

        self.scene = QGraphicsScene()
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        self.setCentralWidget(self.build_ui())

    def build_ui(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)

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

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(controls)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(QLabel("Selected Tile"))
        layout.addWidget(self.selected_tile_info)
        layout.addLayout(progress_layout)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log)
        return root

    def choose_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose raw PNG dataset root")
        if path:
            self.root_path.setText(path)
            self.scan_channels()

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
        finite = np.asarray(array, dtype=np.float32)
        low = float(self.display_min.value())
        high = float(self.display_max.value())
        if high <= low:
            high = low + 1.0
        scaled = ((finite - low) * 255.0 / (high - low)).clip(0, 255).astype(np.uint8)
        height, width = scaled.shape
        return QImage(scaled.data, width, height, width, QImage.Format_Grayscale8).copy()

    def update_live_contrast(self) -> None:
        if not self.tile_graphics:
            return
        for pixmap_item, array in self.tile_graphics:
            pixmap_item.setPixmap(QPixmap.fromImage(self.array_to_qimage(array)))

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
