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
    QFileDialog,
    QGraphicsPixmapItem,
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
        display_min: int,
        display_max: int,
    ) -> None:
        super().__init__()
        self.channel_dir = channel_dir
        self.thumbnail_size = thumbnail_size
        self.overlap_fraction = overlap_fraction
        self.display_min = display_min
        self.display_max = display_max

    def run(self) -> None:
        try:
            grid = discover_channel_grid(self.channel_dir)
            loaded: list[tuple[RawTilePlane, QImage]] = []
            total = len(grid.stacks)
            for index, stack in enumerate(grid.stacks, start=1):
                plane = stack.middle_plane
                if plane is None:
                    continue
                self.progress.emit(index, total, f"Loading {plane.path.name}")
                image = self.load_contrast_image(plane.path)
                thumb = image.scaled(
                    self.thumbnail_size,
                    self.thumbnail_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                loaded.append((plane, thumb))

            tile_width = max((image.width() for _plane, image in loaded), default=self.thumbnail_size)
            tile_height = max((image.height() for _plane, image in loaded), default=self.thumbnail_size)
            step_x = max(1, int(round(tile_width * (1.0 - self.overlap_fraction))))
            step_y = max(1, int(round(tile_height * (1.0 - self.overlap_fraction))))
            items: list[tuple[RawTilePlane, QImage, int, int]] = []
            for plane, image in loaded:
                x = plane.col_index * step_x
                y = plane.row_index * step_y
                items.append((plane, image, x, y))
            self.finished_ok.emit(grid, items, step_x, step_y)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def load_contrast_image(self, path: Path) -> QImage:
        array = iio.imread(path)
        if array.ndim == 3:
            array = array[..., 0]
        finite = np.asarray(array, dtype=np.float32)
        low = float(self.display_min)
        high = float(self.display_max)
        if high <= low:
            high = low + 1.0
        scaled = ((finite - low) * 255.0 / (high - low)).clip(0, 255).astype(np.uint8)
        height, width = scaled.shape
        return QImage(scaled.data, width, height, width, QImage.Format_Grayscale8).copy()


class RawTileGridWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Raw PNG Tile Grid Viewer")
        self.resize(1100, 800)
        self.scan_worker: ScanChannelsWorker | None = None
        self.mosaic_worker: BuildMosaicWorker | None = None
        self.channel_dirs: list[Path] = []
        self.current_grid: RawChannelGrid | None = None

        self.root_path = QLineEdit()
        self.channel_combo = QComboBox()
        self.thumbnail_size = QSpinBox()
        self.thumbnail_size.setRange(32, 1024)
        self.thumbnail_size.setValue(192)
        self.overlap_percent = QSpinBox()
        self.overlap_percent.setRange(0, 50)
        self.overlap_percent.setValue(int(DEFAULT_TILE_OVERLAP_FRACTION * 100))
        self.display_min = QSpinBox()
        self.display_min.setRange(0, 65535)
        self.display_min.setValue(0)
        self.display_max = QSpinBox()
        self.display_max.setRange(1, 65535)
        self.display_max.setValue(500)
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
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
        controls_layout.addWidget(QLabel("Display min"), 4, 0)
        controls_layout.addWidget(self.display_min, 4, 1)
        controls_layout.addWidget(QLabel("Display max"), 5, 0)
        controls_layout.addWidget(self.display_max, 5, 1)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(controls)
        layout.addWidget(self.view, stretch=1)
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
            display_min=self.display_min.value(),
            display_max=self.display_max.value(),
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
        items: list[tuple[RawTilePlane, QImage, int, int]],
        step_x: int,
        step_y: int,
    ) -> None:
        self.current_grid = grid
        self.scene.clear()
        pen = QPen(QColor(0, 180, 255), 1)
        for plane, image, x, y in items:
            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(image))
            pixmap_item.setPos(x, y)
            pixmap_item.setToolTip(
                f"{plane.channel}\n"
                f"stage x,y: {plane.stage_x_raw}, {plane.stage_y_raw}\n"
                f"z rel: {plane.z_rel_raw} ({plane.z_rel_um:.1f} um)\n"
                f"{plane.path}"
            )
            self.scene.addItem(pixmap_item)
            self.scene.addRect(QRectF(x, y, image.width(), image.height()), pen)

        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.set_busy("Ready", False)
        self.log_message(
            f"Rendered {len(items)} tile(s) for {grid.channel}: "
            f"{grid.n_cols} columns x {grid.n_rows} rows, "
            f"thumbnail={self.thumbnail_size.value()}px, step_x={step_x}px, step_y={step_y}px, "
            f"display={self.display_min.value()}-{self.display_max.value()}."
        )

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
