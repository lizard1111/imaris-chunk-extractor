#!/usr/bin/env python3
"""PyQt5 GUI for extracting a centered chunk from an Imaris .ims file."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from imaris_chunk_tool.annotations import save_points_csv as write_points_csv
from imaris_chunk_tool.cli import (
    MIDDLE_CHUNK_CHANNEL,
    MIDDLE_CHUNK_RESOLUTION,
    MIDDLE_CHUNK_SIZE_X,
    MIDDLE_CHUNK_SIZE_Y,
    MIDDLE_CHUNK_SIZE_Z,
    MIDDLE_CHUNK_TIMEPOINT,
)
from imaris_chunk_tool.extraction import (
    extract_center_chunk_to_file,
    load_transform_metadata,
)
from imaris_chunk_tool.imaris_io import (
    iter_data_paths,
    require_h5py,
)
from imaris_chunk_tool.transforms import ChunkTransform


class ImageCanvas(QLabel):
    point_clicked = pyqtSignal(str, int, int)

    def __init__(self, plane: str, title: str) -> None:
        super().__init__()
        self.plane = plane
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(280, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background: #111; border: 1px solid #444;")
        self.source_pixmap: QPixmap | None = None
        self.scaled_pixmap: QPixmap | None = None
        self.pixmap_rect = QRect()
        self.image_width = 0
        self.image_height = 0
        self.points: list[tuple[int, int, int]] = []
        self.current_point = (0, 0, 0)
        self.dragging = False
        self.last_emitted: tuple[int, int] | None = None

    def set_image(self, pixmap: QPixmap, width: int, height: int) -> None:
        self.source_pixmap = pixmap
        self.image_width = width
        self.image_height = height
        self.update_scaled_pixmap()

    def set_points(
        self,
        points: list[tuple[int, int, int]],
        current_point: tuple[int, int, int],
    ) -> None:
        self.points = points
        self.current_point = current_point
        self.update()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self.update_scaled_pixmap()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.LeftButton:
            return
        self.dragging = True
        self.emit_position_from_event(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if not self.dragging or not event.buttons() & Qt.LeftButton:
            return
        self.emit_position_from_event(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def leaveEvent(self, event: Any) -> None:
        self.dragging = False
        super().leaveEvent(event)

    def emit_position_from_event(self, event: Any) -> None:
        if not self.source_pixmap or self.pixmap_rect.width() <= 0 or self.pixmap_rect.height() <= 0:
            return
        if not self.pixmap_rect.contains(event.pos()):
            return
        x = int((event.x() - self.pixmap_rect.x()) * self.image_width / self.pixmap_rect.width())
        y = int((event.y() - self.pixmap_rect.y()) * self.image_height / self.pixmap_rect.height())
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))
        if self.last_emitted == (x, y):
            return
        self.last_emitted = (x, y)
        self.point_clicked.emit(self.plane, x, y)

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if not self.scaled_pixmap:
            return
        painter = QPainter(self)
        painter.drawPixmap(self.pixmap_rect, self.scaled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        cx, cy = self.project_point(self.current_point)
        cross_x = self.pixmap_rect.x() + cx * self.pixmap_rect.width() / self.image_width
        cross_y = self.pixmap_rect.y() + cy * self.pixmap_rect.height() / self.image_height
        painter.setPen(QPen(QColor(80, 210, 255), 1))
        painter.drawLine(
            self.pixmap_rect.left(), int(cross_y), self.pixmap_rect.right(), int(cross_y)
        )
        painter.drawLine(
            int(cross_x), self.pixmap_rect.top(), int(cross_x), self.pixmap_rect.bottom()
        )

        painter.setPen(QPen(QColor(255, 60, 60), 2))
        painter.setBrush(QColor(255, 60, 60, 120))
        for point in self.points:
            if not self.point_is_on_plane(point):
                continue
            x, y = self.project_point(point)
            px = self.pixmap_rect.x() + x * self.pixmap_rect.width() / self.image_width
            py = self.pixmap_rect.y() + y * self.pixmap_rect.height() / self.image_height
            painter.drawEllipse(QPoint(int(px), int(py)), 5, 5)

        painter.setPen(QPen(QColor(240, 240, 240), 1))
        painter.drawText(self.pixmap_rect.adjusted(8, 8, -8, -8), Qt.AlignTop | Qt.AlignLeft, self.title)

    def update_scaled_pixmap(self) -> None:
        if not self.source_pixmap:
            self.scaled_pixmap = None
            self.pixmap_rect = QRect()
            self.update()
            return
        self.scaled_pixmap = self.source_pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        x = (self.width() - self.scaled_pixmap.width()) // 2
        y = (self.height() - self.scaled_pixmap.height()) // 2
        self.pixmap_rect = QRect(x, y, self.scaled_pixmap.width(), self.scaled_pixmap.height())
        self.update()

    def point_is_on_plane(self, point: tuple[int, int, int]) -> bool:
        x, y, z = point
        current_x, current_y, current_z = self.current_point
        if self.plane == "xy":
            return z == current_z
        if self.plane == "xz":
            return y == current_y
        if self.plane == "yz":
            return x == current_x
        return False

    def project_point(self, point: tuple[int, int, int]) -> tuple[int, int]:
        x, y, z = point
        if self.plane == "xy":
            return x, y
        if self.plane == "xz":
            return x, z
        if self.plane == "yz":
            return y, z
        return x, y


class ScanWorker(QThread):
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, ims_path: Path) -> None:
        super().__init__()
        self.ims_path = ims_path

    def run(self) -> None:
        try:
            h5py = require_h5py()
            with h5py.File(self.ims_path, "r") as ims:
                rows = list(iter_data_paths(ims))
            self.finished_ok.emit(rows)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ExtractWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        ims_path: Path,
        output_dir: Path,
        size_x: int,
        size_y: int,
        size_z: int,
        channel: int,
        timepoint: int,
        resolution: int,
        output_format: str,
        center_x: int | None,
        center_y: int | None,
        center_z: int | None,
        angle_x: float,
        angle_y: float,
        angle_z: float,
    ) -> None:
        super().__init__()
        self.ims_path = ims_path
        self.output_dir = output_dir
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.channel = channel
        self.timepoint = timepoint
        self.resolution = resolution
        self.output_format = output_format
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z

    def run(self) -> None:
        try:
            result = extract_center_chunk_to_file(
                ims_path=self.ims_path,
                output_dir=self.output_dir,
                size_x=self.size_x,
                size_y=self.size_y,
                size_z=self.size_z,
                channel=self.channel,
                timepoint=self.timepoint,
                resolution=self.resolution,
                output_format=self.output_format,
                center_x=self.center_x,
                center_y=self.center_y,
                center_z=self.center_z,
                angle_x=self.angle_x,
                angle_y=self.angle_y,
                angle_z=self.angle_z,
            )
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class LoadChunkWorker(QThread):
    finished_ok = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            if self.path.suffix.lower() == ".npy":
                array = np.load(self.path)
            elif self.path.suffix.lower() in {".tif", ".tiff"}:
                import tifffile

                array = tifffile.imread(self.path)
            else:
                raise ValueError("Open a .tif, .tiff, or .npy chunk file.")

            if array.ndim == 2:
                array = array[np.newaxis, :, :]
            if array.ndim != 3:
                raise ValueError(
                    f"Expected a 2D image or 3D z,y,x stack, got shape {array.shape}."
                )
            self.finished_ok.emit(self.path, array)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Imaris Chunk Extractor")
        self.resize(780, 560)
        self.scan_worker: ScanWorker | None = None
        self.extract_worker: ExtractWorker | None = None
        self.load_worker: LoadChunkWorker | None = None
        self.dataset_rows: list[tuple[int, int, int, str, tuple[int, ...]]] = []
        self.chunk_array: np.ndarray | None = None
        self.display_min = 0.0
        self.display_max = 1.0
        self.points: list[tuple[int, int, int]] = []
        self.loaded_transform: ChunkTransform | None = None
        self.current_point = (0, 0, 0)

        self.ims_path = QLineEdit()
        self.output_dir = QLineEdit()
        self.dataset_combo = QComboBox()
        self.size_x = self.spinbox(MIDDLE_CHUNK_SIZE_X, maximum=100000)
        self.size_y = self.spinbox(MIDDLE_CHUNK_SIZE_Y, maximum=100000)
        self.size_z = self.spinbox(MIDDLE_CHUNK_SIZE_Z, maximum=100000)
        self.center_x = self.spinbox(0, maximum=1000000)
        self.center_y = self.spinbox(0, maximum=1000000)
        self.center_z = self.spinbox(0, maximum=1000000)
        self.rotation_x = self.double_spinbox(0.0, minimum=-360.0, maximum=360.0)
        self.rotation_y = self.double_spinbox(0.0, minimum=-360.0, maximum=360.0)
        self.rotation_z = self.double_spinbox(0.0, minimum=-360.0, maximum=360.0)
        self.channel = self.spinbox(MIDDLE_CHUNK_CHANNEL, maximum=10000)
        self.timepoint = self.spinbox(MIDDLE_CHUNK_TIMEPOINT, maximum=10000)
        self.resolution = self.spinbox(MIDDLE_CHUNK_RESOLUTION, maximum=10000)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["tif", "npy"])
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.viewer_path = QLineEdit()
        self.contrast_label = QLabel("Contrast: -")
        self.x_slider = QSlider(Qt.Horizontal)
        self.y_slider = QSlider(Qt.Horizontal)
        self.z_slider = QSlider(Qt.Horizontal)
        self.x_slider.setEnabled(False)
        self.y_slider.setEnabled(False)
        self.z_slider.setEnabled(False)
        self.x_slider.valueChanged.connect(self.on_position_slider_changed)
        self.y_slider.valueChanged.connect(self.on_position_slider_changed)
        self.z_slider.valueChanged.connect(self.on_position_slider_changed)
        self.x_label = QLabel("X: -")
        self.y_label = QLabel("Y: -")
        self.z_label = QLabel("Z: -")
        self.xy_canvas = ImageCanvas("xy", "XY")
        self.xz_canvas = ImageCanvas("xz", "XZ")
        self.yz_canvas = ImageCanvas("yz", "YZ")
        self.xy_canvas.point_clicked.connect(self.move_crosshair_from_view)
        self.xz_canvas.point_clicked.connect(self.move_crosshair_from_view)
        self.yz_canvas.point_clicked.connect(self.move_crosshair_from_view)
        self.points_table = QTableWidget(0, 3)
        self.points_table.setHorizontalHeaderLabels(["x", "y", "z"])
        self.points_table.setMinimumWidth(210)
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

        self.extract_button = QPushButton("Extract Center Chunk")
        self.extract_button.clicked.connect(self.extract_center_chunk)

        self.setCentralWidget(self.build_ui())

    def spinbox(self, value: int, maximum: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(0, maximum)
        widget.setValue(value)
        return widget

    def double_spinbox(self, value: float, minimum: float, maximum: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(2)
        widget.setSingleStep(1.0)
        widget.setValue(value)
        return widget

    def build_ui(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)

        file_group = QGroupBox("Files")
        file_layout = QGridLayout(file_group)
        browse_ims = QPushButton("Browse")
        browse_ims.clicked.connect(self.choose_ims_file)
        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self.choose_output_dir)
        scan_button = QPushButton("Scan Datasets")
        scan_button.clicked.connect(self.scan_datasets)

        file_layout.addWidget(QLabel("Imaris file"), 0, 0)
        file_layout.addWidget(self.ims_path, 0, 1)
        file_layout.addWidget(browse_ims, 0, 2)
        file_layout.addWidget(QLabel("Output folder"), 1, 0)
        file_layout.addWidget(self.output_dir, 1, 1)
        file_layout.addWidget(browse_output, 1, 2)
        file_layout.addWidget(QLabel("Dataset"), 2, 0)
        file_layout.addWidget(self.dataset_combo, 2, 1)
        file_layout.addWidget(scan_button, 2, 2)

        settings_group = QGroupBox("Center Chunk")
        settings_layout = QFormLayout(settings_group)
        reset_center = QPushButton("Reset Center")
        reset_center.clicked.connect(self.reset_center_to_dataset)
        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("X"))
        center_row.addWidget(self.center_x)
        center_row.addWidget(QLabel("Y"))
        center_row.addWidget(self.center_y)
        center_row.addWidget(QLabel("Z"))
        center_row.addWidget(self.center_z)
        center_row.addWidget(reset_center)
        rotation_row = QHBoxLayout()
        rotation_row.addWidget(QLabel("X"))
        rotation_row.addWidget(self.rotation_x)
        rotation_row.addWidget(QLabel("Y"))
        rotation_row.addWidget(self.rotation_y)
        rotation_row.addWidget(QLabel("Z"))
        rotation_row.addWidget(self.rotation_z)
        settings_layout.addRow("Size X", self.size_x)
        settings_layout.addRow("Size Y", self.size_y)
        settings_layout.addRow("Size Z", self.size_z)
        settings_layout.addRow("Center", center_row)
        settings_layout.addRow("Rotation degrees", rotation_row)
        settings_layout.addRow("Channel", self.channel)
        settings_layout.addRow("Timepoint", self.timepoint)
        settings_layout.addRow("Resolution", self.resolution)
        settings_layout.addRow("Output format", self.format_combo)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.extract_button)

        viewer_group = QGroupBox("Viewer And Points")
        viewer_layout = QVBoxLayout(viewer_group)
        viewer_file_layout = QHBoxLayout()
        open_chunk = QPushButton("Open Extract")
        open_chunk.clicked.connect(self.choose_chunk_file)
        save_points = QPushButton("Save CSV")
        save_points.clicked.connect(self.save_points_csv)
        add_current = QPushButton("Add Current Point")
        add_current.clicked.connect(self.add_current_point)
        undo_point = QPushButton("Undo Point")
        undo_point.clicked.connect(self.undo_point)
        clear_points = QPushButton("Clear Points")
        clear_points.clicked.connect(self.clear_points)
        viewer_file_layout.addWidget(QLabel("Chunk"))
        viewer_file_layout.addWidget(self.viewer_path)
        viewer_file_layout.addWidget(open_chunk)
        viewer_file_layout.addWidget(add_current)
        viewer_file_layout.addWidget(save_points)
        viewer_file_layout.addWidget(undo_point)
        viewer_file_layout.addWidget(clear_points)

        x_layout = QHBoxLayout()
        x_layout.addWidget(self.x_label)
        x_layout.addWidget(self.x_slider)
        y_layout = QHBoxLayout()
        y_layout.addWidget(self.y_label)
        y_layout.addWidget(self.y_slider)
        z_layout = QHBoxLayout()
        z_layout.addWidget(self.z_label)
        z_layout.addWidget(self.z_slider)
        contrast_layout = QHBoxLayout()
        contrast_layout.addStretch()
        contrast_layout.addWidget(self.contrast_label)

        image_points_layout = QHBoxLayout()
        ortho_layout = QGridLayout()
        ortho_layout.addWidget(self.xy_canvas, 0, 0)
        ortho_layout.addWidget(self.yz_canvas, 0, 1)
        ortho_layout.addWidget(self.xz_canvas, 1, 0)
        image_points_layout.addLayout(ortho_layout, stretch=1)
        image_points_layout.addWidget(self.points_table)

        viewer_layout.addLayout(viewer_file_layout)
        viewer_layout.addLayout(x_layout)
        viewer_layout.addLayout(y_layout)
        viewer_layout.addLayout(z_layout)
        viewer_layout.addLayout(contrast_layout)
        viewer_layout.addLayout(image_points_layout, stretch=1)

        layout.addWidget(file_group)
        layout.addWidget(settings_group)
        layout.addLayout(buttons)
        layout.addWidget(viewer_group, stretch=1)
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)
        layout.addLayout(progress_layout)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log)

        self.dataset_combo.currentIndexChanged.connect(self.apply_dataset_selection)
        return root

    def choose_chunk_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open extracted chunk",
            "",
            "Chunk files (*.tif *.tiff *.npy);;All files (*)",
        )
        if path:
            self.load_chunk(Path(path))

    def choose_ims_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Imaris file", "", "Imaris files (*.ims);;All files (*)"
        )
        if path:
            self.ims_path.setText(path)
            if not self.output_dir.text():
                self.output_dir.setText(str(Path(path).with_suffix("")) + "_chunks")
            self.scan_datasets()

    def choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if path:
            self.output_dir.setText(path)

    def scan_datasets(self) -> None:
        ims_path = Path(self.ims_path.text()).expanduser()
        if not ims_path.exists():
            self.show_error("Choose an existing .ims file first.")
            return

        self.set_busy(True)
        self.set_status("Scanning datasets...", busy=True)
        self.dataset_combo.clear()
        self.log_message(f"Scanning {ims_path}")
        self.scan_worker = ScanWorker(ims_path)
        self.scan_worker.finished_ok.connect(self.on_scan_finished)
        self.scan_worker.failed.connect(self.on_worker_failed)
        self.scan_worker.start()

    def on_scan_finished(self, rows: list[tuple[int, int, int, str, tuple[int, ...]]]) -> None:
        self.set_busy(False)
        self.set_status("Ready", busy=False)
        self.dataset_rows = rows
        self.dataset_combo.clear()
        if not rows:
            self.log_message("No Imaris image datasets found.")
            return

        for resolution, timepoint, channel, _path, shape in rows:
            shape_text = " x ".join(str(item) for item in shape)
            self.dataset_combo.addItem(
                f"R{resolution} T{timepoint} C{channel} - z,y,x {shape_text}",
                userData=(resolution, timepoint, channel),
            )
        self.log_message(f"Found {len(rows)} dataset(s).")
        self.apply_dataset_selection()

    def apply_dataset_selection(self) -> None:
        data: Any = self.dataset_combo.currentData()
        if not data:
            return
        resolution, timepoint, channel = data
        self.resolution.setValue(resolution)
        self.timepoint.setValue(timepoint)
        self.channel.setValue(channel)
        self.reset_center_to_dataset()

    def reset_center_to_dataset(self) -> None:
        index = self.dataset_combo.currentIndex()
        if index < 0 or index >= len(self.dataset_rows):
            return
        _resolution, _timepoint, _channel, _path, shape = self.dataset_rows[index]
        max_z, max_y, max_x = shape
        self.center_x.setRange(0, max_x - 1)
        self.center_y.setRange(0, max_y - 1)
        self.center_z.setRange(0, max_z - 1)
        self.center_x.setValue(max_x // 2)
        self.center_y.setValue(max_y // 2)
        self.center_z.setValue(max_z // 2)

    def extract_center_chunk(self) -> None:
        ims_path = Path(self.ims_path.text()).expanduser()
        output_dir = Path(self.output_dir.text()).expanduser()
        if not ims_path.exists():
            self.show_error("Choose an existing .ims file first.")
            return
        if not str(output_dir):
            self.show_error("Choose an output folder first.")
            return
        if self.size_x.value() == 0 or self.size_y.value() == 0 or self.size_z.value() == 0:
            self.show_error("Chunk sizes must be greater than zero.")
            return

        self.set_busy(True)
        self.set_status("Extracting chunk...", busy=True)
        self.log_message("Extracting center chunk...")
        self.extract_worker = ExtractWorker(
            ims_path=ims_path,
            output_dir=output_dir,
            size_x=self.size_x.value(),
            size_y=self.size_y.value(),
            size_z=self.size_z.value(),
            channel=self.channel.value(),
            timepoint=self.timepoint.value(),
            resolution=self.resolution.value(),
            output_format=self.format_combo.currentText(),
            center_x=self.center_x.value(),
            center_y=self.center_y.value(),
            center_z=self.center_z.value(),
            angle_x=self.rotation_x.value(),
            angle_y=self.rotation_y.value(),
            angle_z=self.rotation_z.value(),
        )
        self.extract_worker.finished_ok.connect(self.on_extract_finished)
        self.extract_worker.failed.connect(self.on_worker_failed)
        self.extract_worker.start()

    def on_extract_finished(self, result: Any) -> None:
        self.set_busy(False)
        chunk = result.chunk
        max_z, max_y, max_x = result.dataset_shape_zyx
        self.log_message(f"Dataset shape z,y,x: {max_z},{max_y},{max_x}")
        self.log_message(
            "Extracted "
            f"x={chunk.x}:{chunk.x + chunk.size_x}, "
            f"y={chunk.y}:{chunk.y + chunk.size_y}, "
            f"z={chunk.z}:{chunk.z + chunk.size_z}"
        )
        self.log_message(f"Saved: {result.output_path}")
        self.log_message(f"Metadata: {result.metadata_path}")
        self.load_chunk(result.output_path)

    def on_worker_failed(self, details: str) -> None:
        self.set_busy(False)
        self.set_status("Ready", busy=False)
        self.log_message(details)
        self.show_error("The operation failed. See the log for details.")

    def set_busy(self, busy: bool) -> None:
        self.extract_button.setEnabled(not busy)

    def set_status(self, message: str, busy: bool) -> None:
        self.status_label.setText(message)
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def load_chunk(self, path: Path) -> None:
        self.set_busy(True)
        self.set_status("Loading chunk...", busy=True)
        self.log_message(f"Loading chunk: {path}")
        self.load_worker = LoadChunkWorker(path)
        self.load_worker.finished_ok.connect(self.on_chunk_loaded)
        self.load_worker.failed.connect(self.on_worker_failed)
        self.load_worker.start()

    def on_chunk_loaded(self, path: Path, array: np.ndarray) -> None:
        self.set_status("Preparing viewer contrast...", busy=True)
        self.chunk_array = array
        self.loaded_transform = load_transform_metadata(path)
        self.display_min, self.display_max = self.compute_display_range(array)
        self.points = []
        self.viewer_path.setText(str(path))
        self.x_slider.setEnabled(True)
        self.y_slider.setEnabled(True)
        self.z_slider.setEnabled(True)
        max_z, max_y, max_x = array.shape
        self.x_slider.setRange(0, max_x - 1)
        self.y_slider.setRange(0, max_y - 1)
        self.z_slider.setRange(0, max_z - 1)
        self.current_point = (max_x // 2, max_y // 2, max_z // 2)
        self.set_sliders_to_current_point()
        self.refresh_points_table()
        self.update_ortho_views()
        self.log_message(f"Loaded chunk: {path} shape z,y,x={array.shape}")
        self.log_message(
            f"Display range fixed across slices: {self.display_min:.3f} to {self.display_max:.3f}"
        )
        if self.loaded_transform is not None:
            self.log_message("Loaded chunk-to-original transform metadata.")
        else:
            self.log_message("No transform metadata sidecar found for this chunk.")
        self.set_busy(False)
        self.set_status("Ready", busy=False)

    def set_sliders_to_current_point(self) -> None:
        x, y, z = self.current_point
        self.x_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.z_slider.blockSignals(True)
        self.x_slider.setValue(x)
        self.y_slider.setValue(y)
        self.z_slider.setValue(z)
        self.x_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.z_slider.blockSignals(False)

    def on_position_slider_changed(self) -> None:
        if self.chunk_array is None:
            return
        self.current_point = (
            self.x_slider.value(),
            self.y_slider.value(),
            self.z_slider.value(),
        )
        self.update_ortho_views()

    def update_ortho_views(self) -> None:
        if self.chunk_array is None:
            return
        x, y, z = self.current_point
        self.x_label.setText(f"X: {x}")
        self.y_label.setText(f"Y: {y}")
        self.z_label.setText(f"Z: {z}")
        self.contrast_label.setText(
            f"Contrast: {self.display_min:.3g} to {self.display_max:.3g}"
        )
        xy = self.chunk_array[z, :, :]
        xz = self.chunk_array[:, y, :]
        yz = self.chunk_array[:, :, x]
        self.set_canvas_image(self.xy_canvas, xy)
        self.set_canvas_image(self.xz_canvas, xz)
        self.set_canvas_image(self.yz_canvas, yz)
        for canvas in (self.xy_canvas, self.xz_canvas, self.yz_canvas):
            canvas.set_points(self.points, self.current_point)

    def set_canvas_image(self, canvas: ImageCanvas, array: np.ndarray) -> None:
        image = self.array_to_qimage(array)
        pixmap = QPixmap.fromImage(image)
        height, width = array.shape
        canvas.set_image(pixmap, width, height)

    def array_to_qimage(self, array: np.ndarray) -> QImage:
        finite = np.asarray(array, dtype=np.float32)
        minimum = self.display_min
        maximum = self.display_max
        if maximum <= minimum:
            scaled = np.zeros(finite.shape, dtype=np.uint8)
        else:
            scaled = ((finite - minimum) * 255.0 / (maximum - minimum)).clip(0, 255).astype(np.uint8)
        height, width = scaled.shape
        return QImage(scaled.data, width, height, width, QImage.Format_Grayscale8).copy()

    def compute_display_range(self, array: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(array, dtype=np.float32)
        stride_z = max(1, finite.shape[0] // 128)
        stride_y = max(1, finite.shape[1] // 512)
        stride_x = max(1, finite.shape[2] // 512)
        sample = finite[::stride_z, ::stride_y, ::stride_x].reshape(-1)
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            return 0.0, 1.0

        low, high = np.percentile(sample, [0.5, 99.5])
        if high <= low:
            low = float(np.min(sample))
            high = float(np.max(sample))
        if high <= low:
            high = low + 1.0
        return float(low), float(high)

    def move_crosshair_from_view(self, plane: str, image_x: int, image_y: int) -> None:
        if self.chunk_array is None:
            return
        x, y, z = self.current_point
        if plane == "xy":
            x, y = image_x, image_y
        elif plane == "xz":
            x, z = image_x, image_y
        elif plane == "yz":
            y, z = image_x, image_y
        self.current_point = (x, y, z)
        self.set_sliders_to_current_point()
        self.update_ortho_views()

    def add_current_point(self) -> None:
        if self.chunk_array is None:
            return
        x, y, z = self.current_point
        self.points.append(self.current_point)
        self.refresh_points_table()
        self.update_ortho_views()
        self.log_message(f"Added point x={x}, y={y}, z={z}")

    def refresh_points_table(self) -> None:
        self.points_table.setRowCount(len(self.points))
        for row, (x, y, z) in enumerate(self.points):
            self.points_table.setItem(row, 0, QTableWidgetItem(str(x)))
            self.points_table.setItem(row, 1, QTableWidgetItem(str(y)))
            self.points_table.setItem(row, 2, QTableWidgetItem(str(z)))

    def undo_point(self) -> None:
        if not self.points:
            return
        self.points.pop()
        self.refresh_points_table()
        self.update_ortho_views()

    def clear_points(self) -> None:
        self.points = []
        self.refresh_points_table()
        self.update_ortho_views()

    def save_points_csv(self) -> None:
        if not self.points:
            self.show_error("There are no points to save yet.")
            return
        default_path = Path(self.viewer_path.text()).with_suffix(".points.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save points CSV", str(default_path), "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        write_points_csv(Path(path), self.points, self.loaded_transform)
        self.log_message(f"Saved {len(self.points)} point(s): {path}")

    def log_message(self, message: str) -> None:
        self.log.append(message)

    def show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Imaris Chunk Extractor", message)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
