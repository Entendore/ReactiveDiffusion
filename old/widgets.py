import datetime
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, 
    QPushButton, QSlider, QDoubleSpinBox, QComboBox, QLabel, 
    QCheckBox, QSpinBox, QFileDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QMouseEvent

import config

class FastImageViewer(QWidget):
    paint_signal = Signal(int, int, int, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel("Initializing...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.label)
        
        self.drawing = False
        self.brush_radius = 10
        self.brush_strength = 0.5

    def update_image(self, rgb_array):
        h, w, ch = rgb_array.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap.scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self._handle_mouse(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing:
            self._handle_mouse(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def _handle_mouse(self, event):
        img_rect = self.label.pixmap().rect() if self.label.pixmap() else self.label.rect()
        widget_rect = self.label.rect()
        x_offset = (widget_rect.width() - img_rect.width()) / 2
        y_offset = (widget_rect.height() - img_rect.height()) / 2
        
        x = event.position().x() - x_offset
        y = event.position().y() - y_offset
        
        scale_x = 512 / img_rect.width()
        scale_y = 512 / img_rect.height()
        
        sim_x = int(x * scale_x)
        sim_y = int(y * scale_y)
        
        self.paint_signal.emit(sim_x, sim_y, self.brush_radius, self.brush_strength)

class ControlPanel(QWidget):
    params_changed = Signal(dict)
    sim_start_clicked = Signal()
    sim_pause_clicked = Signal()
    sim_reset_clicked = Signal()
    clear_clicked = Signal()
    resolution_changed = Signal(int)
    rec_start_clicked = Signal(str, int) 
    rec_stop_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # System Status
        self.status_group = QGroupBox("System Info")
        status_layout = QVBoxLayout()
        self.mode_label = QLabel("Detecting...")
        self.mode_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        status_layout.addWidget(self.mode_label)
        self.status_group.setLayout(status_layout)
        layout.addWidget(self.status_group)

        # Simulation Controls
        ctrl_group = QGroupBox("Simulation Controls")
        ctrl_layout = QVBoxLayout()
        
        h_btns = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start")
        self.btn_start.setCheckable(True)
        self.btn_reset = QPushButton("⏹ Reset")
        h_btns.addWidget(self.btn_start)
        h_btns.addWidget(self.btn_reset)
        ctrl_layout.addLayout(h_btns)

        h_btns2 = QHBoxLayout()
        self.btn_clear = QPushButton("🚿 Clear")
        h_btns2.addWidget(self.btn_clear)
        ctrl_layout.addLayout(h_btns2)

        # Speed
        h_speed = QHBoxLayout()
        h_speed.addWidget(QLabel("Speed:"))
        self.spin_speed = QSlider(Qt.Horizontal)
        self.spin_speed.setRange(1, 100)
        self.spin_speed.setValue(30)
        h_speed.addWidget(self.spin_speed)
        self.lbl_speed_val = QLabel("30")
        h_speed.addWidget(self.lbl_speed_val)
        ctrl_layout.addLayout(h_speed)

        self.check_auto = QCheckBox("Auto-Disturbance")
        self.check_auto.setChecked(True)
        ctrl_layout.addWidget(self.check_auto)

        # Resolution
        h_res = QHBoxLayout()
        h_res.addWidget(QLabel("Resolution:"))
        self.combo_res = QComboBox()
        self.combo_res.addItems(["256 (Fast)", "512 (Balanced)", "1024 (Detail)"])
        self.combo_res.setCurrentIndex(1) 
        h_res.addWidget(self.combo_res)
        ctrl_layout.addLayout(h_res)

        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # Paint Tools
        paint_group = QGroupBox("Paint Tools")
        paint_layout = QVBoxLayout()
        
        h_brush = QHBoxLayout()
        h_brush.addWidget(QLabel("Brush Size:"))
        self.slider_brush = QSlider(Qt.Horizontal)
        self.slider_brush.setRange(2, 50)
        self.slider_brush.setValue(10)
        h_brush.addWidget(self.slider_brush)
        self.lbl_brush = QLabel("10")
        h_brush.addWidget(self.lbl_brush)
        paint_layout.addLayout(h_brush)

        h_str = QHBoxLayout()
        h_str.addWidget(QLabel("Strength:"))
        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(1, 100)
        self.slider_strength.setValue(90)
        h_str.addWidget(self.slider_strength)
        paint_layout.addLayout(h_str)
        
        paint_group.setLayout(paint_layout)
        layout.addWidget(paint_group)

        # Recording
        rec_group = QGroupBox("Video Recording")
        rec_layout = QVBoxLayout()
        h_dur = QHBoxLayout()
        h_dur.addWidget(QLabel("Duration (s):"))
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(0, 600)
        self.spin_duration.setValue(10)
        self.spin_duration.setSpecialValueText("Infinite")
        h_dur.addWidget(self.spin_duration)
        rec_layout.addLayout(h_dur)

        h_rec_btns = QHBoxLayout()
        self.btn_rec_start = QPushButton("🔴 Start Rec")
        self.btn_rec_stop = QPushButton("⏹ Stop Rec")
        self.btn_rec_stop.setEnabled(False)
        h_rec_btns.addWidget(self.btn_rec_start)
        h_rec_btns.addWidget(self.btn_rec_stop)
        rec_layout.addLayout(h_rec_btns)
        
        self.btn_rec_duration = QPushButton("🎬 Record 10s Now")
        rec_layout.addWidget(self.btn_rec_duration)
        
        self.lbl_rec_status = QLabel("Status: Idle")
        self.lbl_rec_status.setStyleSheet("color: #888;")
        rec_layout.addWidget(self.lbl_rec_status)
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        # Physics Parameters
        param_group = QGroupBox("Physics Parameters")
        form = QFormLayout()
        
        self.spin_F = QDoubleSpinBox()
        self.spin_F.setRange(0.001, 0.1)
        self.spin_F.setSingleStep(0.001)
        self.spin_F.setValue(0.0545)
        self.spin_F.setDecimals(4)

        self.spin_k = QDoubleSpinBox()
        self.spin_k.setRange(0.001, 0.1)
        self.spin_k.setSingleStep(0.001)
        self.spin_k.setValue(0.0620)
        self.spin_k.setDecimals(4)

        form.addRow("Feed Rate (F):", self.spin_F)
        form.addRow("Kill Rate (k):", self.spin_k)
        
        self.combo_map = QComboBox()
        self.combo_map.addItems(list(config.COLORMAPS.keys()))
        form.addRow("Colormap:", self.combo_map)
        
        self.check_sharpen = QCheckBox("Sharpen Output")
        form.addRow(self.check_sharpen)
        
        param_group.setLayout(form)
        layout.addWidget(param_group)

        # Presets
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(config.PRESETS.keys()))
        preset_layout.addWidget(self.combo_preset)
        
        self.combo_seed = QComboBox()
        self.combo_seed.addItems(["Multi-Random", "Single Circle"])
        preset_layout.addWidget(QLabel("Seed Type:"))
        preset_layout.addWidget(self.combo_seed)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        layout.addStretch()

        # Connections
        self.btn_start.clicked.connect(self.on_start_toggle)
        self.btn_reset.clicked.connect(self.sim_reset_clicked.emit)
        self.btn_clear.clicked.connect(self.clear_clicked.emit)
        
        self.spin_F.valueChanged.connect(self.emit_params)
        self.spin_k.valueChanged.connect(self.emit_params)
        self.spin_speed.valueChanged.connect(self.on_speed_change)
        self.combo_map.currentTextChanged.connect(self.emit_params)
        self.combo_res.currentTextChanged.connect(self.on_res_change)
        self.combo_preset.currentTextChanged.connect(self.load_preset)
        self.check_auto.stateChanged.connect(self.emit_params)
        self.check_sharpen.stateChanged.connect(self.emit_params)
        
        self.slider_brush.valueChanged.connect(self.on_brush_change)

        self.btn_rec_start.clicked.connect(self.on_rec_start)
        self.btn_rec_stop.clicked.connect(self.on_rec_stop)
        self.btn_rec_duration.clicked.connect(self.on_rec_duration)

    def on_speed_change(self, val):
        self.lbl_speed_val.setText(str(val))
        self.emit_params()
        
    def on_brush_change(self, val):
        self.lbl_brush.setText(str(val))

    def on_res_change(self, text):
        val = int(text.split(" ")[0])
        self.resolution_changed.emit(val)

    def on_start_toggle(self, checked):
        if checked:
            self.btn_start.setText("⏸ Pause")
            self.sim_start_clicked.emit()
        else:
            self.btn_start.setText("▶ Start")
            self.sim_pause_clicked.emit()

    def emit_params(self):
        d = {
            'F': self.spin_F.value(),
            'k': self.spin_k.value(),
            'speed': self.spin_speed.value(),
            'colormap': config.COLORMAPS.get(self.combo_map.currentText(), cv2.COLORMAP_INFERNO),
            'auto_disturbance': self.check_auto.isChecked(),
            'sharpen': self.check_sharpen.isChecked()
        }
        self.params_changed.emit(d)

    def load_preset(self, name):
        if name in config.PRESETS:
            f, k = config.PRESETS[name]
            self.spin_F.setValue(f)
            self.spin_k.setValue(k)

    def on_rec_start(self): self.rec_start_clicked.emit("", 0)
    def on_rec_stop(self): self.rec_stop_clicked.emit()
    def on_rec_duration(self):
        duration = self.spin_duration.value()
        if duration == 0: duration = 10
        self.rec_start_clicked.emit("", duration)

    def set_recording_state(self, is_rec):
        if is_rec:
            self.btn_rec_start.setEnabled(False)
            self.btn_rec_stop.setEnabled(True)
            self.btn_rec_duration.setEnabled(False)
            self.lbl_rec_status.setText("Status: 🔴 Recording...")
            self.lbl_rec_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        else:
            self.btn_rec_start.setEnabled(True)
            self.btn_rec_stop.setEnabled(False)
            self.btn_rec_duration.setEnabled(True)
            self.lbl_rec_status.setText("Status: Idle")
            self.lbl_rec_status.setStyleSheet("color: #888;")