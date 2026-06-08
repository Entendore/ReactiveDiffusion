import sys
import time
import datetime
import warnings
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QVBoxLayout,
    QHBoxLayout, QGroupBox, QFormLayout, QPushButton, QSlider,
    QDoubleSpinBox, QComboBox, QLabel, QCheckBox, QSpinBox,
    QFileDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread, QMutex, QMutexLocker, QTimer
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QPalette, QColor

# ============================================================================
# CONFIG (was config.py)
# ============================================================================

PRESETS = {
    "Coral":   (0.0545, 0.0620),
    "Mitosis": (0.0367, 0.0649),
    "Spirals": (0.0260, 0.0510),
    "Waves":   (0.0140, 0.0450),
    "Spots":   (0.0350, 0.0650)
}

COLORMAPS = {
    "Inferno": cv2.COLORMAP_INFERNO, "Viridis": cv2.COLORMAP_VIRIDIS,
    "Plasma": cv2.COLORMAP_PLASMA, "Magma": cv2.COLORMAP_MAGMA,
    "Cividis": cv2.COLORMAP_CIVIDIS, "Turbo": cv2.COLORMAP_TURBO,
    "Jet": cv2.COLORMAP_JET, "Rainbow": cv2.COLORMAP_RAINBOW,
    "Hot": cv2.COLORMAP_HOT, "Cool": cv2.COLORMAP_COOL,
    "Ocean": cv2.COLORMAP_OCEAN, "Twilight": cv2.COLORMAP_TWILIGHT,
    "DeepGreen": cv2.COLORMAP_DEEPGREEN, "Bone": cv2.COLORMAP_BONE
}

STYLESHEET = """
    QMainWindow { background-color: #1e1e1e; }
    QWidget { color: #ddd; font-size: 12px; }
    QGroupBox {
        font-weight: bold;
        border: 1px solid #3a3a3a;
        border-radius: 5px;
        margin-top: 10px;
        padding-top: 10px;
        background-color: #252525;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
    QPushButton {
        background-color: #333;
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #555;
    }
    QPushButton:hover { background-color: #444; border: 1px solid #777; }
    QPushButton:checked { background-color: #005500; border: 1px solid #007700; }
    QPushButton:disabled { background-color: #222; color: #666; }
    QSlider::groove:horizontal { height: 4px; background: #444; }
    QSlider::handle:horizontal { background: #0078d7; width: 14px; margin: -5px 0; border-radius: 7px; }
    QComboBox { background-color: #333; border: 1px solid #555; padding: 5px; }
    QCheckBox::indicator { width: 15px; height: 15px; }
    QSpinBox, QDoubleSpinBox { background-color: #333; border: 1px solid #555; }
    QSplitter::handle { background-color: #333; }
"""

# ============================================================================
# ENGINE (was engine.py)
# ============================================================================

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy found: GPU Acceleration Enabled.")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found: Running on CPU (Numba Optimized).")

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator

warnings.filterwarnings('ignore')


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def numba_step_buffered(U_src, V_src, U_dst, V_dst, Du, Dv, F, k, dt):
    rows, cols = U_src.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            Lu = (U_src[i+1, j] + U_src[i-1, j] + U_src[i, j+1] + U_src[i, j-1] - 4 * U_src[i, j])
            Lv = (V_src[i+1, j] + V_src[i-1, j] + V_src[i, j+1] + V_src[i, j-1] - 4 * V_src[i, j])
            uvv = U_src[i, j] * V_src[i, j] * V_src[i, j]
            U_dst[i, j] = U_src[i, j] + (Du * Lu - uvv + F * (1 - U_src[i, j])) * dt
            V_dst[i, j] = V_src[i, j] + (Dv * Lv + uvv - (F + k) * V_src[i, j]) * dt


class SimulationEngine:
    def __init__(self, width=256, height=256, use_gpu=True):
        self.width = width
        self.height = height
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.mutex = QMutex()
        self.F = 0.0545
        self.k = 0.0620
        self.Du = 0.2
        self.Dv = 0.1
        self.reset()

    def _allocate_arrays(self):
        lib = cp if self.use_gpu else np
        shape = (self.height, self.width)
        self.U_src = lib.ones(shape, dtype=np.float32)
        self.V_src = lib.zeros(shape, dtype=np.float32)
        self.U_dst = lib.ones(shape, dtype=np.float32)
        self.V_dst = lib.zeros(shape, dtype=np.float32)

    def reset(self, seed='multi_random'):
        self._allocate_arrays()
        lib = cp if self.use_gpu else np
        y, x = lib.meshgrid(lib.arange(self.height), lib.arange(self.width), indexing='ij')
        if seed == 'multi_random':
            for _ in range(15):
                cx = np.random.randint(0, self.width)
                cy = np.random.randint(0, self.height)
                r = np.random.randint(5, 25)
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                val = np.random.uniform(0.2, 0.5)
                self.U_src[mask] = 0.5
                self.V_src[mask] = val
        elif seed == 'circle':
            cx, cy = self.width // 2, self.height // 2
            r = self.height // 8
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            self.U_src[mask] = 0.5
            self.V_src[mask] = 0.25
        lib.copyto(self.U_dst, self.U_src)
        lib.copyto(self.V_dst, self.V_src)

    def clear(self):
        lib = cp if self.use_gpu else np
        self.U_src.fill(1.0)
        self.V_src.fill(0.0)
        self.U_dst.fill(1.0)
        self.V_dst.fill(0.0)

    def add_random_disturbance(self, num_blobs=5):
        with QMutexLocker(self.mutex):
            lib = cp if self.use_gpu else np
            y, x = lib.meshgrid(lib.arange(self.height), lib.arange(self.width), indexing='ij')
            for _ in range(num_blobs):
                cx = int(lib.random.randint(0, self.width))
                cy = int(lib.random.randint(0, self.height))
                r = int(lib.random.randint(3, 10))
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                self.U_src[mask] = 0.5
                self.V_src[mask] = lib.random.uniform(0.2, 0.4)

    def paint(self, x, y, radius=10, strength=0.5):
        with QMutexLocker(self.mutex):
            lib = cp if self.use_gpu else np
            y_coords, x_coords = lib.meshgrid(lib.arange(self.height), lib.arange(self.width), indexing='ij')
            dist = lib.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            mask = dist < radius
            self.V_src[mask] = strength
            self.V_dst[mask] = strength

    def step(self, steps=10, dt=1.0):
        if self.use_gpu:
            self._step_gpu(steps, dt)
        else:
            self._step_cpu(steps, dt)

    def _step_gpu(self, steps, dt):
        for _ in range(steps):
            Lu = (cp.roll(self.U_src, 1, 0) + cp.roll(self.U_src, -1, 0) +
                  cp.roll(self.U_src, 1, 1) + cp.roll(self.U_src, -1, 1) - 4 * self.U_src)
            Lv = (cp.roll(self.V_src, 1, 0) + cp.roll(self.V_src, -1, 0) +
                  cp.roll(self.V_src, 1, 1) + cp.roll(self.V_src, -1, 1) - 4 * self.V_src)
            uvv = self.U_src * self.V_src * self.V_src
            self.U_dst = self.U_src + (self.Du * Lu - uvv + self.F * (1 - self.U_src)) * dt
            self.V_dst = self.V_src + (self.Dv * Lv + uvv - (self.F + self.k) * self.V_src) * dt
            self.U_src, self.U_dst = self.U_dst, self.U_src
            self.V_src, self.V_dst = self.V_dst, self.V_src

    def _step_cpu(self, steps, dt):
        for _ in range(steps):
            numba_step_buffered(self.U_src, self.V_src, self.U_dst, self.V_dst,
                                self.Du, self.Dv, self.F, self.k, dt)
            self.U_src, self.U_dst = self.U_dst, self.U_src
            self.V_src, self.V_dst = self.V_dst, self.V_src

    def get_image(self, colormap=cv2.COLORMAP_INFERNO, sharpen=False):
        with QMutexLocker(self.mutex):
            v_img = cp.asnumpy(self.V_src) if self.use_gpu else self.V_src.copy()
        v_img = np.power(v_img, 0.6)
        v_norm = cv2.normalize(v_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        if sharpen:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            v_norm = cv2.filter2D(v_norm, -1, kernel)
        colored = cv2.applyColorMap(v_norm, colormap)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ============================================================================
# WORKER (was worker.py)
# ============================================================================

class SimulationWorker(QThread):
    frame_ready = Signal(np.ndarray)
    stats_ready = Signal(float, int)
    recording_stopped = Signal()

    def __init__(self, engine, target_fps=60):
        super().__init__()
        self.engine = engine
        self.running = True
        self.paused = False
        self.steps_per_frame = 30
        self.target_fps = target_fps
        self.colormap = cv2.COLORMAP_INFERNO
        self.auto_disturbance = True
        self.sharpen = False
        self.is_recording = False
        self.video_writer = None
        self.record_start_time = 0
        self.record_duration = 0

    def run(self):
        last_time = time.time()
        frames = 0
        disturbance_timer = 0
        while self.running:
            if not self.paused:
                self.engine.step(self.steps_per_frame)
                if self.auto_disturbance:
                    disturbance_timer += 1
                    if disturbance_timer > 150:
                        self.engine.add_random_disturbance(num_blobs=2)
                        disturbance_timer = 0
                img = self.engine.get_image(self.colormap, self.sharpen)
                self.frame_ready.emit(img)
                if self.is_recording and self.video_writer is not None:
                    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(bgr_img)
                    if self.record_duration > 0 and (time.time() - self.record_start_time >= self.record_duration):
                        self.stop_recording()
                frames += 1
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed >= 1.0:
                    fps = frames / elapsed
                    self.stats_ready.emit(fps, self.engine.width)
                    frames = 0
                    last_time = current_time
                sleep_time = (1.0 / self.target_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.1)

    def start_recording(self, filepath, duration=0):
        if self.is_recording:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (self.engine.width, self.engine.height))
        self.is_recording = True
        self.record_start_time = time.time()
        self.record_duration = duration

    def stop_recording(self):
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            self.recording_stopped.emit()

    def stop(self):
        self.running = False
        if self.is_recording:
            self.stop_recording()
        self.wait()


# ============================================================================
# WIDGETS (was widgets.py)
# ============================================================================

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
        self.combo_map.addItems(list(COLORMAPS.keys()))
        form.addRow("Colormap:", self.combo_map)

        self.check_sharpen = QCheckBox("Sharpen Output")
        form.addRow(self.check_sharpen)

        param_group.setLayout(form)
        layout.addWidget(param_group)

        # Presets
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS.keys()))
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
            'colormap': COLORMAPS.get(self.combo_map.currentText(), cv2.COLORMAP_INFERNO),
            'auto_disturbance': self.check_auto.isChecked(),
            'sharpen': self.check_sharpen.isChecked()
        }
        self.params_changed.emit(d)

    def load_preset(self, name):
        if name in PRESETS:
            f, k = PRESETS[name]
            self.spin_F.setValue(f)
            self.spin_k.setValue(k)

    def on_rec_start(self):
        self.rec_start_clicked.emit("", 0)

    def on_rec_stop(self):
        self.rec_stop_clicked.emit()

    def on_rec_duration(self):
        duration = self.spin_duration.value()
        if duration == 0:
            duration = 10
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


# ============================================================================
# MAIN WINDOW (was app.py)
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RD Simulator Pro MAX")
        self.setGeometry(100, 100, 1300, 850)

        self.use_gpu = GPU_AVAILABLE
        self.current_res = 512

        self.engine = SimulationEngine(self.current_res, self.current_res, use_gpu=self.use_gpu)

        self.setup_ui()
        self.setup_threads()

        QTimer.singleShot(100, self.start_sim)

    def setup_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        self.control_panel = ControlPanel()
        self.control_panel.setMinimumWidth(280)
        self.control_panel.setMaximumWidth(320)

        self.viewer = FastImageViewer()

        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.viewer)
        splitter.setSizes([280, 1020])

        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def setup_threads(self):
        self.worker = SimulationWorker(self.engine, target_fps=60)
        self.worker.frame_ready.connect(self.viewer.update_image)
        self.worker.stats_ready.connect(self.update_status)
        self.worker.recording_stopped.connect(self.on_rec_stopped)

        self.control_panel.params_changed.connect(self.update_params)
        self.control_panel.sim_start_clicked.connect(self.start_sim)
        self.control_panel.sim_pause_clicked.connect(self.pause_sim)
        self.control_panel.sim_reset_clicked.connect(self.reset_sim)
        self.control_panel.clear_clicked.connect(self.clear_sim)
        self.control_panel.resolution_changed.connect(self.change_resolution)

        self.viewer.paint_signal.connect(self.paint_sim)

        self.control_panel.rec_start_clicked.connect(self.start_recording)
        self.control_panel.rec_stop_clicked.connect(self.stop_recording)

        mode = "GPU (CuPy)" if self.use_gpu else "CPU (Numba Optimized)"
        self.control_panel.mode_label.setText(f"Engine: {mode}")

    def update_params(self, params):
        self.engine.F = params['F']
        self.engine.k = params['k']
        self.worker.steps_per_frame = params['speed']
        self.worker.colormap = params['colormap']
        self.worker.auto_disturbance = params['auto_disturbance']
        self.worker.sharpen = params['sharpen']

    def paint_sim(self, x, y, radius, strength):
        scale = self.current_res / 512.0
        self.engine.paint(int(x * scale), int(y * scale), radius=int(radius * scale), strength=strength)

    def start_sim(self):
        if not self.worker.isRunning():
            self.worker.start()
            self.control_panel.btn_start.setChecked(True)
            self.control_panel.btn_start.setText("⏸ Pause")

    def pause_sim(self):
        self.worker.paused = True
        self.statusBar().showMessage("Paused")

    def clear_sim(self):
        self.engine.clear()

    def change_resolution(self, res):
        was_running = self.worker.isRunning()
        if was_running:
            self.worker.stop()

        self.current_res = res
        self.engine = SimulationEngine(res, res, use_gpu=self.use_gpu)

        self.worker = SimulationWorker(self.engine, target_fps=60)
        self.worker.frame_ready.connect(self.viewer.update_image)
        self.worker.stats_ready.connect(self.update_status)
        self.worker.recording_stopped.connect(self.on_rec_stopped)
        self.viewer.paint_signal.connect(self.paint_sim)

        self.control_panel.emit_params()

        if was_running:
            self.worker.start()

    def reset_sim(self):
        self.worker.paused = True
        seed_text = self.control_panel.combo_seed.currentText()
        seed_key = 'multi_random' if 'Multi' in seed_text else 'circle'
        self.engine.reset(seed=seed_key)

        img = self.engine.get_image()
        self.viewer.update_image(img)

        self.control_panel.btn_start.setChecked(False)
        self.control_panel.btn_start.setText("▶ Start")
        self.statusBar().showMessage("Reset Complete")

    def start_recording(self, filepath, duration):
        if filepath == "":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"rd_sim_{timestamp}.mp4"
            fn, _ = QFileDialog.getSaveFileName(self, "Save Video", filepath, "MP4 Files (*.mp4)")
            if not fn:
                return
            filepath = fn

        if not self.worker.isRunning():
            self.start_sim()
            self.worker.paused = False

        self.worker.start_recording(filepath, duration)
        self.control_panel.set_recording_state(True)

    def stop_recording(self):
        self.worker.stop_recording()

    def on_rec_stopped(self):
        self.control_panel.set_recording_state(False)
        self.statusBar().showMessage("Recording Saved.")

    def update_status(self, fps, resolution):
        mode = "GPU" if self.use_gpu else "CPU"
        rec = " [REC]" if self.worker.is_recording else ""
        self.statusBar().showMessage(f"Mode: {mode} | Res: {resolution}x{resolution} | FPS: {fps:.1f}{rec}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    app.setPalette(palette)

    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())