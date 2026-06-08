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
    QFileDialog, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QThread, QMutex, QMutexLocker, QTimer, QPointF
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QPalette, QColor, QPainter, QLinearGradient, QPen

# ============================================================================
# CONFIG
# ============================================================================

PRESETS = {
    "Coral":   (0.0545, 0.0620),
    "Mitosis": (0.0367, 0.0649),
    "Spirals": (0.0260, 0.0510),
    "Waves":   (0.0140, 0.0450),
    "Spots":   (0.0350, 0.0650),
    "Stripes": (0.0420, 0.0590),
    "Maze":    (0.0290, 0.0570)
}

SEEDS = ["Multi-Random", "Single Circle", "Ring", "Grid Dots", "Noise Field", "Spiral", "Image Import"]

BOUNDARIES = ["No-Flux", "Periodic (Toroidal)"]

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
        font-weight: bold; border: 1px solid #3a3a3a; border-radius: 5px;
        margin-top: 10px; padding-top: 10px; background-color: #252525;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
    QPushButton {
        background-color: #333; padding: 6px; border-radius: 4px; border: 1px solid #555;
    }
    QPushButton:hover { background-color: #444; border: 1px solid #777; }
    QPushButton:checked { background-color: #005500; border: 1px solid #007700; }
    QPushButton:disabled { background-color: #222; color: #666; }
    QSlider::groove:horizontal { height: 4px; background: #444; }
    QSlider::handle:horizontal { background: #0078d7; width: 14px; margin: -5px 0; border-radius: 7px; }
    QComboBox { background-color: #333; border: 1px solid #555; padding: 4px; }
    QCheckBox::indicator { width: 15px; height: 15px; }
    QSpinBox, QDoubleSpinBox { background-color: #333; border: 1px solid #555; padding: 2px; }
    QSplitter::handle { background-color: #333; }
    QScrollArea { border: none; background-color: #1e1e1e; }
"""

# ============================================================================
# ENGINE
# ============================================================================

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

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
def numba_step_general(U_src, V_src, U_dst, V_dst, Du_x, Du_y, Dv_x, Dv_y, 
                       F_fld, k_fld, dt, bc_type, use_9pt):
    rows, cols = U_src.shape
    for i in prange(rows):
        for j in range(cols):
            # Boundary handling
            if bc_type == 1: # Periodic
                ip = (i + 1) % rows
                im = (i - 1 + rows) % rows
                jp = (j + 1) % cols
                jm = (j - 1 + cols) % cols
            else: # No-Flux
                ip = min(i + 1, rows - 1)
                im = max(i - 1, 0)
                jp = min(j + 1, cols - 1)
                jm = max(j - 1, 0)

            Uij = U_src[i, j]
            Uip, Uim, Ujp, Ujm = U_src[ip, j], U_src[im, j], U_src[i, jp], U_src[i, jm]
            Vij = V_src[i, j]
            Vip, Vim, Vjp, Vjm = V_src[ip, j], V_src[im, j], V_src[i, jp], V_src[i, jm]

            # Anisotropic Axial Laplacian
            Lu = Du_x * (Ujp - 2*Uij + Ujm) + Du_y * (Uip - 2*Uij + Uim)
            Lv = Dv_x * (Vjp - 2*Vij + Vjm) + Dv_y * (Vip - 2*Vij + Vim)

            if use_9pt:
                Uipp, Uimm, Uipm, Uimp = U_src[ip, jp], U_src[im, jm], U_src[ip, jm], U_src[im, jp]
                Vipp, Vimm, Vipm, Vimp = V_src[ip, jp], V_src[im, jm], V_src[ip, jm], V_src[im, jp]
                Lu_diag = Uipp + Uimm + Uipm + Uimp - 4*Uij
                Lv_diag = Vipp + Vimm + Vipm + Vimp - 4*Vij
                Lu += 0.05 * (Du_x + Du_y) * Lu_diag
                Lv += 0.05 * (Dv_x + Dv_y) * Lv_diag

            F = F_fld[i, j]
            k = k_fld[i, j]

            uvv = Uij * Vij * Vij
            U_dst[i, j] = Uij + (Lu - uvv + F * (1 - Uij)) * dt
            V_dst[i, j] = Vij + (Lv + uvv - (F + k) * Vij) * dt


class SimulationEngine:
    def __init__(self, width=256, height=256, use_gpu=True):
        self.width = width
        self.height = height
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.mutex = QMutex()
        
        # Params
        self.F = 0.0545
        self.k = 0.0620
        self.Du_x, self.Du_y = 0.2, 0.2
        self.Dv_x, self.Dv_y = 0.1, 0.1
        self.boundary = 0 # 0=NoFlux, 1=Periodic
        self.use_9pt = False
        self.noise_level = 0.0
        
        # Gradients
        self.F_gradient = None
        self.k_gradient = None
        
        self.iteration = 0
        self.reset()

    def _lib(self):
        return cp if self.use_gpu else np

    def _allocate_arrays(self):
        lib = self._lib()
        shape = (self.height, self.width)
        self.U_src = lib.ones(shape, dtype=np.float32)
        self.V_src = lib.zeros(shape, dtype=np.float32)
        self.U_dst = lib.ones(shape, dtype=np.float32)
        self.V_dst = lib.zeros(shape, dtype=np.float32)
        self._rebuild_gradients()

    def _rebuild_gradients(self):
        lib = self._lib()
        if self.F_gradient is not None:
            t = lib.linspace(0, 1, self.width)
            grad = lib.tile(t, (self.height, 1))
            self.F_field = self.F + (grad - 0.5) * self.F_gradient
        else:
            self.F_field = lib.full((self.height, self.width), self.F, dtype=np.float32)
            
        if self.k_gradient is not None:
            t = lib.linspace(0, 1, self.width)
            grad = lib.tile(t, (self.height, 1))
            self.k_field = self.k + (grad - 0.5) * self.k_gradient
        else:
            self.k_field = lib.full((self.height, self.width), self.k, dtype=np.float32)

    def reset(self, seed='Multi-Random', image_path=None):
        with QMutexLocker(self.mutex):
            self._allocate_arrays()
            lib = self._lib()
            self.iteration = 0
            
            y, x = lib.meshgrid(lib.arange(self.height), lib.arange(self.width), indexing='ij')
            cx, cy = self.width // 2, self.height // 2
            r = self.height // 8

            if seed == 'Multi-Random':
                for _ in range(15):
                    cxi = np.random.randint(0, self.width)
                    cyi = np.random.randint(0, self.height)
                    ri = np.random.randint(5, 25)
                    mask = (x - cxi)**2 + (y - cyi)**2 <= ri**2
                    self.V_src[mask] = np.random.uniform(0.2, 0.5)
                    self.U_src[mask] = 0.5
            elif seed == 'Single Circle':
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                self.V_src[mask] = 0.25
                self.U_src[mask] = 0.5
            elif seed == 'Ring':
                dist = lib.sqrt((x - cx)**2 + (y - cy)**2)
                mask = (dist > r*0.7) & (dist < r)
                self.V_src[mask] = 0.25
                self.U_src[mask] = 0.5
            elif seed == 'Grid Dots':
                spacing = self.width // 10
                for gi in range(5, self.width, spacing):
                    for gj in range(5, self.height, spacing):
                        mask = (x - gi)**2 + (y - gj)**2 <= 9
                        self.V_src[mask] = 0.25
                        self.U_src[mask] = 0.5
            elif seed == 'Noise Field':
                noise = np.random.rand(self.height, self.width).astype(np.float32)
                mask = noise > 0.98
                self.V_src[mask] = 0.25
                self.U_src[mask] = 0.5
            elif seed == 'Spiral':
                theta = lib.arctan2(y - cy, x - cx)
                dist = lib.sqrt((x - cx)**2 + (y - cy)**2)
                mask = (lib.sin(theta * 3 + dist * 0.1) > 0.7) & (dist < r * 2)
                self.V_src[mask] = 0.25
                self.U_src[mask] = 0.5
            elif seed == 'Image Import' and image_path:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (self.width, self.height))
                    v_vals = img.astype(np.float32) / 255.0 * 0.5
                    self.V_src = lib.array(v_vals)
                    self.U_src = lib.ones_like(v_vals) * (1.0 - v_vals * 0.5)

            lib.copyto(self.U_dst, self.U_src)
            lib.copyto(self.V_dst, self.V_src)

    def clear(self):
        with QMutexLocker(self.mutex):
            lib = self._lib()
            self.U_src.fill(1.0); self.V_src.fill(0.0)
            self.U_dst.fill(1.0); self.V_dst.fill(0.0)
            self.iteration = 0

    def add_random_disturbance(self, num_blobs=5):
        with QMutexLocker(self.mutex):
            lib = self._lib()
            y, x = lib.meshgrid(lib.arange(self.height), lib.arange(self.width), indexing='ij')
            for _ in range(num_blobs):
                cx = int(lib.random.randint(0, self.width))
                cy = int(lib.random.randint(0, self.height))
                r = int(lib.random.randint(3, 10))
                mask = (x - cx)**2 + (y - cy)**2 <= r**2
                self.V_src[mask] = lib.random.uniform(0.2, 0.4)
                self.U_src[mask] = 0.5

    def paint(self, x, y, radius=10, strength=0.5):
        with QMutexLocker(self.mutex):
            lib = self._lib()
            y_coords, x_coords = lib.meshgrid(lib.arange(self.height), lib.arange(self.width), indexing='ij')
            dist = lib.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            mask = dist < radius
            self.V_src[mask] = strength
            self.V_dst[mask] = strength

    def step(self, steps=10, dt=1.0):
        # Clamp dt for stability (Von Neumann criterion)
        max_D = max(self.Du_x, self.Du_y, self.Dv_x, self.Dv_y)
        dt = min(dt, 0.9 / (2.0 * max_D)) if max_D > 0 else dt
        
        with QMutexLocker(self.mutex):
            if self.use_gpu:
                self._step_gpu(steps, dt)
            else:
                self._step_cpu(steps, dt)
            self.iteration += steps

    def _step_gpu(self, steps, dt):
        pad_mode = 'wrap' if self.boundary == 1 else 'edge'
        
        for _ in range(steps):
            U_pad = cp.pad(self.U_src, 1, mode=pad_mode)
            V_pad = cp.pad(self.V_src, 1, mode=pad_mode)
            
            U_c = U_pad[1:-1, 1:-1]
            U_n, U_s = U_pad[:-2, 1:-1], U_pad[2:, 1:-1]
            U_e, U_w = U_pad[1:-1, 2:], U_pad[1:-1, :-2]
            
            V_c = V_pad[1:-1, 1:-1]
            V_n, V_s = V_pad[:-2, 1:-1], V_pad[2:, 1:-1]
            V_e, V_w = V_pad[1:-1, 2:], V_pad[1:-1, :-2]

            Lu = self.Du_x * (U_e + U_w - 2*U_c) + self.Du_y * (U_n + U_s - 2*U_c)
            Lv = self.Dv_x * (V_e + V_w - 2*V_c) + self.Dv_y * (V_n + V_s - 2*V_c)

            if self.use_9pt:
                U_ne, U_nw = U_pad[:-2, 2:], U_pad[:-2, :-2]
                U_se, U_sw = U_pad[2:, 2:], U_pad[2:, :-2]
                Lu += 0.05 * (self.Du_x + self.Du_y) * (U_ne + U_nw + U_se + U_sw - 4*U_c)
                
                V_ne, V_nw = V_pad[:-2, 2:], V_pad[:-2, :-2]
                V_se, V_sw = V_pad[2:, 2:], V_pad[2:, :-2]
                Lv += 0.05 * (self.Dv_x + self.Dv_y) * (V_ne + V_nw + V_se + V_sw - 4*V_c)

            uvv = self.U_src * self.V_src * self.V_src
            self.U_dst = self.U_src + (Lu - uvv + self.F_field * (1 - self.U_src)) * dt
            self.V_dst = self.V_src + (Lv + uvv - (self.F_field + self.k_field) * self.V_src) * dt
            
            if self.noise_level > 0:
                self.U_dst += cp.random.normal(0, self.noise_level, self.U_src.shape).astype(cp.float32) * dt
                self.V_dst += cp.random.normal(0, self.noise_level, self.V_src.shape).astype(cp.float32) * dt

            self.U_src, self.U_dst = self.U_dst, self.U_src
            self.V_src, self.V_dst = self.V_dst, self.V_src

    def _step_cpu(self, steps, dt):
        for _ in range(steps):
            numba_step_general(self.U_src, self.V_src, self.U_dst, self.V_dst,
                               self.Du_x, self.Du_y, self.Dv_x, self.Dv_y,
                               self.F_field, self.k_field, dt, self.boundary, self.use_9pt)
            self.U_src, self.U_dst = self.U_dst, self.U_src
            self.V_src, self.V_dst = self.V_dst, self.V_src
            
            if self.noise_level > 0:
                noise_u = np.random.normal(0, self.noise_level, self.U_src.shape).astype(np.float32) * dt
                noise_v = np.random.normal(0, self.noise_level, self.V_src.shape).astype(np.float32) * dt
                self.U_src += noise_u
                self.V_src += noise_v

    def get_image(self, colormap=cv2.COLORMAP_INFERNO, gamma=0.6, sharpen=False, 
                  bloom_strength=0.0, dual_channel=False, colormap_u=cv2.COLORMAP_COOL):
        with QMutexLocker(self.mutex):
            v_img = cp.asnumpy(self.V_src) if self.use_gpu else self.V_src.copy()
            u_img = cp.asnumpy(self.U_src) if self.use_gpu else self.U_src.copy()

        if dual_channel:
            u_norm = cv2.normalize(u_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            v_norm = cv2.normalize(v_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            u_colored = cv2.applyColorMap(u_norm, colormap_u)
            v_colored = cv2.applyColorMap(v_norm, colormap)
            alpha = v_norm.astype(np.float32) / 255.0
            alpha_3ch = np.stack([alpha]*3, axis=-1)
            blended = (u_colored * (1 - alpha_3ch) + v_colored * alpha_3ch)
            colored = blended.astype(np.uint8)
        else:
            v_img = np.power(np.clip(v_img, 0, 1), gamma)
            v_norm = cv2.normalize(v_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            colored = cv2.applyColorMap(v_norm, colormap)

        if sharpen:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            colored = cv2.filter2D(colored, -1, kernel)

        if bloom_strength > 0:
            bright_mask = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(bright_mask, 180, 255, cv2.THRESH_BINARY)
            bloom = cv2.GaussianBlur(colored, (0, 0), 15)
            colored = cv2.addWeighted(colored, 1.0, bloom, bloom_strength, 0)

        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ============================================================================
# WORKER
# ============================================================================

class SimulationWorker(QThread):
    frame_ready = Signal(np.ndarray)
    stats_ready = Signal(float, int, int)
    recording_stopped = Signal()

    def __init__(self, engine, target_fps=60):
        super().__init__()
        self.engine = engine
        self.running = True
        self.paused = False
        self.steps_per_frame = 30
        self.target_fps = target_fps
        
        # Render Config
        self.colormap = cv2.COLORMAP_INFERNO
        self.gamma = 0.6
        self.sharpen = False
        self.bloom = 0.0
        self.dual_channel = False
        self.colormap_u = cv2.COLORMAP_COOL
        
        self.auto_disturbance = True
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
                
                img = self.engine.get_image(
                    colormap=self.colormap, gamma=self.gamma, sharpen=self.sharpen,
                    bloom_strength=self.bloom, dual_channel=self.dual_channel,
                    colormap_u=self.colormap_u
                )
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
                    self.stats_ready.emit(fps, self.engine.width, self.engine.iteration)
                    frames = 0
                    last_time = current_time
                
                sleep_time = (1.0 / self.target_fps) - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                time.sleep(0.1)

    def start_recording(self, filepath, duration=0):
        if self.is_recording: return
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
        if self.is_recording: self.stop_recording()
        self.wait()


# ============================================================================
# WIDGETS
# ============================================================================

class PhaseDiagramWidget(QWidget):
    params_clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setFixedSize(220, 180)
        self.F_min, self.F_max = 0.01, 0.10
        self.k_min, self.k_max = 0.03, 0.075
        self.current_F = 0.0545
        self.current_k = 0.062

    def map_to_fk(self, x, y):
        f = self.F_min + (self.F_max - self.F_min) * (x / self.width())
        k = self.k_min + (self.k_max - self.k_min) * (y / self.height())
        return f, k

    def map_to_xy(self, f, k):
        x = (f - self.F_min) / (self.F_max - self.F_min) * self.width()
        y = (k - self.k_min) / (self.k_max - self.k_min) * self.height()
        return int(x), int(y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background rough parameter space zones
        grad = QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QColor(20, 0, 40))
        grad.setColorAt(0.4, QColor(0, 40, 40))
        grad.setColorAt(1.0, QColor(40, 20, 0))
        painter.fillRect(self.rect(), grad)

        # Draw preset points
        painter.setPen(QPen(Qt.white, 1))
        for name, (f, k) in PRESETS.items():
            x, y = self.map_to_xy(f, k)
            painter.drawEllipse(x-3, y-3, 6, 6)
            painter.drawText(x+5, y+4, name)

        # Draw current position
        cx, cy = self.map_to_xy(self.current_F, self.current_k)
        painter.setPen(QPen(Qt.red, 2))
        painter.drawLine(cx-5, cy, cx+5, cy)
        painter.drawLine(cx, cy-5, cx, cy+5)

    def mousePressEvent(self, event):
        f, k = self.map_to_fk(event.position().x(), event.position().y())
        self.params_clicked.emit(f, k)


class FastImageViewer(QWidget):
    paint_signal = Signal(int, int, int, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        
        self.current_pixmap = QPixmap()
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._last_mouse_pos = QPointF()
        self._is_panning = False
        
        self.drawing = False
        self.brush_radius = 10
        self.brush_strength = 0.5

    def update_image(self, rgb_array):
        h, w, ch = rgb_array.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(q_img)
        self.update()

    def paintEvent(self, event):
        if self.current_pixmap.isNull(): return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Center the image initially
        target_rect = self.rect()
        source_size = self.current_pixmap.size() * self._zoom
        
        # Calculate offset to keep center aligned during zoom
        dx = (target_rect.width() - source_size.width()) / 2 + self._pan.x()
        dy = (target_rect.height() - source_size.height()) / 2 + self._pan.y()
        
        painter.translate(dx, dy)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self.current_pixmap)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self._zoom = max(0.25, min(self._zoom * factor, 10.0))
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()
        elif event.button() == Qt.LeftButton:
            self.drawing = True
            self._handle_mouse(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_panning:
            delta = event.position() - self._last_mouse_pos
            self._pan += delta
            self._last_mouse_pos = event.position()
            self.update()
        elif self.drawing:
            self._handle_mouse(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton or event.button() == Qt.RightButton:
            self._is_panning = False
        elif event.button() == Qt.LeftButton:
            self.drawing = False

    def _handle_mouse(self, event):
        # Convert screen coords back to pixmap coords
        target_rect = self.rect()
        source_size = self.current_pixmap.size() * self._zoom
        dx = (target_rect.width() - source_size.width()) / 2 + self._pan.x()
        dy = (target_rect.height() - source_size.height()) / 2 + self._pan.y()
        
        local_x = (event.position().x() - dx) / self._zoom
        local_y = (event.position().y() - dy) / self._zoom
        
        if 0 <= local_x < self.current_pixmap.width() and 0 <= local_y < self.current_pixmap.height():
            self.paint_signal.emit(int(local_x), int(local_y), self.brush_radius, self.brush_strength)


class ControlPanel(QWidget):
    params_changed = Signal(dict)
    sim_start_clicked = Signal()
    sim_pause_clicked = Signal()
    sim_reset_clicked = Signal()
    clear_clicked = Signal()
    resolution_changed = Signal(int)
    rec_start_clicked = Signal(str, int)
    rec_stop_clicked = Signal()
    save_state_clicked = Signal()
    load_state_clicked = Signal()
    export_snap_clicked = Signal()
    import_seed_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        scroll.setWidget(container)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.addWidget(scroll)

        # System Status
        self.status_group = QGroupBox("System Info")
        sl = QVBoxLayout()
        self.mode_label = QLabel("Detecting...")
        self.mode_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        sl.addWidget(self.mode_label)
        self.status_group.setLayout(sl)
        layout.addWidget(self.status_group)

        # File I/O
        io_group = QGroupBox("File / State")
        io_l = QHBoxLayout()
        self.btn_save = QPushButton("💾 Save")
        self.btn_load = QPushButton("📂 Load")
        self.btn_snap = QPushButton("📸 Snap")
        io_l.addWidget(self.btn_save)
        io_l.addWidget(self.btn_load)
        io_l.addWidget(self.btn_snap)
        io_group.setLayout(io_l)
        layout.addWidget(io_group)

        # Simulation Controls
        ctrl_group = QGroupBox("Simulation Controls")
        ctrl_layout = QVBoxLayout()

        h_btns = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start"); self.btn_start.setCheckable(True)
        self.btn_reset = QPushButton("⏹ Reset")
        h_btns.addWidget(self.btn_start); h_btns.addWidget(self.btn_reset)
        ctrl_layout.addLayout(h_btns)

        h_btns2 = QHBoxLayout()
        self.btn_clear = QPushButton("🚿 Clear")
        h_btns2.addWidget(self.btn_clear)
        ctrl_layout.addLayout(h_btns2)

        ctrl_layout.addLayout(self._create_slider("Speed:", "spin_speed", 1, 100, 30, "lbl_speed_val"))
        self.check_auto = QCheckBox("Auto-Disturbance"); self.check_auto.setChecked(True)
        ctrl_layout.addWidget(self.check_auto)

        h_res = QHBoxLayout(); h_res.addWidget(QLabel("Resolution:"))
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
        paint_layout.addLayout(self._create_slider("Brush Size:", "slider_brush", 2, 50, 10, "lbl_brush"))
        paint_layout.addLayout(self._create_slider("Strength:", "slider_strength", 1, 100, 90, None))
        paint_group.setLayout(paint_layout)
        layout.addWidget(paint_group)

        # Recording
        rec_group = QGroupBox("Video Recording")
        rec_layout = QVBoxLayout()
        h_dur = QHBoxLayout(); h_dur.addWidget(QLabel("Duration (s):"))
        self.spin_duration = QSpinBox(); self.spin_duration.setRange(0, 600); self.spin_duration.setValue(10)
        h_dur.addWidget(self.spin_duration)
        rec_layout.addLayout(h_dur)
        h_rec_btns = QHBoxLayout()
        self.btn_rec_start = QPushButton("🔴 Rec"); self.btn_rec_stop = QPushButton("⏹ Stop")
        self.btn_rec_stop.setEnabled(False)
        h_rec_btns.addWidget(self.btn_rec_start); h_rec_btns.addWidget(self.btn_rec_stop)
        rec_layout.addLayout(h_rec_btns)
        self.lbl_rec_status = QLabel("Status: Idle"); self.lbl_rec_status.setStyleSheet("color: #888;")
        rec_layout.addWidget(self.lbl_rec_status)
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        # Physics Parameters
        param_group = QGroupBox("Physics Parameters")
        form = QFormLayout()
        
        self.spin_F = QDoubleSpinBox(); self.spin_F.setRange(0.001, 0.1); self.spin_F.setSingleStep(0.001); self.spin_F.setValue(0.0545); self.spin_F.setDecimals(4)
        self.spin_k = QDoubleSpinBox(); self.spin_k.setRange(0.001, 0.1); self.spin_k.setSingleStep(0.001); self.spin_k.setValue(0.0620); self.spin_k.setDecimals(4)
        
        self.combo_bc = QComboBox(); self.combo_bc.addItems(BOUNDARIES)
        self.check_9pt = QCheckBox("9-Point Laplacian")
        
        form.addRow("Feed (F):", self.spin_F)
        form.addRow("Kill (k):", self.spin_k)
        form.addRow("Boundary:", self.combo_bc)
        form.addRow(self.check_9pt)
        
        # Anisotropy
        form.addRow(QLabel("Anisotropy (X/Y):"))
        self.spin_Du_x = QDoubleSpinBox(); self.spin_Du_x.setRange(0.01, 1.0); self.spin_Du_x.setSingleStep(0.01); self.spin_Du_x.setValue(0.20)
        self.spin_Du_y = QDoubleSpinBox(); self.spin_Du_y.setRange(0.01, 1.0); self.spin_Du_y.setSingleStep(0.01); self.spin_Du_y.setValue(0.20)
        self.spin_Dv_x = QDoubleSpinBox(); self.spin_Dv_x.setRange(0.01, 1.0); self.spin_Dv_x.setSingleStep(0.01); self.spin_Dv_x.setValue(0.10)
        self.spin_Dv_y = QDoubleSpinBox(); self.spin_Dv_y.setRange(0.01, 1.0); self.spin_Dv_y.setSingleStep(0.01); self.spin_Dv_y.setValue(0.10)
        
        form.addRow("Du_x:", self.spin_Du_x); form.addRow("Du_y:", self.spin_Du_y)
        form.addRow("Dv_x:", self.spin_Dv_x); form.addRow("Dv_y:", self.spin_Dv_y)
        
        param_group.setLayout(form)
        layout.addWidget(param_group)

        # Visuals
        vis_group = QGroupBox("Visual Settings")
        vis_form = QFormLayout()
        self.combo_map = QComboBox(); self.combo_map.addItems(list(COLORMAPS.keys()))
        vis_form.addRow("Colormap:", self.combo_map)
        vis_form.addRow(self._create_slider("Gamma:", "slider_gamma", 10, 300, 60, "lbl_gamma"))
        self.check_sharpen = QCheckBox("Sharpen")
        vis_form.addRow(self.check_sharpen)
        vis_form.addRow(self._create_slider("Bloom:", "slider_bloom", 0, 100, 0, None))
        self.check_dual = QCheckBox("Dual Channel (U+V)")
        vis_form.addRow(self.check_dual)
        self.combo_map_u = QComboBox(); self.combo_map_u.addItems(list(COLORMAPS.keys())); self.combo_map_u.setCurrentText("Cool")
        vis_form.addRow("U Colormap:", self.combo_map_u)
        vis_group.setLayout(vis_form)
        layout.addWidget(vis_group)

        # Advanced
        adv_group = QGroupBox("Advanced / Noise / Gradient")
        adv_l = QFormLayout()
        adv_l.addRow(self._create_slider("Noise:", "slider_noise", 0, 50, 0, "lbl_noise"))
        adv_l.addRow(self._create_slider("F Gradient:", "slider_fgrad", 0, 50, 0, "lbl_fgrad"))
        adv_l.addRow(self._create_slider("k Gradient:", "slider_kgrad", 0, 50, 0, "lbl_kgrad"))
        adv_group.setLayout(adv_l)
        layout.addWidget(adv_group)

        # Presets & Phase
        preset_group = QGroupBox("Presets & Phase Space")
        preset_layout = QVBoxLayout()
        self.combo_preset = QComboBox(); self.combo_preset.addItems(list(PRESETS.keys()))
        preset_layout.addWidget(self.combo_preset)

        self.combo_seed = QComboBox(); self.combo_seed.addItems(SEEDS)
        preset_layout.addWidget(QLabel("Seed Type:"))
        preset_layout.addWidget(self.combo_seed)
        self.btn_import_seed = QPushButton("🖼 Load Image Seed")
        preset_layout.addWidget(self.btn_import_seed)

        self.phase_widget = PhaseDiagramWidget()
        preset_layout.addWidget(self.phase_widget)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        layout.addStretch()

        # Connections
        self.btn_save.clicked.connect(self.save_state_clicked.emit)
        self.btn_load.clicked.connect(self.load_state_clicked.emit)
        self.btn_snap.clicked.connect(self.export_snap_clicked.emit)
        self.btn_import_seed.clicked.connect(self.import_seed_clicked.emit)
        
        self.btn_start.clicked.connect(self.on_start_toggle)
        self.btn_reset.clicked.connect(self.sim_reset_clicked.emit)
        self.btn_clear.clicked.connect(self.clear_clicked.emit)

        for w in [self.spin_F, self.spin_k, self.spin_Du_x, self.spin_Du_y, self.spin_Dv_x, self.spin_Dv_y]:
            w.valueChanged.connect(self.emit_params)
        for w in [self.spin_speed, self.slider_gamma, self.slider_bloom, self.slider_noise, self.slider_fgrad, self.slider_kgrad]:
            w.valueChanged.connect(self.emit_params)
        for w in [self.combo_map, self.combo_map_u, self.combo_bc]:
            w.currentTextChanged.connect(self.emit_params)
        for w in [self.check_auto, self.check_sharpen, self.check_dual, self.check_9pt]:
            w.stateChanged.connect(self.emit_params)

        self.combo_res.currentTextChanged.connect(self.on_res_change)
        self.combo_preset.currentTextChanged.connect(self.load_preset)
        self.slider_brush.valueChanged.connect(self.on_brush_change)

        self.btn_rec_start.clicked.connect(self.on_rec_start)
        self.btn_rec_stop.clicked.connect(self.rec_stop_clicked.emit)
        
        self.phase_widget.params_clicked.connect(self.on_phase_click)

    def _create_slider(self, label_text, slider_attr, min_val, max_val, default, label_attr):
        h = QHBoxLayout()
        h.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        h.addWidget(slider)
        setattr(self, slider_attr, slider)
        
        if label_attr is not None:
            lbl = QLabel(str(default))
            h.addWidget(lbl)
            setattr(self, label_attr, lbl)
            slider.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
        return h

    def on_brush_change(self, val): pass

    def on_res_change(self, text):
        self.resolution_changed.emit(int(text.split(" ")[0]))

    def on_start_toggle(self, checked):
        if checked:
            self.btn_start.setText("⏸ Pause"); self.sim_start_clicked.emit()
        else:
            self.btn_start.setText("▶ Start"); self.sim_pause_clicked.emit()

    def on_phase_click(self, f, k):
        self.spin_F.setValue(f)
        self.spin_k.setValue(k)

    def emit_params(self):
        d = {
            'F': self.spin_F.value(), 'k': self.spin_k.value(),
            'speed': self.spin_speed.value(),
            'colormap': COLORMAPS.get(self.combo_map.currentText(), cv2.COLORMAP_INFERNO),
            'colormap_u': COLORMAPS.get(self.combo_map_u.currentText(), cv2.COLORMAP_COOL),
            'auto_disturbance': self.check_auto.isChecked(),
            'sharpen': self.check_sharpen.isChecked(),
            'gamma': self.slider_gamma.value() / 100.0,
            'bloom': self.slider_bloom.value() / 100.0,
            'dual_channel': self.check_dual.isChecked(),
            'Du_x': self.spin_Du_x.value(), 'Du_y': self.spin_Du_y.value(),
            'Dv_x': self.spin_Dv_x.value(), 'Dv_y': self.spin_Dv_y.value(),
            'boundary': self.combo_bc.currentIndex(),
            'use_9pt': self.check_9pt.isChecked(),
            'noise_level': self.slider_noise.value() / 10000.0,
            'F_gradient': self.slider_fgrad.value() / 100.0 if self.slider_fgrad.value() > 0 else None,
            'k_gradient': self.slider_kgrad.value() / 100.0 if self.slider_kgrad.value() > 0 else None,
        }
        self.params_changed.emit(d)

    def load_preset(self, name):
        if name in PRESETS:
            f, k = PRESETS[name]
            self.spin_F.setValue(f)
            self.spin_k.setValue(k)

    def on_rec_start(self): self.rec_start_clicked.emit("", 0)
    def set_recording_state(self, is_rec):
        if is_rec:
            self.btn_rec_start.setEnabled(False); self.btn_rec_stop.setEnabled(True)
            self.lbl_rec_status.setText("Status: 🔴 Recording..."); self.lbl_rec_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        else:
            self.btn_rec_start.setEnabled(True); self.btn_rec_stop.setEnabled(False)
            self.lbl_rec_status.setText("Status: Idle"); self.lbl_rec_status.setStyleSheet("color: #888;")


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RD Simulator Pro MAX ++")
        self.setGeometry(100, 100, 1400, 900)

        self.use_gpu = GPU_AVAILABLE
        self.current_res = 512
        self.engine = SimulationEngine(self.current_res, self.current_res, use_gpu=self.use_gpu)

        self.setup_ui()
        self.setup_threads()
        QTimer.singleShot(100, self.start_sim)

    def setup_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.control_panel = ControlPanel()
        self.control_panel.setMinimumWidth(300)
        self.control_panel.setMaximumWidth(340)
        self.viewer = FastImageViewer()
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.viewer)
        splitter.setSizes([300, 1100])
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
        self.control_panel.rec_start_clicked.connect(self.start_recording)
        self.control_panel.rec_stop_clicked.connect(self.stop_recording)
        
        self.control_panel.save_state_clicked.connect(self.save_state)
        self.control_panel.load_state_clicked.connect(self.load_state)
        self.control_panel.export_snap_clicked.connect(self.export_snapshot)
        self.control_panel.import_seed_clicked.connect(self.import_seed_image)

        self.viewer.paint_signal.connect(self.paint_sim)

        mode = "GPU (CuPy)" if self.use_gpu else "CPU (Numba Optimized)"
        self.control_panel.mode_label.setText(f"Engine: {mode}")

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

    def update_params(self, p):
        self.engine.F = p['F']
        self.engine.k = p['k']
        self.engine.Du_x = p['Du_x']; self.engine.Du_y = p['Du_y']
        self.engine.Dv_x = p['Dv_x']; self.engine.Dv_y = p['Dv_y']
        self.engine.boundary = p['boundary']
        self.engine.use_9pt = p['use_9pt']
        self.engine.noise_level = p['noise_level']
        
        # Check gradients
        if self.engine.F_gradient != p['F_gradient']:
            self.engine.F_gradient = p['F_gradient']
            self.engine._rebuild_gradients()
        if self.engine.k_gradient != p['k_gradient']:
            self.engine.k_gradient = p['k_gradient']
            self.engine._rebuild_gradients()

        # Worker render configs
        self.worker.steps_per_frame = p['speed']
        self.worker.colormap = p['colormap']
        self.worker.colormap_u = p['colormap_u']
        self.worker.auto_disturbance = p['auto_disturbance']
        self.worker.sharpen = p['sharpen']
        self.worker.gamma = p['gamma']
        self.worker.bloom = p['bloom']
        self.worker.dual_channel = p['dual_channel']

        self.control_panel.phase_widget.current_F = p['F']
        self.control_panel.phase_widget.current_k = p['k']
        self.control_panel.phase_widget.update()

    def paint_sim(self, x, y, radius, strength):
        self.engine.paint(x, y, radius=radius, strength=strength)

    def start_sim(self):
        if not self.worker.isRunning():
            self.worker.start()
            self.control_panel.btn_start.setChecked(True)
            self.control_panel.btn_start.setText("⏸ Pause")

    def pause_sim(self):
        self.worker.paused = True
        self.statusBar().showMessage("Paused")

    def clear_sim(self): self.engine.clear()

    def change_resolution(self, res):
        was_running = self.worker.isRunning()
        if was_running: self.worker.stop()
        self.current_res = res
        self.engine = SimulationEngine(res, res, use_gpu=self.use_gpu)
        
        self.worker = SimulationWorker(self.engine, target_fps=60)
        self.worker.frame_ready.connect(self.viewer.update_image)
        self.worker.stats_ready.connect(self.update_status)
        self.worker.recording_stopped.connect(self.on_rec_stopped)
        self.viewer.paint_signal.connect(self.paint_sim)
        self.control_panel.emit_params()
        if was_running: self.worker.start()

    def reset_sim(self):
        self.worker.paused = True
        seed_text = self.control_panel.combo_seed.currentText()
        self.engine.reset(seed=seed_text)
        img = self.engine.get_image()
        self.viewer.update_image(img)
        self.control_panel.btn_start.setChecked(False)
        self.control_panel.btn_start.setText("▶ Start")
        self.statusBar().showMessage("Reset Complete")

    def save_state(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save State", "", "NumPy (*.npz)")
        if path:
            u = cp.asnumpy(self.engine.U_src) if self.use_gpu else self.engine.U_src
            v = cp.asnumpy(self.engine.V_src) if self.use_gpu else self.engine.V_src
            np.savez_compressed(path, U=u, V=v, F=self.engine.F, k=self.engine.k, 
                                Du_x=self.engine.Du_x, Du_y=self.engine.Du_y,
                                Dv_x=self.engine.Dv_x, Dv_y=self.engine.Dv_y)

    def load_state(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load State", "", "NumPy (*.npz)")
        if path:
            was_running = self.worker.isRunning()
            if was_running: self.worker.stop()
            
            data = np.load(path)
            lib = cp if self.use_gpu else np
            self.engine.U_src = lib.array(data['U']); self.engine.V_src = lib.array(data['V'])
            self.engine.U_dst = lib.array(data['U']); self.engine.V_dst = lib.array(data['V'])
            self.engine.F = float(data['F']); self.engine.k = float(data['k'])
            self.engine.Du_x = float(data['Du_x']); self.engine.Du_y = float(data['Du_y'])
            self.engine.Dv_x = float(data['Dv_x']); self.engine.Dv_y = float(data['Dv_y'])
            self.engine._rebuild_gradients()
            
            self.control_panel.spin_F.setValue(self.engine.F)
            self.control_panel.spin_k.setValue(self.engine.k)
            
            if was_running: self.worker.start()

    def export_snapshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG (*.png)")
        if path:
            img = self.engine.get_image(self.worker.colormap, self.worker.gamma, self.worker.sharpen, self.worker.bloom, self.worker.dual_channel, self.worker.colormap_u)
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def import_seed_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Seed Image", "", "Images (*.png *.jpg *.bmp)")
        if path:
            self.engine.reset(seed='Image Import', image_path=path)
            img = self.engine.get_image()
            self.viewer.update_image(img)

    def start_recording(self, filepath, duration):
        if filepath == "":
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"rd_sim_{timestamp}.mp4"
            fn, _ = QFileDialog.getSaveFileName(self, "Save Video", filepath, "MP4 Files (*.mp4)")
            if not fn: return
            filepath = fn
        if not self.worker.isRunning():
            self.start_sim(); self.worker.paused = False
        self.worker.start_recording(filepath, duration)
        self.control_panel.set_recording_state(True)

    def stop_recording(self): self.worker.stop_recording()
    def on_rec_stopped(self):
        self.control_panel.set_recording_state(False)
        self.statusBar().showMessage("Recording Saved.")

    def update_status(self, fps, resolution, iteration):
        mode = "GPU" if self.use_gpu else "CPU"
        rec = " [REC]" if self.worker.is_recording else ""
        self.statusBar().showMessage(
            f"Mode: {mode} | Res: {resolution}x{resolution} | FPS: {fps:.1f} | Iter: {iteration}{rec}"
        )

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