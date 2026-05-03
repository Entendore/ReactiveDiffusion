import warnings
import numpy as np
import cv2
from PySide6.QtCore import QMutex, QMutexLocker

# Hardware Acceleration Setup
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
    """Optimized CPU kernel using double buffering."""
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
        
        # Default parameters (Coral)
        self.F = 0.0545
        self.k = 0.0620
        self.Du = 0.2
        self.Dv = 0.1
        self.reset()

    def _allocate_arrays(self):
        """Initialize arrays with double buffering."""
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
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            v_norm = cv2.filter2D(v_norm, -1, kernel)
            
        colored = cv2.applyColorMap(v_norm, colormap)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)