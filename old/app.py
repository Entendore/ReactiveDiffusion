import sys
import datetime
from PySide6.QtWidgets import QApplication, QMainWindow, QSplitter, QStatusBar, QFileDialog
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPalette, QColor

# Local imports
import config
from engine import SimulationEngine, GPU_AVAILABLE
from worker import SimulationWorker
from widgets import ControlPanel, FastImageViewer

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
        self.engine.paint(int(x*scale), int(y*scale), radius=int(radius*scale), strength=strength)

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
            if not fn: return
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    app.setPalette(palette)
    
    app.setStyleSheet(config.STYLESHEET)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())