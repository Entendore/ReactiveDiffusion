import time
import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal

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
        if self.is_recording:
            self.stop_recording()
        self.wait()