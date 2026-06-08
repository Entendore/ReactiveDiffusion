import cv2

# Simulation Presets (Feed, Kill)
PRESETS = {
    "Coral":   (0.0545, 0.0620),
    "Mitosis": (0.0367, 0.0649),
    "Spirals": (0.0260, 0.0510),
    "Waves":   (0.0140, 0.0450),
    "Spots":   (0.0350, 0.0650)
}

# Mapping of UI names to OpenCV colormap constants
COLORMAPS = {
    "Inferno": cv2.COLORMAP_INFERNO, "Viridis": cv2.COLORMAP_VIRIDIS,
    "Plasma": cv2.COLORMAP_PLASMA, "Magma": cv2.COLORMAP_MAGMA,
    "Cividis": cv2.COLORMAP_CIVIDIS, "Turbo": cv2.COLORMAP_TURBO,
    "Jet": cv2.COLORMAP_JET, "Rainbow": cv2.COLORMAP_RAINBOW,
    "Hot": cv2.COLORMAP_HOT, "Cool": cv2.COLORMAP_COOL,
    "Ocean": cv2.COLORMAP_OCEAN, "Twilight": cv2.COLORMAP_TWILIGHT,
    "DeepGreen": cv2.COLORMAP_DEEPGREEN, "Bone": cv2.COLORMAP_BONE
}

# Stylesheet for the application
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