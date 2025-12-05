import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import random
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
import io
import base64
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Reactive Diffusion Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* General styles */
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        color: #1a3a6c;
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid #eaeaea;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, rgba(26,58,108,0.05) 0%, rgba(44,82,130,0.1) 50%, rgba(26,58,108,0.05) 100%);
        border-radius: 10px;
    }
    
    /* Card styles */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #eaeaea;
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        transform: translateY(-3px);
    }
    
    /* Pattern card styles */
    .pattern-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        border: 1px solid #eaeaea;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    .pattern-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: var(--pattern-color);
    }
    .pattern-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    .pattern-card.active {
        box-shadow: 0 0 0 2px var(--pattern-color);
    }
    .pattern-title {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Parameter container */
    .parameter-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    .status-running {
        background: linear-gradient(135deg, #10b981, #34d399);
        color: white;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
    }
    .status-paused {
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        color: white;
        box-shadow: 0 4px 8px rgba(245, 158, 11, 0.3);
    }
    
    /* Control buttons */
    .control-button {
        width: 100%;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #1a3a6c, #2c5282);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.8rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .control-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .control-button:active {
        transform: translateY(0);
    }
    
    /* Animation container */
    .animation-container {
        position: relative;
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        background: #000;
    }
    
    .animation-frame {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 12px;
    }
    
    /* Progress bar */
    .progress-container {
        width: 100%;
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Slider styles */
    .stSlider {
        padding: 0.8rem 0;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e8f4fc, #e0f2fe);
        border-left: 4px solid #2196f3;
        padding: 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #eaeaea;
        margin-top: 2rem;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Performance indicator */
    .performance-indicator {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        z-index: 10;
    }
</style>
""", unsafe_allow_html=True)

# Gray-Scott model implementation
class GrayScottSimulator:
    def __init__(self, width=200, height=200, Du=0.2, Dv=0.1, F=0.04, k=0.06, dt=1.0):
        self.width = width
        self.height = height
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.dt = dt
        
        # Initialize chemical concentrations
        self.U = np.ones((height, width))
        self.V = np.zeros((height, width))
        
        # Create initial disturbance
        center_x, center_y = width//2, height//2
        size = 15
        self.U[center_y-size:center_y+size, center_x-size:center_x+size] = 0.5
        self.V[center_y-size:center_y+size, center_x-size:center_x+size] = 0.25
    
    def laplacian(self, field):
        """Calculate Laplacian with periodic boundary conditions"""
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4 * field
        )
    
    def step(self):
        """Advance simulation by one timestep"""
        lap_U = self.laplacian(self.U)
        lap_V = self.laplacian(self.V)
        
        # Gray-Scott equations
        UV2 = self.U * self.V**2
        self.U += self.dt * (self.Du * lap_U - UV2 + self.F * (1 - self.U))
        self.V += self.dt * (self.Dv * lap_V + UV2 - (self.F + self.k) * self.V)
        
        # Ensure numerical stability
        np.clip(self.U, 0, 1, out=self.U)
        np.clip(self.V, 0, 1, out=self.V)
        
    def reset(self):
        """Reset the simulation to initial state"""
        self.U = np.ones((self.height, self.width))
        self.V = np.zeros((self.height, self.width))
        
        # Create initial disturbance
        center_x, center_y = self.width//2, self.height//2
        size = 15
        self.U[center_y-size:center_y+size, center_x-size:center_x+size] = 0.5
        self.V[center_y-size:center_y+size, center_x-size:center_x+size] = 0.25
    
    def add_random_disturbance(self, num_points=5, size=5):
        """Add random disturbances to the simulation"""
        for _ in range(num_points):
            x = random.randint(size, self.width - size)
            y = random.randint(size, self.height - size)
            self.U[y-size:y+size, x-size:x+size] = random.uniform(0.4, 0.6)
            self.V[y-size:y+size, x-size:x+size] = random.uniform(0.2, 0.3)

# Predefined color palettes
PREDEFINED_PALETTES = {
    "Default": {
        "colors": ["#0d0887", "#7d01a6", "#cb4679", "#f89540", "#f0f921"],
        "type": "linear",
        "description": "Original perceptually uniform palette"
    },
    "Plasma": {
        "cmap": "plasma",
        "type": "matplotlib",
        "description": "Scientific visualization standard"
    },
    "Viridis": {
        "cmap": "viridis",
        "type": "matplotlib",
        "description": "Perceptually uniform, colorblind-friendly"
    },
    "Turbo": {
        "cmap": "turbo",
        "type": "matplotlib",
        "description": "High-contrast spectrum visualization"
    },
    "Inferno": {
        "cmap": "inferno",
        "type": "matplotlib",
        "description": "Dark background optimized palette"
    },
    "Rainbow": {
        "colors": ["#ff0000", "#ff7f00", "#ffff00", "#00ff00", "#0000ff", "#8b00ff"],
        "type": "linear",
        "description": "Classic rainbow spectrum"
    },
    "Ocean": {
        "colors": ["#000033", "#000066", "#003399", "#0066cc", "#0099cc", "#33ccff", "#99ffff"],
        "type": "linear",
        "description": "Cool blue aquatic tones"
    },
    "Fire": {
        "colors": ["#000000", "#330000", "#660000", "#990000", "#cc0000", "#ff3300", "#ff6600", "#ff9900", "#ffcc00", "#ffff00"],
        "type": "linear",
        "description": "Hot temperature gradient"
    },
    "Monochrome": {
        "colors": ["#000000", "#444444", "#888888", "#cccccc", "#ffffff"],
        "type": "linear",
        "description": "Simple grayscale palette"
    },
    "Neon": {
        "colors": ["#000000", "#ff00ff", "#00ffff", "#ffff00", "#ff00ff"],
        "type": "linear",
        "description": "Bright neon colors"
    }
}

# Create example patterns with palette mappings
PATTERNS = {
    "Spots": {
        "description": "Self-replicating spots that form stable patterns",
        "F": 0.035,
        "k": 0.065,
        "Du": 0.2,
        "Dv": 0.1,
        "color": "#3b82f6",
        "recommended_palette": "Default"
    },
    "Worms": {
        "description": "Evolving worm-like structures that branch and merge",
        "F": 0.026,
        "k": 0.055,
        "Du": 0.2,
        "Dv": 0.1,
        "color": "#10b981",
        "recommended_palette": "Plasma"
    },
    "Maze": {
        "description": "Intricate maze-like patterns with persistent channels",
        "F": 0.022,
        "k": 0.051,
        "Du": 0.2,
        "Dv": 0.1,
        "color": "#8b5cf6",
        "recommended_palette": "Viridis"
    },
    "Spirals": {
        "description": "Rotating spiral patterns that emerge from initial conditions",
        "F": 0.014,
        "k": 0.045,
        "Du": 0.16,
        "Dv": 0.08,
        "color": "#f59e0b",
        "recommended_palette": "Turbo"
    },
    "Chaos": {
        "description": "Chaotic patterns with constantly changing structures",
        "F": 0.026,
        "k": 0.061,
        "Du": 0.16,
        "Dv": 0.08,
        "color": "#ef4444",
        "recommended_palette": "Inferno"
    }
}

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = GrayScottSimulator()
    st.session_state.running = False
    st.session_state.current_pattern = "Custom"
    st.session_state.frame_count = 0
    st.session_state.last_update = time.time()
    st.session_state.current_palette = "Default"
    st.session_state.random_palette = None
    st.session_state.palette_history = ["Default"]
    st.session_state.animation_speed = 1.0
    st.session_state.steps_per_frame = 5
    st.session_state.simulation_quality = "Standard"
    st.session_state.frame_history = []
    st.session_state.max_history = 10
    st.session_state.show_grid = False
    st.session_state.show_trails = False
    st.session_state.current_frame = None
    st.session_state.fps = 30
    st.session_state.last_frame_time = 0

# Function to generate a random palette
def generate_random_palette(num_colors=5):
    """Generate a random color palette with harmonious colors"""
    # Base hue for harmonious colors
    base_hue = random.randint(0, 360)
    saturation_range = (60, 90)
    value_range = (70, 95)
    
    colors = []
    for i in range(num_colors):
        # Create harmonious colors by varying hue around base
        hue = (base_hue + i * 360 // num_colors) % 360
        saturation = random.randint(*saturation_range)
        value = random.randint(*value_range)
        
        # Convert HSV to RGB
        c = value / 100.0
        x = c * (1 - abs((hue / 60.0) % 2 - 1))
        m = value / 100.0 - c
        
        if 0 <= hue < 60:
            r, g, b = c, x, 0
        elif 60 <= hue < 120:
            r, g, b = x, c, 0
        elif 120 <= hue < 180:
            r, g, b = 0, c, x
        elif 180 <= hue < 240:
            r, g, b = 0, x, c
        elif 240 <= hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    
    return {
        "colors": colors,
        "type": "linear",
        "description": f"Random palette with {num_colors} harmonious colors"
    }

# Function to get matplotlib colormap from palette definition
def get_cmap_from_palette(palette_name):
    """Convert palette definition to matplotlib colormap using modern API"""
    if palette_name == "Random" and st.session_state.random_palette:
        return mcolors.LinearSegmentedColormap.from_list("random", st.session_state.random_palette["colors"])
    
    palette = PREDEFINED_PALETTES.get(palette_name, PREDEFINED_PALETTES["Default"])
    
    if palette["type"] == "matplotlib":
        # Modern colormap access using registry
        return colormaps[palette["cmap"]]
    elif palette["type"] == "linear":
        return mcolors.LinearSegmentedColormap.from_list(
            palette_name, 
            palette["colors"]
        )
    return colormaps["viridis"]  # Fallback to viridis

# Function to apply pattern
def apply_pattern(pattern_name):
    pattern = PATTERNS[pattern_name]
    st.session_state.simulator.F = pattern["F"]
    st.session_state.simulator.k = pattern["k"]
    st.session_state.simulator.Du = pattern["Du"]
    st.session_state.simulator.Dv = pattern["Dv"]
    st.session_state.simulator.reset()
    st.session_state.current_pattern = pattern_name
    st.session_state.frame_count = 0
    st.session_state.frame_history = []
    st.session_state.current_frame = None
    
    # Apply recommended palette for this pattern
    st.session_state.current_palette = pattern["recommended_palette"]

# Function to generate random palette and apply it
def apply_random_palette():
    num_colors = random.randint(3, 7)
    random_palette = generate_random_palette(num_colors)
    st.session_state.random_palette = random_palette
    st.session_state.current_palette = "Random"
    st.session_state.palette_history.append("Random")

# Function to update simulator parameters
def update_simulator_params():
    if st.session_state.current_pattern == "Custom":
        st.session_state.simulator.F = st.session_state.F_val
        st.session_state.simulator.k = st.session_state.k_val
        st.session_state.simulator.Du = st.session_state.Du_val
        st.session_state.simulator.Dv = st.session_state.Dv_val

# Function to create and display the animation frame
def create_animation_frame():
    """Create and display the current animation frame"""
    # Get the appropriate colormap
    cmap = get_cmap_from_palette(st.session_state.current_palette)
    
    # Create figure with appropriate size based on quality
    quality = st.session_state.simulation_quality
    if quality == "High":
        figsize = (12, 10)
        dpi = 150
    elif quality == "Medium":
        figsize = (10, 8)
        dpi = 120
    else:  # Standard
        figsize = (8, 6)
        dpi = 100
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    
    # Get current state
    V = st.session_state.simulator.V
    
    # Apply trails effect if enabled
    if st.session_state.show_trails and len(st.session_state.frame_history) > 0:
        # Create a composite image with trails
        trail_weight = 0.7
        V_composite = V.copy()
        
        for i, past_V in enumerate(st.session_state.frame_history):
            weight = trail_weight * (i + 1) / len(st.session_state.frame_history)
            V_composite = weight * V_composite + (1 - weight) * past_V
        
        V_display = V_composite
    else:
        V_display = V
    
    # Create the visualization
    im = ax.imshow(V_display, cmap=cmap, vmin=0, vmax=1, interpolation='bilinear')
    
    # Add grid if enabled
    if st.session_state.show_grid:
        ax.grid(True, alpha=0.2, color='white', linestyle='--')
    
    # Set title with current parameters
    ax.set_title(
        f'Reactive Diffusion Simulation | Frame: {st.session_state.frame_count} | Pattern: {st.session_state.current_pattern}',
        fontsize=14, fontweight='bold', pad=20, color='white'
    )
    
    # Add pattern-specific styling
    if st.session_state.current_pattern in PATTERNS:
        pattern_color = PATTERNS[st.session_state.current_pattern]["color"]
        ax.set_title(ax.get_title(), color=pattern_color)
    
    # Remove axes
    ax.axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Concentration of V', fontsize=12, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Add timestamp
    ax.text(0.5, 0.01, f'Time: {st.session_state.frame_count * st.session_state.simulator.dt:.1f}', 
            transform=ax.transAxes, ha='center', fontsize=10, color='white',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    
    # Add performance indicator
    current_time = time.time()
    if st.session_state.last_frame_time > 0:
        fps = 1 / (current_time - st.session_state.last_frame_time)
        ax.text(0.98, 0.98, f'FPS: {fps:.1f}', 
                transform=ax.transAxes, ha='right', va='top', fontsize=10, color='white',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    st.session_state.last_frame_time = current_time
    
    # Save the figure to session state
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor='black')
    buf.seek(0)
    
    # Convert to PIL Image for better performance
    img = Image.open(buf)
    st.session_state.current_frame = img
    
    plt.close(fig)
    
    return img

# Header
st.markdown('<h1 class="main-header">Reactive Diffusion Simulator 🔬</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <strong>Explore pattern formation in reaction-diffusion systems!</strong> This simulation models how chemical reactions coupled with diffusion can spontaneously form complex patterns found in nature - from animal markings to chemical gardens.
</div>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["Simulation", "Controls", "About"])

with tab1:
    # Main visualization area
    st.markdown("### Pattern Visualization")
    
    # Animation container
    animation_container = st.container()
    
    # Animation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⏯️ Start/Stop", key="start_stop", use_container_width=True):
            st.session_state.running = not st.session_state.running
    
    with col2:
        # Animation speed control
        speed = st.slider("Animation Speed", 0.1, 3.0, st.session_state.animation_speed, 0.1)
        st.session_state.animation_speed = speed
        
        # Steps per frame control
        steps = st.slider("Steps per Frame", 1, 20, st.session_state.steps_per_frame, 1)
        st.session_state.steps_per_frame = steps
        
        # FPS control
        fps = st.slider("Target FPS", 5, 60, st.session_state.fps, 5)
        st.session_state.fps = fps
    
    with col3:
        if st.button("⟳ Reset", key="reset", use_container_width=True):
            st.session_state.simulator.reset()
            st.session_state.frame_count = 0
            st.session_state.frame_history = []
            st.session_state.current_frame = None
    
    # Status display
    status_text = "Running" if st.session_state.running else "Paused"
    status_class = "status-running" if st.session_state.running else "status-paused"
    st.markdown(f'<div class="status-badge {status_class}">{status_text}</div>', unsafe_allow_html=True)
    
    # Progress bar
    progress = min(st.session_state.frame_count % 100, 100)
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress}%"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the current frame
    with animation_container:
        if st.session_state.current_frame is not None:
            st.image(st.session_state.current_frame, use_container_width=True, output_format="PNG")
        else:
            # Create initial frame
            img = create_animation_frame()
            st.image(img, use_container_width=True, output_format="PNG")
    
    # Frame history controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Frame", key="save_frame"):
            if len(st.session_state.frame_history) >= st.session_state.max_history:
                st.session_state.frame_history.pop(0)
            st.session_state.frame_history.append(st.session_state.simulator.V.copy())
            st.success(f"Frame saved! History: {len(st.session_state.frame_history)}/{st.session_state.max_history}")
    
    with col2:
        if st.button("Clear History", key="clear_history"):
            st.session_state.frame_history = []
            st.success("Frame history cleared!")

with tab2:
    # Pattern selection
    st.markdown("### Preset Patterns")
    
    # Create pattern cards in a grid
    cols = st.columns(3)
    for i, (pattern_name, pattern_data) in enumerate(PATTERNS.items()):
        with cols[i % 3]:
            is_active = st.session_state.current_pattern == pattern_name
            active_class = "active" if is_active else ""
            
            st.markdown(f"""
            <div class="pattern-card {active_class}" style="--pattern-color: {pattern_data['color']};">
                <div class="pattern-title" style="color: {pattern_data['color']};">
                    {pattern_name}
                </div>
                <p>{pattern_data['description']}</p>
                <strong>Parameters:</strong> F = {pattern_data['F']:.3f}, k = {pattern_data['k']:.3f}<br>
                <strong>Recommended Palette:</strong> {pattern_data['recommended_palette']}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Apply {pattern_name}", key=f"pattern_{pattern_name}", use_container_width=True):
                apply_pattern(pattern_name)
    
    # Custom parameter controls
    st.markdown("### Custom Parameters")
    
    with st.expander("Advanced Settings", expanded=st.session_state.current_pattern == "Custom"):
        # Initialize sliders with current values
        if "F_val" not in st.session_state:
            st.session_state.F_val = st.session_state.simulator.F
        if "k_val" not in st.session_state:
            st.session_state.k_val = st.session_state.simulator.k
        if "Du_val" not in st.session_state:
            st.session_state.Du_val = st.session_state.simulator.Du
        if "Dv_val" not in st.session_state:
            st.session_state.Dv_val = st.session_state.simulator.Dv
        
        # Create sliders with immediate updates
        F_val = st.slider("Feed Rate (F)", 0.01, 0.06, st.session_state.F_val, 0.001, 
                         key="F_slider", on_change=update_simulator_params)
        k_val = st.slider("Kill Rate (k)", 0.03, 0.08, st.session_state.k_val, 0.001,
                         key="k_slider", on_change=update_simulator_params)
        Du_val = st.slider("Diffusion U", 0.1, 0.3, st.session_state.Du_val, 0.01,
                          key="Du_slider", on_change=update_simulator_params)
        Dv_val = st.slider("Diffusion V", 0.05, 0.2, st.session_state.Dv_val, 0.01,
                          key="Dv_slider", on_change=update_simulator_params)
        
        # Update session state
        st.session_state.F_val = F_val
        st.session_state.k_val = k_val
        st.session_state.Du_val = Du_val
        st.session_state.Dv_val = Dv_val
        
        # Set pattern to Custom if any parameter is changed
        if st.session_state.current_pattern != "Custom":
            st.session_state.current_pattern = "Custom"
    
    # Color palette selection
    st.markdown("### Color Palettes")
    
    # Display current palette
    current_palette_name = st.session_state.current_palette
    if current_palette_name == "Random" and st.session_state.random_palette:
        current_palette = st.session_state.random_palette
        palette_desc = current_palette["description"]
        palette_colors = current_palette["colors"]
    elif current_palette_name in PREDEFINED_PALETTES:
        current_palette = PREDEFINED_PALETTES[current_palette_name]
        palette_desc = current_palette["description"]
        palette_colors = current_palette["colors"] if current_palette["type"] == "linear" else []
    else:
        current_palette = PREDEFINED_PALETTES["Default"]
        palette_desc = current_palette["description"]
        palette_colors = current_palette["colors"]
    
    # Create palette preview
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.imshow(gradient, aspect='auto', cmap=get_cmap_from_palette(current_palette_name))
    ax.set_axis_off()
    st.pyplot(fig, width='stretch', clear_figure=True)
    
    st.caption(f"Current: {current_palette_name} - {palette_desc}")
    
    # Palette selection
    palette_options = list(PREDEFINED_PALETTES.keys())
    selected_palette = st.selectbox("Select predefined palette", palette_options, 
                                  index=palette_options.index(st.session_state.current_palette) 
                                  if st.session_state.current_palette in palette_options else 0)
    
    if selected_palette != st.session_state.current_palette:
        st.session_state.current_palette = selected_palette
        if selected_palette in st.session_state.palette_history:
            st.session_state.palette_history.remove(selected_palette)
        st.session_state.palette_history.insert(0, selected_palette)
    
    # Random palette generator
    if st.button("✨ Generate Random Palette", key="random_palette_btn"):
        apply_random_palette()
    
    # Visualization options
    st.markdown("### Visualization Options")
    
    # Quality setting
    quality = st.selectbox("Simulation Quality", ["Standard", "Medium", "High"], 
                          index=["Standard", "Medium", "High"].index(st.session_state.simulation_quality))
    st.session_state.simulation_quality = quality
    
    # Grid toggle
    st.session_state.show_grid = st.checkbox("Show Grid", value=st.session_state.show_grid)
    
    # Trails toggle
    st.session_state.show_trails = st.checkbox("Show Trails", value=st.session_state.show_trails,
                                           help="Show a trail effect by blending with previous frames")
    
    # Initial disturbance options
    st.markdown("### Initial Conditions")
    
    if st.button("Add Random Disturbance"):
        st.session_state.simulator.add_random_disturbance()
        st.success("Added random disturbance to the simulation!")

with tab3:
    # Scientific explanation
    st.markdown("""
    ## How does this work?
    
    **The Gray-Scott Model** simulates two chemicals (U and V) reacting and diffusing through space:
    
    - **U** (substrate): Continuously fed into the system at rate F
    - **V** (activator): Consumes U to replicate itself
    - **Reaction**: U + 2V → 3V (V catalyzes its own production)
    - **Depletion**: V is removed at rate (F + k)
    
    This creates a system where small fluctuations can amplify into stable patterns through a process called **Turing instability**. The balance between reaction rates and diffusion coefficients determines the resulting patterns:
    
    - **Low k**: Spots form and replicate
    - **Medium k**: Worm-like structures emerge
    - **High k**: Maze-like patterns develop
    
    These patterns mirror biological morphogenesis seen in animal coats, coral formations, and chemical gardens.
    
    ## Animation Features
    
    This enhanced simulator includes several animation features:
    
    - **Adjustable Speed**: Control how fast the simulation runs
    - **Steps per Frame**: Adjust the computational steps between visual updates
    - **Frame History**: Save and blend previous frames to create trail effects
    - **Quality Settings**: Balance between performance and visual quality
    - **FPS Control**: Target specific frame rates for smooth animation
    - **Performance Indicator**: Real-time FPS display
    
    ## Color Palettes
    
    **Color Palettes** play a crucial role in visualizing these patterns:
    - **Perceptually uniform** palettes (Viridis, Plasma) help accurately represent data
    - **Thematic palettes** (Fire, Ocean) enhance visual storytelling
    - **Random palettes** can reveal hidden structures through color contrast
    
    Try different palettes to highlight different aspects of the patterns!
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>Reactive Diffusion Simulator | Based on the Gray-Scott Model</p>
    <p>This simulation demonstrates how simple reaction-diffusion equations can generate complex biological patterns.</p>
    <p>💡 Tip: Try different color palettes to reveal hidden structures in the patterns!</p>
</div>
""", unsafe_allow_html=True)

# Animation loop with proper frame timing
if st.session_state.running:
    # Calculate time since last frame
    current_time = time.time()
    frame_interval = 1.0 / st.session_state.fps
    
    # Only update if enough time has passed
    if 'last_animation_time' not in st.session_state:
        st.session_state.last_animation_time = current_time
    
    if current_time - st.session_state.last_animation_time >= frame_interval:
        # Run simulation steps based on animation speed and steps per frame
        num_steps = max(1, int(st.session_state.steps_per_frame * st.session_state.animation_speed))
        
        for _ in range(num_steps):
            st.session_state.simulator.step()
            st.session_state.frame_count += 1
        
        # Create and display the new frame
        create_animation_frame()
        
        # Update the last animation time
        st.session_state.last_animation_time = current_time
        
        # Force a rerun to display the new frame
        st.rerun()
    else:
        # Wait for the next frame
        time.sleep(0.01)
        st.rerun()