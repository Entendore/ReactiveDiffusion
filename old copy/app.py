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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.integrate import odeint
import json
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import threading
import queue
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import networkx as nx
from scipy.spatial import Delaunay
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd
from datetime import datetime
import hashlib
import os
from pathlib import Path
import tempfile
import zipfile
import pickle
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Ultra-Advanced Reactive Diffusion Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    /* Advanced styling */
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        color: #1a3a6c;
        text-align: center;
        padding: 2rem 0;
        border-bottom: 3px solid #eaeaea;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(26,58,108,0.1) 0%, rgba(44,82,130,0.2) 50%, rgba(26,58,108,0.1) 100%);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Advanced card styles */
    .advanced-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .advanced-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    }
    
    .advanced-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* ML panel styles */
    .ml-panel {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #0ea5e9;
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.1);
    }
    
    /* Neural network visualization */
    .neural-network {
        background: #1e293b;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: inset 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Progress indicators */
    .ml-progress {
        width: 100%;
        height: 12px;
        background: #e2e8f0;
        border-radius: 6px;
        overflow: hidden;
        margin: 1rem 0;
        position: relative;
    }
    
    .ml-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 6px;
        transition: width 0.5s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Confidence meter */
    .confidence-meter {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Feature importance */
    .feature-bar {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: #f8fafc;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .feature-bar:hover {
        background: #e2e8f0;
        transform: translateX(5px);
    }
    
    .feature-name {
        flex: 1;
        font-weight: 500;
        color: #374151;
    }
    
    .feature-value {
        width: 60px;
        text-align: right;
        font-weight: 700;
        color: #1e40af;
    }
    
    /* Tutorial styles */
    .tutorial-step {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #f59e0b;
        position: relative;
    }
    
    .tutorial-step::before {
        content: attr(data-step);
        position: absolute;
        top: -10px;
        left: -10px;
        width: 30px;
        height: 30px;
        background: #f59e0b;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    /* Performance metrics */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Chemical network visualization */
    .chemical-node {
        fill: #3b82f6;
        stroke: #1e40af;
        stroke-width: 2px;
    }
    
    .chemical-edge {
        stroke: #9ca3af;
        stroke-width: 2px;
        marker-end: url(#arrowhead);
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 3px solid #e2e8f0;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 1rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Advanced controls */
    .control-group {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .control-label {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Simulation canvas */
    .simulation-canvas {
        border: 2px solid #3b82f6;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
        background: #000;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f9fafb;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    
    /* Template card */
    .template-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .template-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    
    /* Export options */
    .export-option {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .export-option:hover {
        background: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Data structures for advanced features
@dataclass
class ChemicalSpecies:
    """Represents a chemical species in reaction network"""
    name: str
    concentration: np.ndarray
    diffusion_rate: float
    color: str
    description: str

@dataclass
class Reaction:
    """Represents a reaction between chemical species"""
    reactants: List[str]
    products: List[str]
    rate_constant: float
    description: str

@dataclass
class SimulationState:
    """Complete state of simulation"""
    chemicals: Dict[str, ChemicalSpecies]
    reactions: List[Reaction]
    parameters: Dict[str, float]
    time: float
    frame: int

@dataclass
class SimulationTemplate:
    """Template for saving/loading simulations"""
    name: str
    description: str
    parameters: Dict[str, float]
    chemicals: Dict[str, float]
    reactions: List[Dict]
    tags: List[str]
    created_at: str

class MultiChemicalSimulator:
    """Advanced simulator supporting multiple chemical species"""
    
    def __init__(self, width=200, height=200, num_chemicals=2):
        self.width = width
        self.height = height
        self.num_chemicals = num_chemicals
        self.chemicals = {}
        self.reactions = []
        self.time = 0.0
        self.frame = 0
        
        # Initialize default chemicals
        self._initialize_default_chemicals()
        
    def _initialize_default_chemicals(self):
        """Initialize default chemical species"""
        # Substrate
        self.chemicals['U'] = ChemicalSpecies(
            name='U',
            concentration=np.ones((self.height, self.width)),
            diffusion_rate=0.2,
            color='#3b82f6',
            description='Substrate chemical'
        )
        
        # Activator
        self.chemicals['V'] = ChemicalSpecies(
            name='V',
            concentration=np.zeros((self.height, self.width)),
            diffusion_rate=0.1,
            color='#ef4444',
            description='Activator chemical'
        )
        
        # Add third chemical for advanced models
        if self.num_chemicals > 2:
            self.chemicals['W'] = ChemicalSpecies(
                name='W',
                concentration=np.zeros((self.height, self.width)),
                diffusion_rate=0.15,
                color='#10b981',
                description='Inhibitor chemical'
            )
    
    def add_chemical(self, name: str, initial_concentration: float = 0.0, 
                   diffusion_rate: float = 0.1, color: str = '#8b5cf6'):
        """Add a new chemical species"""
        self.chemicals[name] = ChemicalSpecies(
            name=name,
            concentration=np.full((self.height, self.width), initial_concentration),
            diffusion_rate=diffusion_rate,
            color=color,
            description=f'Chemical {name}'
        )
    
    def add_reaction(self, reactants: List[str], products: List[str], 
                   rate_constant: float = 0.1, description: str = ""):
        """Add a reaction to the network"""
        reaction = Reaction(
            reactants=reactants,
            products=products,
            rate_constant=rate_constant,
            description=description
        )
        self.reactions.append(reaction)
    
    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Calculate Laplacian with periodic boundary conditions"""
        return (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4 * field
        )
    
    def step(self, dt: float = 1.0):
        """Advance simulation by one timestep"""
        # Calculate diffusion for each chemical
        diffusion_terms = {}
        for name, chemical in self.chemicals.items():
            diffusion_terms[name] = chemical.diffusion_rate * self.laplacian(chemical.concentration)
        
        # Calculate reaction terms
        reaction_terms = {}
        for name in self.chemicals:
            reaction_terms[name] = np.zeros((self.height, self.width))
        
        for reaction in self.reactions:
            # Calculate reaction rate
            rate = reaction.rate_constant
            for reactant in reaction.reactants:
                rate *= self.chemicals[reactant].concentration
            
            # Update reactants and products
            for reactant in reaction.reactants:
                reaction_terms[reactant] -= rate
            for product in reaction.products:
                reaction_terms[product] += rate
        
        # Update concentrations
        for name, chemical in self.chemicals.items():
            chemical.concentration += dt * (diffusion_terms[name] + reaction_terms[name])
            # Ensure numerical stability
            np.clip(chemical.concentration, 0, 1, out=chemical.concentration)
        
        self.time += dt
        self.frame += 1
    
    def get_combined_concentration(self) -> np.ndarray:
        """Get combined concentration for visualization"""
        # Weighted combination of all chemicals
        combined = np.zeros((self.height, self.width))
        weights = {'U': 0.3, 'V': 0.5, 'W': 0.2}
        
        for name, chemical in self.chemicals.items():
            weight = weights.get(name, 0.1)
            combined += weight * chemical.concentration
        
        return combined
    
    def analyze_network(self) -> Dict:
        """Analyze the reaction network"""
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes
        for name in self.chemicals:
            G.add_node(name)
        
        # Add edges for reactions
        for reaction in self.reactions:
            for reactant in reaction.reactants:
                for product in reaction.products:
                    G.add_edge(reactant, product, weight=reaction.rate_constant)
        
        # Calculate network metrics
        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'clustering_coefficient': nx.average_clustering(G.to_undirected()),
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'eigenvalue_centrality': nx.eigenvector_centrality(G)
        }
        
        return metrics
    
    def save_state(self) -> Dict:
        """Save current simulation state"""
        state = {
            'chemicals': {
                name: {
                    'concentration': chemical.concentration.tolist(),
                    'diffusion_rate': chemical.diffusion_rate,
                    'color': chemical.color,
                    'description': chemical.description
                }
                for name, chemical in self.chemicals.items()
            },
            'reactions': [
                {
                    'reactants': reaction.reactants,
                    'products': reaction.products,
                    'rate_constant': reaction.rate_constant,
                    'description': reaction.description
                }
                for reaction in self.reactions
            ],
            'time': self.time,
            'frame': self.frame,
            'width': self.width,
            'height': self.height
        }
        return state
    
    def load_state(self, state: Dict):
        """Load simulation state"""
        self.width = state['width']
        self.height = state['height']
        self.time = state['time']
        self.frame = state['frame']
        
        # Load chemicals
        self.chemicals = {}
        for name, chem_data in state['chemicals'].items():
            self.chemicals[name] = ChemicalSpecies(
                name=name,
                concentration=np.array(chem_data['concentration']),
                diffusion_rate=chem_data['diffusion_rate'],
                color=chem_data['color'],
                description=chem_data['description']
            )
        
        # Load reactions
        self.reactions = []
        for reaction_data in state['reactions']:
            self.reactions.append(Reaction(
                reactants=reaction_data['reactants'],
                products=reaction_data['products'],
                rate_constant=reaction_data['rate_constant'],
                description=reaction_data['description']
            ))

class PatternPredictionModel:
    """Machine learning model for pattern prediction"""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.regressor = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['F', 'k', 'Du', 'Dv', 'initial_variance', 'initial_mean']
        
    def extract_features(self, simulator) -> np.ndarray:
        """Extract features from simulation state"""
        features = []
        
        # Parameter features
        features.extend([
            getattr(simulator, 'F', 0.04),
            getattr(simulator, 'k', 0.06),
            getattr(simulator, 'Du', 0.2),
            getattr(simulator, 'Dv', 0.1)
        ])
        
        # Statistical features
        if hasattr(simulator, 'V'):
            features.extend([
                np.var(simulator.V),
                np.mean(simulator.V)
            ])
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features)
    
    def train_pattern_classifier(self, training_data: List[Tuple]):
        """Train pattern classifier"""
        if len(training_data) < 10:
            return False
        
        X = np.array([self.extract_features(sim) for sim, _ in training_data])
        y = np.array([label for _, label in training_data])
        
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        return True
    
    def predict_pattern(self, simulator) -> Dict:
        """Predict pattern type"""
        if not self.is_trained:
            return {'pattern': 'Unknown', 'confidence': 0.0}
        
        features = self.extract_features(simulator)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return {
            'pattern': prediction,
            'confidence': confidence,
            'probabilities': dict(zip(self.classifier.classes_, probabilities))
        }
    
    def predict_parameters(self, desired_pattern: str, num_samples: int = 100) -> List[Dict]:
        """Predict parameters for desired pattern"""
        if not self.is_trained:
            return []
        
        # Generate parameter combinations
        F_range = np.linspace(0.01, 0.06, num_samples)
        k_range = np.linspace(0.03, 0.08, num_samples)
        
        predictions = []
        for F in F_range:
            for k in k_range:
                # Create dummy simulator for prediction
                dummy_sim = type('DummySim', (), {})()
                dummy_sim.F = F
                dummy_sim.k = k
                dummy_sim.Du = 0.2
                dummy_sim.Dv = 0.1
                
                pred = self.predict_pattern(dummy_sim)
                if pred['pattern'] == desired_pattern:
                    predictions.append({
                        'F': F,
                        'k': k,
                        'confidence': pred['confidence']
                    })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions[:10]  # Return top 10

class TemplateManager:
    """Manages simulation templates"""
    
    def __init__(self):
        self.templates = []
        self.templates_dir = Path("templates")
        self.templates_dir.mkdir(exist_ok=True)
        self.load_templates()
    
    def load_templates(self):
        """Load templates from files"""
        self.templates = []
        
        # Load built-in templates
        self.templates.extend(self.get_builtin_templates())
        
        # Load user templates
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                    template = SimulationTemplate(**template_data)
                    self.templates.append(template)
            except Exception as e:
                st.error(f"Error loading template {template_file}: {e}")
    
    def get_builtin_templates(self) -> List[SimulationTemplate]:
        """Get built-in templates"""
        return [
            SimulationTemplate(
                name="Classic Spots",
                description="Self-replicating spots pattern",
                parameters={"F": 0.035, "k": 0.065, "Du": 0.2, "Dv": 0.1},
                chemicals={"U": 1.0, "V": 0.0},
                reactions=[{"reactants": ["U", "V", "V"], "products": ["V", "V", "V"], "rate": 1.0}],
                tags=["spots", "classic", "stable"],
                created_at=datetime.now().isoformat()
            ),
            SimulationTemplate(
                name="Worm Maze",
                description="Worm-like maze patterns",
                parameters={"F": 0.026, "k": 0.055, "Du": 0.2, "Dv": 0.1},
                chemicals={"U": 1.0, "V": 0.0},
                reactions=[{"reactants": ["U", "V", "V"], "products": ["V", "V", "V"], "rate": 1.0}],
                tags=["worms", "maze", "complex"],
                created_at=datetime.now().isoformat()
            ),
            SimulationTemplate(
                name="Spiral Waves",
                description="Rotating spiral wave patterns",
                parameters={"F": 0.014, "k": 0.045, "Du": 0.16, "Dv": 0.08},
                chemicals={"U": 1.0, "V": 0.0},
                reactions=[{"reactants": ["U", "V", "V"], "products": ["V", "V", "V"], "rate": 1.0}],
                tags=["spirals", "waves", "dynamic"],
                created_at=datetime.now().isoformat()
            ),
            SimulationTemplate(
                name="Chaos",
                description="Chaotic pattern formation",
                parameters={"F": 0.026, "k": 0.061, "Du": 0.16, "Dv": 0.08},
                chemicals={"U": 1.0, "V": 0.0},
                reactions=[{"reactants": ["U", "V", "V"], "products": ["V", "V", "V"], "rate": 1.0}],
                tags=["chaos", "unstable", "complex"],
                created_at=datetime.now().isoformat()
            )
        ]
    
    def save_template(self, template: SimulationTemplate):
        """Save template to file"""
        template_file = self.templates_dir / f"{template.name.replace(' ', '_')}.json"
        
        with open(template_file, 'w') as f:
            json.dump(template.__dict__, f, indent=2)
        
        self.templates.append(template)
    
    def get_template(self, name: str) -> Optional[SimulationTemplate]:
        """Get template by name"""
        for template in self.templates:
            if template.name == name:
                return template
        return None
    
    def search_templates(self, query: str) -> List[SimulationTemplate]:
        """Search templates by name or tags"""
        query = query.lower()
        results = []
        
        for template in self.templates:
            if (query in template.name.lower() or 
                query in template.description.lower() or
                any(query in tag.lower() for tag in template.tags)):
                results.append(template)
        
        return results

# Initialize advanced session state
if 'multi_simulator' not in st.session_state:
    st.session_state.multi_simulator = MultiChemicalSimulator(num_chemicals=3)
    st.session_state.prediction_model = PatternPredictionModel()
    st.session_state.template_manager = TemplateManager()
    st.session_state.ml_training_data = []
    st.session_state.show_ml_panel = False
    st.session_state.show_tutorial = False
    st.session_state.show_performance = False
    st.session_state.simulation_history = []
    st.session_state.bookmarked_patterns = []
    st.session_state.export_queue = []
    st.session_state.import_queue = []
    st.session_state.current_template = None

# Advanced chemical models
ADVANCED_MODELS = {
    "Gray-Scott": {
        "chemicals": ["U", "V"],
        "reactions": [
            {"reactants": ["U", "V", "V"], "products": ["V", "V", "V"], "rate": 1.0},
            {"reactants": ["V"], "products": [], "rate": "F + k"}
        ],
        "description": "Classic two-chemical model"
    },
    "FitzHugh-Nagumo": {
        "chemicals": ["U", "V"],
        "reactions": [
            {"reactants": ["V"], "products": ["U"], "rate": "a"},
            {"reactants": ["U", "U", "U"], "products": ["U", "U", "U", "U"], "rate": "b"}
        ],
        "description": "Neural excitation model"
    },
    "Oregonator": {
        "chemicals": ["X", "Y", "Z"],
        "reactions": [
            {"reactants": ["X", "Y"], "products": ["X", "X"], "rate": "k1"},
            {"reactants": ["X", "X"], "products": ["X", "X", "X"], "rate": "k2"},
            {"reactants": ["X", "Z"], "products": ["Y", "Z"], "rate": "k3"}
        ],
        "description": "Belousov-Zhabotinsky reaction model"
    },
    "Brusselator": {
        "chemicals": ["X", "Y"],
        "reactions": [
            {"reactants": [], "products": ["X"], "rate": "A"},
            {"reactants": ["X"], "products": ["Y"], "rate": "B"},
            {"reactants": ["X", "X", "Y"], "products": ["X", "X", "X"], "rate": "C"}
        ],
        "description": "Auto-catalytic reaction model"
    }
}

# Function to create multi-chemical visualization
def create_multi_chemical_visualization():
    """Create visualization for multi-chemical simulation"""
    st.markdown("### Chemical Concentration Visualization")
    
    # Get simulator
    sim = st.session_state.multi_simulator
    
    # Visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        view_mode = st.selectbox(
            "View Mode",
            ["Combined View", "Individual Chemicals", "3D Surface Plot", "Statistical Analysis"]
        )
    
    with col2:
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Twilight"]
        )
    
    # Create visualization based on selected mode
    if view_mode == "Combined View":
        # Combined view of all chemicals
        combined = sim.get_combined_concentration()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(combined, cmap=color_scheme.lower())
        plt.colorbar(im, ax=ax, label="Combined Concentration")
        ax.set_title(f"Combined Chemical Concentration (Frame {sim.frame})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        st.pyplot(fig)
        plt.close()
        
    elif view_mode == "Individual Chemicals":
        # Individual chemical views
        num_chemicals = len(sim.chemicals)
        cols = min(3, num_chemicals)
        rows = (num_chemicals + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (name, chemical) in enumerate(sim.chemicals.items()):
            if i < len(axes):
                im = axes[i].imshow(chemical.concentration, cmap=color_scheme.lower())
                axes[i].set_title(f"Chemical {name}")
                plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplots
        for i in range(num_chemicals, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    elif view_mode == "3D Surface Plot":
        # 3D surface plot of combined concentration
        combined = sim.get_combined_concentration()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        x = np.arange(0, sim.width, 5)  # Sample every 5th point for performance
        y = np.arange(0, sim.height, 5)
        X, Y = np.meshgrid(x, y)
        Z = combined[::5, ::5]
        
        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=color_scheme.lower(), 
                              linewidth=0, antialiased=True, alpha=0.8)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Concentration")
        ax.set_title(f"3D Surface Plot (Frame {sim.frame})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Concentration")
        
        st.pyplot(fig)
        plt.close()
        
    elif view_mode == "Statistical Analysis":
        # Statistical analysis of chemical concentrations
        st.markdown("#### Statistical Analysis")
        
        # Create dataframe with statistics
        stats_data = []
        for name, chemical in sim.chemicals.items():
            stats_data.append({
                'Chemical': name,
                'Mean': np.mean(chemical.concentration),
                'Std Dev': np.std(chemical.concentration),
                'Min': np.min(chemical.concentration),
                'Max': np.max(chemical.concentration),
                'Median': np.median(chemical.concentration)
            })
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, width='stretch')
        
        # Create histogram plots
        fig, axes = plt.subplots(1, len(sim.chemicals), figsize=(15, 5))
        if len(sim.chemicals) == 1:
            axes = [axes]
        
        for i, (name, chemical) in enumerate(sim.chemicals.items()):
            axes[i].hist(chemical.concentration.flatten(), bins=50, alpha=0.7)
            axes[i].set_title(f"Distribution of {name}")
            axes[i].set_xlabel("Concentration")
            axes[i].set_ylabel("Frequency")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Simulation info
    st.markdown("#### Simulation Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Time Steps", sim.frame)
    
    with col2:
        st.metric("Chemicals", len(sim.chemicals))
    
    with col3:
        st.metric("Reactions", len(sim.reactions))

# Function to create reaction network visualization
def create_reaction_network_viz():
    """Create visualization of reaction network"""
    st.markdown("### Reaction Network Visualization")
    
    # Get simulator
    sim = st.session_state.multi_simulator
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add nodes
    for name in sim.chemicals:
        G.add_node(name)
    
    # Add edges for reactions
    for reaction in sim.reactions:
        for reactant in reaction.reactants:
            for product in reaction.products:
                G.add_edge(reactant, product, weight=reaction.rate_constant)
    
    # Network analysis
    metrics = sim.analyze_network()
    
    # Display metrics
    st.markdown("#### Network Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nodes", metrics['num_nodes'])
    
    with col2:
        st.metric("Edges", metrics['num_edges'])
    
    with col3:
        st.metric("Density", f"{metrics['density']:.3f}")
    
    # Visualize network
    st.markdown("#### Network Graph")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    
    # Draw edges with width based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights], 
                          edge_color='gray', alpha=0.7, arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    plt.title("Reaction Network")
    plt.axis('off')
    
    st.pyplot(plt)
    plt.close()
    
    # Display reaction details
    st.markdown("#### Reaction Details")
    
    for i, reaction in enumerate(sim.reactions):
        with st.expander(f"Reaction {i+1}: {reaction.description or 'No description'}"):
            st.markdown(f"**Reactants:** {', '.join(reaction.reactants)}")
            st.markdown(f"**Products:** {', '.join(reaction.products)}")
            st.markdown(f"**Rate Constant:** {reaction.rate_constant}")

# Function to create ML training interface
def create_ml_training_interface():
    """Create interface for training ML models"""
    st.markdown("### Machine Learning Training")
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Data")
        
        # Option to generate synthetic training data
        if st.button("Generate Synthetic Training Data", width='stretch'):
            generate_synthetic_training_data()
            st.success("Generated synthetic training data")
        
        # Option to upload training data
        uploaded_file = st.file_uploader(
            "Upload Training Data",
            type=['json', 'csv'],
            help="Upload a file with simulation parameters and pattern labels"
        )
        
        if uploaded_file is not None:
            process_uploaded_training_data(uploaded_file)
    
    with col2:
        st.markdown("#### Model Configuration")
        
        # Classifier options
        n_estimators = st.slider("Random Forest Estimators", 10, 200, 100)
        max_depth = st.slider("Max Depth", 1, 20, 10)
        
        # Regressor options
        hidden_layers = st.selectbox(
            "MLP Hidden Layers",
            [(50,), (100,), (100, 50), (100, 50, 25)]
        )
    
    # Training button
    if st.button("Train Pattern Prediction Model", width='stretch'):
        if len(st.session_state.ml_training_data) > 0:
            with st.spinner("Training model..."):
                # Update model parameters
                st.session_state.prediction_model.classifier.n_estimators = n_estimators
                st.session_state.prediction_model.classifier.max_depth = max_depth
                st.session_state.prediction_model.regressor.hidden_layer_sizes = hidden_layers
                
                # Train model
                success = st.session_state.prediction_model.train_pattern_classifier(
                    st.session_state.ml_training_data
                )
                
                if success:
                    st.success("Model trained successfully!")
                    st.session_state.show_ml_panel = True
                else:
                    st.error("Training failed. Not enough training data.")
        else:
            st.error("No training data available. Please generate or upload training data.")
    
    # Training data preview
    if st.session_state.ml_training_data:
        st.markdown("#### Training Data Preview")
        
        # Create dataframe from training data
        preview_data = []
        for i, (sim, label) in enumerate(st.session_state.ml_training_data[:5]):  # Show first 5
            features = st.session_state.prediction_model.extract_features(sim)
            row = {'Pattern': label}
            for j, name in enumerate(st.session_state.prediction_model.feature_names):
                row[name] = features[j]
            preview_data.append(row)
        
        df = pd.DataFrame(preview_data)
        st.dataframe(df, width='stretch')
        
        st.info(f"Showing 5 of {len(st.session_state.ml_training_data)} training samples")

# Function to generate synthetic training data
def generate_synthetic_training_data():
    """Generate synthetic training data for ML model"""
    # Define parameter ranges for different patterns
    pattern_params = {
        "Spots": {"F_range": (0.03, 0.04), "k_range": (0.06, 0.07)},
        "Stripes": {"F_range": (0.025, 0.035), "k_range": (0.05, 0.06)},
        "Maze": {"F_range": (0.02, 0.03), "k_range": (0.045, 0.055)},
        "Chaos": {"F_range": (0.035, 0.045), "k_range": (0.055, 0.065)},
        "Waves": {"F_range": (0.015, 0.025), "k_range": (0.045, 0.055)}
    }
    
    # Generate training samples
    for pattern, params in pattern_params.items():
        for _ in range(20):  # 20 samples per pattern
            # Random parameters within range
            F = np.random.uniform(*params["F_range"])
            k = np.random.uniform(*params["k_range"])
            Du = np.random.uniform(0.15, 0.25)
            Dv = np.random.uniform(0.05, 0.15)
            
            # Create simulator with these parameters
            sim = MultiChemicalSimulator(width=100, height=100)
            
            # Set parameters
            sim.F = F
            sim.k = k
            sim.Du = Du
            sim.Dv = Dv
            
            # Run simulation for a few steps
            for _ in range(10):
                sim.step()
            
            # Add to training data
            st.session_state.ml_training_data.append((sim, pattern))

# Function to process uploaded training data
def process_uploaded_training_data(uploaded_file):
    """Process uploaded training data file"""
    try:
        if uploaded_file.name.endswith('.json'):
            # Process JSON file
            data = json.load(uploaded_file)
            
            for item in data:
                # Create simulator from item
                sim = MultiChemicalSimulator(
                    width=item.get('width', 100),
                    height=item.get('height', 100)
                )
                
                # Set parameters
                for param, value in item.get('parameters', {}).items():
                    setattr(sim, param, value)
                
                # Add to training data
                st.session_state.ml_training_data.append((sim, item.get('pattern', 'Unknown')))
                
        elif uploaded_file.name.endswith('.csv'):
            # Process CSV file
            df = pd.read_csv(uploaded_file)
            
            for _, row in df.iterrows():
                # Create simulator from row
                sim = MultiChemicalSimulator(
                    width=row.get('width', 100),
                    height=row.get('height', 100)
                )
                
                # Set parameters
                for param in ['F', 'k', 'Du', 'Dv']:
                    if param in row:
                        setattr(sim, param, row[param])
                
                # Add to training data
                st.session_state.ml_training_data.append((sim, row.get('pattern', 'Unknown')))
        
        st.success(f"Loaded {len(data) if uploaded_file.name.endswith('.json') else len(df)} training samples")
        
    except Exception as e:
        st.error(f"Error processing training data: {str(e)}")

# Function to create data export interface
def create_data_export_interface():
    """Create comprehensive data export interface"""
    st.markdown("### Data Export & Management")
    
    # Export options
    st.markdown("#### Export Current Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export format selection
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "Pickle", "CSV Data", "Image Sequence", "Video"]
        )
        
        # Export options
        include_analysis = st.checkbox("Include Analysis Data", value=True)
        include_history = st.checkbox("Include Frame History", value=False)
        compress_data = st.checkbox("Compress Export", value=True)
    
    with col2:
        # Export destination
        export_path = st.text_input("Export Path", value="./exports/")
        filename_prefix = st.text_input("Filename Prefix", value="simulation_export")
        
        # Export button
        if st.button("📤 Export Simulation", width='stretch'):
            perform_export(export_format, export_path, filename_prefix, 
                         include_analysis, include_history, compress_data)
    
    # Batch export
    st.markdown("#### Batch Export")
    
    if st.session_state.export_queue:
        st.info(f"Items in export queue: {len(st.session_state.export_queue)}")
        
        if st.button("Process Export Queue", width='stretch'):
            process_export_queue()
    
    # Export history
    st.markdown("#### Export History")
    
    # Mock export history
    export_history = [
        {"filename": "simulation_001.json", "date": "2024-01-15", "size": "2.3 MB"},
        {"filename": "simulation_002.zip", "date": "2024-01-14", "size": "15.7 MB"},
        {"filename": "video_export.mp4", "date": "2024-01-13", "size": "45.2 MB"}
    ]
    
    for export_item in export_history:
        with st.expander(f"{export_item['filename']} ({export_item['date']})"):
            st.write(f"Size: {export_item['size']}")
            st.write(f"Date: {export_item['date']}")
            col1, col2 = st.columns(2)
            with col1:
                st.button("📥 Download", key=f"download_{export_item['filename']}")
            with col2:
                st.button("🗑️ Delete", key=f"delete_{export_item['filename']}")

# Function to create data import interface
def create_data_import_interface():
    """Create comprehensive data import interface"""
    st.markdown("### Data Import & Loading")
    
    # File upload
    st.markdown("#### Import Simulation File")
    
    uploaded_file = st.file_uploader(
        "Choose a simulation file",
        type=['json', 'pkl', 'zip'],
        help="Upload a previously saved simulation"
    )
    
    if uploaded_file is not None:
        # Process uploaded file
        file_type = uploaded_file.name.split('.')[-1]
        
        if file_type == 'json':
            process_json_import(uploaded_file)
        elif file_type == 'pkl':
            process_pickle_import(uploaded_file)
        elif file_type == 'zip':
            process_zip_import(uploaded_file)
    
    # Template loading
    st.markdown("#### Load from Templates")
    
    # Template search
    search_query = st.text_input("Search Templates", placeholder="Search by name or tags...")
    
    if search_query:
        matching_templates = st.session_state.template_manager.search_templates(search_query)
    else:
        matching_templates = st.session_state.template_manager.templates
    
    # Display templates
    for template in matching_templates:
        with st.expander(f"{template.name} - {template.description}"):
            # Template details
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Parameters:**")
                for param, value in template.parameters.items():
                    st.write(f"  {param}: {value}")
            
            with col2:
                st.write("**Tags:**")
                st.write(", ".join(template.tags))
                st.write(f"**Created:** {template.created_at[:10]}")
            
            # Load button
            if st.button(f"Load Template", key=f"load_{template.name}"):
                load_template(template)

# Function to create template management interface
def create_template_management_interface():
    """Create template creation and management interface"""
    st.markdown("### Template Management")
    
    # Create new template
    st.markdown("#### Create New Template")
    
    with st.form("create_template"):
        template_name = st.text_input("Template Name*", placeholder="Enter template name...")
        template_description = st.text_area("Description*", placeholder="Describe this template...")
        
        # Tags
        tags_input = st.text_input("Tags (comma-separated)", placeholder="spots, stable, classic")
        tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()] if tags_input else []
        
        # Save current state as template
        if st.form_submit_button("💾 Save as Template"):
            if template_name and template_description:
                # Create template from current simulation
                current_sim = st.session_state.multi_simulator
                
                template = SimulationTemplate(
                    name=template_name,
                    description=template_description,
                    parameters={
                        'width': current_sim.width,
                        'height': current_sim.height,
                        'time': current_sim.time,
                        'frame': current_sim.frame
                    },
                    chemicals={
                        name: np.mean(chem.concentration).item()
                        for name, chem in current_sim.chemicals.items()
                    },
                    reactions=[
                        {
                            'reactants': reaction.reactants,
                            'products': reaction.products,
                            'rate_constant': reaction.rate_constant,
                            'description': reaction.description
                        }
                        for reaction in current_sim.reactions
                    ],
                    tags=tags,
                    created_at=datetime.now().isoformat()
                )
                
                # Save template
                st.session_state.template_manager.save_template(template)
                st.success(f"Template '{template_name}' saved successfully!")
            else:
                st.error("Please fill in all required fields")
    
    # Manage existing templates
    st.markdown("#### Manage Templates")
    
    # Template list
    for template in st.session_state.template_manager.templates:
        with st.expander(f"{template.name}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 Edit", key=f"edit_{template.name}"):
                    st.info("Edit functionality would be implemented here")
            
            with col2:
                if st.button("📋 Copy", key=f"copy_{template.name}"):
                    st.info("Copy functionality would be implemented here")
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{template.name}"):
                    st.info("Delete functionality would be implemented here")

# Helper functions for export/import
def perform_export(format_type: str, path: str, prefix: str, 
                 include_analysis: bool, include_history: bool, compress: bool):
    """Perform the actual export operation"""
    try:
        # Create export directory
        os.makedirs(path, exist_ok=True)
        
        # Get current simulation state
        sim = st.session_state.multi_simulator
        state = sim.save_state()
        
        # Add metadata
        state['metadata'] = {
            'exported_at': datetime.now().isoformat(),
            'version': '1.0',
            'format': format_type,
            'include_analysis': include_analysis,
            'include_history': include_history
        }
        
        # Add analysis data if requested
        if include_analysis:
            state['analysis'] = sim.analyze_network()
        
        # Add history if requested
        if include_history and st.session_state.simulation_history:
            state['history'] = st.session_state.simulation_history[-10:]  # Last 10 frames
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}"
        
        # Export based on format
        if format_type == "JSON":
            filepath = os.path.join(path, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
        
        elif format_type == "Pickle":
            filepath = os.path.join(path, f"{filename}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
        
        elif format_type == "CSV Data":
            filepath = os.path.join(path, f"{filename}.csv")
            # Export chemical concentrations as CSV
            df_data = []
            for name, chemical in sim.chemicals.items():
                df_data.append({
                    'chemical': name,
                    'mean_concentration': np.mean(chemical.concentration),
                    'std_concentration': np.std(chemical.concentration),
                    'max_concentration': np.max(chemical.concentration),
                    'min_concentration': np.min(chemical.concentration)
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(filepath, index=False)
        
        # Compress if requested
        if compress and format_type in ["JSON", "Pickle"]:
            zip_filepath = os.path.join(path, f"{filename}.zip")
            with zipfile.ZipFile(zip_filepath, 'w') as zipf:
                zipf.write(filepath, os.path.basename(filepath))
            os.remove(filepath)  # Remove original file
            filepath = zip_filepath
        
        st.success(f"Export completed: {filepath}")
        
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def process_json_import(uploaded_file):
    """Process JSON import"""
    try:
        content = json.load(uploaded_file)
        
        # Validate and load state
        if 'chemicals' in content:
            st.session_state.multi_simulator.load_state(content)
            st.success("Simulation loaded successfully!")
        else:
            st.error("Invalid simulation file format")
            
    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")

def process_pickle_import(uploaded_file):
    """Process pickle import"""
    try:
        content = pickle.load(uploaded_file)
        
        # Validate and load state
        if hasattr(content, 'chemicals') or isinstance(content, dict):
            if isinstance(content, dict):
                st.session_state.multi_simulator.load_state(content)
            else:
                # Handle pickle object
                st.session_state.multi_simulator = content
            st.success("Simulation loaded successfully!")
        else:
            st.error("Invalid simulation file format")
            
    except Exception as e:
        st.error(f"Failed to load file: {str(e)}")

def process_zip_import(uploaded_file):
    """Process ZIP import"""
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zipf:
            # Find JSON or pickle file in zip
            json_files = [f for f in zipf.namelist() if f.endswith('.json')]
            pickle_files = [f for f in zipf.namelist() if f.endswith('.pkl')]
            
            if json_files:
                # Load JSON
                with zipf.open(json_files[0]) as f:
                    content = json.load(f)
                    st.session_state.multi_simulator.load_state(content)
                    st.success("Simulation loaded from ZIP!")
            elif pickle_files:
                # Load pickle
                with zipf.open(pickle_files[0]) as f:
                    content = pickle.load(f)
                    st.session_state.multi_simulator = content
                    st.success("Simulation loaded from ZIP!")
            else:
                st.error("No valid simulation file found in ZIP")
                
    except Exception as e:
        st.error(f"Failed to load ZIP file: {str(e)}")

def load_template(template: SimulationTemplate):
    """Load a template into the simulator"""
    try:
        # Create new simulator with template parameters
        sim = MultiChemicalSimulator(
            width=template.parameters.get('width', 200),
            height=template.parameters.get('height', 200),
            num_chemicals=len(template.chemicals)
        )
        
        # Set chemical concentrations
        for name, concentration in template.chemicals.items():
            if name in sim.chemicals:
                sim.chemicals[name].concentration.fill(concentration)
        
        # Set reactions
        sim.reactions = []
        for reaction_data in template.reactions:
            sim.add_reaction(
                reaction_data['reactants'],
                reaction_data['products'],
                reaction_data['rate_constant'],
                reaction_data.get('description', '')
            )
        
        # Update session state
        st.session_state.multi_simulator = sim
        st.session_state.current_template = template.name
        
        st.success(f"Template '{template.name}' loaded successfully!")
        
    except Exception as e:
        st.error(f"Failed to load template: {str(e)}")

def process_export_queue():
    """Process items in export queue"""
    if st.session_state.export_queue:
        # Process first item in queue
        export_item = st.session_state.export_queue.pop(0)
        
        # Perform export
        perform_export(**export_item)
        
        st.success(f"Processed export: {export_item.get('filename', 'Unknown')}")

# Main application header
st.markdown('<h1 class="main-header">Ultra-Advanced Reactive Diffusion Simulator 🔬</h1>', unsafe_allow_html=True)

# Create advanced tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Multi-Chemical", "ML Prediction", "Data Export", "Data Import", 
    "Templates", "Advanced Models", "Local Sharing"
])

with tab1:
    st.markdown("### Multi-Chemical Simulation")
    
    # Model selection
    model_name = st.selectbox("Select Chemical Model", list(ADVANCED_MODELS.keys()))
    model_info = ADVANCED_MODELS[model_name]
    
    st.info(model_info['description'])
    
    # Initialize simulator with selected model
    if st.button(f"Initialize {model_name} Model", width='stretch'):
        sim = MultiChemicalSimulator(num_chemicals=len(model_info['chemicals']))
        
        # Add chemicals
        for chem_name in model_info['chemicals']:
            if chem_name not in sim.chemicals:
                sim.add_chemical(chem_name)
        
        # Add reactions
        for reaction in model_info['reactions']:
            sim.add_reaction(
                reaction['reactants'],
                reaction['products'],
                reaction['rate']
            )
        
        st.session_state.multi_simulator = sim
        st.success(f"Initialized {model_name} model with {len(sim.chemicals)} chemicals")
    
    # Simulation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Run Step", width='stretch'):
            st.session_state.multi_simulator.step()
            st.rerun()
    
    with col2:
        if st.button("🔄 Run 100 Steps", width='stretch'):
            for _ in range(100):
                st.session_state.multi_simulator.step()
            st.rerun()
    
    with col3:
        if st.button("⟳ Reset", width='stretch'):
            st.session_state.multi_simulator = MultiChemicalSimulator(
                num_chemicals=len(model_info['chemicals'])
            )
            st.rerun()
    
    # Visualization
    create_multi_chemical_visualization()
    create_reaction_network_viz()

with tab2:
    st.markdown("### Machine Learning Pattern Prediction")
    
    # ML training interface
    create_ml_training_interface()
    
    # Pattern prediction
    if st.session_state.prediction_model.is_trained:
        st.markdown("#### Pattern Prediction")
        
        # Current pattern prediction
        prediction = st.session_state.prediction_model.predict_pattern(
            st.session_state.multi_simulator
        )
        
        st.markdown(f"""
        <div class="ml-panel">
            <h4>Predicted Pattern: {prediction['pattern']}</h4>
            <div class="confidence-meter">
                <div class="confidence-fill" style="width: {prediction['confidence']*100}%"></div>
            </div>
            <p>Confidence: {prediction['confidence']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # Data export interface
    create_data_export_interface()

with tab4:
    # Data import interface
    create_data_import_interface()

with tab5:
    # Template management interface
    create_template_management_interface()

with tab6:
    st.markdown("### Advanced Chemical Models")
    
    # Model comparison
    st.markdown("#### Model Comparison")
    
    comparison_data = []
    for name, model in ADVANCED_MODELS.items():
        comparison_data.append({
            'Model': name,
            'Chemicals': len(model['chemicals']),
            'Reactions': len(model['reactions']),
            'Complexity': 'High' if len(model['chemicals']) > 2 else 'Medium'
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, width='stretch')
    
    # Custom model creation
    st.markdown("#### Create Custom Model")
    
    with st.expander("Custom Reaction Network"):
        # Add chemicals
        st.markdown("##### Add Chemicals")
        num_chemicals = st.number_input("Number of Chemicals", 2, 10, 3)
        
        # Define reactions
        st.markdown("##### Define Reactions")
        num_reactions = st.number_input("Number of Reactions", 1, 20, 3)
        
        if st.button("Create Custom Model", width='stretch'):
            st.info("Custom model creation would be implemented here")

with tab7:
    st.markdown("### Local Sharing & Collaboration")
    
    # Local file sharing
    st.markdown("#### Local File Sharing")
    
    st.markdown("""
    <div class="upload-area">
        <h4>📁 Share Simulation Files</h4>
        <p>Drag and drop simulation files here or click to browse</p>
        <p>Supported formats: JSON, Pickle, ZIP</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Share options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export for Sharing", width='stretch'):
            # Create shareable package
            share_data = {
                'simulation': st.session_state.multi_simulator.save_state(),
                'metadata': {
                    'shared_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'description': 'Shared via local export'
                }
            }
            
            # Convert to JSON for sharing
            share_json = json.dumps(share_data, indent=2)
            
            # Provide download
            st.download_button(
                label="📥 Download Shareable File",
                data=share_json,
                file_name=f"shared_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📋 Generate Share Link", width='stretch'):
            st.info("Local sharing link would be generated here")
    
    # Recent shared files
    st.markdown("#### Recently Shared")
    
    # Mock shared files
    shared_files = [
        {"name": "Amazing Spots Pattern", "date": "2024-01-15", "size": "2.1 MB"},
        {"name": "Complex Maze Structure", "date": "2024-01-14", "size": "3.7 MB"},
        {"name": "Spiral Wave Simulation", "date": "2024-01-13", "size": "4.2 MB"}
    ]
    
    for file in shared_files:
        with st.expander(f"{file['name']} ({file['date']})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"Size: {file['size']}")
            
            with col2:
                st.write(f"Date: {file['date']}")
            
            with col3:
                if st.button("📥 Download", key=f"share_download_{file['name']}"):
                    st.success(f"Downloading {file['name']}...")
    
    # Collaboration features
    st.markdown("#### Local Collaboration")
    
    st.markdown("""
    <div class="advanced-card">
        <h4>🤝 Collaborative Features</h4>
        <p>Enable local collaboration by sharing simulation files through:</p>
        <ul>
            <li>Network drives</li>
            <li>USB storage devices</li>
            <li>Local file servers</li>
            <li>Email attachments</li>
        </ul>
        <p>All collaboration happens locally - no cloud services required!</p>
    </div>
    """, unsafe_allow_html=True)

# Advanced sidebar
st.sidebar.markdown("### Quick Actions")

# Quick export
if st.sidebar.button("📤 Quick Export", width='stretch'):
    perform_export("JSON", "./exports/", "quick_export", True, False, True)

# Quick template save
if st.sidebar.button("💾 Save as Template", width='stretch'):
    st.info("Template saving interface opened in Templates tab")

# Recent files
st.sidebar.markdown("#### Recent Files")
recent_files = ["simulation_001.json", "template_spots.json", "export_20240115.zip"]
for file in recent_files:
    if st.sidebar.button(f"📄 {file}", key=f"recent_{file}"):
        st.info(f"Loading {file}...")

# Footer
st.markdown("""
<div class="footer">
    <p>Ultra-Advanced Reactive Diffusion Simulator (Local Version)</p>
    <p>Featuring comprehensive data management, templates, and local sharing capabilities</p>
    <p>🔒 All data stays local - no cloud services required</p>
</div>
""", unsafe_allow_html=True)