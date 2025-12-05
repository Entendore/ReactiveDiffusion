import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random

# --- Collection folder ---
output_dir = "reaction_diffusion_collection_auto"
os.makedirs(output_dir, exist_ok=True)

# --- Grid size ---
size = 200

# --- Laplacian function ---
def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, (0, -1), (0, 1))
        + np.roll(Z, (0, 1), (0, 1))
        + np.roll(Z, (-1, 0), (0, 1))
        + np.roll(Z, (1, 0), (0, 1))
    )

# --- Simulation step ---
def update(U, V, Du, Dv, F, k):
    Lu = laplacian(U)
    Lv = laplacian(V)
    UVV = U * V * V
    U += Du * Lu - UVV + F * (1 - U)
    V += Dv * Lv + UVV - (F + k) * V
    return U, V

# --- Seed patterns ---
def seed_pattern(U, V, pattern="square"):
    r = random.randint(10, 30)
    if pattern == "square":
        U[size//2-r:size//2+r, size//2-r:size//2+r] = 0.50
        V[size//2-r:size//2+r, size//2-r:size//2+r] = 0.25
    elif pattern == "circle":
        y, x = np.ogrid[:size, :size]
        mask = (x - size//2)**2 + (y - size//2)**2 <= r**2
        U[mask] = 0.50
        V[mask] = 0.25
    elif pattern == "cross":
        U[size//2-2:size//2+2, :] = 0.50
        V[size//2-2:size//2+2, :] = 0.25
        U[:, size//2-2:size//2+2] = 0.50
        V[:, size//2-2:size//2+2] = 0.25
    return U, V

# --- Random parameter generator ---
def random_config():
    Du = round(random.uniform(0.1, 0.2), 3)
    Dv = round(random.uniform(0.05, 0.1), 3)
    F = round(random.uniform(0.02, 0.07), 3)
    k = round(random.uniform(0.045, 0.07), 3)
    seed = random.choice(["square", "circle", "cross"])
    return {"Du": Du, "Dv": Dv, "F": F, "k": k, "seed": seed}

# --- Number of patterns ---
num_patterns = 100
metadata = []

for idx in range(num_patterns):
    cfg = random_config()
    
    U = np.ones((size, size))
    V = np.zeros((size, size))
    U, V = seed_pattern(U, V, cfg["seed"])
    
    # Run simulation
    for _ in range(5000):
        U, V = update(U, V, cfg["Du"], cfg["Dv"], cfg["F"], cfg["k"])
    
    # Save result
    filename = f"pattern_{idx+1}_{cfg['seed']}.png"
    plt.imshow(U, cmap='inferno', interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save metadata
    cfg["filename"] = filename
    metadata.append(cfg)

# Save metadata JSON
with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Generated {num_patterns} patterns in '{output_dir}'")
