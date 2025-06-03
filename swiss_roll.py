#%%
import numpy as np
import plotly.io as pio
pio.renderers.default = "notebook"  # or "iframe", "png", or "browser" if needed
import plotly.graph_objs as go
from sklearn.datasets import make_swiss_roll

# Generate Swiss roll dataset
n_samples = 5000
X, color = make_swiss_roll(n_samples, noise=0.2)

# Select a point to highlight
index = 3500  # You can choose any index
origin = X[index]
origin_color = color[index]

# Define a random direction vector
np.random.seed(42)  # For reproducibility
direction = np.random.randn(3)
direction = direction / np.linalg.norm(direction)  # Normalize
length = 5
arrow_tip = origin + length * direction

# Scatter plot for all points (except the selected one)
scatter_points = go.Scatter3d(
    x=X[:, 0],
    y=X[:, 1],
    z=X[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=color,
        colorscale='Viridis',
        opacity=0.7
    ),
    name='Swiss Roll'
)

# Red point to highlight
highlight = go.Scatter3d(
    x=[origin[0]],
    y=[origin[1]],
    z=[origin[2]],
    mode='markers',
    marker=dict(
        size=7,
        color='red'
    ),
    name='Selected Point'
)

# Step 1: Find neighbors within radius
radius = 2.0  # You can adjust this
distances = np.linalg.norm(X - origin, axis=1)
neighbor_mask = (distances < radius) & (distances > 0)  # exclude the origin itself
neighbors = X[neighbor_mask]

# Step 2: Plot neighbors in light red
neighbor_scatter = go.Scatter3d(
    x=neighbors[:, 0],
    y=neighbors[:, 1],
    z=neighbors[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color='rgba(255,0,0,0.1)'  # semi-transparent red
    ),
    name=f'Neighbors (r<{radius})'
)

# Arrow shaft (as a line)
arrow_line = go.Scatter3d(
    x=[origin[0], arrow_tip[0]],
    y=[origin[1], arrow_tip[1]],
    z=[origin[2], arrow_tip[2]],
    mode='lines',
    line=dict(color='red', width=5),
    name='Arrow'
)

# Step 3: Covariance
centered_neighbors = neighbors - neighbors.mean(axis=0)
cov_matrix = np.cov(centered_neighbors.T)

# Step 4: Transform the original direction vector using the covariance matrix
transformed_dir = cov_matrix @ direction
transformed_dir = transformed_dir / np.linalg.norm(transformed_dir)  # normalize

# Define a new arrow tip
length2 = 5
transformed_tip = origin + length2 * transformed_dir

# Step 5: Draw the transformed direction arrow
transformed_arrow = go.Scatter3d(
    x=[origin[0], transformed_tip[0]],
    y=[origin[1], transformed_tip[1]],
    z=[origin[2], transformed_tip[2]],
    mode='lines',
    line=dict(color='orange', width=5),
    name='Transformed Arrow'
)

# Combine all into one figure
fig = go.Figure(data=[scatter_points,
                      highlight,
                      neighbor_scatter,
                      arrow_line,
                      transformed_arrow])

# Layout
fig.update_layout(
    title="Swiss Roll with Highlighted Point and Direction Vector",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

fig.show()

# %%

