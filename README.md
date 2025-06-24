# Chaotic Systems Visualization

An interactive Python application for exploring chaotic dynamical systems, featuring the Lorenz attractor, double pendulum, and other fascinating nonlinear systems.

## Features

- **Lorenz Attractor**: Solve and visualize the famous butterfly attractor
- **Double Pendulum**: Simulate the chaotic motion of a double pendulum
- **Interactive 3D Visualization**: Explore attractors from different viewpoints
- **Trajectory Divergence**: Observe how initially close trajectories diverge exponentially
- **Parameter Exploration**: Modify system parameters to see different behaviors
- **Multiple Chaotic Systems**: Extensible framework for adding new systems

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface
```bash
# Run Lorenz attractor visualization
python main.py --system lorenz

# Run double pendulum simulation
python main.py --system double_pendulum

# Interactive web dashboard
python dashboard.py
```

### Jupyter Notebooks
Explore the interactive notebooks in the `notebooks/` directory:
- `lorenz_explorer.ipynb`: Interactive Lorenz attractor exploration
- `double_pendulum.ipynb`: Double pendulum analysis
- `chaos_comparison.ipynb`: Compare different chaotic systems

## Project Structure

```
The_Lorenz_attractor/
├── src/
│   ├── chaotic_systems/
│   │   ├── __init__.py
│   │   ├── lorenz.py           # Lorenz attractor implementation
│   │   ├── double_pendulum.py  # Double pendulum system
│   │   └── base_system.py      # Base class for chaotic systems
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── interactive_3d.py   # Interactive 3D plotting
│   │   ├── trajectory_analysis.py # Divergence analysis
│   │   └── plotly_viz.py       # Plotly-based visualizations
│   └── utils/
│       ├── __init__.py
│       ├── ode_solver.py       # ODE integration utilities
│       └── math_utils.py       # Mathematical utilities
├── notebooks/
│   ├── lorenz_explorer.ipynb
│   ├── double_pendulum.ipynb
│   └── chaos_comparison.ipynb
├── main.py                     # Command-line interface
├── dashboard.py               # Web dashboard
├── requirements.txt
└── README.md
```

## Mathematical Background

### Lorenz Attractor
The Lorenz system is defined by:
- dx/dt = σ(y - x)
- dy/dt = x(ρ - z) - y  
- dz/dt = xy - βz

### Double Pendulum
A system of two coupled pendulums exhibiting chaotic motion for certain initial conditions.

## License

MIT License
