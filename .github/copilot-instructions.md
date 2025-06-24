<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Chaotic Systems Visualization Project

This project focuses on implementing and visualizing chaotic dynamical systems with emphasis on:

## Code Style Guidelines
- Use NumPy for numerical computations
- Prefer scipy.integrate for ODE solving
- Use type hints for all function parameters and returns
- Follow PEP 8 styling conventions
- Document classes and functions with comprehensive docstrings

## Mathematical Implementation
- Implement ODE systems as classes inheriting from BaseChaoticSystem
- Use consistent parameter naming (sigma, rho, beta for Lorenz)
- Include mathematical documentation in docstrings with LaTeX notation
- Validate numerical stability and integration accuracy

## Visualization Standards
- Use Plotly for interactive 3D visualizations
- Implement consistent color schemes across all plots
- Include parameter controls for real-time exploration
- Provide both static matplotlib and interactive Plotly options

## Project Structure
- Keep chaotic systems in separate modules under src/chaotic_systems/
- Centralize visualization utilities in src/visualization/
- Use Jupyter notebooks for exploratory analysis and demonstrations
- Maintain clean separation between computation and visualization logic
