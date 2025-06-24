"""
Interactive 3D visualization using Plotly.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class Interactive3DVisualizer:
    """Interactive 3D visualization for chaotic systems."""
    
    def __init__(self, title: str = "Chaotic System Visualization"):
        """
        Initialize the visualizer.
        
        Args:
            title: Title for the visualization
        """
        self.title = title
        self.color_palette = px.colors.qualitative.Set1
    
    def plot_trajectory_3d(self, 
                          solution: Dict[str, Any],
                          labels: Tuple[str, str, str] = ('X', 'Y', 'Z'),
                          color_by_time: bool = True,
                          show_initial_point: bool = True,
                          show_final_point: bool = True) -> go.Figure:
        """
        Create interactive 3D trajectory plot.
        
        Args:
            solution: Solution dictionary from chaotic system
            labels: Axis labels (x, y, z)
            color_by_time: Whether to color trajectory by time
            show_initial_point: Whether to highlight initial point
            show_final_point: Whether to highlight final point
            
        Returns:
            Plotly figure object
        """
        x, y, z = solution['y'][:3]  # Take first 3 dimensions
        t = solution['t']
        
        fig = go.Figure()
        
        # Main trajectory
        if color_by_time:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(
                    color=t,
                    colorscale='Viridis',
                    width=2,
                    colorbar=dict(title="Time")
                ),
                name="Trajectory",
                hovertemplate="<b>%{fullData.name}</b><br>" +
                            f"{labels[0]}: %{{x:.3f}}<br>" +
                            f"{labels[1]}: %{{y:.3f}}<br>" +
                            f"{labels[2]}: %{{z:.3f}}<br>" +
                            "Time: %{marker.color:.3f}<extra></extra>"
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='blue', width=2),
                name="Trajectory"
            ))
        
        # Initial point
        if show_initial_point:
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                marker=dict(size=8, color='green'),
                name="Initial Point"
            ))
        
        # Final point
        if show_final_point:
            fig.add_trace(go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]],
                mode='markers',
                marker=dict(size=8, color='red'),
                name="Final Point"
            ))
        
        fig.update_layout(
            title=self.title,
            scene=dict(
                xaxis_title=labels[0],
                yaxis_title=labels[1],
                zaxis_title=labels[2],
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            showlegend=True
        )
        
        return fig
    
    def plot_divergence_analysis(self, 
                               divergence_data: Dict[str, Any],
                               labels: Tuple[str, str, str] = ('X', 'Y', 'Z')) -> go.Figure:
        """
        Plot trajectory divergence analysis.
        
        Args:
            divergence_data: Divergence data from compute_divergence()
            labels: Axis labels
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D Trajectories', 'Distance vs Time', 
                          'Log Distance vs Time', 'Phase Space (X-Y)'),
            specs=[[{"type": "scatter3d"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        traj1 = divergence_data['trajectory_1']
        traj2 = divergence_data['trajectory_2']
        t = divergence_data['t']
        distance = divergence_data['distance']
        log_distance = divergence_data['log_distance']
        
        # 3D trajectories
        fig.add_trace(go.Scatter3d(
            x=traj1[0], y=traj1[1], z=traj1[2],
            mode='lines',
            line=dict(color='blue', width=3),
            name='Trajectory 1'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter3d(
            x=traj2[0], y=traj2[1], z=traj2[2],
            mode='lines',
            line=dict(color='red', width=3),
            name='Trajectory 2'
        ), row=1, col=1)
        
        # Distance vs time
        fig.add_trace(go.Scatter(
            x=t, y=distance,
            mode='lines',
            line=dict(color='purple'),
            name='Distance'
        ), row=1, col=2)
        
        # Log distance vs time  
        valid_idx = distance > 0
        if np.any(valid_idx):
            fig.add_trace(go.Scatter(
                x=t[valid_idx], y=log_distance[valid_idx],
                mode='lines',
                line=dict(color='orange'),
                name='Log Distance'
            ), row=2, col=1)
            
            # Linear fit for Lyapunov exponent
            lyapunov = divergence_data['lyapunov_estimate']
            fit_line = lyapunov * t[valid_idx] + log_distance[valid_idx][0]
            fig.add_trace(go.Scatter(
                x=t[valid_idx], y=fit_line,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'Linear Fit (λ ≈ {lyapunov:.3f})'
            ), row=2, col=1)
        
        # Phase space projection
        fig.add_trace(go.Scatter(
            x=traj1[0], y=traj1[1],
            mode='lines',
            line=dict(color='blue'),
            name='Trajectory 1 (X-Y)'
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=traj2[0], y=traj2[1],
            mode='lines',
            line=dict(color='red'),
            name='Trajectory 2 (X-Y)'
        ), row=2, col=2)
        
        fig.update_layout(
            title="Trajectory Divergence Analysis",
            height=800
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Distance", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Log Distance", row=2, col=1)
        fig.update_xaxes(title_text=labels[0], row=2, col=2)
        fig.update_yaxes(title_text=labels[1], row=2, col=2)
        
        return fig
    
    def plot_phase_space_2d(self, 
                           solution: Dict[str, Any],
                           dims: Tuple[int, int] = (0, 1),
                           labels: Optional[Tuple[str, str]] = None) -> go.Figure:
        """
        Create 2D phase space plot.
        
        Args:
            solution: Solution dictionary
            dims: Which dimensions to plot (indices)
            labels: Axis labels
            
        Returns:
            Plotly figure
        """
        if labels is None:
            labels = (f"Dimension {dims[0]}", f"Dimension {dims[1]}")
        
        x = solution['y'][dims[0]]
        y = solution['y'][dims[1]]
        t = solution['t']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(
                color=t,
                colorscale='Viridis',
                width=2,
                colorbar=dict(title="Time")
            ),
            name="Phase Space Trajectory"
        ))
        
        # Initial and final points
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[y[0]],
            mode='markers',
            marker=dict(size=10, color='green', symbol='circle'),
            name="Initial Point"
        ))
        
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name="Final Point"
        ))
        
        fig.update_layout(
            title="Phase Space Projection",
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            showlegend=True
        )
        
        return fig
    
    def plot_time_series(self, solution: Dict[str, Any]) -> go.Figure:
        """
        Plot time series of all variables.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Plotly figure with subplots
        """
        n_vars = solution['y'].shape[0]
        
        fig = make_subplots(
            rows=n_vars, cols=1,
            subplot_titles=[f'Variable {i+1}' for i in range(n_vars)],
            shared_xaxes=True
        )
        
        t = solution['t']
        colors = px.colors.qualitative.Set1
        
        for i in range(n_vars):
            fig.add_trace(go.Scatter(
                x=t, y=solution['y'][i],
                mode='lines',
                line=dict(color=colors[i % len(colors)]),
                name=f'Variable {i+1}'
            ), row=i+1, col=1)
        
        fig.update_layout(
            title="Time Series",
            height=200 * n_vars,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time", row=n_vars, col=1)
        
        return fig
