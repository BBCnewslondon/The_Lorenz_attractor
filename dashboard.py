#!/usr/bin/env python3
"""
Interactive web dashboard for exploring chaotic systems.
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chaotic_systems import LorenzAttractor, DoublePendulum
from visualization import Interactive3DVisualizer, TrajectoryAnalyzer

# Initialize Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Chaotic Systems Explorer", style={'textAlign': 'center'}),
    
    # System selection
    html.Div([
        html.Label("Select Chaotic System:"),
        dcc.Dropdown(
            id='system-dropdown',
            options=[
                {'label': 'Lorenz Attractor', 'value': 'lorenz'},
                {'label': 'Double Pendulum', 'value': 'double_pendulum'}
            ],
            value='lorenz'
        )
    ], style={'width': '30%', 'margin': '20px'}),
    
    # Parameter controls
    html.Div(id='parameter-controls'),
    
    # Simulation controls
    html.Div([
        html.Label("Time Span:"),
        dcc.Slider(
            id='time-span-slider',
            min=10, max=100, value=50, step=10,
            marks={i: str(i) for i in range(10, 101, 20)}
        ),
        html.Br(),
        html.Label("Number of Points:"),
        dcc.Slider(
            id='num-points-slider',
            min=1000, max=20000, value=10000, step=1000,
            marks={i: str(i) for i in range(1000, 20001, 5000)}
        ),
        html.Br(),
        html.Div([
            html.Button('Generate Trajectory', id='generate-btn', n_clicks=0),
            html.Button('Analyze Divergence', id='divergence-btn', n_clicks=0)
        ], style={'margin': '20px'})
    ], style={'margin': '20px'}),
    
    # Plots
    html.Div([
        dcc.Graph(id='main-plot'),
        dcc.Graph(id='phase-plot'),
        dcc.Graph(id='analysis-plot')
    ])
])


@app.callback(
    Output('parameter-controls', 'children'),
    Input('system-dropdown', 'value')
)
def update_parameter_controls(system):
    """Update parameter controls based on selected system."""
    if system == 'lorenz':
        return html.Div([
            html.H3("Lorenz Parameters"),
            html.Label("σ (sigma):"),
            dcc.Slider(id='sigma-slider', min=1, max=20, value=10, step=0.1,
                      marks={i: str(i) for i in range(1, 21, 5)}),
            html.Label("ρ (rho):"),
            dcc.Slider(id='rho-slider', min=10, max=50, value=28, step=0.1,
                      marks={i: str(i) for i in range(10, 51, 10)}),
            html.Label("β (beta):"),
            dcc.Slider(id='beta-slider', min=1, max=5, value=8/3, step=0.1,
                      marks={i: str(round(i, 1)) for i in np.arange(1, 5.1, 1)})
        ])
    else:  # double_pendulum
        return html.Div([
            html.H3("Double Pendulum Parameters"),
            html.Label("Mass 1:"),
            dcc.Slider(id='m1-slider', min=0.5, max=3, value=1, step=0.1,
                      marks={i: str(i) for i in [0.5, 1, 1.5, 2, 2.5, 3]}),
            html.Label("Mass 2:"),
            dcc.Slider(id='m2-slider', min=0.5, max=3, value=1, step=0.1,
                      marks={i: str(i) for i in [0.5, 1, 1.5, 2, 2.5, 3]}),
            html.Label("Length 1:"),
            dcc.Slider(id='L1-slider', min=0.5, max=2, value=1, step=0.1,
                      marks={i: str(i) for i in [0.5, 1, 1.5, 2]}),
            html.Label("Length 2:"),
            dcc.Slider(id='L2-slider', min=0.5, max=2, value=1, step=0.1,
                      marks={i: str(i) for i in [0.5, 1, 1.5, 2]}),
            html.Label("Initial θ₁ (radians):"),
            dcc.Slider(id='theta1-slider', min=0, max=2*np.pi, value=np.pi/2, step=0.1,
                      marks={i: f"{i:.1f}" for i in np.arange(0, 2*np.pi+0.1, np.pi/2)}),
            html.Label("Initial θ₂ (radians):"),
            dcc.Slider(id='theta2-slider', min=0, max=2*np.pi, value=np.pi/2, step=0.1,
                      marks={i: f"{i:.1f}" for i in np.arange(0, 2*np.pi+0.1, np.pi/2)})
        ])


@app.callback(
    [Output('main-plot', 'figure'),
     Output('phase-plot', 'figure'),
     Output('analysis-plot', 'figure')],
    [Input('generate-btn', 'n_clicks'),
     Input('divergence-btn', 'n_clicks')],
    [State('system-dropdown', 'value'),
     State('time-span-slider', 'value'),
     State('num-points-slider', 'value')] +
    [State(f'{param}-slider', 'value') for param in 
     ['sigma', 'rho', 'beta', 'm1', 'm2', 'L1', 'L2', 'theta1', 'theta2']]
)
def update_plots(generate_clicks, divergence_clicks, system, time_span, num_points,
                sigma, rho, beta, m1, m2, L1, L2, theta1, theta2):
    """Update plots based on button clicks and parameters."""
    
    # Determine which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, {}, {}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Set up time array
    t_eval = np.linspace(0, time_span, num_points)
    
    if system == 'lorenz':
        # Create Lorenz system
        lorenz = LorenzAttractor(sigma=sigma, rho=rho, beta=beta)
        
        if button_id == 'divergence-btn':
            # Divergence analysis
            ic1 = np.array([1.0, 1.0, 1.0])
            ic2 = ic1 + 1e-8 * np.random.randn(3)
            
            divergence_data = lorenz.compute_divergence(ic1, ic2, (0, time_span), t_eval)
            
            visualizer = Interactive3DVisualizer()
            main_fig = visualizer.plot_divergence_analysis(divergence_data, ('X', 'Y', 'Z'))
            
            # Phase space plot
            phase_fig = visualizer.plot_phase_space_2d(
                {'y': divergence_data['trajectory_1'], 't': divergence_data['t']}, 
                (0, 1), ('X', 'Y')
            )
            
            # Analysis plot - Lyapunov exponent
            analyzer = TrajectoryAnalyzer()
            lyap_result = analyzer.compute_lyapunov_exponent(divergence_data)
            
            analysis_fig = go.Figure()
            valid_mask = divergence_data['distance'] > 0
            analysis_fig.add_trace(go.Scatter(
                x=divergence_data['t'][valid_mask],
                y=divergence_data['log_distance'][valid_mask],
                mode='lines',
                name='Log Distance'
            ))
            analysis_fig.update_layout(
                title=f"Lyapunov Exponent ≈ {lyap_result['lyapunov']:.4f}",
                xaxis_title="Time",
                yaxis_title="Log Distance"
            )
            
        else:
            # Single trajectory
            solution = lorenz.solve((0, time_span), np.array([1.0, 1.0, 1.0]), t_eval)
            
            visualizer = Interactive3DVisualizer()
            main_fig = visualizer.plot_trajectory_3d(solution, ('X', 'Y', 'Z'))
            phase_fig = visualizer.plot_phase_space_2d(solution, (0, 1), ('X', 'Y'))
            
            # Time series plot
            analysis_fig = visualizer.plot_time_series(solution)
    
    else:  # double_pendulum
        # Create double pendulum
        pendulum = DoublePendulum(m1=m1, m2=m2, L1=L1, L2=L2)
        ic = np.array([theta1, theta2, 0.0, 0.0])
        
        if button_id == 'divergence-btn':
            # Divergence analysis
            ic2 = ic + 1e-6 * np.random.randn(4)
            
            divergence_data = pendulum.compute_divergence(ic, ic2, (0, time_span), t_eval)
            
            # Use first 3 dimensions for 3D plot
            div_data_3d = {
                'trajectory_1': divergence_data['trajectory_1'][:3],
                'trajectory_2': divergence_data['trajectory_2'][:3],
                't': divergence_data['t'],
                'distance': divergence_data['distance'],
                'log_distance': divergence_data['log_distance'],
                'lyapunov_estimate': divergence_data['lyapunov_estimate']
            }
            
            visualizer = Interactive3DVisualizer()
            main_fig = visualizer.plot_divergence_analysis(div_data_3d, ('θ₁', 'θ₂', 'ω₁'))
            
            # Phase space
            phase_fig = visualizer.plot_phase_space_2d(
                {'y': divergence_data['trajectory_1'], 't': divergence_data['t']},
                (0, 2), ('θ₁', 'ω₁')
            )
            
            # Lyapunov plot
            analyzer = TrajectoryAnalyzer()
            lyap_result = analyzer.compute_lyapunov_exponent(divergence_data)
            
            analysis_fig = go.Figure()
            valid_mask = divergence_data['distance'] > 0
            analysis_fig.add_trace(go.Scatter(
                x=divergence_data['t'][valid_mask],
                y=divergence_data['log_distance'][valid_mask],
                mode='lines',
                name='Log Distance'
            ))
            analysis_fig.update_layout(
                title=f"Lyapunov Exponent ≈ {lyap_result['lyapunov']:.4f}",
                xaxis_title="Time",
                yaxis_title="Log Distance"
            )
            
        else:
            # Single trajectory
            solution = pendulum.solve((0, time_span), ic, t_eval)
            
            visualizer = Interactive3DVisualizer()
            main_fig = visualizer.plot_trajectory_3d(solution, ('θ₁', 'θ₂', 'ω₁'))
            phase_fig = visualizer.plot_phase_space_2d(solution, (0, 2), ('θ₁', 'ω₁'))
            
            # Energy plot
            energy = pendulum.compute_energy(solution)
            analysis_fig = go.Figure()
            analysis_fig.add_trace(go.Scatter(
                x=energy['time'], y=energy['kinetic'],
                mode='lines', name='Kinetic Energy'
            ))
            analysis_fig.add_trace(go.Scatter(
                x=energy['time'], y=energy['potential'],
                mode='lines', name='Potential Energy'
            ))
            analysis_fig.add_trace(go.Scatter(
                x=energy['time'], y=energy['total'],
                mode='lines', name='Total Energy'
            ))
            analysis_fig.update_layout(
                title="Energy Analysis",
                xaxis_title="Time",
                yaxis_title="Energy"
            )
    
    return main_fig, phase_fig, analysis_fig


if __name__ == '__main__':
    print("Starting Chaotic Systems Dashboard...")
    print("Open your browser to http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)
