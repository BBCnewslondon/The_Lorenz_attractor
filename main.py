#!/usr/bin/env python3
"""
Command-line interface for chaotic systems visualization.
"""
import argparse
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chaotic_systems import LorenzAttractor, DoublePendulum
from visualization import Interactive3DVisualizer, TrajectoryAnalyzer


def run_lorenz(args):
    """Run Lorenz attractor simulation."""
    print("Running Lorenz attractor simulation...")
    
    # Create system
    lorenz = LorenzAttractor(sigma=args.sigma, rho=args.rho, beta=args.beta)
    
    # Set up initial conditions
    if args.divergence:
        ic1 = np.array([1.0, 1.0, 1.0])
        ic2 = ic1 + args.perturbation * np.random.randn(3)
        
        # Compute divergence
        divergence_data = lorenz.compute_divergence(
            ic1, ic2, (0, args.time_span), 
            np.linspace(0, args.time_span, args.num_points)
        )
        
        # Visualize divergence
        visualizer = Interactive3DVisualizer("Lorenz Attractor - Trajectory Divergence")
        fig = visualizer.plot_divergence_analysis(divergence_data, ('X', 'Y', 'Z'))
        fig.show()
        
        # Analyze Lyapunov exponent
        analyzer = TrajectoryAnalyzer()
        lyap_result = analyzer.compute_lyapunov_exponent(divergence_data)
        print(f"Estimated Lyapunov exponent: {lyap_result['lyapunov']:.4f}")
        print(f"R-squared: {lyap_result['r_squared']:.4f}")
        
    else:
        # Single trajectory
        solution = lorenz.solve(
            (0, args.time_span),
            np.array([1.0, 1.0, 1.0]),
            np.linspace(0, args.time_span, args.num_points)
        )
        
        # Visualize
        visualizer = Interactive3DVisualizer("Lorenz Attractor")
        fig = visualizer.plot_trajectory_3d(solution, ('X', 'Y', 'Z'))
        fig.show()
        
        # Phase space
        fig_phase = visualizer.plot_phase_space_2d(solution, (0, 1), ('X', 'Y'))
        fig_phase.show()


def run_double_pendulum(args):
    """Run double pendulum simulation.""" 
    print("Running double pendulum simulation...")
    
    # Create system
    pendulum = DoublePendulum(
        m1=args.m1, m2=args.m2, 
        L1=args.L1, L2=args.L2,
        g=args.g, damping=args.damping
    )
    
    # Initial conditions
    ic = np.array([args.theta1, args.theta2, args.omega1, args.omega2])
    
    if args.divergence:
        ic2 = ic + args.perturbation * np.random.randn(4)
        
        # Compute divergence
        divergence_data = pendulum.compute_divergence(
            ic, ic2, (0, args.time_span),
            np.linspace(0, args.time_span, args.num_points)
        )
        
        # Visualize (using first 3 dimensions)
        visualizer = Interactive3DVisualizer("Double Pendulum - Trajectory Divergence")
        
        # Create modified divergence data for 3D visualization
        div_data_3d = divergence_data.copy()
        div_data_3d['trajectory_1'] = div_data_3d['trajectory_1'][:3]
        div_data_3d['trajectory_2'] = div_data_3d['trajectory_2'][:3]
        
        fig = visualizer.plot_divergence_analysis(div_data_3d, ('θ₁', 'θ₂', 'ω₁'))
        fig.show()
        
    else:
        # Single trajectory
        solution = pendulum.solve(
            (0, args.time_span), ic,
            np.linspace(0, args.time_span, args.num_points)
        )
        
        # Visualize in angle space
        visualizer = Interactive3DVisualizer("Double Pendulum")
        fig = visualizer.plot_trajectory_3d(solution, ('θ₁', 'θ₂', 'ω₁'))
        fig.show()
        
        # Phase space
        fig_phase = visualizer.plot_phase_space_2d(solution, (0, 2), ('θ₁', 'ω₁'))
        fig_phase.show()
        
        # Energy analysis
        energy = pendulum.compute_energy(solution)
        
        import plotly.graph_objects as go
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(
            x=energy['time'], y=energy['kinetic'],
            mode='lines', name='Kinetic Energy'
        ))
        fig_energy.add_trace(go.Scatter(
            x=energy['time'], y=energy['potential'],
            mode='lines', name='Potential Energy'
        ))
        fig_energy.add_trace(go.Scatter(
            x=energy['time'], y=energy['total'],
            mode='lines', name='Total Energy'
        ))
        fig_energy.update_layout(
            title="Double Pendulum Energy",
            xaxis_title="Time",
            yaxis_title="Energy"
        )
        fig_energy.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Chaotic Systems Visualization")
    parser.add_argument('--system', choices=['lorenz', 'double_pendulum'], 
                       required=True, help='Chaotic system to simulate')
    parser.add_argument('--time-span', type=float, default=50.0,
                       help='Time span for simulation')
    parser.add_argument('--num-points', type=int, default=10000,
                       help='Number of time points')
    parser.add_argument('--divergence', action='store_true',
                       help='Analyze trajectory divergence')
    parser.add_argument('--perturbation', type=float, default=1e-8,
                       help='Perturbation size for divergence analysis')
    
    # Lorenz parameters
    parser.add_argument('--sigma', type=float, default=10.0,
                       help='Lorenz sigma parameter')
    parser.add_argument('--rho', type=float, default=28.0,
                       help='Lorenz rho parameter')
    parser.add_argument('--beta', type=float, default=8.0/3.0,
                       help='Lorenz beta parameter')
    
    # Double pendulum parameters
    parser.add_argument('--m1', type=float, default=1.0,
                       help='Mass of first pendulum')
    parser.add_argument('--m2', type=float, default=1.0,
                       help='Mass of second pendulum')
    parser.add_argument('--L1', type=float, default=1.0,
                       help='Length of first pendulum')
    parser.add_argument('--L2', type=float, default=1.0,
                       help='Length of second pendulum')
    parser.add_argument('--g', type=float, default=9.81,
                       help='Gravitational acceleration')
    parser.add_argument('--damping', type=float, default=0.0,
                       help='Damping coefficient')
    parser.add_argument('--theta1', type=float, default=np.pi/2,
                       help='Initial angle of first pendulum')
    parser.add_argument('--theta2', type=float, default=np.pi/2,
                       help='Initial angle of second pendulum')
    parser.add_argument('--omega1', type=float, default=0.0,
                       help='Initial angular velocity of first pendulum')
    parser.add_argument('--omega2', type=float, default=0.0,
                       help='Initial angular velocity of second pendulum')
    
    args = parser.parse_args()
    
    try:
        if args.system == 'lorenz':
            run_lorenz(args)
        elif args.system == 'double_pendulum':
            run_double_pendulum(args)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
