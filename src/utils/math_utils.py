"""
Mathematical utilities for chaotic systems.
"""
import numpy as np
from typing import Tuple, Optional


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """
    Normalize angles to [-π, π] range.
    
    Args:
        angles: Array of angles in radians
        
    Returns:
        Normalized angles
    """
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def runge_kutta_4(func, y0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    4th order Runge-Kutta integration.
    
    Args:
        func: Function defining dy/dt = func(t, y)
        y0: Initial conditions
        t: Time array
        
    Returns:
        Solution array with shape (len(y0), len(t))
    """
    n = len(t)
    y = np.zeros((len(y0), n))
    y[:, 0] = y0
    
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        k1 = func(t[i], y[:, i])
        k2 = func(t[i] + h/2, y[:, i] + h*k1/2)
        k3 = func(t[i] + h/2, y[:, i] + h*k2/2)
        k4 = func(t[i] + h, y[:, i] + h*k3)
        
        y[:, i + 1] = y[:, i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return y


def compute_jacobian(func, state: np.ndarray, t: float = 0.0, 
                    eps: float = 1e-8) -> np.ndarray:
    """
    Compute Jacobian matrix numerically.
    
    Args:
        func: Function defining the system
        state: State vector
        t: Time (for non-autonomous systems)
        eps: Step size for numerical differentiation
        
    Returns:
        Jacobian matrix
    """
    n = len(state)
    jacobian = np.zeros((n, n))
    f0 = func(t, state)
    
    for i in range(n):
        state_plus = state.copy()
        state_plus[i] += eps
        f_plus = func(t, state_plus)
        
        state_minus = state.copy()
        state_minus[i] -= eps
        f_minus = func(t, state_minus)
        
        jacobian[:, i] = (f_plus - f_minus) / (2 * eps)
    
    return jacobian


def find_fixed_points(func, search_bounds: Tuple[np.ndarray, np.ndarray],
                     tolerance: float = 1e-6, max_iterations: int = 1000) -> list:
    """
    Find fixed points of a dynamical system.
    
    Args:
        func: Function defining the system
        search_bounds: (lower_bounds, upper_bounds) for search
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations for Newton's method
        
    Returns:
        List of fixed points
    """
    from scipy.optimize import fsolve
    
    lower, upper = search_bounds
    n_dim = len(lower)
    
    # Generate initial guesses
    n_guesses = 10
    fixed_points = []
    
    for _ in range(n_guesses):
        # Random initial guess
        guess = lower + np.random.rand(n_dim) * (upper - lower)
        
        try:
            # Solve func(t, x) = 0
            solution = fsolve(lambda x: func(0, x), guess, xtol=tolerance)
            
            # Check if it's actually a fixed point
            if np.linalg.norm(func(0, solution)) < tolerance:
                # Check if we already found this fixed point
                is_new = True
                for existing_fp in fixed_points:
                    if np.linalg.norm(solution - existing_fp) < tolerance:
                        is_new = False
                        break
                
                if is_new:
                    fixed_points.append(solution)
        
        except:
            continue
    
    return fixed_points


def lyapunov_exponents(func, state: np.ndarray, t_span: Tuple[float, float],
                      n_steps: int = 1000) -> np.ndarray:
    """
    Compute Lyapunov exponents using the method of Benettin et al.
    
    Args:
        func: Function defining the system
        state: Initial state
        t_span: Time span for integration
        n_steps: Number of time steps
        
    Returns:
        Array of Lyapunov exponents
    """
    from scipy.integrate import solve_ivp
    
    n = len(state)
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    dt = t_eval[1] - t_eval[0]
    
    # Initialize orthonormal basis
    tangent_vectors = np.eye(n)
    lyap_sum = np.zeros(n)
    
    current_state = state.copy()
    
    for i in range(1, n_steps):
        t = t_eval[i-1]
        
        # Integrate main trajectory
        sol = solve_ivp(func, [t, t + dt], current_state, dense_output=True)
        current_state = sol.y[:, -1]
        
        # Integrate tangent vectors
        jacobian = compute_jacobian(func, current_state, t)
        tangent_vectors = tangent_vectors + dt * jacobian @ tangent_vectors
        
        # Gram-Schmidt orthonormalization
        norms = np.zeros(n)
        for j in range(n):
            # Orthogonalize against previous vectors
            for k in range(j):
                tangent_vectors[:, j] -= np.dot(tangent_vectors[:, j], 
                                               tangent_vectors[:, k]) * tangent_vectors[:, k]
            
            # Normalize
            norms[j] = np.linalg.norm(tangent_vectors[:, j])
            if norms[j] > 0:
                tangent_vectors[:, j] /= norms[j]
        
        # Accumulate Lyapunov exponents
        lyap_sum += np.log(norms)
    
    # Average over time
    lyapunov_exp = lyap_sum / (t_span[1] - t_span[0])
    
    return lyapunov_exp


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
    """
    Compute mutual information between two time series.
    
    Args:
        x: First time series
        y: Second time series  
        bins: Number of bins for histogram
        
    Returns:
        Mutual information value
    """
    # Create 2D histogram
    hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    hist_x, _ = np.histogram(x, bins=x_edges)
    hist_y, _ = np.histogram(y, bins=y_edges)
    
    # Normalize to get probabilities
    p_xy = hist_xy / np.sum(hist_xy)
    p_x = hist_x / np.sum(hist_x)
    p_y = hist_y / np.sum(hist_y)
    
    # Compute mutual information
    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi
