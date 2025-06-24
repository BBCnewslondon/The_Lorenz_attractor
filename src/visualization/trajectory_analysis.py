"""
Trajectory analysis utilities for chaotic systems.
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import linregress
import matplotlib.pyplot as plt


class TrajectoryAnalyzer:
    """Analyze properties of chaotic trajectories."""
    
    def __init__(self):
        """Initialize the trajectory analyzer."""
        pass
    
    def compute_lyapunov_exponent(self, 
                                divergence_data: Dict[str, Any],
                                fit_range: Tuple[float, float] = None) -> Dict[str, float]:
        """
        Estimate the largest Lyapunov exponent from trajectory divergence.
        
        Args:
            divergence_data: Output from compute_divergence()
            fit_range: Time range for linear fit (start, end)
            
        Returns:
            Dictionary with Lyapunov exponent estimates
        """
        t = divergence_data['t']
        distance = divergence_data['distance']
        log_distance = divergence_data['log_distance']
        
        # Filter valid data points
        valid_mask = (distance > 0) & np.isfinite(log_distance)
        
        if not np.any(valid_mask):
            return {'lyapunov': np.nan, 'r_squared': np.nan, 'fit_range': None}
        
        t_valid = t[valid_mask]
        log_dist_valid = log_distance[valid_mask]
        
        # Determine fit range
        if fit_range is None:
            # Use middle 50% of the data to avoid initial transients and saturation
            start_idx = len(t_valid) // 4
            end_idx = 3 * len(t_valid) // 4
            t_fit = t_valid[start_idx:end_idx]
            log_dist_fit = log_dist_valid[start_idx:end_idx]
        else:
            mask = (t_valid >= fit_range[0]) & (t_valid <= fit_range[1])
            t_fit = t_valid[mask]
            log_dist_fit = log_dist_valid[mask]
        
        if len(t_fit) < 2:
            return {'lyapunov': np.nan, 'r_squared': np.nan, 'fit_range': fit_range}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(t_fit, log_dist_fit)
        
        return {
            'lyapunov': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'fit_range': (t_fit[0], t_fit[-1]),
            'intercept': intercept
        }
    
    def analyze_periodicity(self, 
                          solution: Dict[str, Any],
                          variable_idx: int = 0) -> Dict[str, Any]:
        """
        Analyze periodicity in a time series using FFT.
        
        Args:
            solution: Solution dictionary
            variable_idx: Which variable to analyze
            
        Returns:
            Dictionary with frequency analysis
        """
        t = solution['t']
        y = solution['y'][variable_idx]
        
        # Ensure uniform time spacing
        dt = np.mean(np.diff(t))
        
        # Compute FFT
        fft_vals = np.fft.fft(y - np.mean(y))  # Remove DC component
        freqs = np.fft.fftfreq(len(y), dt)
        
        # Take only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power = np.abs(fft_vals[pos_mask])**2
        
        # Find dominant frequency
        dominant_idx = np.argmax(power)
        dominant_freq = freqs_pos[dominant_idx]
        dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf
        
        return {
            'frequencies': freqs_pos,
            'power_spectrum': power,
            'dominant_frequency': dominant_freq,
            'dominant_period': dominant_period,
            'time_spacing': dt
        }
    
    def compute_fractal_dimension(self, 
                                solution: Dict[str, Any],
                                max_embedding_dim: int = 10) -> Dict[str, Any]:
        """
        Estimate fractal dimension using box-counting method.
        
        Args:
            solution: Solution dictionary
            max_embedding_dim: Maximum embedding dimension to test
            
        Returns:
            Dictionary with fractal dimension estimate
        """
        # Use first 3 dimensions if available
        n_dims = min(3, solution['y'].shape[0])
        trajectory = solution['y'][:n_dims].T
        
        # Normalize trajectory to unit cube
        trajectory_norm = (trajectory - trajectory.min(axis=0)) / (trajectory.max(axis=0) - trajectory.min(axis=0))
        
        # Box sizes (powers of 2)
        box_sizes = 2.0**(-np.arange(2, 8))
        box_counts = []
        
        for box_size in box_sizes:
            # Count boxes that contain trajectory points
            boxes = np.floor(trajectory_norm / box_size).astype(int)
            unique_boxes = np.unique(boxes, axis=0)
            box_counts.append(len(unique_boxes))
        
        box_counts = np.array(box_counts)
        
        # Linear fit in log-log space
        valid_mask = box_counts > 0
        if np.sum(valid_mask) >= 2:
            log_sizes = np.log(box_sizes[valid_mask])
            log_counts = np.log(box_counts[valid_mask])
            slope, intercept = np.polyfit(log_sizes, log_counts, 1)
            fractal_dim = -slope
        else:
            fractal_dim = np.nan
        
        return {
            'box_sizes': box_sizes,
            'box_counts': box_counts,
            'fractal_dimension': fractal_dim,
            'r_squared': np.corrcoef(np.log(box_sizes[valid_mask]), 
                                   np.log(box_counts[valid_mask]))[0,1]**2 if np.sum(valid_mask) >= 2 else np.nan
        }
    
    def compute_poincare_section(self, 
                               solution: Dict[str, Any],
                               plane_normal: Tuple[float, float, float] = (0, 0, 1),
                               plane_point: Tuple[float, float, float] = (0, 0, 0)) -> Dict[str, Any]:
        """
        Compute Poincaré section of the trajectory.
        
        Args:
            solution: Solution dictionary
            plane_normal: Normal vector to the Poincaré plane
            plane_point: Point on the Poincaré plane
            
        Returns:
            Dictionary with Poincaré section data
        """
        if solution['y'].shape[0] < 3:
            raise ValueError("Need at least 3D trajectory for Poincaré section")
        
        trajectory = solution['y'][:3].T  # Shape: (n_points, 3)
        t = solution['t']
        
        normal = np.array(plane_normal)
        point = np.array(plane_point)
        
        # Compute signed distance from each point to the plane
        distances = np.dot(trajectory - point, normal)
        
        # Find crossings (sign changes)
        crossings = []
        crossing_times = []
        
        for i in range(len(distances) - 1):
            if distances[i] * distances[i+1] < 0:  # Sign change
                # Linear interpolation to find exact crossing
                alpha = -distances[i] / (distances[i+1] - distances[i])
                crossing_point = trajectory[i] + alpha * (trajectory[i+1] - trajectory[i])
                crossing_time = t[i] + alpha * (t[i+1] - t[i])
                
                crossings.append(crossing_point)
                crossing_times.append(crossing_time)
        
        crossings = np.array(crossings) if crossings else np.empty((0, 3))
        crossing_times = np.array(crossing_times) if crossing_times else np.empty(0)
        
        return {
            'crossings': crossings,
            'crossing_times': crossing_times,
            'plane_normal': normal,
            'plane_point': point
        }
    
    def estimate_correlation_dimension(self, 
                                     solution: Dict[str, Any],
                                     max_points: int = 1000) -> Dict[str, Any]:
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.
        
        Args:
            solution: Solution dictionary
            max_points: Maximum number of points to use (for computational efficiency)
            
        Returns:
            Dictionary with correlation dimension estimate
        """
        # Use first 3 dimensions if available
        n_dims = min(3, solution['y'].shape[0])
        trajectory = solution['y'][:n_dims].T
        
        # Subsample if necessary
        if len(trajectory) > max_points:
            indices = np.linspace(0, len(trajectory)-1, max_points, dtype=int)
            trajectory = trajectory[indices]
        
        n_points = len(trajectory)
        
        # Range of radius values
        distances = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.linalg.norm(trajectory[i] - trajectory[j])
                distances.append(dist)
        
        distances = np.array(distances)
        radii = np.logspace(np.log10(np.min(distances[distances > 0])), 
                           np.log10(np.max(distances)), 20)
        
        correlation_sums = []
        for r in radii:
            count = np.sum(distances <= r)
            correlation_sum = 2 * count / (n_points * (n_points - 1))
            correlation_sums.append(correlation_sum)
        
        correlation_sums = np.array(correlation_sums)
        
        # Estimate correlation dimension from slope
        valid_mask = correlation_sums > 0
        if np.sum(valid_mask) >= 2:
            log_radii = np.log(radii[valid_mask])
            log_corr = np.log(correlation_sums[valid_mask])
            slope, intercept = np.polyfit(log_radii, log_corr, 1)
            corr_dim = slope
        else:
            corr_dim = np.nan
        
        return {
            'radii': radii,
            'correlation_sums': correlation_sums,
            'correlation_dimension': corr_dim
        }
