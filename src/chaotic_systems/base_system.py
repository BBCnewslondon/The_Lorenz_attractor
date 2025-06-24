"""
Base class for chaotic dynamical systems.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.integrate import solve_ivp


class BaseChaoticSystem(ABC):
    """Abstract base class for chaotic dynamical systems."""
    
    def __init__(self, parameters: Dict[str, float]):
        """
        Initialize the chaotic system with parameters.
        
        Args:
            parameters: Dictionary of system parameters
        """
        self.parameters = parameters
        self.default_initial_conditions = self.get_default_initial_conditions()
    
    @abstractmethod
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Define the system of differential equations.
        
        Args:
            t: Time variable
            state: Current state vector
            
        Returns:
            Derivative of the state vector
        """
        pass
    
    @abstractmethod
    def get_default_initial_conditions(self) -> np.ndarray:
        """Return default initial conditions for the system."""
        pass
    
    @abstractmethod
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return valid parameter ranges for the system."""
        pass
    
    def solve(self, 
              t_span: Tuple[float, float],
              initial_conditions: Optional[np.ndarray] = None,
              t_eval: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Solve the differential equation system.
        
        Args:
            t_span: Time span (start, end)
            initial_conditions: Initial state vector
            t_eval: Time points to evaluate solution
            **kwargs: Additional arguments for solve_ivp
            
        Returns:
            Dictionary containing solution data
        """
        if initial_conditions is None:
            initial_conditions = self.default_initial_conditions
            
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 10000)
        
        # Default solver options
        solver_options = {
            'method': 'RK45',
            'rtol': 1e-8,
            'atol': 1e-11
        }
        solver_options.update(kwargs)
        
        solution = solve_ivp(
            self.equations,
            t_span,
            initial_conditions,
            t_eval=t_eval,
            **solver_options
        )
        
        return {
            't': solution.t,
            'y': solution.y,
            'success': solution.success,
            'message': solution.message,
            'parameters': self.parameters.copy(),
            'initial_conditions': initial_conditions.copy()
        }
    
    def compute_divergence(self,
                          initial_conditions_1: np.ndarray,
                          initial_conditions_2: np.ndarray,
                          t_span: Tuple[float, float],
                          t_eval: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute trajectory divergence between two nearby initial conditions.
        
        Args:
            initial_conditions_1: First initial condition
            initial_conditions_2: Second initial condition
            t_span: Time span for integration
            t_eval: Time points to evaluate
            
        Returns:
            Dictionary with divergence analysis
        """
        sol1 = self.solve(t_span, initial_conditions_1, t_eval)
        sol2 = self.solve(t_span, initial_conditions_2, t_eval)
        
        if not (sol1['success'] and sol2['success']):
            raise RuntimeError("Failed to solve ODE system for divergence analysis")
        
        # Calculate distance between trajectories
        distance = np.linalg.norm(sol1['y'] - sol2['y'], axis=0)
        initial_distance = np.linalg.norm(initial_conditions_1 - initial_conditions_2)
        
        # Estimate Lyapunov exponent
        log_distance = np.log(distance / initial_distance)
        
        return {
            't': sol1['t'],
            'trajectory_1': sol1['y'],
            'trajectory_2': sol2['y'], 
            'distance': distance,
            'log_distance': log_distance,
            'initial_distance': initial_distance,
            'lyapunov_estimate': np.polyfit(sol1['t'][distance > 0], 
                                          log_distance[distance > 0], 1)[0]
        }
