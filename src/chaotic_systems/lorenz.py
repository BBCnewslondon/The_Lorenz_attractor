"""
Lorenz attractor implementation.

The Lorenz system is a system of ordinary differential equations first studied by 
Edward Lorenz. It is notable for having chaotic solutions for certain parameter 
values and initial conditions.

Mathematical definition:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

Where σ, ρ, and β are system parameters.
"""
import numpy as np
from typing import Dict, Tuple
from .base_system import BaseChaoticSystem


class LorenzAttractor(BaseChaoticSystem):
    """
    Implementation of the Lorenz attractor system.
    
    The classic parameters that produce chaotic behavior are:
    σ = 10, ρ = 28, β = 8/3
    """
    
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0):
        """
        Initialize the Lorenz system.
        
        Args:
            sigma: Prandtl number (σ)
            rho: Rayleigh number (ρ) 
            beta: Geometric factor (β)
        """
        parameters = {
            'sigma': sigma,
            'rho': rho,
            'beta': beta
        }
        super().__init__(parameters)
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Lorenz equations.
        
        Args:
            t: Time (not used in autonomous system)
            state: [x, y, z] state vector
            
        Returns:
            [dx/dt, dy/dt, dz/dt] derivative vector
        """
        x, y, z = state
        sigma = self.parameters['sigma']
        rho = self.parameters['rho']
        beta = self.parameters['beta']
        
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def get_default_initial_conditions(self) -> np.ndarray:
        """Default initial conditions near the attractor."""
        return np.array([1.0, 1.0, 1.0])
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Valid parameter ranges for the Lorenz system."""
        return {
            'sigma': (0.1, 50.0),
            'rho': (0.1, 50.0), 
            'beta': (0.1, 10.0)
        }
    
    def get_equilibrium_points(self) -> Dict[str, np.ndarray]:
        """
        Calculate equilibrium points of the Lorenz system.
        
        Returns:
            Dictionary of equilibrium points
        """
        rho = self.parameters['rho']
        beta = self.parameters['beta']
        
        # Origin (always an equilibrium)
        origin = np.array([0.0, 0.0, 0.0])
        
        equilibria = {'origin': origin}
        
        # Two symmetric equilibria exist when ρ > 1
        if rho > 1:
            sqrt_beta_rho_minus_1 = np.sqrt(beta * (rho - 1))
            
            equilibria['positive'] = np.array([
                sqrt_beta_rho_minus_1,
                sqrt_beta_rho_minus_1,
                rho - 1
            ])
            
            equilibria['negative'] = np.array([
                -sqrt_beta_rho_minus_1,
                -sqrt_beta_rho_minus_1,
                rho - 1
            ])
        
        return equilibria
    
    def is_chaotic_parameters(self) -> bool:
        """Check if current parameters likely produce chaotic behavior."""
        sigma = self.parameters['sigma']
        rho = self.parameters['rho']
        beta = self.parameters['beta']
        
        # Rough criteria for chaos in Lorenz system
        return (sigma > 0 and rho > 24.74 and beta > 0)
    
    @classmethod
    def create_classic(cls) -> 'LorenzAttractor':
        """Create Lorenz system with classic chaotic parameters."""
        return cls(sigma=10.0, rho=28.0, beta=8.0/3.0)
    
    @classmethod
    def create_periodic(cls) -> 'LorenzAttractor':
        """Create Lorenz system with parameters that produce periodic behavior."""
        return cls(sigma=10.0, rho=24.0, beta=8.0/3.0)
