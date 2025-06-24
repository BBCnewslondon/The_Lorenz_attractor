"""
Double pendulum chaotic system implementation.

The double pendulum consists of two pendulums attached end to end, and is 
one of the simplest dynamical systems that exhibits chaotic behavior.

The system has four state variables:
- θ₁: angle of first pendulum
- θ₂: angle of second pendulum  
- ω₁: angular velocity of first pendulum
- ω₂: angular velocity of second pendulum
"""
import numpy as np
from typing import Dict, Tuple
from .base_system import BaseChaoticSystem


class DoublePendulum(BaseChaoticSystem):
    """
    Implementation of the double pendulum system.
    
    The equations of motion are derived from Lagrangian mechanics and
    result in a complex system of coupled nonlinear ODEs.
    """
    
    def __init__(self, 
                 m1: float = 1.0, 
                 m2: float = 1.0,
                 L1: float = 1.0, 
                 L2: float = 1.0,
                 g: float = 9.81,
                 damping: float = 0.0):
        """
        Initialize the double pendulum system.
        
        Args:
            m1: Mass of first pendulum bob
            m2: Mass of second pendulum bob
            L1: Length of first pendulum
            L2: Length of second pendulum
            g: Gravitational acceleration
            damping: Damping coefficient
        """
        parameters = {
            'm1': m1,
            'm2': m2,
            'L1': L1,
            'L2': L2,
            'g': g,
            'damping': damping
        }
        super().__init__(parameters)
    
    def equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Double pendulum equations of motion.
        
        Args:
            t: Time (not used in autonomous system)
            state: [θ₁, θ₂, ω₁, ω₂] state vector
            
        Returns:
            [dθ₁/dt, dθ₂/dt, dω₁/dt, dω₂/dt] derivative vector
        """
        theta1, theta2, omega1, omega2 = state
        
        m1 = self.parameters['m1']
        m2 = self.parameters['m2']
        L1 = self.parameters['L1']
        L2 = self.parameters['L2']
        g = self.parameters['g']
        damping = self.parameters['damping']
        
        # Differences in angles
        delta_theta = theta2 - theta1
        cos_delta = np.cos(delta_theta)
        sin_delta = np.sin(delta_theta)
        
        # Denominators for the equations
        denominator1 = (m1 + m2) * L1 - m2 * L1 * cos_delta * cos_delta
        denominator2 = (L2 / L1) * denominator1
        
        # Angular accelerations
        numerator1 = (-m2 * L1 * omega1**2 * sin_delta * cos_delta +
                     m2 * g * np.sin(theta2) * cos_delta +
                     m2 * L2 * omega2**2 * sin_delta -
                     (m1 + m2) * g * np.sin(theta1) -
                     damping * omega1)
        
        numerator2 = (-m2 * L2 * omega2**2 * sin_delta * cos_delta +
                     (m1 + m2) * g * np.sin(theta1) * cos_delta +
                     (m1 + m2) * L1 * omega1**2 * sin_delta -
                     (m1 + m2) * g * np.sin(theta2) -
                     damping * omega2)
        
        domega1_dt = numerator1 / denominator1
        domega2_dt = numerator2 / denominator2
        
        return np.array([omega1, omega2, domega1_dt, domega2_dt])
    
    def get_default_initial_conditions(self) -> np.ndarray:
        """Default initial conditions with small perturbation from vertical."""
        return np.array([np.pi/2, np.pi/2, 0.0, 0.0])  # θ₁, θ₂, ω₁, ω₂
    
    def get_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Valid parameter ranges for the double pendulum."""
        return {
            'm1': (0.1, 10.0),
            'm2': (0.1, 10.0),
            'L1': (0.1, 5.0),
            'L2': (0.1, 5.0),
            'g': (1.0, 20.0),
            'damping': (0.0, 1.0)
        }
    
    def get_cartesian_positions(self, solution: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert angular positions to Cartesian coordinates.
        
        Args:
            solution: Solution dictionary from solve()
            
        Returns:
            Tuple of (positions_1, positions_2) arrays with shape (2, N)
            where positions_i[0] = x and positions_i[1] = y
        """
        theta1 = solution['y'][0]
        theta2 = solution['y'][1]
        
        L1 = self.parameters['L1']
        L2 = self.parameters['L2']
        
        # Position of first pendulum bob
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        
        # Position of second pendulum bob
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)
        
        positions_1 = np.array([x1, y1])
        positions_2 = np.array([x2, y2])
        
        return positions_1, positions_2
    
    def compute_energy(self, solution: Dict) -> Dict[str, np.ndarray]:
        """
        Compute kinetic, potential, and total energy of the system.
        
        Args:
            solution: Solution dictionary from solve()
            
        Returns:
            Dictionary with energy components
        """
        theta1, theta2, omega1, omega2 = solution['y']
        
        m1 = self.parameters['m1']
        m2 = self.parameters['m2']
        L1 = self.parameters['L1']
        L2 = self.parameters['L2']
        g = self.parameters['g']
        
        # Kinetic energy
        T1 = 0.5 * m1 * (L1 * omega1)**2
        T2 = 0.5 * m2 * ((L1 * omega1)**2 + (L2 * omega2)**2 + 
                         2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2))
        kinetic_energy = T1 + T2
        
        # Potential energy (taking lowest point as zero)
        V1 = -m1 * g * L1 * np.cos(theta1)
        V2 = -m2 * g * (L1 * np.cos(theta1) + L2 * np.cos(theta2))
        potential_energy = V1 + V2
        
        total_energy = kinetic_energy + potential_energy
        
        return {
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': total_energy,
            'time': solution['t']
        }
    
    @classmethod
    def create_chaotic(cls) -> 'DoublePendulum':
        """Create double pendulum with parameters that exhibit chaotic behavior."""
        return cls(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81, damping=0.0)
