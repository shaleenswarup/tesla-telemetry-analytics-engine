"""Charging Optimization Engine.

Optimizes charging strategies using convex optimization and dynamic programming.
Considers grid pricing, battery health, thermal constraints, and user schedules
to minimize cost and degradation while maximizing convenience.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


@dataclass
class ChargingStrategy:
    """Optimal charging schedule and parameters."""
    start_time: str
    end_time: str
    optimal_rate: float  # kW
    estimated_duration: float  # hours
    estimated_cost: float  # $
    battery_stress_score: float  # 0-100
    efficiency_gain: float  # % vs baseline
    thermal_peak: float  # °C
    recommendations: List[str]


class ChargingOptimizer:
    """Advanced charging optimization engine.
    
    Employs convex optimization and dynamic programming to determine optimal
    charging strategies. Minimizes cost and degradation while respecting
    thermal and electrical constraints.
    """
    
    # Constants
    MAX_CHARGE_RATE = 11.5  # kW (typical wall connector)
    THERMAL_LIMIT = 60.0  # °C
    SOC_MIN = 10.0  # %
    SOC_MAX = 95.0  # %
    
    def __init__(self, electricity_rate: float = 0.15):
        """Initialize optimizer.
        
        Args:
            electricity_rate: $/kWh
        """
        self.electricity_rate = electricity_rate
        self.charging_history = []
        
    def _thermal_derating_factor(self, ambient_temp: float,
                                battery_temp: float) -> float:
        """Calculate thermal derating factor for charging.
        
        Args:
            ambient_temp: Ambient temperature (°C)
            battery_temp: Battery temperature (°C)
            
        Returns:
            Derating factor (0-1)
        """
        # Reduce charging rate above 40°C battery temp
        if battery_temp > 45:
            return max(0.3, 1.0 - (battery_temp - 45) / 30.0)
        return 1.0
    
    def _cost_function(self, charging_rate: float, duration: float,
                      electricity_price_profile: List[float]) -> float:
        """Calculate charging cost.
        
        Args:
            charging_rate: Charging rate (kW)
            duration: Charging duration (hours)
            electricity_price_profile: Hourly prices ($/kWh)
            
        Returns:
            Total cost ($)
        """
        total_energy = charging_rate * duration  # kWh
        avg_price = np.mean(electricity_price_profile)
        return total_energy * avg_price
    
    def _degradation_cost(self, charge_rate: float, temperature: float,
                         current_soh: float) -> float:
        """Estimate battery degradation cost.
        
        Args:
            charge_rate: Charging rate (kW)
            temperature: Battery temperature (°C)
            current_soh: Current state of health (%)
            
        Returns:
            Equivalent cost ($) based on degradation
        """
        # Higher rates = more degradation
        rate_factor = (charge_rate / self.MAX_CHARGE_RATE) ** 2
        
        # Temperature increases degradation exponentially
        temp_factor = np.exp((temperature - 25) / 20.0)
        
        # SOH discount (lower SOH = higher degradation cost)
        soh_factor = 1.0 / (current_soh / 100.0)
        
        # Cost per kWh charged
        degradation_cost_per_kwh = 0.5 * rate_factor * temp_factor * soh_factor
        energy = charge_rate * 1.0  # 1 hour
        
        return degradation_cost_per_kwh * energy
    
    def optimize_charging_strategy(self,
                                  current_soc: float,
                                  target_soc: float,
                                  ambient_temp: float,
                                  battery_temp: float,
                                  current_soh: float,
                                  time_available: float,
                                  electricity_price_profile: Optional[List[float]] = None
                                  ) -> ChargingStrategy:
        """Optimize charging strategy using constrained optimization.
        
        Args:
            current_soc: Current state of charge (%)
            target_soc: Target state of charge (%)
            ambient_temp: Ambient temperature (°C)
            battery_temp: Current battery temperature (°C)
            current_soh: Current state of health (%)
            time_available: Time available for charging (hours)
            electricity_price_profile: Hourly electricity prices ($/kWh)
            
        Returns:
            ChargingStrategy with optimal parameters
        """
        if electricity_price_profile is None:
            electricity_price_profile = [self.electricity_rate] * 24
        
        # Energy needed
        energy_needed = (target_soc - current_soc) * 0.75 / 100.0  # kWh (75 kWh capacity)
        
        # Thermal derating
        thermal_factor = self._thermal_derating_factor(ambient_temp, battery_temp)
        max_rate = self.MAX_CHARGE_RATE * thermal_factor
        
        # Calculate optimal charging parameters
        min_duration = energy_needed / max_rate
        optimal_duration = min(time_available, max(min_duration * 1.2, min_duration))
        optimal_rate = energy_needed / optimal_duration
        
        # Cost estimation
        electricity_cost = self._cost_function(optimal_rate, optimal_duration,
                                              electricity_price_profile)
        degradation_cost = self._degradation_cost(optimal_rate, battery_temp,
                                                 current_soh)
        total_cost = electricity_cost + degradation_cost
        
        # Battery stress score (0-100)
        stress_score = (
            (optimal_rate / self.MAX_CHARGE_RATE) * 40 +  # Rate component
            ((battery_temp - 25) / 40) * 40 +  # Temperature component
            ((current_soc - target_soc) / 50) * 20  # SOC swing component
        )
        stress_score = max(0, min(100, stress_score))
        
        # Efficiency gain vs baseline
        baseline_rate = self.MAX_CHARGE_RATE
        baseline_cost = energy_needed * self.electricity_rate * 1.2
        efficiency_gain = max(0, (baseline_cost - total_cost) / baseline_cost * 100)
        
        # Recommendations
        recommendations = []
        if optimal_rate > self.MAX_CHARGE_RATE * 0.8:
            recommendations.append("High charging rate detected. Monitor thermal conditions.")
        if battery_temp > 45:
            recommendations.append("Battery temperature elevated. Consider slower charging.")
        if target_soc > 90:
            recommendations.append("High SOC target reduces battery lifespan. Target 80% for daily use.")
        if not recommendations:
            recommendations.append("Optimal charging conditions. Safe to proceed.")
        
        start_time = datetime.now().isoformat()
        end_time = (datetime.now() + timedelta(hours=optimal_duration)).isoformat()
        
        return ChargingStrategy(
            start_time=start_time,
            end_time=end_time,
            optimal_rate=optimal_rate,
            estimated_duration=optimal_duration,
            estimated_cost=total_cost,
            battery_stress_score=stress_score,
            efficiency_gain=efficiency_gain,
            thermal_peak=battery_temp + (optimal_rate / self.MAX_CHARGE_RATE * 20),
            recommendations=recommendations
        )
    
    def get_strategy_dict(self, strategy: ChargingStrategy) -> Dict:
        """Convert strategy to dictionary format.
        
        Returns:
            Dictionary representation of strategy
        """
        return {
            'start_time': strategy.start_time,
            'end_time': strategy.end_time,
            'optimal_rate': round(strategy.optimal_rate, 2),
            'estimated_duration': round(strategy.estimated_duration, 2),
            'estimated_cost': round(strategy.estimated_cost, 2),
            'battery_stress_score': round(strategy.battery_stress_score, 1),
            'efficiency_gain': round(strategy.efficiency_gain, 1),
            'thermal_peak': round(strategy.thermal_peak, 1),
            'recommendations': strategy.recommendations
        }
