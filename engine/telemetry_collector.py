"""Tesla Telemetry Data Collector Module.

Responsible for real-time collection and streaming of vehicle telemetry data
from Tesla vehicles through simulated API integration.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ChargeState(Enum):
    """Vehicle charging states."""
    DISCONNECTED = "disconnected"
    CONNECTED_IDLE = "connected_idle"
    CHARGING = "charging"
    COMPLETE = "complete"
    ERROR = "error"


class DriveMode(Enum):
    """Vehicle drive modes."""
    PARK = "park"
    REVERSE = "reverse"
    NEUTRAL = "neutral"
    DRIVE = "drive"


@dataclass
class BatteryMetrics:
    """Battery state and health metrics."""
    soc: float  # State of charge percentage
    voltage: float  # Battery voltage (V)
    current: float  # Current draw (A)
    power: float  # Power (kW)
    temperature: float  # Battery temperature (°C)
    capacity: float  # Total capacity (kWh)
    degradation: float  # Capacity degradation percentage
    cell_voltages: List[float]  # Individual cell voltages
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VehicleState:
    """Current vehicle operational state."""
    timestamp: str
    odometer: float  # km
    speed: float  # km/h
    drive_mode: DriveMode
    charge_state: ChargeState
    inside_temp: float  # °C
    outside_temp: float  # °C
    battery: BatteryMetrics
    elevation: float  # meters
    latitude: float
    longitude: float
    efficiency: float  # Wh/km
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['drive_mode'] = self.drive_mode.value
        data['charge_state'] = self.charge_state.value
        data['battery'] = self.battery.to_dict()
        return data


class TelemetryCollector:
    """Real-time telemetry data collection engine for Tesla vehicles.
    
    Collects comprehensive vehicle metrics including battery state, thermal
    conditions, drivetrain efficiency, and operational parameters. Implements
    adaptive sampling based on vehicle state and data significance.
    """
    
    def __init__(self, vehicle_id: str, api_key: Optional[str] = None,
                 sample_rate: float = 1.0):
        """Initialize collector.
        
        Args:
            vehicle_id: Unique Tesla vehicle identifier
            api_key: API authentication key (simulated)
            sample_rate: Data collection frequency (Hz)
        """
        self.vehicle_id = vehicle_id
        self.api_key = api_key or "demo_key"
        self.sample_rate = sample_rate
        self.collection_interval = 1.0 / sample_rate
        self.data_buffer: List[VehicleState] = []
        self.is_collecting = False
        
    def _simulate_sensor_reading(self, base_value: float,
                                noise_std: float = 0.05) -> float:
        """Add realistic sensor noise to readings."""
        return base_value + np.random.normal(0, noise_std * base_value)
    
    def _generate_battery_metrics(self, soc: float,
                                 thermal_stress: float) -> BatteryMetrics:
        """Generate realistic battery state metrics.
        
        Args:
            soc: State of charge (0-100%)
            thermal_stress: Thermal stress factor (0-1)
            
        Returns:
            BatteryMetrics with realistic values
        """
        capacity = 75.0  # kWh
        degradation = 0.5 + (thermal_stress * 2.0)  # 0.5-2.5%
        
        # Temperature rises with thermal stress
        base_temp = 25.0 + (thermal_stress * 30.0)
        temperature = self._simulate_sensor_reading(base_temp, noise_std=0.02)
        
        # Voltage follows SOC curve with polynomial fit
        voltage = 320.0 + (soc / 100.0 * 80.0)
        voltage = self._simulate_sensor_reading(voltage, noise_std=0.01)
        
        # Current varies with thermal and charge state
        current = (thermal_stress * 250.0) if thermal_stress > 0 else 0
        current = self._simulate_sensor_reading(current, noise_std=0.05)
        
        power = (voltage * current) / 1000.0  # Convert to kW
        
        # Cell voltages with realistic variation
        cell_voltages = [voltage / 96.0 + np.random.normal(0, 0.01)
                        for _ in range(96)]
        
        return BatteryMetrics(
            soc=max(0, min(100, soc)),
            voltage=voltage,
            current=current,
            power=power,
            temperature=temperature,
            capacity=capacity * (1 - degradation / 100.0),
            degradation=degradation,
            cell_voltages=cell_voltages
        )
    
    def collect_snapshot(self, 
                        odometer: float,
                        speed: float,
                        soc: float,
                        drive_mode: DriveMode,
                        charge_state: ChargeState,
                        latitude: float = 0.0,
                        longitude: float = 0.0) -> VehicleState:
        """Collect single telemetry snapshot.
        
        Args:
            odometer: Current odometer reading (km)
            speed: Vehicle speed (km/h)
            soc: Battery state of charge (%)
            drive_mode: Current drive mode
            charge_state: Current charge state
            latitude: GPS latitude
            longitude: GPS longitude
            
        Returns:
            VehicleState object with all collected metrics
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate thermal stress from driving conditions
        thermal_stress = min(1.0, (speed / 200.0) + (max(0, soc - 80) / 20.0))
        
        # Efficiency calculation (Wh/km)
        efficiency = 150.0 + (speed / 100.0 * 50.0) + (thermal_stress * 30.0)
        efficiency = self._simulate_sensor_reading(efficiency, noise_std=0.05)
        
        battery_metrics = self._generate_battery_metrics(soc, thermal_stress)
        
        state = VehicleState(
            timestamp=timestamp,
            odometer=self._simulate_sensor_reading(odometer, noise_std=0.0001),
            speed=self._simulate_sensor_reading(speed, noise_std=0.02),
            drive_mode=drive_mode,
            charge_state=charge_state,
            inside_temp=self._simulate_sensor_reading(22.0, noise_std=0.05),
            outside_temp=self._simulate_sensor_reading(15.0, noise_std=0.10),
            battery=battery_metrics,
            elevation=self._simulate_sensor_reading(100.0, noise_std=0.01),
            latitude=latitude,
            longitude=longitude,
            efficiency=efficiency
        )
        
        self.data_buffer.append(state)
        return state
    
    def get_buffer_as_dataframe(self) -> pd.DataFrame:
        """Convert collected data buffer to pandas DataFrame.
        
        Returns:
            DataFrame with flattened telemetry data
        """
        if not self.data_buffer:
            return pd.DataFrame()
        
        records = []
        for state in self.data_buffer:
            record = state.to_dict()
            records.append(record)
        
        return pd.DataFrame(records)
    
    def clear_buffer(self) -> None:
        """Clear the data collection buffer."""
        self.data_buffer.clear()
    
    def get_latest_state(self) -> Optional[VehicleState]:
        """Get the most recent telemetry snapshot."""
        return self.data_buffer[-1] if self.data_buffer else None
