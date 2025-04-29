"""
This script is used to control the altitude of the Crazyflie drone using a PID controller.
"""

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import time
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PIDConfig:
    """Configuration for PID controller"""
    Kp: float
    Ki: float
    Kd: float
    output_limits: Tuple[float, float] = (-1.0, 1.0)
    anti_windup: bool = True

class PIDController:
    """
    PID controller with anti-windup protection and output clamping.
    """
    def __init__(self, config: PIDConfig, setpoint: float = 0.0):
        self.config = config
        self.setpoint = setpoint
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update_setpoint(self, setpoint: float) -> None:
        """Update the target setpoint"""
        self.setpoint = setpoint

    def compute(self, process_variable: float, dt: Optional[float] = None) -> float:
        """
        Compute PID output.
        
        Args:
            process_variable: Current process value
            dt: Time step (optional, will use time.time() if not provided)
            
        Returns:
            float: PID controller output
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        P_out = self.config.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        if self.config.anti_windup:
            # Simple anti-windup: limit integral term
            max_integral = (self.config.output_limits[1] - P_out) / self.config.Ki
            min_integral = (self.config.output_limits[0] - P_out) / self.config.Ki
            self.integral = np.clip(self.integral, min_integral, max_integral)
        I_out = self.config.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        D_out = self.config.Kd * derivative
        
        # Compute total output with clamping
        output = P_out + I_out + D_out
        output = np.clip(output, *self.config.output_limits)
        
        # Update previous error
        self.previous_error = error
        
        return output

def reset_estimator(cf) -> None:
    """Reset the position estimator"""
    if not cf:
        raise RuntimeError("Drone not connected")
    
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    print("Estimator reset")
    wait_for_position_estimator(cf)


def wait_for_position_estimator(scf):
    print("Waiting for estimator to find position...")

    log_config = LogConfig(name="Kalman Variance", period_in_ms=100)
    log_config.add_variable("kalman.varPX", "float")
    log_config.add_variable("kalman.varPY", "float")
    log_config.add_variable("kalman.varPZ", "float")

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.2

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data["kalman.varPX"])
            var_x_history.pop(0)
            var_y_history.append(data["kalman.varPY"])
            var_y_history.pop(0)
            var_z_history.append(data["kalman.varPZ"])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            print("dx:{} dy:{} dz:{}".format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (
                (max_x - min_x) < threshold
                and (max_y - min_y) < threshold
                and (max_z - min_z) < threshold
            ):
                break

def log_stab_callback(timestamp, data, logconf):
    return data["kalman.stateZ"]

def simple_log_async(scf, logconf):
    cf = scf.cf
    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(log_stab_callback)
    logconf.start()
    time.sleep(5)
    logconf.stop()

class DroneController:
    """Handles drone control and logging"""
    def __init__(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.find_crazyflies()
        self.cf = Crazyflie(rw_cache="./cache")
        self.log_vec: List[List[float]] = []
        self.z_target = 0.0
        self.log_config = LogConfig(name="Altitude", period_in_ms=10)
        self.log_config.add_variable("kalman.stateZ", "float")

    def find_crazyflies(self) -> None:
        print("Scanning interfaces for Crazyflies...")
        available = cflib.crtp.scan_interfaces()
        assert len(available) == 1, print("can not find crazyflie, try again")
        print("Crazyflies found.")
        self.uri = available[0][0]

    # Create and run controller
    def run_experiment(self, sequence: List[float], pid_config: PIDConfig) -> None:
        with SyncCrazyflie(self.uri, cf=self.cf) as scf:
            reset_estimator(self.cf)
            self.set_gains()
            self.start_position_logging()
            self.run_sequence(sequence, pid_config)

    # set PID gains
    def set_gains(self):
        # Default gain
        # posCtlPid.xKp: 2.0
        # posCtlPid.xKi: 0.0
        # posCtlPid.xKd: 0.0
        # posCtlPid.yKp: 2.0
        # posCtlPid.yKi: 0.0
        # posCtlPid.yKd: 0.0
        # posCtlPid.zKp: 2.0
        # posCtlPid.zKi: 0.5
        # posCtlPid.zKd: 0.0

        # Modify Position Gains
        self.cf.param.set_value("posCtlPid.xKp", 2.0)
        self.cf.param.set_value("posCtlPid.xKd", 0.5)
        self.cf.param.set_value("posCtlPid.yKp", 2.0)
        self.cf.param.set_value("posCtlPid.yKd", 0.5)
        self.cf.param.set_value("posCtlPid.zKp", 2.0)
        self.cf.param.set_value("posCtlPid.zKd", 0.5)

        # Modify Velocity Gains
        self.cf.param.set_value("velCtlPid.vxKp", 10.0)
        self.cf.param.set_value("velCtlPid.vxKi", 1.0)
        self.cf.param.set_value("velCtlPid.vyKp", 10.0)
        self.cf.param.set_value("velCtlPid.vyKi", 1.0)
        self.cf.param.set_value("velCtlPid.vzKp", 15.0)
        self.cf.param.set_value("velCtlPid.vzKi", 15.0)


    def disconnect(self) -> None:
        """Disconnect from the drone"""
        if self.scf:
            self.scf.close_link()



    def position_callback(self, timestamp: float, data: dict, logconf: LogConfig) -> None:
        """Callback for position logging"""
        x = data["kalman.stateX"]
        y = data["kalman.stateY"]
        z = data["kalman.stateZ"]
        logger.info(f"Position: ({x:.3f}, {y:.3f}, {z:.3f})")
        self.log_vec.append([timestamp, x, y, z, self.z_target])

    def start_position_logging(self) -> None:
        """Start logging position data"""
        log_conf = LogConfig(name="Position", period_in_ms=10)
        log_conf.add_variable("kalman.stateX", "float")
        log_conf.add_variable("kalman.stateY", "float")
        log_conf.add_variable("kalman.stateZ", "float")

        self.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self.position_callback)
        log_conf.start()

    def run_sequence(self, z_sequence: List[float], pid_config: PIDConfig) -> None:
        """
        Run a sequence of altitude commands using PID control.
        
        Args:
            z_sequence: List of target altitudes
            pid_config: PID controller configuration
        """
        if not self.cf:
            raise RuntimeError("Drone not connected")

        pid_controller = PIDController(pid_config)
        t_old = time.time()

        try:
            # Takeoff gently
            initial_position = z_sequence[0]
            for y in range(10):
                self.cf.commander.send_position_setpoint(0, 0, y / 10.0 * initial_position, 0)
                self.z_target = initial_position / 25
                time.sleep(0.1)

            # Main control loop
            for altitude_target in z_sequence:
                pid_controller.update_setpoint(altitude_target)
                logger.info(f"Setting altitude target to {altitude_target}")
                
                for _ in range(50):
                    t = time.time()
                    dt = t - t_old
                    t_old = t
                    
                    z = self.cf.state.z
                    thrust_output = pid_controller.compute(z, dt)
                    
                    self.cf.commander.send_setpoint(0, 0, 0, thrust_output)
                    self.z_target = altitude_target
                    # time.sleep(0.01)

        except Exception as e:
            logger.error(f"Error during sequence execution: {e}")
            raise
        finally:
            self.cf.commander.send_stop_setpoint()
            time.sleep(0.1)

def main():
    # PID configuration
    pid_config = PIDConfig(
        Kp=9500,  # meters
        Ki=0,     # meters second
        Kd=45000  # meters/second
    )


    # Extract z values from sequence
    z_sequence = [0, 0.2, 1, 0.2, 0]

    controller = DroneController()
    controller.run_experiment(z_sequence, pid_config)

if __name__ == "__main__":
    main()





