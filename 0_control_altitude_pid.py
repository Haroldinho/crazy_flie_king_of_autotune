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
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configure logging
logging.basicConfig(
    filename='logs/z_pid_position.log', 
    encoding='utf-8',
    level=logging.INFO, 
    format='%(asctime)s, %(levelname)s, %(message)s,', 
    filemode='a',
    datefmt="%Y-%m-%d %H:%M",
    )
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# CONSTANTS
MASS = 0.031  # 0.0313 (NO LED)     0.03834 (WITH LED)
GRAVITY = 9.81
LEARNING_RATE = 0.2
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

    def update_target(self, setpoint: float) -> None:
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
        if self.config.Ki > 0 and self.config.anti_windup:
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
            )
                break

def convert_rad_to_deg(angle_in_rad: float)-> float:
    """
    Function to generate angle in radians to degrees

    """
    return angle_in_rad/math.PI  * 180


class DroneController:
    """Handles drone control and logging"""
    def __init__(self):
        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.find_crazyflies()
        self.cf = Crazyflie(rw_cache="./cache")
        self.log_vec: List[List[float]] = []
        self.log_config = LogConfig(name="Altitude", period_in_ms=10)
        self.log_config.add_variable("stateEstimate.x", "FP16")
        self.log_config.add_variable("stateEstimate.y", "FP16")
        self.log_config.add_variable("stateEstimate.z", "FP16")
        self.log_config.add_variable("stateEstimate.roll", "FP16")
        self.log_config.add_variable("stateEstimate.pitch", "FP16")
        self.log_config.add_variable("stateEstimateZ.rateYaw", "FP16") # yaw rate
        self.z_target = 0
        self.thrust_input = 0
        self.pause_in_s_between_setpoints = 3 # seconds
        

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

    def position_callback(self, timestamp: float, data: dict, logconf: LogConfig) -> None:
        """Callback for position logging"""
        x = data["kalman.stateX"]
        y = data["kalman.stateY"]
        z = data["kalman.stateZ"]
        z_target = self.z_target
        print(f"Position: ({x:.3f}, {y:.3f}, {z:.3f}), \tTarget: {z_target:.3f} \tThrust: {self.thrust_input}")
        logging.info(f"Position: ({x:.3f}, {y:.3f}, {z:.3f}), \tTarget: {z_target:.3f} \tThrust: {self.thrust_input}")
        self.log_vec.append([timestamp, x, y, z])

    def start_position_logging(self) -> None:
        """Start logging position data"""
        log_conf = LogConfig(name="Position", period_in_ms=10)
        log_conf.add_variable("kalman.stateX", "float")
        log_conf.add_variable("kalman.stateY", "float")
        log_conf.add_variable("kalman.stateZ", "float")

        self.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self.position_callback)
        log_conf.start()

    def visualize_states(self, log_data: dict, z_sequence: List[float], thrust_data: List[float]) -> None:
        """
        Visualize the logged states against wall clock time.
        
        Args:
            log_data: Dictionary containing logged state data
            z_sequence: List of target z positions
            thrust_data: List of thrust values
        """
        # Create time array (assuming 10ms logging period)
        time_array = np.arange(0, len(log_data["stateEstimate.x"])) * 0.01  # 10ms = 0.01s
        
        # Create figure with subplots for states
        fig1 = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig1)
        
        # Position subplot
        ax_pos = fig1.add_subplot(gs[0, :])
        ax_pos.plot(time_array, log_data["stateEstimate.x"], label='X position')
        ax_pos.plot(time_array, log_data["stateEstimate.y"], label='Y position')
        ax_pos.plot(time_array, log_data["stateEstimate.z"], label='Z position', linewidth=2)
        
        # Add target z positions
        target_times = []
        current_time = 0
        for z_target in z_sequence:
            target_times.append(current_time)
            current_time += self.pause_in_s_between_setpoints
        
        # Plot target z positions as horizontal lines
        for i, (t_start, z_target) in enumerate(zip(target_times[:-1], z_sequence[:-1])):
            t_end = target_times[i + 1]
            ax_pos.hlines(y=z_target, xmin=t_start, xmax=t_end, 
                         colors='r', linestyles='--', label='Target Z' if i == 0 else "")
        
        # Plot last target position
        if len(z_sequence) > 0:
            ax_pos.hlines(y=z_sequence[-1], xmin=target_times[-1], xmax=time_array[-1],
                         colors='r', linestyles='--', label='Target Z' if len(z_sequence) == 1 else "")
        
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.set_title('Drone Position')
        ax_pos.legend()
        ax_pos.grid(True)
        
        # Attitude subplot
        ax_att = fig1.add_subplot(gs[1, :])
        ax_att.plot(time_array, log_data["stateEstimate.roll"], label='Roll')
        ax_att.plot(time_array, log_data["stateEstimate.pitch"], label='Pitch')
        ax_att.set_xlabel('Time (s)')
        ax_att.set_ylabel('Angle (rad)')
        ax_att.set_title('Drone Attitude')
        ax_att.legend()
        ax_att.grid(True)
        
        # Yaw rate subplot
        ax_yaw = fig1.add_subplot(gs[2, :])
        ax_yaw.plot(time_array, log_data["stateEstimateZ.rateYaw"], label='Yaw Rate')
        ax_yaw.set_xlabel('Time (s)')
        ax_yaw.set_ylabel('Yaw Rate (rad/s)')
        ax_yaw.set_title('Yaw Rate')
        ax_yaw.legend()
        ax_yaw.grid(True)
        
        plt.tight_layout()
        plt.savefig('logs/drone_states.png')
        plt.close()

        # Create separate figure for thrust vs z-position
        fig2, ax1 = plt.subplots(figsize=(15, 6))
        
        # Plot z-position on primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Z Position (m)', color=color)
        line1 = ax1.plot(time_array, log_data["stateEstimate.z"], color=color, label='Z Position', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Add target z positions
        for i, (t_start, z_target) in enumerate(zip(target_times[:-1], z_sequence[:-1])):
            t_end = target_times[i + 1]
            ax1.hlines(y=z_target, xmin=t_start, xmax=t_end, 
                      colors='r', linestyles='--', label='Target Z' if i == 0 else "")
        
        # Plot last target position
        if len(z_sequence) > 0:
            ax1.hlines(y=z_sequence[-1], xmin=target_times[-1], xmax=time_array[-1],
                      colors='r', linestyles='--', label='Target Z' if len(z_sequence) == 1 else "")
        
        # Create secondary y-axis for thrust
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Thrust', color=color)
        line2 = ax2.plot(time_array, thrust_data, color=color, label='Thrust', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title('Z Position and Thrust vs Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('logs/thrust_vs_zposition.png')
        plt.close()

    def run_sequence(self, z_sequence: List[float], pid_config: PIDConfig) -> None:
        """
        Run a sequence of altitude commands using PID control.
        send a new setpoint every second
        
        Args:
            z_sequence: List of target altitudes
            pid_config: PID controller configuration
        """


        if not self.cf:
            raise RuntimeError("Drone not connected")
                # setpoint positions logging
        log_data = {}
        log_data["stateEstimate.x"] = []
        log_data["stateEstimate.y"] = []
        log_data["stateEstimate.z"] = []
        pid_controller = PIDController(pid_config)
        lg_stab2 = LogConfig(name="Stabilizer2", period_in_ms=10)
        lg_stab2.add_variable("stabilizer.thrust", "float")
        lg_stab2.add_variable("pm.vbatMV", "float")
        lg_stab2.add_variable("stateEstimate.x", "FP16")
        lg_stab2.add_variable("stateEstimate.y", "FP16")
        lg_stab2.add_variable("stateEstimate.z", "FP16")
        # lg_stab2.add_variable("stateEstimate.roll", "FP16")
        # lg_stab2.add_variable("stateEstimate.pitch", "FP16")
        # lg_stab2.add_variable("stateEstimateZ.rateYaw", "FP16") # yaw rate
        # add variables for the 
        # hover logging position
        log_data2 = {}
        log_data2["stabilizer.thrust"] = []
        log_data2["pm.vbatMV"] = []
        log_data2["stateEstimate.x"] = []
        log_data2["stateEstimate.y"] = []
        log_data2["stateEstimate.z"] = []
        position_index = 0
        initial_z_position = z_sequence[position_index]
        counter_thrust = 0
        self.z_target = initial_z_position
        # ====================== INITIALIZE ===========================
        with SyncLogger(self.cf, lg_stab2) as logger:
            for log_entry in logger:
                for key, value in log_entry[1].items():
                    if str(key) in log_data2:
                        log_data2[str(key)].append(value)

                self.cf.commander.send_hover_setpoint(0, 0, 0, initial_z_position)
                time.sleep(0.1)
                counter_thrust = counter_thrust + 1

                if counter_thrust >= 50:
                    break

        
        # ======================= PID  ===========================
        steady_thrust = 1.02 * np.mean(
            log_data2["stabilizer.thrust"][-50:]
        )
        t_old = time.time()
        
        last_recorded_time = t_old
        max_thrust = 65535 * (MASS * 9.81) / (steady_thrust)
        thrust = steady_thrust
        
        with SyncLogger(self.cf, self.log_config) as logger:
            t_start = time.time()
            # Store thrust values for visualization
            thrust_history = []
            
            for log_entry in logger:
                self.thrust_input = thrust
                data = log_entry[1]
                # read all tracked state variables
                for key, value in data.items():
                    log_data[str(key)].append(value)
                zz = np.round(log_data["stateEstimate.z"][-1], 6)
                theta = np.round(log_data["stateEstimate.roll"][-1], 6)
                phi = np.round(log_data['stateEstimate.pitch'][-1], 6)
                yaw_rate = np.round(log_data['stateEstimateZ.rateYaw'][-1], 6)


                altitude_target = z_sequence[min(position_index, len(z_sequence) -1)]
                # update target position
                pid_controller.update_target(altitude_target)
                self.z_target = altitude_target
                print(f"Setting altitude target to {altitude_target}")
                t = time.time()
                dt = t - t_old
                t_old = t
                old_thrust = thrust
                pid_thrust_output = pid_controller.compute(zz, dt)    
                # # ACTUATION LIMIT
                thrust_output = pid_thrust_output * MASS + MASS * GRAVITY
                if (65535 * thrust_output) / max_thrust > 65535:
                    thrust = 65535
                else:
                    thrust = (65535 * thrust_output) / max_thrust
                    if thrust < 0:
                        thrust = 0
                thrust = LEARNING_RATE * thrust + (1 - LEARNING_RATE) * old_thrust

                # thrust must be an integer
                # send attitude controller target
                
                # Send a new control setpoint for roll/pitch/yaw_Rate/thrust to the copter
                # pitch (deg), roll (deg), yaw rate (deg/s) and thrust
                self.cf.commander.send_setpoint(phi, theta, yaw_rate, int(thrust))
                # time.sleep(0.01)
                # break when we get to the last setpoint
                # if position_index == len(z_sequence) - 1:
                #     break
                # every 3 seconds, send a new setpoint
                if  (t - last_recorded_time) > self.pause_in_s_between_setpoints and abs(altitude_target - zz)< 0.1:
                    # set a new position

                    position_index = position_index + 1
                    print(f"\nUpdating z positiong to {z_sequence[position_index]}")
                    logging.info(f"Updating positiong to {z_sequence[position_index]} ")
                    # reset the time
                    last_recorded_time = t
 
                    

                """ =    END OF SIMULATION   =   !! """
                # if ((time.time() - t_start) > max(len(z_sequence) * (self.pause_in_s_between_setpoints + 1), 30)):
                if (time.time() - t_start) > 300:
                    # disable the low level controller and return to the high level controller
                    self.cf.commander.send_notify_setpoint_stop()
                    logging.info("\nTERMINATING\n")
                    NUM_STEPS = 60
                    for ik in range(NUM_STEPS):
                        final_height = zz * (1 - (ik + 1)/float(NUM_STEPS))

                        if final_height < 0.15:
                            final_height = 0.0
                            self.cf.commander.send_hover_setpoint(0, 0, 0, final_height)
                            ik = 60
                            break

                        self.cf.commander.send_hover_setpoint(0, 0, 0, final_height)
                        time.sleep(0.075)

                time_simulation = time.time() - t_start
                print(f"Time simulation: {time_simulation:.2f} seconds")

                # Store thrust value
                thrust_history.append(thrust)

            # After the experiment is complete, visualize the states
            self.visualize_states(log_data, z_sequence, thrust_history)

def main():
    # PID configuration
    pid_config = PIDConfig(
        Kp=9500,  # meters
        Ki=0.5,     # meters second
        Kd=45000  # meters/second
    )


    # Extract z values from sequence
    # z_sequence = [0.2, 0.5, 1, 0.8,  0.5, 0.2]
    z_sequence = [0.2, 1]

    controller = DroneController()
    controller.run_experiment(z_sequence, pid_config)

if __name__ == "__main__":
    main()





