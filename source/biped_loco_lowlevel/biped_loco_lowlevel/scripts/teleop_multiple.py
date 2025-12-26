import sys
import os
import time
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped_loco_lowlevel.recoil.core import ST3215Driver
from move_actuator_util import start_continuous_motor_control, update_motor_angles, stop_continuous_motor_control

def servo_units_to_rad(servo_units):
    """
    Convert servo units (0-4095) to radians.
    Inverse of rad_to_servo_units function.
    """
    scale = 4095 / (2 * np.pi)
    return (servo_units - 2046) / scale


def get_motor_angle(driver, motor_id):
    """
    Read the current angle of a motor in radians.
    
    Args:
        driver: ST3215Driver instance
        motor_id: Motor ID (1-10)
    
    Returns:
        float: Motor angle in radians, or None if read fails
    """
    position = driver.read_position(motor_id)
    if position is not None:
        angle = servo_units_to_rad(position)
        return angle
    return None


def read_humanoid_motor_angle(bus, motor_id):
    """
    Read the current angle of a humanoid motor in radians.
    
    Args:
        bus: recoil.Bus instance
        motor_id: Motor ID
    
    Returns:
        float: Motor angle in radians, or None if read fails
    """
    position = bus.read_position_measured(motor_id)
    if position is not None:
        return position
    return None

# Initialize motor control
motor_bus_mapping = {
    2: "can0",  
    4: "can0",  
    6: "can0",  
    8: "can0",  
    10: "can0",
}

control_thread = None
try:
    # Start with no motors - they will be added dynamically when update_motor_angles is called
    # Pass the motor-to-bus mapping so motors are initialized on the correct CAN bus
    control_thread = start_continuous_motor_control(
        initial_motors=None, 
        update_interval_ms=50,
        motor_bus_mapping=motor_bus_mapping
    )
    print("Motor control started. Press 'q' to quit.")
    print("Motor bus mapping: can0 (motors 3, 7), can1 (motors 4, 8)")
    print("You can update any motor ID using update_motor_angles([(motor_id, angle), ...])")
except Exception as e:
    print(f"Warning: Could not start motor control: {e}")
    print("Continuing with vision only...")


try:


                # Motors will be automatically initialized if they haven't been initialized yet
                if control_thread is not None:
                    try:
                        # Update any motor IDs you want - they will be initialized automatically
                        update_motor_angles([(2, left_shoulder_angle),(4, -right_shoulder_angle),(8, left_elbow_angle),(10, -right_elbow_angle)])
                    except Exception as e:
                        print(f"Warning: Could not update motor angles: {e}")
                                     
    
except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    # Stop motor control
    if control_thread is not None:
        try:
            stop_continuous_motor_control()
            print("Motor control stopped.")
        except Exception as e:
            print(f"Warning: Error stopping motor control: {e}")