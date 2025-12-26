#!/usr/bin/env python3
"""
Test Teleop - Synchronize servo motor movements with humanoid motors
Reads angle changes from servo motor and applies them to humanoid motor
"""

import sys
import os
import time
import numpy as np
import argparse

# Parse our arguments first before importing modules that use get_args()
parser = argparse.ArgumentParser(description='Test teleop: synchronize servo motor with humanoid motor')
parser.add_argument('servo_motor_id', type=int, nargs='?', default=4,
                    help='Servo motor ID (default: 4)')
parser.add_argument('humanoid_motor_id', type=int, nargs='?', default=4,
                    help='Humanoid motor ID (default: 4)')
# Parse known args - this will leave unknown args for recoil.util.get_args() to handle
args, remaining_argv = parser.parse_known_args()

# Save our parsed arguments
servo_motor_id = args.servo_motor_id
humanoid_motor_id = args.humanoid_motor_id

# Modify sys.argv to only include remaining args for get_args()
# This prevents get_args() from seeing our arguments
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]] + remaining_argv

# Add the parent directory to the path to import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from biped_loco_lowlevel.recoil.core import ST3215Driver

# Add vision_to_teleop directory to path for imports
vision_teleop_dir = os.path.join(parent_dir, 'vision_to_teleop')
sys.path.append(vision_teleop_dir)
from move_actuator_util import (
    start_continuous_motor_control,
    update_motor_angles,
    stop_continuous_motor_control,
    DEFAULT_MOTOR_BUS_MAPPING
)
import berkeley_humanoid_lite_lowlevel.recoil as recoil

# Keep sys.argv modified so get_args() doesn't see our arguments
# We've already parsed our arguments, so we don't need to restore sys.argv


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


def initialize_humanoid_motor(bus, motor_id):
    """
    Initialize a humanoid motor with appropriate settings.
    
    Args:
        bus: recoil.Bus instance
        motor_id: Motor ID
    
    Returns:
        bool: True if initialization successful
    """
    try:
        if motor_id < 5:
            kp = 50.0
            kd = 2.0
            torque_limit = 6.0
        elif motor_id == 7 or motor_id == 8:
            kp = 20.0
            kd = 5.0
            torque_limit = 2.5
        elif motor_id == 5 or motor_id == 6:
            kp = 20.0
            kd = 4.0
            torque_limit = 4.0
        else:
            kp = 20.0
            kd = 4.0
            torque_limit = 4.0
        
        bus.write_position_kp(motor_id, kp)
        bus.write_position_kd(motor_id, kd)
        bus.write_torque_limit(motor_id, torque_limit)
        bus.write_gear_ratio(motor_id, -15.0)
        bus.write_position_limit_upper(motor_id, np.inf)
        bus.write_position_limit_lower(motor_id, -np.inf)
        
        bus.set_mode(motor_id, recoil.Mode.POSITION)
        bus.feed(motor_id)
        
        return True
    except Exception as e:
        print(f"Error initializing humanoid motor {motor_id}: {e}")
        return False


def main(servo_motor_id, humanoid_motor_id):
    print(f"Teleop Test: Servo Motor {servo_motor_id} -> Humanoid Motor {humanoid_motor_id}")
    print("Initializing and setting zero position...")
    
    servo_driver = None
    humanoid_bus = None
    control_thread = None
    
    try:
        # Initialize servo driver
        servo_driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        print(f"✓ Servo driver initialized")
        
        # Initialize humanoid motor bus
        bus_channel = DEFAULT_MOTOR_BUS_MAPPING.get(humanoid_motor_id, "can0")
        humanoid_bus = recoil.Bus(channel=bus_channel, bitrate=1000000)
        print(f"✓ Humanoid bus initialized on {bus_channel}")
        
        # Initialize humanoid motor
        if not initialize_humanoid_motor(humanoid_bus, humanoid_motor_id):
            print("Failed to initialize humanoid motor")
            return 1
        print(f"✓ Humanoid motor {humanoid_motor_id} initialized")
        
        # Read initial angles from both motors
        print("\nReading initial angles...")
        time.sleep(0.5)  # Give motors time to settle
        
        initial_servo_angle = None
        initial_humanoid_angle = None
        
        # Try reading initial angles a few times
        for attempt in range(5):
            initial_servo_angle = get_motor_angle(servo_driver, servo_motor_id)
            initial_humanoid_angle = read_humanoid_motor_angle(humanoid_bus, humanoid_motor_id)
            
            if initial_servo_angle is not None and initial_humanoid_angle is not None:
                break
            time.sleep(0.1)
        
        if initial_servo_angle is None:
            print(f"Error: Could not read initial angle from servo motor {servo_motor_id}")
            return 1
        
        if initial_humanoid_angle is None:
            print(f"Error: Could not read initial angle from humanoid motor {humanoid_motor_id}")
            return 1
        
        print(f"Initial servo angle: {initial_servo_angle:.4f} rad ({np.degrees(initial_servo_angle):.2f}°)")
        print(f"Initial humanoid angle: {initial_humanoid_angle:.4f} rad ({np.degrees(initial_humanoid_angle):.2f}°)")
        
        # Set zero position - make them equal
        # The offset is the difference between initial angles
        # We'll maintain this offset so movements are relative
        angle_offset = initial_humanoid_angle - initial_servo_angle
        print(f"Angle offset: {angle_offset:.4f} rad ({np.degrees(angle_offset):.2f}°)")
        print("\nZero position set. Starting teleop control...")
        print("Move the servo motor to control the humanoid motor.")
        print("Press Ctrl+C to exit\n")
        
        # Start continuous motor control
        # Initialize with current humanoid position to maintain it
        control_thread = start_continuous_motor_control(
            initial_motors=[(humanoid_motor_id, initial_humanoid_angle)],
            update_interval_ms=50,
            motor_bus_mapping={humanoid_motor_id: bus_channel}
        )
        
        time.sleep(0.2)  # Give control thread time to start
        
        # Main control loop
        rate = 50.0  # 50 Hz update rate
        sleep_time = 1.0 / rate
        
        while True:
            # Read current servo angle
            current_servo_angle = get_motor_angle(servo_driver, servo_motor_id)
            
            if current_servo_angle is not None:
                # Calculate change from initial servo angle
                servo_angle_change = current_servo_angle - initial_servo_angle
                
                # Calculate target humanoid angle: initial + change + offset
                # This ensures they start equal and move together
                target_humanoid_angle = initial_humanoid_angle + servo_angle_change
                
                # Update humanoid motor target
                update_motor_angles([(humanoid_motor_id, target_humanoid_angle)])
                
                # Optional: print status
                print(f"Servo: {current_servo_angle:.4f} rad | "
                      f"Change: {servo_angle_change:+.4f} rad | "
                      f"Humanoid target: {target_humanoid_angle:.4f} rad")
            else:
                print(f"Warning: Failed to read servo angle")
            
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if control_thread is not None:
            stop_continuous_motor_control()
            time.sleep(0.2)
        
        if humanoid_bus is not None:
            try:
                humanoid_bus.set_mode(humanoid_motor_id, recoil.Mode.IDLE)
                humanoid_bus.stop()
            except:
                pass
        
        if servo_driver is not None:
            try:
                servo_driver.close()
            except:
                pass
    
    return 0


if __name__ == "__main__":
    exit_code = main(servo_motor_id, humanoid_motor_id)
    sys.exit(exit_code)
