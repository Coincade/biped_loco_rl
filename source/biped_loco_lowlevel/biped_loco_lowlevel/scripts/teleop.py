#!/usr/bin/env python3
"""
Teleop - Synchronize multiple servo motor movements with humanoid motors
Reads angle changes from servo motors and applies them to humanoid motors
"""

import sys
import os
import time
import numpy as np
import argparse

# Parse our arguments first before importing modules that use get_args()
parser = argparse.ArgumentParser(description='Teleop: synchronize servo arm with humanoid arm')
# Parse known args - this will leave unknown args for recoil.util.get_args() to handle
args, remaining_argv = parser.parse_known_args()

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

# Motor IDs: 2, 4, 6, 8, 10 for both servo (leader) and humanoid (follower) motors
SERVO_MOTOR_IDS = [2, 4, 6, 8, 10]
HUMANOID_MOTOR_IDS = [2, 4, 6, 8, 10]
# Motor IDs: 2, 4, 6, 8, 10 for both servo (leader) and humanoid (follower) motors
# SERVO_MOTOR_IDS = [1, 3, 5, 7, 9]
# HUMANOID_MOTOR_IDS = [1, 3, 5, 7, 9]


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
        elif motor_id == 7:
            kp = 20.0
            kd = 5.0
            torque_limit = 3.0
        elif motor_id == 8:
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


def main():
    print("=" * 70)
    print("Arm Teleop - Synchronize Servo Arm with Humanoid Arm")
    print("=" * 70)
    print(f"Servo motor IDs (leader): {SERVO_MOTOR_IDS}")
    print(f"Humanoid motor IDs (follower): {HUMANOID_MOTOR_IDS}")
    print("=" * 70)
    print("Initializing and setting zero position...\n")
    
    servo_driver = None
    control_thread = None
    
    try:
        # Initialize servo driver
        servo_driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        print("✓ Servo driver initialized")
        
        # Read initial angles from servo motors (like vision_logic.py pattern)
        # We'll use servo angles directly to set zero position
        print("\nReading initial servo angles...")
        time.sleep(0.5)  # Give motors time to settle
        
        initial_servo_angles = {}
        
        # Try reading initial servo angles a few times
        for attempt in range(5):
            all_read = True
            for servo_id in SERVO_MOTOR_IDS:
                if servo_id not in initial_servo_angles:
                    angle = get_motor_angle(servo_driver, servo_id)
                    if angle is not None:
                        initial_servo_angles[servo_id] = angle
                    else:
                        all_read = False
            
            # Check if we got all angles
            if len(initial_servo_angles) == len(SERVO_MOTOR_IDS):
                break
            time.sleep(0.1)
        
        # Verify we got all initial servo angles
        missing_servo = [sid for sid in SERVO_MOTOR_IDS if sid not in initial_servo_angles]
        
        if missing_servo:
            print(f"Error: Could not read initial angles from servo motors: {missing_servo}")
            return 1
        
        # Print initial servo angles
        print("\nInitial servo angles (zero position):")
        for servo_id, humanoid_id in zip(SERVO_MOTOR_IDS, HUMANOID_MOTOR_IDS):
            servo_angle = initial_servo_angles[servo_id]
            print(f"  Motor {servo_id}->{humanoid_id}: Servo={servo_angle:.4f} rad ({np.degrees(servo_angle):.2f}°)")
        
        # Create bus mapping for continuous motor control (like vision_logic.py)
        motor_bus_mapping = {
            motor_id: DEFAULT_MOTOR_BUS_MAPPING.get(motor_id, "can0")
            for motor_id in HUMANOID_MOTOR_IDS
        }
        
        # Start continuous motor control with no initial motors (like vision_logic.py)
        # Motors will be initialized dynamically when update_motor_angles is first called
        print("\nStarting continuous motor control...")
        control_thread = start_continuous_motor_control(
            initial_motors=None,  # Start with no motors - they'll be added dynamically
            update_interval_ms=50,
            motor_bus_mapping=motor_bus_mapping
        )
        print("✓ Motor control started")
        
        # Set zero position by updating motors with initial servo angles
        # This will initialize all motors and set them to match servo positions
        print("\nSetting zero position (initializing motors with servo angles)...")
        initial_motor_updates = [(hid, initial_servo_angles[sid]) 
                                  for sid, hid in zip(SERVO_MOTOR_IDS, HUMANOID_MOTOR_IDS)]
        update_motor_angles(initial_motor_updates)
        time.sleep(1.0)  # Give motors time to initialize and move to initial positions
        
        print("\nZero position set. Starting teleop control...")
        print("Move the servo arm to control the humanoid arm.")
        print("Press Ctrl+C to exit\n")
        
        # Main control loop (like vision_logic.py)
        # Read servo angles and update all motors together in a single call
        while True:
            try:
                # Read current angles from all servo motors
                motor_updates = []
                all_read = True
                
                for servo_id, humanoid_id in zip(SERVO_MOTOR_IDS, HUMANOID_MOTOR_IDS):
                    current_servo_angle = get_motor_angle(servo_driver, servo_id)
                    
                    if current_servo_angle is not None:
                        # Calculate change from initial servo angle
                        servo_angle_change = current_servo_angle - initial_servo_angles[servo_id]
                        
                        # Calculate target humanoid angle: initial servo position + change
                        # This ensures they start equal and move together
                        if servo_id == 4:
                            target_humanoid_angle = initial_servo_angles[servo_id] + servo_angle_change
                        else:
                            target_humanoid_angle = - (initial_servo_angles[servo_id] + servo_angle_change)
                        
                        motor_updates.append((humanoid_id, target_humanoid_angle))
                    else:
                        all_read = False
                
                # Update all motors together in a single call (like vision_logic.py)
                if all_read and motor_updates and control_thread is not None:
                    try:
                        update_motor_angles(motor_updates)
                    except Exception as e:
                        print(f"Warning: Could not update motor angles: {e}")
                elif not all_read:
                    print("Warning: Failed to read some servo angles")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.02)  # ~50 Hz update rate
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup (like vision_logic.py)
        if control_thread is not None:
            try:
                stop_continuous_motor_control()
                print("Motor control stopped.")
            except Exception as e:
                print(f"Warning: Error stopping motor control: {e}")
        
        if servo_driver is not None:
            try:
                servo_driver.close()
            except:
                pass
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
