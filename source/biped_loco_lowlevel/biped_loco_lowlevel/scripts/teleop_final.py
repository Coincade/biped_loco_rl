#!/usr/bin/env python3
"""
Teleop Final - Synchronize both servo arms with humanoid arms
Reads angle changes from servo motors and applies them to humanoid motors for both arms
"""

import sys
import os
import time
import numpy as np
import argparse

# Parse our arguments first before importing modules that use get_args()
parser = argparse.ArgumentParser(description='Teleop Final: synchronize both servo arms with humanoid arms')
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

# Motor IDs for both arms
# Left arm servo motors (leader) -> Left arm humanoid motors (follower)
LEFT_SERVO_MOTOR_IDS = [2, 4, 6, 8, 10]
LEFT_HUMANOID_MOTOR_IDS = [2, 4, 6, 8, 10]

# Right arm servo motors (leader) -> Right arm humanoid motors (follower)
RIGHT_SERVO_MOTOR_IDS = [1, 3, 5, 7, 9]
RIGHT_HUMANOID_MOTOR_IDS = [1, 3, 5, 7, 9]

# Combine all motor IDs (remove duplicates)
ALL_SERVO_MOTOR_IDS = sorted(list(set(LEFT_SERVO_MOTOR_IDS + RIGHT_SERVO_MOTOR_IDS)))
ALL_HUMANOID_MOTOR_IDS = sorted(list(set(LEFT_HUMANOID_MOTOR_IDS + RIGHT_HUMANOID_MOTOR_IDS)))

# Create mapping: servo_id -> humanoid_id
# For left arm
SERVO_TO_HUMANOID_MAPPING = {}
for servo_id, humanoid_id in zip(LEFT_SERVO_MOTOR_IDS, LEFT_HUMANOID_MOTOR_IDS):
    SERVO_TO_HUMANOID_MAPPING[servo_id] = humanoid_id
# For right arm
for servo_id, humanoid_id in zip(RIGHT_SERVO_MOTOR_IDS, RIGHT_HUMANOID_MOTOR_IDS):
    SERVO_TO_HUMANOID_MAPPING[servo_id] = humanoid_id


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


def main():
    print("=" * 70)
    print("Bimanual Teleop - Synchronize Both Servo Arms with Humanoid Arms")
    print("=" * 70)
    print(f"Left arm - Servo motor IDs (leader): {LEFT_SERVO_MOTOR_IDS}")
    print(f"Left arm - Humanoid motor IDs (follower): {LEFT_HUMANOID_MOTOR_IDS}")
    print(f"Right arm - Servo motor IDs (leader): {RIGHT_SERVO_MOTOR_IDS}")
    print(f"Right arm - Humanoid motor IDs (follower): {RIGHT_HUMANOID_MOTOR_IDS}")
    print("=" * 70)
    print("Initializing and setting zero position...\n")
    
    servo_driver = None
    control_thread = None
    
    try:
        # Initialize servo driver
        servo_driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        print("✓ Servo driver initialized")
        
        # Read initial angles from all servo motors (like vision_logic.py pattern)
        # We'll use servo angles directly to set zero position
        print("\nReading initial servo angles from all motors...")
        time.sleep(0.5)  # Give motors time to settle
        
        initial_servo_angles = {}
        
        # Try reading initial servo angles a few times
        for attempt in range(5):
            all_read = True
            for servo_id in ALL_SERVO_MOTOR_IDS:
                if servo_id not in initial_servo_angles:
                    angle = get_motor_angle(servo_driver, servo_id)
                    if angle is not None:
                        initial_servo_angles[servo_id] = angle
                    else:
                        all_read = False
            
            # Check if we got all angles
            if len(initial_servo_angles) == len(ALL_SERVO_MOTOR_IDS):
                break
            time.sleep(0.1)
        
        # Verify we got all initial servo angles
        missing_servo = [sid for sid in ALL_SERVO_MOTOR_IDS if sid not in initial_servo_angles]
        
        if missing_servo:
            print(f"Error: Could not read initial angles from servo motors: {missing_servo}")
            return 1
        
        # Print initial servo angles
        print("\nInitial servo angles (zero position):")
        print("Left arm:")
        for servo_id, humanoid_id in zip(LEFT_SERVO_MOTOR_IDS, LEFT_HUMANOID_MOTOR_IDS):
            servo_angle = initial_servo_angles[servo_id]
            print(f"  Motor {servo_id}->{humanoid_id}: Servo={servo_angle:.4f} rad ({np.degrees(servo_angle):.2f}°)")
        print("Right arm:")
        for servo_id, humanoid_id in zip(RIGHT_SERVO_MOTOR_IDS, RIGHT_HUMANOID_MOTOR_IDS):
            servo_angle = initial_servo_angles[servo_id]
            print(f"  Motor {servo_id}->{humanoid_id}: Servo={servo_angle:.4f} rad ({np.degrees(servo_angle):.2f}°)")
        
        # Create bus mapping for continuous motor control
        # Override DEFAULT_MOTOR_BUS_MAPPING to match physical setup:
        # Right arm motors (2, 4, 6, 8, 10) are on can1
        # Left arm motors (1, 3, 5, 7, 9) are on can0
        motor_bus_mapping = {}
        for motor_id in LEFT_HUMANOID_MOTOR_IDS:
            motor_bus_mapping[motor_id] = "can1"  # Left arm (2, 4, 6, 8, 10) on can1
        for motor_id in RIGHT_HUMANOID_MOTOR_IDS:
            motor_bus_mapping[motor_id] = "can0"  # Right arm (1, 3, 5, 7, 9) on can0
        
        # Print bus mapping for debugging
        print("\nMotor bus mapping:")
        print("  Left arm motors (can0):", [m for m in LEFT_HUMANOID_MOTOR_IDS if motor_bus_mapping.get(m) == "can0"])
        print("  Left arm motors (can1):", [m for m in LEFT_HUMANOID_MOTOR_IDS if motor_bus_mapping.get(m) == "can1"])
        print("  Right arm motors (can0):", [m for m in RIGHT_HUMANOID_MOTOR_IDS if motor_bus_mapping.get(m) == "can0"])
        print("  Right arm motors (can1):", [m for m in RIGHT_HUMANOID_MOTOR_IDS if motor_bus_mapping.get(m) == "can1"])
        for motor_id, bus in sorted(motor_bus_mapping.items()):
            print(f"    Motor {motor_id} -> {bus}")
        
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
        # Initialize in two steps: can0 first, then can1, to ensure proper initialization
        print("\nSetting zero position (initializing motors with servo angles)...")
        
        # First, initialize can0 motors (left arm: 1, 3, 5, 7, 9)
        print("  Initializing can0 motors (left arm: {})...".format(RIGHT_HUMANOID_MOTOR_IDS))
        can0_motor_updates = []
        for servo_id, humanoid_id in zip(RIGHT_SERVO_MOTOR_IDS, RIGHT_HUMANOID_MOTOR_IDS):
            # Left arm: no sign inversion (direct mapping)
            can0_motor_updates.append((humanoid_id, initial_servo_angles[servo_id]))
        if can0_motor_updates:
            update_motor_angles(can0_motor_updates)
            time.sleep(0.5)  # Give can0 motors time to initialize
        
        # Then, initialize can1 motors (right arm: 2, 4, 6, 8, 10)
        print("  Initializing can1 motors (right arm: {})...".format(LEFT_HUMANOID_MOTOR_IDS))
        can1_motor_updates = []
        for servo_id, humanoid_id in zip(LEFT_SERVO_MOTOR_IDS, LEFT_HUMANOID_MOTOR_IDS):
            # Right arm: apply sign inversion (except motor 4, based on teleop.py pattern)
            if servo_id == 4:
                can1_motor_updates.append((humanoid_id, initial_servo_angles[servo_id]))
            else:
                can1_motor_updates.append((humanoid_id, -initial_servo_angles[servo_id]))
        if can1_motor_updates:
            update_motor_angles(can1_motor_updates)
            time.sleep(1.0)  # Give can1 motors extra time to initialize
        
        print("  ✓ All motors initialized")
        
        print("\nZero position set. Starting teleop control...")
        print("Move the servo arms to control the humanoid arms.")
        print("Press Ctrl+C to exit\n")
        
        # Main control loop (like vision_logic.py)
        # Read servo angles and update all motors together in a single call
        while True:
            try:
                # Read current angles from all servo motors
                motor_updates = []
                all_read = True
                
                # Process left arm motors (1, 3, 5, 7, 9) on can0
                for servo_id, humanoid_id in zip(RIGHT_SERVO_MOTOR_IDS, RIGHT_HUMANOID_MOTOR_IDS):
                    current_servo_angle = get_motor_angle(servo_driver, servo_id)
                    
                    if current_servo_angle is not None:
                        # Calculate change from initial servo angle
                        servo_angle_change = current_servo_angle - initial_servo_angles[servo_id]
                        
                        # Calculate target humanoid angle: initial servo position + change
                        # Left arm: no sign inversion (direct mapping)
                        target_humanoid_angle = -(initial_servo_angles[servo_id] + servo_angle_change)
                        
                        motor_updates.append((humanoid_id, target_humanoid_angle))
                    else:
                        all_read = False
                
                # Process right arm motors (2, 4, 6, 8, 10) on can1
                for servo_id, humanoid_id in zip(LEFT_SERVO_MOTOR_IDS, LEFT_HUMANOID_MOTOR_IDS):
                    current_servo_angle = get_motor_angle(servo_driver, servo_id)
                    
                    if current_servo_angle is not None:
                        # Calculate change from initial servo angle
                        servo_angle_change = current_servo_angle - initial_servo_angles[servo_id]
                        
                        # Calculate target humanoid angle: initial servo position + change
                        # Right arm: apply sign inversion (except motor 4, based on teleop.py pattern)
                        if servo_id == 4:
                            target_humanoid_angle = initial_servo_angles[servo_id] + servo_angle_change
                        else:
                            target_humanoid_angle = -(initial_servo_angles[servo_id] + servo_angle_change)
                        
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
