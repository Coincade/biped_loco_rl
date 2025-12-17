#!/usr/bin/env python3
"""
Read Servo Angle - Continuously read and print motor angle for a given motor ID
"""

import sys
import os
import time
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped_loco_lowlevel.recoil.core import ST3215Driver


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
    # Default motor ID, can be changed via command line argument
    motor_id = 4
    if len(sys.argv) > 1:
        try:
            motor_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid motor ID: {sys.argv[1]}. Using default motor ID: {motor_id}")
    
    print(f"Reading motor angle for Motor ID: {motor_id}")
    print("Move the motor physically to see angle changes...")
    print("Press Ctrl+C to exit\n")
    
    try:
        driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        
        # Continuous loop to read and print motor angle
        while True:
            angle = get_motor_angle(driver, motor_id)
            if angle is not None:
                print(f"Motor {motor_id} angle: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
            else:
                print(f"Failed to read angle for Motor {motor_id}")
            
            time.sleep(0.1)  # Read every 100ms
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        try:
            driver.close()
        except:
            pass
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
