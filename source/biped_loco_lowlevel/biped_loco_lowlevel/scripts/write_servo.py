#!/usr/bin/env python3
"""
Write Servo Angle - Move a servo motor to a specified angle
"""

import sys
import os
import time
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped_loco_lowlevel.recoil.core import ST3215Driver


def rad_to_servo_units(angle):
    """
    Convert angle in radians to servo units (0-4095).
    
    Args:
        angle: Angle in radians
    
    Returns:
        int: Servo position in units
    """
    scale = 4095 / (2 * np.pi)
    return int(angle * scale + 2046)


def set_motor_angle(driver, motor_id, angle, speed=255, acc=50):
    """
    Move a motor to the specified angle in radians.
    
    Args:
        driver: ST3215Driver instance
        motor_id: Motor ID (1-10)
        angle: Target angle in radians
        speed: Servo speed (default: 255)
        acc: Servo acceleration (default: 50)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        position = rad_to_servo_units(angle)
        driver.move_servo(motor_id, position, speed=speed, acc=acc)
        return True
    except Exception as e:
        print(f"Error moving motor {motor_id}: {e}")
        return False


def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python write_servo.py <motor_id> <angle_radians> [speed] [acc]")
        print("Example: python write_servo.py 4 1.57")
        print("Example: python write_servo.py 4 1.57 255 50")
        return 1
    
    try:
        motor_id = int(sys.argv[1])
        angle = float(sys.argv[2])
        speed = int(sys.argv[3]) if len(sys.argv) > 3 else 255
        acc = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    except ValueError as e:
        print(f"Error: Invalid argument - {e}")
        print("Usage: python write_servo.py <motor_id> <angle_radians> [speed] [acc]")
        return 1
    
    print(f"Moving Motor {motor_id} to angle: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
    print(f"Speed: {speed}, Acceleration: {acc}")
    
    try:
        driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        
        # Move motor to target angle
        if set_motor_angle(driver, motor_id, angle, speed=speed, acc=acc):
            print(f"✓ Command sent to Motor {motor_id}")
            print("Waiting for motor to reach position...")
            time.sleep(2.0)  # Wait for motor to reach position
            
            # Optionally read back the position to verify
            position = driver.read_position(motor_id)
            if position is not None:
                scale = 4095 / (2 * np.pi)
                actual_angle = (position - 2046) / scale
                print(f"Actual position: {actual_angle:.4f} rad ({np.degrees(actual_angle):.2f}°)")
            else:
                print("Could not read back position")
        else:
            print(f"✗ Failed to move Motor {motor_id}")
            return 1
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
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
