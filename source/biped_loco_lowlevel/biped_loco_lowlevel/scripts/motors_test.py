#!/usr/bin/env python3
"""
Simple Motor Test - Move all motors to position 1023
"""

import sys
import os
import time
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped_loco_lowlevel.recoil.core import ST3215Driver

def rad_to_servo_units(angle):
    scale = 4095/(2*np.pi)
    return int(angle*scale+2046)

def main():
    MOTOR_ANGLE_1 = 0
    MOTOR_ANGLE_2 = 0
    MOTOR_ANGLE_3 = 0
    MOTOR_ANGLE_4 = 0
    MOTOR_ANGLE_5 = 0
    MOTOR_ANGLE_6 = 0
    MOTOR_ANGLE_7 = 0
    MOTOR_ANGLE_8 = 0
    MOTOR_ANGLE_9 = 0
    MOTOR_ANGLE_10 = 0
    
    TEST_POSITION_1 = rad_to_servo_units(MOTOR_ANGLE_1)
    TEST_POSITION_2 = rad_to_servo_units(MOTOR_ANGLE_2)
    TEST_POSITION_3 = rad_to_servo_units(MOTOR_ANGLE_3)
    TEST_POSITION_4 = rad_to_servo_units(MOTOR_ANGLE_4)
    TEST_POSITION_5 = rad_to_servo_units(MOTOR_ANGLE_5)
    TEST_POSITION_6 = rad_to_servo_units(MOTOR_ANGLE_6)
    TEST_POSITION_7 = rad_to_servo_units(MOTOR_ANGLE_7)
    TEST_POSITION_8 = rad_to_servo_units(MOTOR_ANGLE_8)
    TEST_POSITION_9 = rad_to_servo_units(MOTOR_ANGLE_9)
    TEST_POSITION_10 = rad_to_servo_units(MOTOR_ANGLE_10)
    
    # Motor IDs (1-10)
    MOTORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    try:
        driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        
        # Move all motors to test position
        positions = [TEST_POSITION_1, TEST_POSITION_2, TEST_POSITION_3, TEST_POSITION_4, TEST_POSITION_5,
                    TEST_POSITION_6, TEST_POSITION_7, TEST_POSITION_8, TEST_POSITION_9, TEST_POSITION_10]
        
        for i, motor_id in enumerate(MOTORS):
            driver.move_servo(motor_id, positions[i], speed=255, acc=50)
            time.sleep(0.1)
        
        time.sleep(2.0)  # Wait for motors to reach position
        
        # Keep connection alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        pass
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
