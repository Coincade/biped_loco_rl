#!/usr/bin/env python3
"""
Teleop Control - Synchronize servo motor movements with RobStride motors
Reads angle changes from servo motors and applies them to RobStride motors
"""

import sys
import os
import time
import numpy as np
import argparse

# Parse our arguments first before importing modules that use get_args()
parser = argparse.ArgumentParser(description='Teleop Control: synchronize servo arm with RobStride arm')
# Parse known args - this will leave unknown args for other modules to handle
args, remaining_argv = parser.parse_known_args()

# Modify sys.argv to only include remaining args for get_args()
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]] + remaining_argv

# Add the parent directory to the path to import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from biped_loco_lowlevel.recoil.core import ST3215Driver

# Add robstride_control to path for imports
# Path structure: scripts/ -> biped_loco_lowlevel/ -> source/ -> robstride_control/
source_dir = os.path.dirname(os.path.dirname(parent_dir))  # Go up to source/
robstride_control_dir = os.path.join(source_dir, 'robstride_control')
sys.path.append(robstride_control_dir)

try:
    from robstride_dynamics import RobstrideBus, Motor, ParameterType
except ImportError:
    try:
        from robstride_dynamics.bus import RobstrideBus, Motor
        from robstride_dynamics.protocol import ParameterType
    except ImportError as e:
        print(f"❌ Failed to import RobStride SDK: {e}")
        print(f"   Make sure robstride_control is installed and in the correct location.")
        print(f"   Expected path: {robstride_control_dir}")
        sys.exit(1)

# Keep sys.argv modified so get_args() doesn't see our arguments

# Motor IDs: 2, 4, 6, 8, 10 for both servo (leader) and RobStride (follower) motors
SERVO_MOTOR_IDS = [2, 4, 6, 8, 10]
ROBSTRIDE_MOTOR_IDS = [2, 4, 6, 8, 10]

# Motor model mapping (from robostride_control.py)
MOTOR_MODEL_MAP = {
    2: "rs-03",
    4: "rs-03",
    6: "rs-06",
    8: "rs-06",
    10: "rs-02"
}

# Per-motor control parameters (from robostride_control.py)
MOTOR_KP = {
    2: 100.0,
    4: 100.0,
    6: 28.0,
    8: 28.0,
    10: 28.0
}

MOTOR_KD = {
    2: 18.0,
    4: 18.0,
    6: 6.0,
    8: 5.0,
    10: 6.0
}

MOTOR_TORQUE_LIMIT = {
    2: 4.0,
    4: 6.0,
    6: 3.0,
    8: 4.0,
    10: 4.0
}


def servo_units_to_rad(servo_units):
    """
    Convert servo units (0-4095) to radians.
    Inverse of rad_to_servo_units function.
    """
    scale = 4095 / (2 * np.pi)
    return (servo_units - 2046) / scale


def get_motor_angle(driver, motor_id):
    """
    Read the current angle of a servo motor in radians.
    
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


def initialize_robstride_motors(bus, motor_ids, motor_names):
    """
    Initialize RobStride motors with appropriate settings.
    
    Args:
        bus: RobstrideBus instance
        motor_ids: List of motor IDs
        motor_names: List of motor names
    
    Returns:
        bool: True if initialization successful
    """
    try:
        print("⚙️ Setting MIT mode control parameters...")
        for motor_id, motor_name in zip(motor_ids, motor_names):
            bus.write(motor_name, ParameterType.POSITION_KP, MOTOR_KP[motor_id])
            time.sleep(0.1)
            bus.write(motor_name, ParameterType.VELOCITY_KP, MOTOR_KD[motor_id])
            time.sleep(0.1)
            bus.write(motor_name, ParameterType.TORQUE_LIMIT, MOTOR_TORQUE_LIMIT[motor_id])
            time.sleep(0.1)
        
        print("⚙️ Setting mode to MIT (Mode 0)...")
        for motor_name in motor_names:
            bus.write(motor_name, ParameterType.MODE, 0)
            time.sleep(0.1)
        
        time.sleep(0.2)
        
        # Verify mode
        for motor_name in motor_names:
            mode = bus.read(motor_name, ParameterType.MODE)
            if mode != 0:
                print(f"⚠️ Warning: Motor {motor_name} mode is {mode}, expected 0")
                bus.disable(motor_name)
                time.sleep(0.1)
                bus.write(motor_name, ParameterType.MODE, 0)
                time.sleep(0.1)
                bus.enable(motor_name)
                time.sleep(0.1)
        
        print("✅ RobStride motors initialized")
        return True
    except Exception as e:
        print(f"❌ Error initializing RobStride motors: {e}")
        return False


def main():
    print("=" * 70)
    print("Teleop Control - Synchronize Servo Arm with RobStride Arm")
    print("=" * 70)
    print(f"Servo motor IDs (leader): {SERVO_MOTOR_IDS}")
    print(f"RobStride motor IDs (follower): {ROBSTRIDE_MOTOR_IDS}")
    print("=" * 70)
    print("Initializing and setting zero position...\n")
    
    servo_driver = None
    robstride_bus = None
    motor_names = [f"motor_{motor_id}" for motor_id in ROBSTRIDE_MOTOR_IDS]
    
    try:
        # Initialize servo driver
        servo_driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        print("✓ Servo driver initialized")
        
        # Read initial angles from servo motors
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
        for servo_id, robstride_id in zip(SERVO_MOTOR_IDS, ROBSTRIDE_MOTOR_IDS):
            servo_angle = initial_servo_angles[servo_id]
            print(f"  Motor {servo_id}->{robstride_id}: Servo={servo_angle:.4f} rad ({np.degrees(servo_angle):.2f}°)")
        
        # Initialize RobStride bus and motors
        print("\n🔍 Connecting to RobStride CAN bus (can0)...")
        
        # Define motors with model mapping
        motors = {}
        for motor_id, motor_name in zip(ROBSTRIDE_MOTOR_IDS, motor_names):
            model = MOTOR_MODEL_MAP.get(motor_id, "rs-02")
            motors[motor_name] = Motor(id=motor_id, model=model)
        
        # Calibration parameters
        calibration = {}
        for motor_name in motor_names:
            calibration[motor_name] = {"direction": 1, "homing_offset": 0.0}
        
        # Create and connect bus
        robstride_bus = RobstrideBus("can0", motors, calibration)
        robstride_bus.connect(handshake=True)
        print("✓ RobStride bus connected")
        
        # Enable all motors
        print("\n⚡ Activating RobStride motors...")
        for motor_name in motor_names:
            robstride_bus.enable(motor_name)
            time.sleep(0.5)
        
        # Initialize motors with control parameters
        if not initialize_robstride_motors(robstride_bus, ROBSTRIDE_MOTOR_IDS, motor_names):
            print("Failed to initialize RobStride motors")
            return 1
        
        # Set zero position by sending initial servo angles to RobStride motors
        print("\nSetting zero position (moving RobStride motors to servo positions)...")
        for servo_id, robstride_id, motor_name in zip(SERVO_MOTOR_IDS, ROBSTRIDE_MOTOR_IDS, motor_names):
            target_angle = initial_servo_angles[servo_id]
            # Apply sign inversion (except motor 4, based on teleop.py pattern)
            if servo_id == 4:
                final_target = target_angle
            else:
                final_target = -target_angle
            
            robstride_bus.write_operation_frame(
                motor_name,
                final_target,
                MOTOR_KP[robstride_id],
                MOTOR_KD[robstride_id],
                0.0,  # target_velocity
                0.0   # torque_feedforward
            )
            time.sleep(0.1)
        
        time.sleep(1.0)  # Give motors time to move to initial positions
        
        print("\nZero position set. Starting teleop control...")
        print("Move the servo arm to control the RobStride arm.")
        print("Press Ctrl+C to exit\n")
        
        # Main control loop
        rate = 50.0  # 50 Hz update rate
        sleep_time = 1.0 / rate
        
        while True:
            try:
                # Read current angles from all servo motors
                motor_updates = []
                all_read = True
                
                for servo_id, robstride_id, motor_name in zip(SERVO_MOTOR_IDS, ROBSTRIDE_MOTOR_IDS, motor_names):
                    current_servo_angle = get_motor_angle(servo_driver, servo_id)
                    
                    if current_servo_angle is not None:
                        # Calculate change from initial servo angle
                        servo_angle_change = current_servo_angle - initial_servo_angles[servo_id]
                        
                        # Calculate target RobStride angle: initial servo position + change
                        # Apply sign inversion (except motor 4, based on teleop.py pattern)
                        if servo_id == 4:
                            target_robstride_angle = -(initial_servo_angles[servo_id] + servo_angle_change)
                        elif servo_id == 8:
                            target_robstride_angle = initial_servo_angles[servo_id] + servo_angle_change + 1.57    
                        else:
                            target_robstride_angle = initial_servo_angles[servo_id] + servo_angle_change
                        
                        # Send command to RobStride motor
                        robstride_bus.write_operation_frame(
                            motor_name,
                            target_robstride_angle,
                            MOTOR_KP[robstride_id],
                            MOTOR_KD[robstride_id],
                            0.0,  # target_velocity
                            0.0   # torque_feedforward
                        )
                    else:
                        all_read = False
                
                if not all_read:
                    print("Warning: Failed to read some servo angles")
                
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if robstride_bus is not None:
            try:
                print("\n🛑 Stopping RobStride motors...")
                # Return to zero position
                for motor_name in motor_names:
                    robstride_bus.write_operation_frame(
                        motor_name,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    )
                time.sleep(1.0)
                
                # Disable all motors
                print("🚫 Disabling motors...")
                for motor_name in motor_names:
                    robstride_bus.disable(motor_name)
                
                robstride_bus.disconnect()
                print("✓ RobStride bus disconnected")
            except Exception as e:
                print(f"Warning: Error during RobStride cleanup: {e}")
        
        if servo_driver is not None:
            try:
                servo_driver.close()
                print("✓ Servo driver closed")
            except:
                pass
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
