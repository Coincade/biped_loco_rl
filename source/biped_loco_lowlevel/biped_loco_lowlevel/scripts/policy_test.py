#!/usr/bin/env python3
"""
Simple Policy Servos - Move servos with policy values (like move_servos.py)

This is a simple version that continuously moves your servos using policy-derived
joint angles. It's like your move_servos.py but with dynamic policy values.

Usage:
    python simple_policy_servos.py

The servos will move continuously with policy-derived joint angles.
"""

import sys
import os
import time
import numpy as np

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your servo driver
from biped_loco_lowlevel.recoil.core import ST3215Driver


def rad_to_servo_units(angle):
    """
    Convert joint angle from radians to servo units.
    Same function as in your move_servos.py
    """
    scale = 4095/(2*np.pi)
    return int(angle*scale+2046)


def get_policy_joint_angles(command, time_step):
    """
    Get joint angles from policy simulation.
    
    This simulates what your biped policy would output for different commands.
    
    Args:
        command (list): [vx, vy, wz] - forward, lateral, turning velocities
        time_step (float): Current time step for animation
        
    Returns:
        list: Joint angles in radians [10 joints]
    """
    # Default robot pose (standing position) - SAME AS IN biped.py
    default_pos = np.array([
        0.0,    # left_hip_pitch_joint   - Forward/backward hip movement
        0.0,    # left_hip_roll_joint    - Side-to-side hip movement  
        0.0,   # left_knee_roll_joint   - Knee bending (slightly bent)
        0.0,    # left_ankle_roll_joint  - Ankle angle (compensating for knee)
        0.0,    # left_foot_joint        - Foot rotation
        0.0,    # right_hip_pitch_joint  - Forward/backward hip movement
        0.0,    # right_hip_roll_joint   - Side-to-side hip movement
        0.0,   # right_knee_roll_joint  - Knee bending (slightly bent)
        0.0,    # right_ankle_roll_joint - Ankle angle (compensating for knee)
        0.0     # right_foot_joint       - Foot rotation
    ])
    
    # Action scaling - SAME AS IN biped.py
    action_scale = 1
    
    # Simulate policy action based on command
    if command[0] > 0.5:  # Forward movement
        # Policy wants to move forward - create walking motion
        walking_phase = np.sin(time_step * 2.0)  # Walking cycle
        
        action = np.array([
            0.1 * walking_phase,     # left_hip_pitch  - forward/back swing
            0.0,                     # left_hip_roll   - no side movement
            -0.2 + 0.1 * walking_phase,  # left_knee    - knee bend for step
            0.3 - 0.1 * walking_phase,   # left_ankle   - ankle adjustment
            0.0,                     # left_foot       - no foot rotation
            0.1 * (-walking_phase),  # right_hip_pitch - opposite phase
            0.0,                     # right_hip_roll  - no side movement
            -0.2 - 0.1 * walking_phase,  # right_knee   - opposite knee
            0.3 + 0.1 * walking_phase,   # right_ankle  - opposite ankle
            0.0                      # right_foot      - no foot rotation
        ])
    elif command[2] > 0.3:  # Turn left
        # Policy wants to turn left - lean left
        action = np.array([
            0.0,                     # left_hip_pitch  - no forward movement
            0.2,                     # left_hip_roll   - lean left
            0.0,                     # left_knee       - no knee change
            0.0,                     # left_ankle      - no ankle change
            0.0,                     # left_foot       - no foot rotation
            0.0,                     # right_hip_pitch - no forward movement
            -0.2,                    # right_hip_roll  - lean right (counterbalance)
            0.0,                     # right_knee      - no knee change
            0.0,                     # right_ankle     - no ankle change
            0.0                      # right_foot      - no foot rotation
        ])
    elif command[2] < -0.3:  # Turn right
        # Policy wants to turn right - lean right
        action = np.array([
            0.0,                     # left_hip_pitch  - no forward movement
            -0.2,                    # left_hip_roll   - lean right
            0.0,                     # left_knee       - no knee change
            0.0,                     # left_ankle      - no ankle change
            0.0,                     # left_foot       - no foot rotation
            0.0,                     # right_hip_pitch - no forward movement
            0.2,                     # right_hip_roll  - lean left (counterbalance)
            0.0,                     # right_knee      - no knee change
            0.0,                     # right_ankle     - no ankle change
            0.0                      # right_foot      - no foot rotation
        ])
    else:  # Standing still
        # Policy wants to maintain balance - small adjustments
        balance_phase = np.sin(time_step * 0.5)  # Slow balance adjustments
        
        action = np.array([
            0.0,                     # left_hip_pitch  - no movement
            0.0,                     # left_hip_roll   - no movement
            0.0,                     # left_knee       - no movement
            0.0,                     # left_ankle      - no movement
            0.0,                     # left_foot       - no movement
            0.0,                     # right_hip_pitch - no movement
            0.0,                     # right_hip_roll  - no movement
            0.0,                     # right_knee      - no movement
            0.0,                     # right_ankle     - no movement
            0.0                      # right_foot      - no movement
        ])
    
    # Compute final joint positions (SAME AS biped.py line 184)
    final_joint_pos = default_pos + (action * action_scale)
    
    return final_joint_pos.tolist()


def main():
    """
    Main function - continuously move servos with policy values.
    """
    print("Simple Policy Servos - Moving servos with policy-derived joint angles")
    print("=" * 70)
    
    # Motor IDs (1-10)
    MOTORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Joint names for reference
    joint_names = [
        "left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle", "left_foot",
        "right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle", "right_foot"
    ]
    
    # Demo commands sequence
    demo_commands = [
        ([0.0, 0.0, 0.0], "STANDING STILL", 3.0),
        ([1.0, 0.0, 0.0], "MOVE FORWARD", 5.0),
        ([0.0, 0.0, 0.0], "STANDING STILL", 2.0),
        ([0.0, 0.0, 0.5], "TURN LEFT", 4.0),
        ([0.0, 0.0, 0.0], "STANDING STILL", 2.0),
        ([0.0, 0.0, -0.5], "TURN RIGHT", 4.0),
        ([0.0, 0.0, 0.0], "STANDING STILL", 2.0),
        ([0.5, 0.0, 0.0], "SLOW FORWARD", 4.0),
        ([0.0, 0.0, 0.0], "STANDING STILL", 2.0),
    ]
    
    try:
        # Initialize servo driver
        print("Initializing servo driver...")
        driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        
        # Move to default pose first
        print("Moving to default pose...")
        default_angles = get_policy_joint_angles([0.0, 0.0, 0.0], 0.0)
        default_positions = [rad_to_servo_units(angle) for angle in default_angles]
        
        for i, motor_id in enumerate(MOTORS):
            driver.move_servo(motor_id, default_positions[i], speed=100, acc=50)
            time.sleep(0.05)
        
        time.sleep(2.0)  # Wait for servos to reach position
        print("Default pose set!")
        
        # Run demo sequence
        print("\nStarting demo sequence...")
        time_step = 0.0
        
        for command, name, duration in demo_commands:
            print(f"\n{'='*20} {name} {'='*20}")
            print(f"Command: {command}")
            print(f"Duration: {duration} seconds")
            
            # Run this command for the specified duration
            start_time = time.time()
            while time.time() - start_time < duration:
                # Get policy joint angles for current time step
                joint_angles = get_policy_joint_angles(command, time_step)
                
                # Convert to servo positions
                servo_positions = [rad_to_servo_units(angle) for angle in joint_angles]
                
                # Send to all servos
                for i, motor_id in enumerate(MOTORS):
                    driver.move_servo(motor_id, servo_positions[i], speed=255, acc=50)
                
                # Debug output every 50 cycles (1 second at 50Hz)
                if int(time_step * 50) % 50 == 0:
                    print(f"Time: {time_step:.1f}s")
                    print(f"Joint angles (rad): {[f'{angle:.3f}' for angle in joint_angles]}")
                    print(f"Servo positions: {servo_positions}")
                    print(f"Joint names: {joint_names}")
                    print("-" * 50)
                
                # Update time step and wait
                time_step += 0.02  # 20ms timestep (50Hz)
                time.sleep(0.02)
        
        print("\nDemo sequence complete!")
        print("Press Ctrl+C to exit...")
        
        # Keep running with standing still
        while True:
            joint_angles = get_policy_joint_angles([0.0, 0.0, 0.0], time_step)
            servo_positions = [rad_to_servo_units(angle) for angle in joint_angles]
            
            for i, motor_id in enumerate(MOTORS):
                driver.move_servo(motor_id, servo_positions[i], speed=255, acc=50)
            
            time_step += 0.02
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
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
