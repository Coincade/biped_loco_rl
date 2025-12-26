#!/usr/bin/env python3
"""
Policy Servo Runner - Move servos with policy-derived joint angles

This file continuously moves your servos using joint angles computed from the biped policy.
It's like move_servos.py but with dynamic policy values instead of static angles.

Usage:
    python policy_servo_runner.py

The servos will move continuously based on policy commands (forward, turn, etc.)
"""

import sys
import os
import time
import numpy as np
import threading
import torch
import yaml
from typing import Optional

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your servo driver
from biped_loco_lowlevel.recoil.core import ST3215Driver


class PolicyServoRunner:
    """
    Simple servo runner that moves servos with policy-derived joint angles.
    
    This class simulates the policy output and sends joint angles to your servos
    in real-time, just like move_servos.py but with dynamic values.
    """
    
    def __init__(self, port="/dev/ttyACM0", baudrate=1000000):
        """
        Initialize the policy servo runner.
        
        Args:
            port (str): Serial port for servo communication
            baudrate (int): Baud rate for servo communication
        """
        # Servo driver setup
        self.port = port
        self.baudrate = baudrate
        self.driver = None
        
        # Motor IDs (10 servos for 10 joints)
        self.motor_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # Joint names for reference
        self.joint_names = [
            "left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle", "left_foot",
            "right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle", "right_foot"
        ]
        
        # Default robot pose (standing position) - SAME AS IN biped.py
        self.default_pos = np.array([
            0.0,    # left_hip_pitch_joint   - Forward/backward hip movement
            0.0,    # left_hip_roll_joint    - Side-to-side hip movement  
            -0.3,   # left_knee_roll_joint   - Knee bending (slightly bent)
            0.6,    # left_ankle_roll_joint  - Ankle angle (compensating for knee)
            0.0,    # left_foot_joint        - Foot rotation
            0.0,    # right_hip_pitch_joint  - Forward/backward hip movement
            0.0,    # right_hip_roll_joint   - Side-to-side hip movement
            -0.3,   # right_knee_roll_joint  - Knee bending (slightly bent)
            0.6,    # right_ankle_roll_joint - Ankle angle (compensating for knee)
            0.0     # right_foot_joint       - Foot rotation
        ])
        
        # Action scaling - SAME AS IN biped.py
        self._action_scale = 0.005
        
        # Control parameters
        self.control_frequency = 50  # Hz (50Hz = 20ms per cycle)
        self.dt = 1.0 / self.control_frequency
        
        # Current state
        self.current_command = np.array([0.0, 0.0, 0.0])  # [vx, vy, wz]
        self.is_running = False
        self.control_thread = None
        
        # Policy and environment config
        self.policy = None
        self.env_config = None
        self._previous_action = np.zeros(10)
        
        # Demo sequence with forward dummy velocities
        self.demo_commands = [
            ([0.0, 0.0, 0.0], "STANDING STILL", 3.0),
            ([1.0, 0.0, 0.0], "MOVE FORWARD", 5.0),
            ([0.8, 0.0, 0.0], "FAST FORWARD", 4.0),
            ([0.5, 0.0, 0.0], "SLOW FORWARD", 4.0),
            ([1.2, 0.0, 0.0], "VERY FAST FORWARD", 3.0),
            ([0.0, 0.0, 0.0], "STANDING STILL", 2.0),
        ]
        
        print("Policy Servo Runner initialized!")
        print(f"Default pose: {self.default_pos}")
        print(f"Action scale: {self._action_scale}")
        print(f"Control frequency: {self.control_frequency}Hz")
        
        # Load real policy
        self.load_real_policy()
    
    def load_real_policy(self):
        """
        Load your actual trained policy from .pt file and environment config.
        
        Make sure you have these files in the same directory:
        - policy.pt (your trained policy)
        - env.yaml (environment configuration)
        """
        try:
            # Load policy file
            if os.path.exists("/home/yash/projects/robotics/biped_loco/configs/policy_latest.yaml"):
                self.policy = torch.jit.load("/home/yash/projects/robotics/biped_loco/logs/rsl_rl/biped_loco_flat/2025-10-06_09-09-57/exported/policy.pt")
                self.policy.eval()
                print("✓ Real policy loaded from policy.pt")
            else:
                print("❌ ERROR: policy.pt not found!")
                print("Please place your trained policy file (policy.pt) in the same directory")
                self.policy = None
            
            # Load environment config
            if os.path.exists("/home/yash/projects/robotics/biped_loco/configs/policy_latest.yaml"):
                with open("/home/yash/projects/robotics/biped_loco/configs/policy_latest.yaml", 'r') as f:
                    self.env_config = yaml.safe_load(f)
                print("✓ Environment config loaded from env.yaml")
            else:
                print("❌ ERROR: env.yaml not found!")
                print("Please place your environment config file (env.yaml) in the same directory")
                self.env_config = None
                
        except Exception as e:
            print(f"❌ Error loading policy: {e}")
            self.policy = None
            self.env_config = None
    
    def _compute_real_observation(self, command):
        """
        Compute observation vector for real policy (42 elements).
        
        This creates the observation that your trained policy expects.
        For hardware, you'll need to implement sensor feedback.
        
        Args:
            command (np.ndarray): Robot command [vx, vy, wz]
            
        Returns:
            np.ndarray: Observation vector (42 elements)
        """
        obs = np.zeros(42)
        
        # Command (velocity_commands) - 3 elements: [v_x, v_y, w_z]
        obs[9:12] = command
        
        # Joint positions relative to default pose - 10 elements
        current_joint_pos = self.get_current_joint_positions()
        obs[12:22] = current_joint_pos - self.default_pos
        
        # Joint velocities - 10 elements
        current_joint_vel = self.get_current_joint_velocities()
        obs[22:32] = current_joint_vel
        
        # Previous actions - 10 elements
        obs[32:42] = self._previous_action
        
        # For hardware, you'll need to implement these from sensors:
        # obs[:3] = base linear velocity (from IMU/odometry)
        # obs[3:6] = base angular velocity (from IMU)
        # obs[6:9] = gravity vector (from IMU orientation)
        
        return obs
    
    def get_current_joint_positions(self):
        """
        Get current joint positions from your robot.
        
        For now, returns default positions. In real implementation,
        you would read from encoders or servo feedback.
        """
        # TODO: Implement real joint position feedback
        return self.default_pos.copy()
    
    def get_current_joint_velocities(self):
        """
        Get current joint velocities from your robot.
        
        For now, returns zeros. In real implementation,
        you would compute from position differences or read from sensors.
        """
        # TODO: Implement real joint velocity feedback
        return np.zeros(10)
    
    def get_real_policy_action(self, command):
        """
        Get action from your actual trained policy.
        
        Args:
            command (np.ndarray): Robot command [vx, vy, wz]
            
        Returns:
            np.ndarray: Policy action (10 joint commands)
        """
        if self.policy is None:
            print("❌ No policy loaded! Cannot get policy action.")
            return np.zeros(10)
        
        try:
            # Create observation vector
            obs = self._compute_real_observation(command)
            
            # Run policy
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).view(1, -1).float()
                action = self.policy(obs_tensor).detach().view(-1).numpy()
            
            return action
            
        except Exception as e:
            print(f"❌ Error running real policy: {e}")
            return np.zeros(10)
    
    def rad_to_servo_units(self, angle_rad):
        """
        Convert joint angle from radians to servo units.
        
        This uses the same conversion as your move_servos.py file.
        
        Args:
            angle_rad (float): Joint angle in radians
            
        Returns:
            int: Servo position in servo units (0-4095)
        """
        scale = 4095 / (2 * np.pi)
        return int(angle_rad * scale + 2046)
    
    
    def compute_final_joint_positions(self, command):
        """
        Compute final joint positions using the same formula as biped.py:
        final_joint_pos = self.default_pos + (self.action * self._action_scale)
        
        Args:
            command (np.ndarray): Robot command [vx, vy, wz]
            
        Returns:
            np.ndarray: Final joint positions in radians
        """
        # Get policy action (real or simulated)
        policy_action = self.get_real_policy_action(command)
        
        # Store for next observation
        self._previous_action = policy_action.copy()
        
        # Compute final joint positions (SAME AS biped.py line 184)
        final_joint_pos = self.default_pos + (policy_action * self._action_scale)
        
        return final_joint_pos, policy_action
    
    def send_servo_commands(self, joint_positions_rad):
        """
        Send joint positions to servo motors.
        
        Args:
            joint_positions_rad (np.ndarray): Joint positions in radians
        """
        if not self.driver:
            return
        
        try:
            # Convert joint positions to servo units
            servo_positions = [self.rad_to_servo_units(angle) for angle in joint_positions_rad]
            
            # Send commands to all servos
            for i, motor_id in enumerate(self.motor_ids):
                self.driver.move_servo(motor_id, servo_positions[i], speed=255, acc=50)
            
        except Exception as e:
            print(f"Error sending servo commands: {e}")
    
    def control_loop(self):
        """
        Main control loop that runs at specified frequency.
        
        This loop:
        1. Computes final joint positions from policy action
        2. Sends commands to servo motors
        3. Maintains timing
        """
        print("Starting control loop...")
        
        cycle_count = 0
        while self.is_running:
            start_time = time.time()
            
            # Compute final joint positions (same as biped.py line 184)
            final_joint_pos, policy_action = self.compute_final_joint_positions(self.current_command)
            
            # Send commands to servos
            self.send_servo_commands(final_joint_pos)
            
            # Debug output (print every 50 cycles = 1 second)
            if cycle_count % 50 == 0:
                servo_positions = [self.rad_to_servo_units(angle) for angle in final_joint_pos]
                if self.policy is not None:
                    print(f"Command: {self.current_command}")
                    print(f"Policy action: {policy_action}")
                    print(f"Final joint pos (rad): {final_joint_pos}")
                    print(f"Servo positions: {servo_positions}")
                    print("-" * 50)
                else:
                    print("❌ No policy loaded - servos will not move")
                    print(f"Command: {self.current_command}")
                    print("-" * 50)
            
            cycle_count += 1
            
            # Maintain timing
            elapsed = time.time() - start_time
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: Control loop running slow! Elapsed: {elapsed:.3f}s, Target: {self.dt:.3f}s")
    
    def initialize_servos(self):
        """
        Initialize servo communication and move to default pose.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            print("Initializing servo driver...")
            self.driver = ST3215Driver(port=self.port, baudrate=self.baudrate)
            
            print("Moving to default pose...")
            self.send_servo_commands(self.default_pos)
            time.sleep(2.0)  # Wait for servos to reach position
            
            print("Servo initialization complete!")
            return True
            
        except Exception as e:
            print(f"Error initializing servos: {e}")
            return False
    
    def start_control_loop(self):
        """
        Start the control loop in a separate thread.
        """
        if self.is_running:
            print("Control loop already running!")
            return
        
        self.is_running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        print("Control loop started!")
    
    def stop_control_loop(self):
        """
        Stop the control loop.
        """
        self.is_running = False
        if self.control_thread:
            self.control_thread.join()
        print("Control loop stopped!")
    
    def run_demo(self):
        """
        Run a demo sequence with different commands.
        """
        if self.policy is None:
            print("❌ Cannot run demo - no policy loaded!")
            print("Please place policy.pt and env.yaml files in the same directory")
            return
        
        print("Starting demo sequence...")
        
        for command, name, duration in self.demo_commands:
            print(f"\n{'='*20} {name} {'='*20}")
            print(f"Command: {command}")
            print(f"Duration: {duration} seconds")
            
            # Set command
            self.current_command = np.array(command)
            
            # Let it run for the specified duration
            time.sleep(duration)
        
        print("\nDemo sequence complete!")
    
    def close(self):
        """
        Close servo communication and cleanup.
        """
        self.stop_control_loop()
        if self.driver:
            self.driver.close()
        print("Policy servo runner closed!")


def main():
    """
    Main function - run the policy servo demo.
    """
    print("Policy Servo Runner - Moving servos with policy-derived joint angles")
    print("=" * 70)
    print("REQUIRED FILES:")
    print("- policy.pt (your trained policy)")
    print("- env.yaml (environment configuration)")
    print("- biped_loco_lowlevel (servo driver)")
    print("=" * 70)
    
    # Create policy servo runner
    runner = PolicyServoRunner()
    
    try:
        # Initialize servos
        if not runner.initialize_servos():
            print("Failed to initialize servos!")
            return 1
        
        # Start control loop
        runner.start_control_loop()
        
        # Run demo sequence
        runner.run_demo()
        
        # Keep running
        print("\nDemo complete! Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        runner.close()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
