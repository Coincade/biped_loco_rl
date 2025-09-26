# robot.py
# Robot class managing 10 servo motors for the biped

import numpy as np
import yaml
from biped_loco_lowlevel.recoil.core import ST3215Driver


class State:
    IDLE = 0
    RL_INIT = 1
    RL_RUNNING = 2


class BipedLoco:
    def __init__(self, port="/dev/ttyACM0", baudrate=1000000, config_path="configs/policy_latest.yaml"):
        # Require hardware connection - no mock mode
        print(f"Attempting to connect to robot hardware on port: {port}")
        try:
            self.driver = ST3215Driver(port=port, baudrate=baudrate)
            print("✅ Successfully connected to robot hardware")
        except Exception as e:
            print(f"❌ Failed to connect to robot hardware: {e}")
            print("\nTroubleshooting steps:")
            print("1. Check if robot is connected to USB port")
            print("2. Verify correct port (current: {})".format(port))
            print("   - List available ports: ls -la /dev/tty*")
            print("3. Check permissions:")
            print("   - Run: sudo usermod -a -G dialout $USER")
            print("   - Then logout and login again")
            print("4. Check if port is in use:")
            print("   - Run: lsof /dev/ttyACM0")
            print("5. Try different port if available")
            raise RuntimeError(f"Cannot connect to robot hardware: {e}")

        # Define 10 joints with unique IDs
        self.joints = {
            1: "left_hip_pitch_joint",
            2: "left_hip_roll_joint",
            3: "left_knee_roll_joint",
            4: "left_ankle_roll_joint",
            5: "left_foot_joint",
            6: "right_hip_pitch_joint",
            7: "right_hip_roll_joint",
            8: "right_knee_roll_joint",
            9: "right_ankle_roll_joint",
            10: "right_foot_joint",
        }

        self.num_joints = len(self.joints)

        # Initialize joint state arrays
        self.joint_targets = np.zeros(self.num_joints, dtype=np.float32)
        self.joint_positions = np.zeros(self.num_joints, dtype=np.float32)

        self.state = State.IDLE
        self.next_state = State.IDLE

        # Load initial positions from policy config file
        self.rl_init_positions = self._load_initial_positions(config_path)
        
        # Initialize observation components
        self.prev_actions = np.zeros(self.num_joints, dtype=np.float32)
        self.joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        self.prev_joint_positions = np.zeros(self.num_joints, dtype=np.float32)
    
    def _load_initial_positions(self, config_path):
        """
        Load initial joint positions from policy config file.
        Converts from normalized coordinates (-1 to +1) to servo units (0-4096).
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Get normalized joint positions from config
            normalized_positions = config.get('default_joint_positions', [])
            
            if len(normalized_positions) != self.num_joints:
                print(f"Warning: Expected {self.num_joints} joint positions, got {len(normalized_positions)}")
                print("Using default neutral positions (2048)")
                return np.array([2048] * self.num_joints, dtype=np.float32)
            
            # Convert from normalized (-1 to +1) to servo units (0-4096)
            # Formula: servo_unit = (normalized * 2048) + 2048
            scale = 4096 / (2 * 3.1416)
            servo_positions = np.array([
                (pos * scale) + 2048 for pos in normalized_positions
            ], dtype=np.float32)
            
            print(f"Loaded initial positions from {config_path}:")
            for i, (joint_name, pos) in enumerate(zip(self.joints.values(), servo_positions)):
                print(f"  {joint_name}: {pos:.1f} (normalized: {normalized_positions[i]:.3f})")
            
            return servo_positions
            
        except Exception as e:
            print(f"Warning: Could not load initial positions from {config_path}: {e}")
            print("Using default neutral positions (2048)")
            return np.array([2048] * self.num_joints, dtype=np.float32)

    def reset(self):
        """
        Reset robot to initial positions from policy config.
        """
        print("Moving robot to initial positions from policy config...")
        for i, servo_id in enumerate(self.joints.keys()):
            pos = int(self.rl_init_positions[i])
            self.driver.move_servo(servo_id, pos)
            read_pos = self.driver.read_position(servo_id)
            print(f" Read pose: {list(self.joints.values())[i]}: {read_pos}")
            print(f"  {list(self.joints.values())[i]}: {pos}")
        
        self.joint_targets[:] = self.rl_init_positions
        
        # Return full observation space (39 dimensions)
        return self._get_full_observations()

    def step(self, actions: np.ndarray):
        """
        Apply RL actions to the servos.
        actions: np.ndarray of shape (num_joints,)
        """
        assert actions.shape[0] == self.num_joints, "Action size mismatch!"

        self.joint_targets[:] = actions

        print(f"Applying RL actions to hardware: {actions}")
        for i, servo_id in enumerate(self.joints.keys()):
            pos = int(self.joint_targets[i])
            self.driver.move_servo(servo_id, pos)
            read_pos = self.driver.read_position(servo_id)
            print(f" Read pose: {list(self.joints.values())[i]}: {read_pos}")
            print(f"  {list(self.joints.values())[i]}: {pos}")

        # Update measured positions
        for i, servo_id in enumerate(self.joints.keys()):
            pos = self.driver.read_position(servo_id)
            if pos is not None:
                self.joint_positions[i] = pos
            else:
                print(f"Warning: Could not read position from servo {servo_id}")

        # Return full observation space (39 dimensions)
        return self._get_full_observations()
    
    def hold_position(self):
        """
        Hold current joint positions (for idle mode).
        """
        for i, servo_id in enumerate(self.joints.keys()):
            pos = int(self.joint_positions[i])
            self.driver.move_servo(servo_id, pos)

    def stop(self):
        """
        Stop robot safely (hold current positions).
        """
        for i, servo_id in enumerate(self.joints.keys()):
            self.driver.move_servo(servo_id, int(self.joint_positions[i]))
    
    def _get_full_observations(self):
        """
        Construct the full 39-dimensional observation space expected by the policy.
        
        Returns:
            np.ndarray: 39-dimensional observation vector
        """
        # Calculate joint velocities (simple finite difference)
        joint_vel = self.joint_positions - self.prev_joint_positions
        
        # Update previous positions for next iteration
        self.prev_joint_positions[:] = self.joint_positions
        
        # Construct observation vector (39 dimensions total)
        obs = np.zeros(39, dtype=np.float32)
        
        # 1. velocity_commands (3D) - zero for now (no command tracking)
        obs[0:3] = 0.0
        
        # 2. base_ang_vel (3D) - zero for now (no IMU data)
        obs[3:6] = 0.0
        
        # 3. projected_gravity (3D) - assume upright [0, 0, -1]
        obs[6:9] = [0.0, 0.0, -1.0]
        
        # 4. joint_pos (10D) - current joint positions (normalized)
        # Convert from servo units (0-4096) to normalized range (-1, 1)
        normalized_joint_pos = (self.joint_positions - 2048) / 2048.0
        obs[9:19] = normalized_joint_pos
        
        # 5. joint_vel (10D) - joint velocities
        obs[19:29] = joint_vel / 100.0  # Scale down velocities
        
        # 6. actions (10D) - previous actions
        obs[29:39] = self.prev_actions
        
        # Update previous actions for next iteration
        self.prev_actions[:] = normalized_joint_pos
        
        # Debug: Print observation info (uncomment for debugging)
        # print(f"Observation shape: {obs.shape}, Joint positions: {self.joint_positions}")
        
        return obs
