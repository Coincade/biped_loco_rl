# robot.py
# Simplified robot class for servo control (based on Berkeley Humanoid Lite approach)

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
            print("4. Check if another process is using the port")
            print("   - Run: lsof | grep {}".format(port))
            raise RuntimeError(f"Hardware connection failed: {e}")

        # Define joint mapping (servo_id: joint_name)
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
            10: "right_foot_joint"
        }
        
        self.num_joints = len(self.joints)
        
        # Initialize state - start in init mode to go to initial position
        self.state = State.RL_INIT
        self.next_state = State.RL_INIT
        
        # Load initial positions from policy config
        self.rl_init_positions = self._load_initial_positions(config_path)
        
        # Initialize position tracking (in servo units)
        self.joint_positions = np.zeros(self.num_joints, dtype=np.float32)
        self.joint_targets = np.zeros(self.num_joints, dtype=np.float32)
        
        # Initialize observation components
        self.prev_actions = np.zeros(self.num_joints, dtype=np.float32)
        self.joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        self.prev_joint_positions = np.zeros(self.num_joints, dtype=np.float32)
        
        print(f"Robot initialized with {self.num_joints} joints")

    def _load_initial_positions(self, config_path):
        """
        Load initial joint positions from policy config file.
        Convert from normalized coordinates to servo units.
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Get normalized joint positions from config (in radians)
            normalized_positions = config.get('default_joint_positions', [])
            
            if len(normalized_positions) != self.num_joints:
                print(f"Warning: Expected {self.num_joints} joint positions, got {len(normalized_positions)}")
                print("Using default neutral positions (2048 servo units)")
                return np.array([2048] * self.num_joints, dtype=np.int32)
            
            # Convert normalized positions to servo units
            servo_positions = self._normalized_to_servo_units(normalized_positions)
            
            print(f"Loaded initial positions from {config_path}:")
            for i, (joint_name, pos) in enumerate(zip(self.joints.values(), servo_positions)):
                print(f"  {joint_name}: {pos} servo units (normalized: {normalized_positions[i]:.3f})")
            
            return servo_positions
            
        except Exception as e:
            print(f"Warning: Could not load initial positions from {config_path}: {e}")
            print("Using default neutral positions (2048 servo units)")
            return np.array([2048] * self.num_joints, dtype=np.int32)

    def _normalized_to_servo_units(self, normalized_positions):
        """
        Convert normalized positions [-1, 1] to servo units [0, 4096].
        This is a simplified conversion - you may need to calibrate for your specific servos.
        """
        servo_units = []
        for pos in normalized_positions:
            # Convert from [-1, 1] to [0, 4096]
            # Assuming ±1.0 normalized = ±π/2 radians = ±90 degrees
            servo_unit = int((pos + 1.0) * 2048)  # Maps [-1,1] to [0,4096]
            servo_unit = np.clip(servo_unit, 0, 4096)
            servo_units.append(servo_unit)
        return np.array(servo_units, dtype=np.int32)

    def _servo_units_to_normalized(self, servo_units):
        """
        Convert servo units [0, 4096] to normalized positions [-1, 1].
        """
        normalized = (servo_units - 2048) / 2048.0
        return np.clip(normalized, -1.0, 1.0)

    def reset(self):
        """
        Reset robot to initial positions from policy config.
        """
        print("Moving robot to initial positions from policy config...")
        
        # Move servos to initial positions
        for i, servo_id in enumerate(self.joints.keys()):
            servo_pos = self.rl_init_positions[i]
            self.driver.move_servo(servo_id, servo_pos)
            
            # Read actual position for verification
            read_pos = self.driver.read_position(servo_id)
            if read_pos is not None:
                self.joint_positions[i] = read_pos
                print(f"  {list(self.joints.values())[i]}: {servo_pos} (read: {read_pos})")
            else:
                print(f"  {list(self.joints.values())[i]}: {servo_pos} (read failed)")
        
        # Store targets in servo units
        self.joint_targets[:] = self.rl_init_positions
        
        # Return full observation space (39 dimensions)
        return self._get_full_observations()

    def step(self, actions: np.ndarray):
        """
        Apply RL actions to the servos.
        actions: np.ndarray of shape (num_joints,) in normalized coordinates [-1, 1]
        """
        assert actions.shape[0] == self.num_joints, "Action size mismatch!"

        # Convert normalized actions to servo units
        servo_positions = self._normalized_to_servo_units(actions)
        
        # Store targets in servo units
        self.joint_targets[:] = servo_positions

        print(f"Applying RL actions to hardware (normalized): {actions}")
        print(f"Servo positions: {servo_positions}")
        
        # Send commands to servos
        for i, servo_id in enumerate(self.joints.keys()):
            servo_pos = servo_positions[i]
            self.driver.move_servo(servo_id, servo_pos)

        # Update measured positions
        for i, servo_id in enumerate(self.joints.keys()):
            servo_pos = self.driver.read_position(servo_id)
            if servo_pos is not None:
                self.joint_positions[i] = servo_pos
            else:
                print(f"Warning: Could not read position from servo {servo_id}")

        # Return full observation space (39 dimensions)
        return self._get_full_observations()
    
    def hold_position(self):
        """
        Hold current joint positions (for idle mode).
        """
        # Send current targets to servos
        for i, servo_id in enumerate(self.joints.keys()):
            servo_pos = int(self.joint_targets[i])
            self.driver.move_servo(servo_id, servo_pos)

    def stop(self):
        """
        Stop robot safely (hold current positions).
        """
        for i, servo_id in enumerate(self.joints.keys()):
            servo_pos = int(self.joint_positions[i])
            self.driver.move_servo(servo_id, servo_pos)
    
    def _get_full_observations(self):
        """
        Construct the full 39-dimensional observation space expected by the policy.
        Uses normalized coordinates like Berkeley approach.
        """
        # Calculate joint velocities (simple finite difference)
        joint_vel = self.joint_positions - self.prev_joint_positions
        
        # Update previous positions for next iteration
        self.prev_joint_positions[:] = self.joint_positions
        
        # Convert servo positions to normalized coordinates
        normalized_joint_pos = self._servo_units_to_normalized(self.joint_positions)
        
        # Construct observation vector (39 dimensions total)
        obs = np.zeros(39, dtype=np.float32)
        
        # 1. velocity_commands (3D) - zero for now (no command tracking)
        obs[0:3] = 0.0
        
        # 2. base_ang_vel (3D) - zero for now (no IMU data)
        obs[3:6] = 0.0
        
        # 3. projected_gravity (3D) - assume upright [0, 0, -1]
        obs[6:9] = [0.0, 0.0, -1.0]
        
        # 4. joint_pos (10D) - current joint positions (normalized)
        obs[9:19] = normalized_joint_pos
        
        # 5. joint_vel (10D) - joint velocities (normalized)
        # Convert velocity from servo units to normalized
        normalized_vel = joint_vel / 2048.0  # Scale by half range
        obs[19:29] = normalized_vel
        
        # 6. actions (10D) - previous actions (normalized)
        obs[29:39] = self.prev_actions
        
        # Update previous actions for next iteration
        self.prev_actions[:] = normalized_joint_pos
        
        return obs