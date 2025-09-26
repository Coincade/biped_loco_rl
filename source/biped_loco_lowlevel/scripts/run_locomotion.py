# run_locomotion.py
# Main RL control loop for servo-based biped (simplified for servo control)

from loop_rate_limiters import RateLimiter
from biped_loco_lowlevel.robot import BipedLoco, State
from biped_loco_lowlevel.policy.rl_controller import RlController
from biped_loco_lowlevel.policy.config import Cfg
from biped_loco_lowlevel.policy.gamepad import Se2Gamepad
import time


def main():
    cfg = Cfg.from_arguments()

    # Initialize robot (requires hardware connection)
    print("Initializing robot...")
    robot = BipedLoco(config_path=cfg.config_path if hasattr(cfg, 'config_path') else "configs/policy_latest.yaml")
    print("Robot initialized with positions from policy config")
    
    # Initialize gamepad controller
    print("Initializing gamepad controller...")
    gamepad = Se2Gamepad()
    gamepad.run()
    print("Gamepad controller ready")
    print("Controls:")
    print("  - A + Right Bumper: Enter RL control mode")
    print("  - A + Left Bumper: Enter init mode") 
    print("  - X or Thumbsticks: Enter idle mode")
    print("  - Left stick: Forward/backward movement")
    print("  - Right stick: Side movement")
    print("  - Right stick X: Yaw rotation")

    # Initialize policy
    print("Loading RL policy...")
    controller = RlController(cfg)
    controller.load_policy()
    print("RL policy loaded")

    # Set control rate based on policy frequency
    policy_freq = 1.0 / cfg.policy_dt
    rate = RateLimiter(policy_freq)
    print(f"Policy frequency: {policy_freq:.1f} Hz")
    
    # Initialize observation
    obs = robot.reset()

    print("\nRobot will automatically go to initial position...")
    print("Waiting for gamepad input...")
    print("Press A + Right Bumper to start RL control")

    try:
        while True:
            # Check gamepad mode
            mode = gamepad.commands.get("mode_switch", 0)
            
            if mode == 3:  # RL control mode
                # Apply RL policy actions (returns normalized positions [-1, 1])
                actions = controller.update(obs)
                obs = robot.step(actions)
                rate.sleep()
            elif mode == 2:  # Init mode
                # Reset robot to initial positions from policy
                print("Resetting robot to initial positions...")
                obs = robot.reset()
                time.sleep(0.1)  # Small delay
            elif mode == 1:  # Idle mode
                # Hold current position
                robot.hold_position()
                time.sleep(0.01)  # Small delay
            else:  # No mode selected (default to init mode on startup)
                # On first run, go to initial position, then hold
                if robot.state == State.RL_INIT:
                    print("Going to initial position...")
                    obs = robot.reset()
                    robot.state = State.IDLE  # Switch to idle after init
                else:
                    # Hold current position
                    robot.hold_position()
                time.sleep(0.01)  # Small delay
                
    except KeyboardInterrupt:
        print("\nShutting down...")
        gamepad.stop()
        robot.stop()


if __name__ == "__main__":
    main()
