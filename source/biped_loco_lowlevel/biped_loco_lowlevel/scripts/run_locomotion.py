# core.py
# Main RL control loop for the biped

from loop_rate_limiters import RateLimiter
from biped_loco_lowlevel.robot import BipedLoco
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

    rate = RateLimiter(1 / cfg.policy_dt)
    
    # Initialize observation
    obs = robot.reset()

    print("\nWaiting for gamepad input...")
    print("Press A + Right Bumper to start RL control")

    try:
        while True:
            # Check gamepad mode
            mode = gamepad.commands.get("mode_switch", 0)
            print(mode)
            if mode == 3:  # RL control mode
                # Apply RL policy actions
                actions = controller.update(obs)
                obs = robot.step(actions)
                rate.sleep()
            elif mode == 2:  # Init mode
                # Reset robot to neutral position
                print("Resetting robot to neutral position...")
                obs = robot.pos_callback()
                time.sleep(0.1)  # Small delay
            elif mode == 1:  # Idle mode
                # Hold current position
                robot.hold_position()
                time.sleep(0.01)  # Small delay
            else:  # No mode selected
                # Hold current position
                pos=2048
                robot.pos_callback(pos)
                time.sleep(0.01)  # Small delay
                
    except KeyboardInterrupt:
        print("\nShutting down...")
        gamepad.stop()
        print("Current mode:", mode)
        robot.stop()


if __name__ == "__main__":
    main()
