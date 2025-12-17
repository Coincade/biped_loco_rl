#!/usr/bin/env python3
"""
Change Servo ID - Change the ID of a connected servo motor
"""

import sys
import os
import time

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biped_loco_lowlevel.recoil.core import ST3215Driver
from biped_loco_lowlevel.STservo_sdk.sts import STS_ID


def change_servo_id(driver, current_id, new_id):
    """
    Change the ID of a servo motor.
    
    Args:
        driver: ST3215Driver instance
        current_id: Current servo ID (1-252)
        new_id: New servo ID (1-252)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if new_id < 1 or new_id > 252:
        print(f"Error: New ID must be between 1 and 252 (got {new_id})")
        return False
    
    if current_id < 1 or current_id > 252:
        print(f"Error: Current ID must be between 1 and 252 (got {current_id})")
        return False
    
    if current_id == new_id:
        print(f"Error: Current ID and new ID are the same ({current_id})")
        return False
    
    try:
        print(f"Changing servo ID from {current_id} to {new_id}...")
        
        # Step 1: Unlock EPROM to allow writing
        print("Step 1: Unlocking EPROM...")
        comm_result, error = driver.servo.unLockEprom(current_id)
        if comm_result != 0:
            print(f"Failed to unlock EPROM: comm_result={comm_result}, error={error}")
            return False
        print("✓ EPROM unlocked")
        time.sleep(0.1)
        
        # Step 2: Write new ID to address STS_ID (5)
        print(f"Step 2: Writing new ID {new_id} to servo...")
        comm_result, error = driver.servo.write1ByteTxRx(current_id, STS_ID, new_id)
        if comm_result != 0:
            print(f"Failed to write new ID: comm_result={comm_result}, error={error}")
            # Try to lock EPROM before returning
            driver.servo.LockEprom(current_id)
            return False
        print("✓ New ID written")
        time.sleep(0.2)  # Give servo time to process
        
        # Step 3: Lock EPROM to protect settings
        print("Step 3: Locking EPROM...")
        comm_result, error = driver.servo.LockEprom(new_id)  # Use new_id since ID has changed
        if comm_result != 0:
            print(f"Warning: Failed to lock EPROM: comm_result={comm_result}, error={error}")
            print("ID change may have succeeded, but EPROM is not locked")
        else:
            print("✓ EPROM locked")
        time.sleep(0.1)
        
        # Step 4: Verify the change by reading the ID
        print(f"Step 4: Verifying new ID...")
        time.sleep(0.2)
        # Try to read position from new ID - if it works, the ID change was successful
        position = driver.read_position(new_id)
        if position is not None:
            print(f"✓ Successfully verified: Servo now responds to ID {new_id}")
            return True
        else:
            print(f"Warning: Could not verify ID change. Servo may not respond to new ID {new_id}")
            print("Try power cycling the servo and testing again.")
            return False
            
    except Exception as e:
        print(f"Error changing servo ID: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python change_id.py <current_id> <new_id>")
        print("Example: python change_id.py 1 5")
        print("\nNote: Servo IDs must be between 1 and 252")
        print("Warning: Make sure only ONE servo is connected when changing ID!")
        return 1
    
    try:
        current_id = int(sys.argv[1])
        new_id = int(sys.argv[2])
    except ValueError as e:
        print(f"Error: Invalid argument - {e}")
        print("Usage: python change_id.py <current_id> <new_id>")
        return 1
    
    print("=" * 60)
    print("Servo ID Change Utility")
    print("=" * 60)
    print(f"Current ID: {current_id}")
    print(f"New ID: {new_id}")
    print("\n⚠️  WARNING: Make sure only ONE servo is connected!")
    print("⚠️  Changing ID on multiple servos will affect all of them!")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return 0
    
    try:
        driver = ST3215Driver(port="/dev/ttyACM0", baudrate=1000000)
        print("\n✓ Connected to servo driver\n")
        
        # Change the servo ID
        if change_servo_id(driver, current_id, new_id):
            print("\n" + "=" * 60)
            print(f"✓ SUCCESS: Servo ID changed from {current_id} to {new_id}")
            print("=" * 60)
            print("\nNote: You may need to power cycle the servo for the change to take full effect.")
            return 0
        else:
            print("\n" + "=" * 60)
            print("✗ FAILED: Could not change servo ID")
            print("=" * 60)
            return 1
        
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
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
