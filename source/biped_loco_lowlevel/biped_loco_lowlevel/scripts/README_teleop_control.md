# Teleop Control - Servo to RobStride Arm Teleoperation

This script enables teleoperation of RobStride motors using servo motors as the leader (input) device. The servo arm movements are read in real-time and translated to control commands for the RobStride arm.

## Overview

- **Leader (Input)**: Servo motors (ST3215) with IDs: 2, 4, 6, 8, 10
- **Follower (Output)**: RobStride motors with IDs: 2, 4, 6, 8, 10
- **Control Frequency**: 50 Hz
- **Communication**: 
  - Servo motors: Serial (USB) via `/dev/ttyACM0`
  - RobStride motors: CAN bus via `can0`

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Python**: Python 3.8 or higher
- **Hardware**:
  - ST3215 servo motors connected via USB (typically `/dev/ttyACM0`)
  - RobStride motors connected via CAN bus (`can0`)
  - CAN interface properly configured

### Required Python Packages

The script requires the following dependencies:

#### From `biped_loco_lowlevel`:
- `pyserial` - For serial communication with servo motors
- `numpy` - For numerical operations
- `STservo_sdk` - ST3215 servo motor SDK (included in project)

#### From `robstride_control`:
- `python-can>=4.0.0` - CAN bus communication
- `numpy>=1.21.0` - Numerical operations

### Installation Steps

1. **Install system dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip can-utils
   ```

2. **Install Python dependencies for biped_loco_lowlevel**:
   ```bash
   cd source/biped_loco_lowlevel/biped_loco_lowlevel
   pip install -r requirements.txt
   ```

3. **Install Python dependencies for robstride_control**:
   ```bash
   cd source/robstride_control
   pip install -r requirements.txt
   ```

4. **Verify CAN interface**:
   ```bash
   # Check if CAN interface exists
   ip link show can0
   
   # If not configured, set up CAN interface
   sudo ip link set can0 type can bitrate 1000000
   sudo ip link set up can0
   ```

5. **Verify USB device permissions**:
   ```bash
   # Check if servo device is accessible
   ls -l /dev/ttyACM0
   
   # If permission denied, add user to dialout group
   sudo usermod -a -G dialout $USER
   # Log out and log back in for changes to take effect
   ```

## Project Structure

The script expects the following directory structure:

```
biped_loco/
├── source/
│   ├── biped_loco_lowlevel/
│   │   └── biped_loco_lowlevel/
│   │       └── scripts/
│   │           └── teleop_control.py  ← This script
│   └── robstride_control/
│       └── robstride_dynamics/
│           ├── __init__.py
│           ├── bus.py
│           └── protocol.py
```

## Usage

### Basic Usage

```bash
cd source/biped_loco_lowlevel/biped_loco_lowlevel/scripts
python3 teleop_control.py
```

### What the Script Does

1. **Initialization Phase**:
   - Connects to servo driver on `/dev/ttyACM0`
   - Reads initial angles from all servo motors (2, 4, 6, 8, 10)
   - Connects to RobStride CAN bus on `can0`
   - Initializes RobStride motors with MIT mode (Mode 0)
   - Sets control parameters (Kp, Kd, torque limits) per motor
   - Moves RobStride motors to match initial servo positions (zero position)

2. **Control Loop** (50 Hz):
   - Reads current angles from servo motors
   - Calculates angle changes from initial positions
   - Applies sign inversion (except motor 4) to match arm kinematics
   - Sends position commands to RobStride motors via CAN bus

3. **Cleanup** (on Ctrl+C):
   - Returns RobStride motors to zero position
   - Disables all motors
   - Disconnects from CAN bus
   - Closes servo driver connection

### Motor Configuration

The script uses the following motor configuration:

| Motor ID | Model | Kp (Nm/rad) | Kd (Nm/rad/s) | Torque Limit (Nm) |
|----------|-------|-------------|---------------|-------------------|
| 2        | rs-03 | 100.0       | 18.0          | 4.0               |
| 4        | rs-03 | 100.0       | 18.0          | 4.0               |
| 6        | rs-06 | 28.0        | 6.0           | 3.0               |
| 8        | rs-06 | 28.0        | 5.0           | 4.0               |
| 10       | rs-02 | 28.0        | 6.0           | 4.0               |

### Sign Inversion

The script applies sign inversion to match arm kinematics:
- **Motor 4**: No inversion (direct mapping)
- **Motors 2, 6, 8, 10**: Inverted (multiplied by -1)

This can be adjusted in the code if needed.

## Troubleshooting

### Import Errors

**Error**: `❌ Failed to import RobStride SDK`

**Solutions**:
1. Verify `robstride_control` is in the correct location relative to the script
2. Check that `robstride_dynamics` module exists:
   ```bash
   ls -la source/robstride_control/robstride_dynamics/
   ```
3. Ensure Python can find the module:
   ```python
   import sys
   sys.path.append('path/to/robstride_control')
   from robstride_dynamics import RobstrideBus
   ```

### Servo Communication Errors

**Error**: Cannot connect to servo motors

**Solutions**:
1. Check USB connection:
   ```bash
   lsusb | grep -i servo
   ```
2. Verify device path:
   ```bash
   ls -l /dev/ttyACM*
   ```
3. Check permissions:
   ```bash
   groups  # Should include 'dialout'
   ```
4. Try different baudrate if needed (default: 1000000)

### CAN Bus Errors

**Error**: Cannot connect to CAN bus

**Solutions**:
1. Check CAN interface status:
   ```bash
   ip link show can0
   ```
2. Verify CAN interface is up:
   ```bash
   sudo ip link set up can0
   ```
3. Test CAN communication:
   ```bash
   candump can0
   ```
4. Check bitrate matches (default: 1000000)

### Motor Not Responding

**Symptoms**: Servo motors read correctly but RobStride motors don't move

**Solutions**:
1. Verify RobStride motors are powered and enabled
2. Check CAN bus connection and bitrate
3. Verify motor IDs match (2, 4, 6, 8, 10)
4. Check motor initialization messages in console
5. Ensure motors are in MIT mode (Mode 0)

### Permission Errors

**Error**: Permission denied accessing `/dev/ttyACM0` or CAN interface

**Solutions**:
1. Add user to dialout group:
   ```bash
   sudo usermod -a -G dialout $USER
   ```
2. Add user to can group (if exists):
   ```bash
   sudo usermod -a -G can $USER
   ```
3. Log out and log back in
4. Alternatively, run with sudo (not recommended for development)

## Configuration

### Changing Motor IDs

To use different motor IDs, modify these constants in the script:

```python
SERVO_MOTOR_IDS = [2, 4, 6, 8, 10]
ROBSTRIDE_MOTOR_IDS = [2, 4, 6, 8, 10]
```

### Changing Control Parameters

Adjust motor-specific parameters:

```python
MOTOR_KP = {
    2: 100.0,  # Position gain
    4: 100.0,
    # ...
}

MOTOR_KD = {
    2: 18.0,   # Damping gain
    4: 18.0,
    # ...
}
```

### Changing CAN Channel

To use a different CAN channel (e.g., `can1`), modify:

```python
robstride_bus = RobstrideBus("can1", motors, calibration)  # Change can0 to can1
```

### Changing Control Frequency

Modify the rate in the main control loop:

```python
rate = 50.0  # Hz (change to desired frequency)
sleep_time = 1.0 / rate
```

## Safety Notes

⚠️ **Important Safety Considerations**:

1. **Emergency Stop**: Always have an emergency stop mechanism ready
2. **Torque Limits**: Verify torque limits are appropriate for your application
3. **Range Limits**: Ensure servo and RobStride motors don't exceed their mechanical limits
4. **Initialization**: Wait for motors to fully initialize before moving the servo arm
5. **Cleanup**: Always let the script complete cleanup (Ctrl+C) before disconnecting hardware

## Performance

- **Control Frequency**: 50 Hz (20 ms loop time)
- **Latency**: ~20-40 ms (servo read + CAN write)
- **Jitter**: Typically < 5 ms

For higher performance, reduce `sleep_time` but ensure CAN bus can handle the increased traffic.

## Example Output

```
======================================================================
Teleop Control - Synchronize Servo Arm with RobStride Arm
======================================================================
Servo motor IDs (leader): [2, 4, 6, 8, 10]
RobStride motor IDs (follower): [2, 4, 6, 8, 10]
======================================================================
Initializing and setting zero position...

✓ Servo driver initialized

Reading initial servo angles...

Initial servo angles (zero position):
  Motor 2->2: Servo=0.1234 rad (7.07°)
  Motor 4->4: Servo=-0.0456 rad (-2.61°)
  ...

🔍 Connecting to RobStride CAN bus (can0)...
✓ RobStride bus connected

⚡ Activating RobStride motors...
⚙️ Setting MIT mode control parameters...
⚙️ Setting mode to MIT (Mode 0)...
✅ RobStride motors initialized

Setting zero position (moving RobStride motors to servo positions)...

Zero position set. Starting teleop control...
Move the servo arm to control the RobStride arm.
Press Ctrl+C to exit
```

## Additional Resources

- **RobStride Control Documentation**: `source/robstride_control/README.md`
- **Servo Motor Documentation**: See `STservo_sdk` documentation
- **CAN Bus Setup**: Linux SocketCAN documentation

## License

This script is part of the biped_loco project. See project license for details.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Check hardware connections and permissions
4. Review console output for specific error messages
