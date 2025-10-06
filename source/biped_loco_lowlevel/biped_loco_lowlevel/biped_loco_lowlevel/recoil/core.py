# driver.py
# Handles low-level communication with ST3215 servos via Waveshare STServo SDK

from biped_loco_lowlevel.STservo_sdk.protocol_packet_handler import protocol_packet_handler as ppp
from biped_loco_lowlevel.STservo_sdk.port_handler import PortHandler
from biped_loco_lowlevel.STservo_sdk.sts import *


class ST3215Driver:
    def __init__(self, port="/dev/ttyACM0", baudrate=1000000):
        """
        Initialize STServo driver on a given port.
        """
        self.port_handler = PortHandler(port)
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {port}")
        if not self.port_handler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate {baudrate}")

        self.servo = sts(self.port_handler)

    def move_servo(self, servo_id: int, position: int, speed: int = 100, acc: int = 50):
        """
        Move the servo to `position` with given `speed` and `acc`.
        - position: 0–4096 (servo position range)
        - speed: servo speed value
        - acc: servo acceleration
        """
        self.servo.WritePosEx(servo_id, position, speed, acc)

    def read_position(self, servo_id: int) -> int:
        """
        Read the current position of a servo (0–4096).
        Returns None if communication fails.
        """
        pos, comm_result, error = self.servo.ReadPos(servo_id)
        if comm_result == 0:  # Communication successful
            return pos
        else:
            print(f"Failed to read position for servo {servo_id}: comm_result={comm_result}, error={error}")
            return None

    def close(self):
        """
        Close the serial port connection.
        """
        self.port_handler.closePort()
