# utils.py
import numpy as np
from typing import Tuple

# Default servo range and mapping assumptions
DEFAULT_SERVO_MIN = 0
DEFAULT_SERVO_MAX = 1000
DEFAULT_SERVO_CENTER = (DEFAULT_SERVO_MIN + DEFAULT_SERVO_MAX) / 2.0
DEFAULT_JOINT_RANGE_RAD = np.pi  # +/- pi maps to min..max by default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rad_to_servo_value(angle_rad: float,
                       servo_min: int = DEFAULT_SERVO_MIN,
                       servo_max: int = DEFAULT_SERVO_MAX,
                       joint_range_rad: float = DEFAULT_JOINT_RANGE_RAD) -> int:
    """
    Convert angle in radians to servo integer units.
    Mapping: -joint_range_rad -> servo_min, 0 -> mid, +joint_range_rad -> servo_max
    """
    mid = (servo_min + servo_max) / 2.0
    half_span = (servo_max - servo_min) / 2.0
    # Normalize angle to [-1, 1] over range [-joint_range_rad, joint_range_rad]
    norm = angle_rad / joint_range_rad
    norm = clamp(norm, -1.0, 1.0)
    val = int(round(mid + norm * half_span))
    return int(clamp(val, servo_min, servo_max))


def servo_value_to_rad(value: int,
                       servo_min: int = DEFAULT_SERVO_MIN,
                       servo_max: int = DEFAULT_SERVO_MAX,
                       joint_range_rad: float = DEFAULT_JOINT_RANGE_RAD) -> float:
    """
    Convert servo integer units back to radians using the inverse mapping.
    """
    mid = (servo_min + servo_max) / 2.0
    half_span = (servo_max - servo_min) / 2.0
    frac = (value - mid) / half_span
    frac = clamp(frac, -1.0, 1.0)
    return float(frac * joint_range_rad)
