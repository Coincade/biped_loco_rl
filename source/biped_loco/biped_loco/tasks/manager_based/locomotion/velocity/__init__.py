# Copyright (c) 2024, Biped Loco Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Import the velocity environment configuration
from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Import biped_loco config to register environments
from . import config

# Make it available at package level
__all__ = ["LocomotionVelocityRoughEnvCfg"]
