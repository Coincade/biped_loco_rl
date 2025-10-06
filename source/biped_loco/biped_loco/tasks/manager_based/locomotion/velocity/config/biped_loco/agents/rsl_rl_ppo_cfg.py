# Copyright (c) 2024, Biped Loco Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BipedLocoRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "biped_loco_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Reduced from 1.0 to prevent instability
        actor_obs_normalization=True,  # Enable observation normalization for stability
        critic_obs_normalization=True,  # Enable observation normalization for stability
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Slightly increased for better exploration
        num_learning_epochs=4,  # Reduced from 5 to prevent overfitting
        num_mini_batches=4,
        learning_rate=3.0e-4,  # Reduced from 1e-3 for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,  # Reduced from 1.0 for more aggressive gradient clipping
    )


@configclass
class BipedLocoFlatPPORunnerCfg(BipedLocoRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "biped_loco_flat"
