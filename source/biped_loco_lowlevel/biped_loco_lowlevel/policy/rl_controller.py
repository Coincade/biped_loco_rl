# rl_controller.py
"""
Policy runner for the servo-based biped.

- Supports Torch (.pt) and ONNX (.onnx) policies
- Policy.forward(observations: np.ndarray) -> np.ndarray of shape (n_actions,)
- RlController.update(robot_observations: np.ndarray) -> np.ndarray actions (radians)
"""

from typing import Union
from abc import ABC, abstractmethod
import numpy as np
try:
    import torch
except ImportError:
    torch = None
try:
    import onnxruntime as ort
except ImportError:
    ort = None


class Policy(ABC):
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    @abstractmethod
    def forward(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class TorchPolicy(Policy):
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        super().__init__(checkpoint_path)
        self.device = device
        # load whole model object (assumes saved with torch.save(model))
        self.model: torch.nn.Module = torch.load(checkpoint_path, map_location=self.device)
        self.model.eval()

    def forward(self, observations: np.ndarray) -> np.ndarray:
        # observations expected shape (1, obs_dim) or (obs_dim,)
        obs = observations
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
        with torch.no_grad():
            t = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            out = self.model(t)
            out = out.detach().cpu().numpy()
            return out.squeeze(0)


class OnnxPolicy(Policy):
    def __init__(self, checkpoint_path: str):
        super().__init__(checkpoint_path)
        self.session = ort.InferenceSession(checkpoint_path, providers=["CPUExecutionProvider"])
        # try to find input name automatically
        inputs = self.session.get_inputs()
        if len(inputs) == 0:
            raise RuntimeError("ONNX model has no inputs")
        self.input_name = inputs[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, observations: np.ndarray) -> np.ndarray:
        obs = observations
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
        out = self.session.run([self.output_name], {self.input_name: obs.astype(np.float32)})
        return np.asarray(out[0]).squeeze(0)


class RlController:
    """
    Wraps a policy and presents update(obs) -> actions API.

    Args:
        cfg: OmegaConf config (contains keys used below)
    """

    def __init__(self, cfg: Union[dict, object]):
        self.cfg = cfg
        self.policy = None
        # sizes
        self.num_actions = int(cfg.num_actions)
        self.num_observations = int(cfg.num_observations)
        self.history_length = int(cfg.history_length)

        # buffer for previous observations (flattened)
        total_obs = int(self.num_observations * (self.history_length + 1))
        self.policy_observations = np.zeros((1, total_obs), dtype=np.float32)
        self.policy_actions = np.zeros((1, self.num_actions), dtype=np.float32)
        self.prev_actions = np.zeros((self.num_actions,), dtype=np.float32)

        # defaults
        self.action_scale = float(cfg.get("action_scale", 1.0))
        self.action_limit_lower = np.array(cfg.get("action_limit_lower", -1.0) * np.ones(self.num_actions), dtype=np.float32)
        self.action_limit_upper = np.array(cfg.get("action_limit_upper", 1.0) * np.ones(self.num_actions), dtype=np.float32)
        self.default_joint_positions = np.array(cfg.get("default_joint_positions", np.zeros(self.num_actions)), dtype=np.float32)

    def load_policy(self) -> None:
        model_checkpoint_path = str(self.cfg.policy_checkpoint_path)
        if model_checkpoint_path.endswith(".pt") or model_checkpoint_path.endswith(".pth"):
            self.policy = TorchPolicy(model_checkpoint_path, device=self.cfg.get("device", "cpu"))
            print("[RlController] Loaded Torch policy:", model_checkpoint_path)
        elif model_checkpoint_path.endswith(".onnx"):
            self.policy = OnnxPolicy(model_checkpoint_path)
            print("[RlController] Loaded ONNX policy:", model_checkpoint_path)
        else:
            raise ValueError("Unsupported policy format: " + model_checkpoint_path)

    def update(self, robot_observations: np.ndarray) -> np.ndarray:
        """
        Runs one policy step.

        Args:
            robot_observations: raw observation vector from RobotInterface.get_observations()
                               expected shape (obs_dim,) where obs_dim matches cfg.num_observations

        Returns:
            actions (np.ndarray) shape (num_actions,) — normalized joint positions [-1, 1]
        """
        if self.policy is None:
            raise RuntimeError("Policy not loaded. Call load_policy() first.")

        # Flatten input observations
        obs_flat = robot_observations.astype(np.float32).reshape(-1)
        
        # Check if observation dimensions match
        if len(obs_flat) != self.num_observations:
            raise ValueError(f"Observation dimension mismatch: expected {self.num_observations}, got {len(obs_flat)}")
        
        # Shift history and append newest observation
        prev = self.policy_observations[0, self.num_observations:]
        new_buf = np.concatenate([prev, obs_flat], axis=0)
        self.policy_observations[0, :] = new_buf

        # Run policy network
        out = self.policy.forward(self.policy_observations)
        out = np.asarray(out).reshape(self.num_actions)

        # Clip to policy action limits
        clipped = np.clip(out, self.action_limit_lower, self.action_limit_upper)

        # Remember previous actions for history
        self.prev_actions[:] = clipped

        # Scale actions and add default joint positions (Berkeley approach)
        # This gives us normalized joint positions [-1, 1]
        actions = clipped * self.action_scale + self.default_joint_positions

        return actions
