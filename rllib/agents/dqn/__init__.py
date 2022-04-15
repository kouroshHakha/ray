from ray.rllib.algorithms.dqn.apex import ApexTrainer, APEX_DEFAULT_CONFIG
from ray.rllib.algorithms.dqn.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.dqn.r2d2 import R2D2Trainer, R2D2_DEFAULT_CONFIG
from ray.rllib.algorithms.dqn.r2d2_torch_policy import R2D2TorchPolicy
from ray.rllib.algorithms.dqn.simple_q import (
    SimpleQTrainer,
    DEFAULT_CONFIG as SIMPLE_Q_DEFAULT_CONFIG,
)
from ray.rllib.algorithms.dqn.simple_q_tf_policy import SimpleQTFPolicy
from ray.rllib.algorithms.dqn.simple_q_torch_policy import SimpleQTorchPolicy

__all__ = [
    "ApexTrainer",
    "APEX_DEFAULT_CONFIG",
    "DQNTFPolicy",
    "DQNTorchPolicy",
    "DQNTrainer",
    "DEFAULT_CONFIG",
    "R2D2TorchPolicy",
    "R2D2Trainer",
    "R2D2_DEFAULT_CONFIG",
    "SIMPLE_Q_DEFAULT_CONFIG",
    "SimpleQTFPolicy",
    "SimpleQTorchPolicy",
    "SimpleQTrainer",
]

from ray.rllib.utils.deprecation import deprecation_warning

deprecation_warning("ray.rllib.agents.dqn", "ray.rllib.algorithms.dqn", error=False)
