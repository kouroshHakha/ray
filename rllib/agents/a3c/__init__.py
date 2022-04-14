from ray.rllib.algorithms.a3c.a3c import A3CTrainer, DEFAULT_CONFIG
from ray.rllib.algorithms.a3c.a2c import A2CTrainer


__all__ = ["A2CTrainer", "A3CTrainer", "DEFAULT_CONFIG"]

from ray.rllib.utils.deprecation import deprecation_warning
deprecation_warning(
    'ray.rllib.agents.a3c',
    'ray.rllib.algorithms.a3c',
    error=False
)
