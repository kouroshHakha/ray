from ray.rllib.algorithms.alpha_star.alpha_star import DEFAULT_CONFIG, AlphaStarTrainer

__all__ = [
    "DEFAULT_CONFIG",
    "AlphaStarTrainer",
]

from ray.rllib.utils.deprecation import deprecation_warning

deprecation_warning(
    "ray.rllib.agents.alpha_star", "ray.rllib.algorithms.alpha_star", error=False
)
