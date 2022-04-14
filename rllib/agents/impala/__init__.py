from ray.rllib.algorithms.impala.impala import DEFAULT_CONFIG, ImpalaTrainer

__all__ = [
    "DEFAULT_CONFIG",
    "ImpalaTrainer",
]


from ray.rllib.utils.deprecation import deprecation_warning
deprecation_warning(
    'ray.rllib.agents.impala',
    'ray.rllib.algorithms.impala',
    error=False
)