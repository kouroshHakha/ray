from ray.rllib.algorithms.bandit.bandit import BanditLinTSTrainer, BanditLinUCBTrainer

__all__ = ["BanditLinTSTrainer", "BanditLinUCBTrainer"]


from ray.rllib.utils.deprecation import deprecation_warning
deprecation_warning(
    'ray.rllib.agents.bandit',
    'ray.rllib.algorithms.bandit',
    error=False
)