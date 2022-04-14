from ray.rllib.algorithms.slateq.slateq import SlateQTrainer, DEFAULT_CONFIG
from ray.rllib.algorithms.slateq.slateq_tf_policy import SlateQTFPolicy
from ray.rllib.algorithms.slateq.slateq_torch_policy import SlateQTorchPolicy

__all__ = [
    "DEFAULT_CONFIG",
    "SlateQTFPolicy",
    "SlateQTorchPolicy",
    "SlateQTrainer",
]


from ray.rllib.utils.deprecation import deprecation_warning
deprecation_warning(
    'ray.rllib.agents.slateq',
    'ray.rllib.algorithms.slateq',
    error=False
)