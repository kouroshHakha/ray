from typing import Dict
import numpy as np

from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.registry import register_env
from ray import air, tune
from ray.rllib.algorithms.callbacks import make_multi_callbacks, DefaultCallbacks
from ray.rllib.connectors.connector import AgentConnector, ConnectorContext
from ray.rllib.connectors import registry
import random
from ray.rllib.fault_injector_connector import FaultInjectorConnector, FaultInjectionConnectorCallback

register_env("multi_cartpole", lambda _: MultiAgentCartPole({"num_agents": 2}))

# Number of policies overall in the PolicyMap.
num_policies = 12
# Number of those policies that should be trained. These are a subset of `num_policies`.
num_trainable = 1

num_envs_per_worker = 1

# Define the config as an APPOConfig object.
config = (
    APPOConfig()
    .framework("tf2", eager_tracing=True)
    .resources(num_gpus=1)
    .environment("multi_cartpole")
    .rollouts(
        num_rollout_workers=16,
        observation_filter="MeanStdFilter",
    )
    .fault_tolerance(
        recreate_failed_workers=True,
        restart_failed_sub_environments=False,
        num_consecutive_worker_failures_tolerance=100,
    )
    .training(
        model={
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "linear",
            "vf_share_layers": True
        },
    )
    .multi_agent(
        # 2 agents per sub-env.
        # This is to avoid excessive swapping during an episode rollout, since
        # Policies are only re-picked at the beginning of each episode.
        count_steps_by="agent_steps",
        policy_map_capacity=2 * num_envs_per_worker,
        policy_states_are_swappable=True,
        policies={f"pol{i}" for i in range(num_policies)},
        # Train only the first n policies.
        policies_to_train=[f"pol{i}" for i in range(num_trainable)],
        # Pick one trainable and one non-trainable policy per episode.
        policy_mapping_fn=(
            lambda aid, eps, worker, **kw: "pol"
            + str(
                np.random.randint(0, num_trainable)
                if aid == 0
                else np.random.randint(num_trainable, num_policies)
            )
        ),
    )
    .callbacks(callbacks_class=make_multi_callbacks([FaultInjectionConnectorCallback]))
)

# # Define some stopping criteria.
# stop = {
#     "evaluation/policy_reward_mean/pol0": 50.0,
#     "timesteps_total": 500000,
# }

# algo = config.build()
# algo.train()
# breakpoint()
tuner = tune.Tuner(
    "APPO",
    param_space=config,
)
tuner.fit()