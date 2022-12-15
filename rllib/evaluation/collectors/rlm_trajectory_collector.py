import collections
from gym.spaces import Space
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Tuple, TYPE_CHECKING, Union

from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.typing import (
    AgentID,
    EpisodeID,
    EnvID,
    PolicyID,
    TensorType,
    ViewRequirementsDict,
)
from ray.util.debug import log_once
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule


class RLModuleTrajectoryCollector:
    """Collects already postprocessed (single agent) trajectories for one RLModule.

    Samples come in through already postprocessed SampleBatches, which
    contain single episode/trajectory data for a single agent and are then
    appended to this policy's buffers.
    """

    def __init__(self):
        """Initializes a RLModuleTrajectoryCollector instance.

        Args:
            policy: The policy object.
        """

        self.batches = []
        # The total timestep count for all agents that use this policy.
        # NOTE: This is not an env-step count (across n agents). AgentA and
        # agentB, both using this policy, acting in the same episode and both
        # doing n steps would increase the count by 2*n.
        self.agent_steps = 0

    def add_postprocessed_batch_for_training(
        self, batch: SampleBatch, view_requirements: ViewRequirementsDict
    ) -> None:
        """Adds a postprocessed SampleBatch (single agent) to our buffers.

        Args:
            batch: An individual agent's (one trajectory)
                SampleBatch to be added to the Policy's buffers.
            view_requirements: The view
                requirements for the policy. This is so we know, whether a
                view-column needs to be copied at all (not needed for
                training).
        """
        # Add the agent's trajectory length to our count.
        self.agent_steps += batch.count
        # And remove columns not needed for training.
        for view_col, view_req in view_requirements.items():
            if view_col in batch and not view_req.used_for_training:
                del batch[view_col]
        self.batches.append(batch)

    def build(self):
        """Builds a SampleBatch for this policy from the collected data.

        Also resets all buffers for further sample collection for this policy.

        Returns:
            SampleBatch: The SampleBatch with all thus-far collected data for
                this policy.
        """
        # Create batch from our buffers.
        batch = concat_samples(self.batches)
        # Clear batches for future samples.
        self.batches = []
        # Reset agent steps to 0.
        self.agent_steps = 0
        return batch


class MultiAgentRLModuleTrajectoryCollector:
    def __init__(self, marl_module: MultiAgentRLModule):
        self.policy_collectors = {
            mid: RLModuleTrajectoryCollector() for mid in marl_module
        }
        # Total env-steps (1 env-step=up to N agents stepped).
        self.env_steps = 0
        # Total agent steps (1 agent-step=1 individual agent (out of N)
        # stepped).
        self.agent_steps = 0
