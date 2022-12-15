import gym
from typing import Any, Mapping, Union

import torch.nn as nn
import torch
from torch.distributions import Categorical
from ray.rllib.models.torch.torch_distributions import TorchDeterministic

from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict


class RandomTorchRLModule(TorchRLModule):
    def __init__(
        self,
        observation_space: "gym.Space",
        action_space: "gym.Space",
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        return self.__get_fwd_output(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        return self.__get_fwd_output(batch)

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        return self.__get_fwd_output(batch)

    @classmethod
    @override(RLModule)
    def from_model_config(
        cls,
        observation_space: "gym.Space",
        action_space: "gym.Space",
        model_config: Mapping[str, Any],
    ) -> Union["RLModule", Mapping[str, Any]]:

        return cls(observation_space, action_space)

    def __get_fwd_output(self, batch):
        return {
            "action_dist": TorchDeterministic(
                loc=torch.from_numpy(self.action_space.sample())[None] # batched
            ),
        }