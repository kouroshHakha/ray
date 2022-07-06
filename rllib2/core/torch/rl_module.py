from typing import Dict

from rllib2.core.torch.torch_rl_module import TorchRLModule, RLModuleOutput
from rllib2.models.torch.pi_distribution import PiDistributionDict

import torch.nn as nn

from dataclasses import dataclass
import torch.nn as nn

from rllib2.models.torch.pi import PiOutput, Pi
from rllib2.utils import NNOutput


"""Examples of TorchRLModules in RLlib --> See under algorithms"""

"""
Examples:
    
    configs: RLModuleConfig = ...
    self.model = PPOModule(configs)
    
    # The user of the following use-cases are RLlib methods. So they should have a 
    # pre-defined signature that is familiar to RLlib.
    
    # Inference during sampling env or during evaluating policy
    out = self.model({'obs': s[None]}, explore=True/False, inference=True)

    # During sample collection for training
    action = out.behavioral_sample()
    # During sample collection during evaluation
    action = out.target_sample()
    
    #TODO: I don't know if we'd need explore=True / False to change the behavior of sampling
    another alternative is to use explore and only have one sampling method action = out.sample()
    
    # computing (e.g. actor/critic) loss during policy update
    # The output in this use-case will be consumed by the loss function which is 
    # defined by the user (the author of the algorithm).
    # So the structure should flexible to accommodate the various user needs. 
    out = self.model(batch, explore=False, inference=False)
    
    # Note: user knows xyz should exist when forward_train() gets called
    print(out.xyz)
    
"""

@dataclass
class RLModuleConfig(NNConfig):
    """dataclass for holding the nested configuration parameters"""
    action_space: Optional[rllib.env.Space] = None
    obs_space: Optional[rllib.env.Space] = None

@dataclass
class RLModuleOutput(NNOutput):
    """dataclass for holding the outputs of RLModule forward_train() calls"""
    pass


class TorchRLModule(nn.Module):

    def __init__(self, configs=None):
        super().__init__()
        self.configs = configs


    def __call__(self, batch, *args, explore=False, inference=False, **kwargs):
        if inference:
            return self.forward(batch, explore=explore, **kwargs)
        return self.forward_train(batch, **kwargs)

    def forward(self, batch: SampleBatch, explore=False, **kwargs) -> PiDistribution:
        """Forward-pass during online sample collection
        Which could be either during training or evaluation based on explore parameter.
        """
        pass

    def forward_train(self, batch: SampleBatch, **kwargs) -> RLModuleOutput:
        """Forward-pass during computing loss function"""
        pass


# class TorchMARLModule(TorchRLModule):

#     def forward(self, batch: MultiAgentBatch, explore=False, **kwargs) -> PiDistributionDict:
#         pass

#     def forward_train(self, batch: MultiAgentBatch, **kwargs) -> RLModuleOutput:
#         pass


# class MARLModule(TorchRLModule):
#     """Base class for MARL"""

#     def __init__(self, rl_module_dict: Dict[str, TorchRLModule]):
#         super(MARLModule, self).__init__()

#         self.rl_module_dict = rl_module_dict


#     def forward(self, batch: MultiAgentBatch, explore=False, **kwargs) -> PiDistributionDict:
#         pass

#     def forward_train(self, batch: MultiAgentBatch, **kwargs) -> MARLModuleOutput:
#         pass


# class DefaultMARLModule(MARLModule):
#     """
#     Independent Agent training for MARL
#     """

#     def forward(self, batch: MultiAgentBatch, explore=False, **kwargs) -> PiDistributionDict:
#         out_dists = {}
#         for mod_name, mod in self.rl_module_dict.items():
#             out_dists[mod_name] = mod.forward(batch[mod_name], explore=explore, **kwargs)
#         return PiDistributionDict(out_dists)

#     def forward_train(self, batch: MultiAgentBatch, **kwargs) -> MARLModuleOutput:
#         outputs = {}
#         for mod_name, mod in self.rl_module_dict.items():
#             outputs[mod_name] = mod.forward_train(batch[mod_name], **kwargs)
#         return MARLModuleOutput(outputs)

