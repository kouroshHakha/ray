

from ray.rllib.algorithms.callbacks import make_multi_callbacks, DefaultCallbacks
from ray.rllib.connectors.connector import AgentConnector, ConnectorContext
from ray.rllib.connectors import registry
import random

class FaultInjectorConnector(AgentConnector):

    def transform(self, ac_data):
        # with a small probablity (0.5), if t == 100, env_id % 10 == 0, and agent_id == 0
        # With a fault injected the rollout worker has to be restarted.
        time = ac_data.data["t"]
        # if (ac_data.agent_id == 0): # time == 100 and (ac_data.env_id % 10 == 0) and 
        if random.random() < 0.5:
            raise ValueError("Fault injected!")
        return ac_data

    def to_state(self):
        return FaultInjectorConnector.__name__, {}

    @staticmethod
    def from_state(ctx: ConnectorContext, params: dict):
        return FaultInjectorConnector(ctx)
    
registry.register_connector(FaultInjectorConnector.__name__, FaultInjectorConnector)

class FaultInjectionConnectorCallback(DefaultCallbacks):

    def on_create_policy(self, *, policy_id, policy) -> None:
        # only apply fault to pol1
        if policy_id != "pol1":
            return
        ctx = ConnectorContext.from_policy(policy)
        assert len(policy.agent_connectors.connectors) > 0
        if type(policy.agent_connectors.connectors[0]) != FaultInjectorConnector:
            connector = FaultInjectorConnector(ctx)
            policy.agent_connectors.prepend(connector)
        return

