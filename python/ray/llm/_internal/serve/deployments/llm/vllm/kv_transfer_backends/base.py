import abc
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ray.llm._internal.serve.configs.server_models import LLMConfig


class BaseConnectorBackend(abc.ABC):
    def __init__(self, llm_config: "LLMConfig"):
        """Base class for connector backends.

        Args:
            llm_config: The llm configuration for this engine
        """
        self.llm_config = llm_config

    @property
    def kv_transfer_config(self) -> Dict[str, Any]:
        engine_kwargs = self.llm_config.engine_kwargs
        kv_transfer_config = engine_kwargs.get("kv_transfer_config")
        assert (
            kv_transfer_config is not None
        ), "In Connector backend, kv_transfer_config is not set"
        return kv_transfer_config

    @abc.abstractmethod
    def setup(self) -> None:
        """Setup the connector backend.

        This method is called to setup the connector backend.
        """
        pass
