"""Tokenization ingress mixin.

Provides HTTP endpoints for tokenize/detokenize operations.
"""

from fastapi.responses import JSONResponse
from starlette.responses import Response

from ray.llm._internal.serve.core.configs.openai_api_models import (
    DetokenizeRequest,
    ErrorResponse,
    TokenizeCompletionRequest,
)
from ray.llm._internal.serve.core.configs.openai_api_models import OpenAIHTTPException
from ray.llm._internal.serve.observability.logging import get_logger

logger = get_logger(__name__)


class TokenizationIngressMixin:
    """Ingress mixin for /tokenize and /detokenize endpoints.

    Adds endpoints for tokenizing text into token IDs and
    detokenizing token IDs back into text.

    Note: This mixin expects to be mixed into a class that has:
        - _get_model_id(model: str) -> str
        - _get_configured_serve_handle(model_id: str) -> DeploymentHandle
    """

    ENDPOINTS = {
        "tokenize": lambda app: app.post("/tokenize"),
        "detokenize": lambda app: app.post("/detokenize"),
    }

    async def tokenize(self, body: TokenizeCompletionRequest) -> Response:
        """Tokenize text into token IDs.

        Args:
            body: Request containing the prompt to tokenize and model ID.

        Returns:
            JSONResponse with TokenizeResponse containing token IDs and count.
        """
        logger.debug("Tokenizing prompt for model: %s", body.model)
        model_id = await self._get_model_id(body.model)
        handle = self._get_configured_serve_handle(model_id)

        # Get the result from the async generator
        results = handle.tokenize.remote(body)
        result = await results.__anext__()

        if isinstance(result, ErrorResponse):
            raise OpenAIHTTPException(
                message=result.error.message,
                status_code=result.error.code,
                type=result.error.type,
            )

        return JSONResponse(content=result.model_dump())

    async def detokenize(self, body: DetokenizeRequest) -> Response:
        """Detokenize token IDs back into text.

        Args:
            body: Request containing the token IDs to detokenize and model ID.

        Returns:
            JSONResponse with DetokenizeResponse containing the decoded text.
        """
        logger.debug("Detokenizing tokens for model: %s", body.model)
        model_id = await self._get_model_id(body.model)
        handle = self._get_configured_serve_handle(model_id)

        # Get the result from the async generator
        results = handle.detokenize.remote(body)
        result = await results.__anext__()

        if isinstance(result, ErrorResponse):
            raise OpenAIHTTPException(
                message=result.error.message,
                status_code=result.error.code,
                type=result.error.type,
            )

        return JSONResponse(content=result.model_dump())

