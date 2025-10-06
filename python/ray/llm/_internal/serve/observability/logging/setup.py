import logging

# TODO: NIGHTLY FIX
# from ray._common.filters import CoreContextFilter
# from ray._common.formatters import JSONFormatter
from ray.serve._private.logging_utils import ServeContextFilter


def _configure_stdlib_logging():
    """Configures stdlib root logger to make sure stdlib loggers (created as
    `logging.getLogger(...)`) are using Ray's `JSONFormatter` with Core and Serve
     context filters.
    """

    handler = logging.StreamHandler()
    # TODO: NIGHTLY FIX
    # handler.addFilter(CoreContextFilter())
    # TODO: NIGHTLY FIX
    # handler.setFormatter(JSONFormatter())
    handler.addFilter(ServeContextFilter())

    root_logger = logging.getLogger()
    # NOTE: It's crucial we reset all the handlers of the root logger,
    #       to make sure that logs aren't emitted twice
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def setup_logging():
    _configure_stdlib_logging()
