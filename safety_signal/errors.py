"""Exception types. Callers catch SafetySignalError to handle anything from this package."""


class SafetySignalError(Exception):
    """Base class for every error raised by this package."""


class ApiError(SafetySignalError):
    def __init__(self, status: int, message: str):
        super().__init__(f"openFDA returned {status}: {message}")
        self.status = status


class NotFoundError(ApiError):
    """openFDA answers 404 when a search matches nothing. That is a normal outcome, not a failure."""

    def __init__(self, message: str = "no matches"):
        super().__init__(404, message)


class RateLimitError(ApiError):
    """Still being throttled (429) after all retries."""


class DataError(SafetySignalError):
    """The data came back but does not make sense (missing fields, inconsistent counts)."""
