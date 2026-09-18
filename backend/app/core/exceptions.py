"""Application-level exceptions mapped to HTTP responses in main.py."""


class PredictIQError(Exception):
    """Base class for expected application errors."""


class FileTooLargeError(PredictIQError):
    pass


class UnsupportedFileTypeError(PredictIQError):
    pass


class MalformedCSVError(PredictIQError):
    pass


class DatasetValidationError(PredictIQError):
    """Raised when a dataset fails hard validation checks.

    Carries the full quality report so the API can return it in the
    error response body instead of just a message.
    """

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


class NotFoundError(PredictIQError):
    pass


class InvalidPromotionError(PredictIQError):
    """A promotion or rollback request violates the promotion policy or
    the current lifecycle state (e.g. promoting a non-candidate, rolling
    back with no previous production model).
    """


class PromotionConflictError(PredictIQError):
    """A concurrent promotion/rollback raced this one and won.

    Raised when the database's partial unique index rejects a write that
    would have produced two rows in the same exclusive stage — the
    application-level checks passed, but another transaction committed
    first. The caller should surface this as HTTP 409 and let the client
    retry against the now-current state.
    """
