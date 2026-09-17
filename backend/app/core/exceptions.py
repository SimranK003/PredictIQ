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
