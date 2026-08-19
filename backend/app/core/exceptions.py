"""HTTP-mapped application errors. Handled in `app.main` as JSON responses."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidInputError(AppError):
    status_code = 400
    code = "invalid_input"


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class UnreadableDocumentError(AppError):
    status_code = 422
    code = "unreadable_document"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"


class LLMServiceError(AppError):
    status_code = 502
    code = "llm_upstream_error"


class ConfigurationError(AppError):
    status_code = 503
    code = "not_configured"
