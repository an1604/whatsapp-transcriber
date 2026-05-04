from __future__ import annotations


class WhatsAppTranscriberError(Exception):
    """Base exception for all whatsapp-transcriber errors."""


class ConfigurationError(WhatsAppTranscriberError):
    """Raised when configuration is invalid, missing, or fails validation."""


class VideoNotFoundError(WhatsAppTranscriberError):
    """Raised when a requested video does not exist in the database."""


class DuplicateVideoError(WhatsAppTranscriberError):
    """Raised when attempting to insert a video whose URL hash already exists."""


class InvalidStatusTransitionError(WhatsAppTranscriberError):
    """Raised when a requested video status transition is not permitted."""


class PlatformNotSupportedError(WhatsAppTranscriberError):
    """Raised when a URL belongs to a platform that is not enabled or recognized."""


class DownloadError(WhatsAppTranscriberError):
    """Raised when a video/audio download fails."""


class TranscriptionError(WhatsAppTranscriberError):
    """Raised when audio transcription fails."""


class SummarizationError(WhatsAppTranscriberError):
    """Raised when LLM summarization fails."""


class URLExtractionError(WhatsAppTranscriberError):
    """Raised when URL extraction from a raw message string fails."""


class URLCanonicalizationError(WhatsAppTranscriberError):
    """Raised when a URL cannot be resolved or canonicalized."""


class JobNotFoundError(WhatsAppTranscriberError):
    """Raised when a requested pipeline job does not exist in the database."""


class AudioNotFoundError(WhatsAppTranscriberError):
    """Raised when an audio file is expected on disk but does not exist."""


class SidecarError(WhatsAppTranscriberError):
    """Raised when communication with the WhatsApp Node.js sidecar fails."""


class SidecarNotReadyError(SidecarError):
    """Raised when the sidecar is up but WhatsApp is not yet authenticated."""


class PipelineNotConfiguredError(WhatsAppTranscriberError):
    """Raised when a pipeline operation is attempted but no orchestrator is wired in."""


class VideoAlreadyProcessingError(WhatsAppTranscriberError):
    """Raised when attempting to process a video that is not in 'discovered' status."""
