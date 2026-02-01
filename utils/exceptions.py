"""Custom exception classes for better error handling."""


class BotError(Exception):
    """Base exception for all bot errors."""
    pass


class DatabaseError(BotError):
    """Raised when database operations fail."""
    pass


class CacheError(BotError):
    """Raised when cache operations fail."""
    pass


class RateLimitError(BotError):
    """Raised when user exceeds rate limit."""
    
    def __init__(self, message: str = "Rate limit exceeded", cooldown_seconds: int = 0):
        super().__init__(message)
        self.cooldown_seconds = cooldown_seconds


class ValidationError(BotError):
    """Raised when input validation fails."""
    pass


class NotFoundError(BotError):
    """Raised when a requested resource is not found."""
    pass


class ConfigurationError(BotError):
    """Raised when configuration is invalid or missing."""
    pass


class ExternalServiceError(BotError):
    """Raised when external service (LeetCode API, etc.) fails."""
    pass
