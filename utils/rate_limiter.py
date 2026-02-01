"""Rate limiter for per-user command cooldowns."""
import time
import logging
from typing import Optional
from functools import wraps
from telegram import Update
from telegram.ext import ContextTypes
from utils.exceptions import RateLimitError

logger = logging.getLogger(__name__)


class RateLimiter:
    """In-memory rate limiter with per-user tracking."""
    
    def __init__(self, default_cooldown: int = 3):
        """
        Initialize rate limiter.
        
        Args:
            default_cooldown: Default cooldown in seconds between commands
        """
        self.default_cooldown = default_cooldown
        self._user_timestamps: dict[str, dict[str, float]] = {}
    
    def check_rate_limit(
        self, 
        user_id: str, 
        command: str, 
        cooldown: Optional[int] = None
    ) -> tuple[bool, int]:
        """
        Check if user is rate limited.
        
        Returns:
            Tuple of (is_allowed, remaining_cooldown_seconds)
        """
        cooldown = cooldown or self.default_cooldown
        current_time = time.time()
        
        # Initialize user tracking if needed
        if user_id not in self._user_timestamps:
            self._user_timestamps[user_id] = {}
        
        user_commands = self._user_timestamps[user_id]
        
        # Check if command was used recently
        if command in user_commands:
            last_used = user_commands[command]
            elapsed = current_time - last_used
            
            if elapsed < cooldown:
                remaining = int(cooldown - elapsed)
                return False, remaining
        
        # Update timestamp
        user_commands[command] = current_time
        return True, 0
    
    def reset_user(self, user_id: str) -> None:
        """Reset rate limit for a user."""
        if user_id in self._user_timestamps:
            del self._user_timestamps[user_id]
    
    def cleanup_old_entries(self, max_age: int = 3600) -> None:
        """Remove entries older than max_age seconds."""
        current_time = time.time()
        cutoff = current_time - max_age
        
        for user_id in list(self._user_timestamps.keys()):
            user_commands = self._user_timestamps[user_id]
            for command in list(user_commands.keys()):
                if user_commands[command] < cutoff:
                    del user_commands[command]
            
            if not user_commands:
                del self._user_timestamps[user_id]


# Global rate limiter instance
rate_limiter = RateLimiter(default_cooldown=2)


def rate_limit(cooldown: int = 2):
    """
    Decorator to apply rate limiting to command handlers.
    
    Args:
        cooldown: Cooldown in seconds between uses
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user = update.effective_user
            if user is None:
                return await func(update, context)
            
            command = func.__name__
            user_id = str(user.id)
            
            is_allowed, remaining = rate_limiter.check_rate_limit(
                user_id, command, cooldown
            )
            
            if not is_allowed:
                logger.info(f"Rate limited user {user_id} for command {command}")
                await update.message.reply_text(
                    f"⏳ Please wait {remaining} second{'s' if remaining != 1 else ''} before using this command again."
                )
                return
            
            return await func(update, context)
        return wrapper
    return decorator
