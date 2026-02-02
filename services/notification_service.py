"""Notification service for scheduled daily questions."""
import logging
from datetime import datetime
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Bot
from telegram.error import TelegramError
from telegram.utils.helpers import escape_markdown

from services.question_service import question_service
from services.user_service import user_service
from database.database import AsyncSessionLocal
from database.db_models import QuestionDB
from sqlalchemy import select

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for scheduled notifications."""
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.bot: Optional[Bot] = None
    
    def initialize(self, bot: Bot) -> None:
        """Initialize the notification service with bot instance."""
        self.bot = bot
        self.scheduler = AsyncIOScheduler()
        
        # Schedule cleanup job every hour
        self.scheduler.add_job(
            self._cleanup_job,
            CronTrigger(minute=0),
            id="cleanup_job",
            replace_existing=True
        )
        
        # Schedule daily notification check every minute
        # This checks for users whose notification time matches current time
        self.scheduler.add_job(
            self._send_daily_notifications,
            CronTrigger(minute="*"),  # Run every minute
            id="daily_notifications",
            replace_existing=True
        )
        
        logger.info("Notification service initialized")
    
    def start(self) -> None:
        """Start the scheduler."""
        if self.scheduler and not self.scheduler.running:
            self.scheduler.start()
            logger.info("Notification scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Notification scheduler stopped")
    
    async def _cleanup_job(self) -> None:
        """Periodic cleanup of stale data."""
        from utils.rate_limiter import rate_limiter
        rate_limiter.cleanup_old_entries()
        logger.debug("Completed cleanup job")
    
    async def _send_daily_notifications(self) -> None:
        """Check and send daily notifications to users."""
        if not self.bot:
            return
        
        current_time = datetime.utcnow().strftime("%H:%M")
        
        # Get users with matching notification time
        users = await user_service.get_users_by_notification_time(current_time)
        
        for user in users:
            try:
                await self._send_daily_question(user.telegram_id)
            except Exception as e:
                logger.error(f"Failed to send notification to {user.telegram_id}: {e}")
    
    async def _send_daily_question(self, telegram_id: str) -> None:
        """Send daily question to a specific user."""
        if not self.bot:
            return
        
        try:
            user_data = await user_service.get_user(telegram_id)
            if not user_data:
                return
            
            question = await question_service.get_daily_question(user_data.difficulty)
            
            message = (
                f"🌅 *Good morning! Here's your daily LeetCode question:*\n\n"
                f"*{question.title}* ({question.difficulty.value})\n"
                f"Category: {question.category}\n"
            )
            if question.solution_approach:
                message += f"Solution Approach: {question.solution_approach}\n"
            message += f"\n{escape_markdown(question.description)}\n\n"
            if question.external_link:
                message += f"🔗 [View on LeetCode]({question.external_link})\n\n"
            message += "Use /solution to get the solution when you're ready!"
            
            await self.bot.send_message(
                chat_id=int(telegram_id),
                text=message,
                parse_mode="Markdown",
                disable_web_page_preview=True
            )
            logger.info(f"Sent daily question to user {telegram_id}")
            
        except TelegramError as e:
            logger.error(f"Telegram error sending to {telegram_id}: {e}")
        except Exception as e:
            logger.error(f"Error sending daily question to {telegram_id}: {e}")
    
    async def send_test_notification(self, telegram_id: str) -> bool:
        """Send a test notification to verify delivery works."""
        if not self.bot:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=int(telegram_id),
                text="🔔 *Test Notification*\n\nYour notifications are working correctly!",
                parse_mode="Markdown"
            )
            return True
        except Exception as e:
            logger.error(f"Test notification failed: {e}")
            return False


notification_service = NotificationService()
