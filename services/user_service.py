"""User service for managing user data and preferences."""
import json
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from models import User, Difficulty
from database.database import AsyncSessionLocal
from database.db_models import UserDB, UserQuestionDB
from utils.cache import cache

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


class UserService:
    def __init__(self):
        self._users_cache: dict[str, User] = {}

    def _db_to_model(self, db_user: UserDB) -> User:
        """Convert database model to Pydantic model."""
        return User(
            id=db_user.id,
            telegram_id=db_user.telegram_id,
            username=db_user.username,
            timezone=db_user.timezone,
            difficulty=Difficulty(db_user.difficulty),
            notification_time=db_user.notification_time,
            streak=db_user.streak,
            total_questions=db_user.total_questions,
            created_at=db_user.created_at,
            last_active=db_user.last_active,
        )

    async def get_user(self, telegram_id: str) -> Optional[User]:
        """Get user by telegram ID."""
        cache_key = f"user_{telegram_id}"

        # Check cache first
        cached_data = await cache.get(cache_key)
        if cached_data:
            user_data = json.loads(cached_data)
            user = User(**user_data)
            self._users_cache[telegram_id] = user
            return user

        # Check in-memory cache
        if telegram_id in self._users_cache:
            return self._users_cache[telegram_id]

        # Query database
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(UserDB).where(UserDB.telegram_id == telegram_id)
                )
                db_user = result.scalar_one_or_none()
                if db_user:
                    user = self._db_to_model(db_user)
                    await cache.set(cache_key, json.dumps(user.model_dump(), cls=DateTimeEncoder))
                    self._users_cache[telegram_id] = user
                    return user
        except Exception as e:
            logger.warning(f"Database query failed: {e}")

        return None

    async def create_user(
        self, telegram_id: str, username: Optional[str] = None
    ) -> User:
        """Create a new user."""
        try:
            async with AsyncSessionLocal() as session:
                db_user = UserDB(
                    telegram_id=telegram_id,
                    username=username,
                    created_at=datetime.utcnow(),
                    last_active=datetime.utcnow(),
                )
                session.add(db_user)
                await session.commit()
                await session.refresh(db_user)
                
                user = self._db_to_model(db_user)
                cache_key = f"user_{telegram_id}"
                await cache.set(cache_key, json.dumps(user.model_dump(), cls=DateTimeEncoder))
                self._users_cache[telegram_id] = user
                logger.info(f"Created new user: {telegram_id}")
                return user
        except Exception as e:
            logger.error(f"Failed to create user in database: {e}")
            # Fallback to cache-only user
            user = User(
                telegram_id=telegram_id,
                username=username,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
            )
            cache_key = f"user_{telegram_id}"
            await cache.set(cache_key, json.dumps(user.model_dump(), cls=DateTimeEncoder))
            self._users_cache[telegram_id] = user
            return user

    async def update_user(self, user: User) -> User:
        """Update user data."""
        user.last_active = datetime.utcnow()
        
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(UserDB)
                    .where(UserDB.telegram_id == user.telegram_id)
                    .values(
                        username=user.username,
                        timezone=user.timezone,
                        difficulty=user.difficulty.value,
                        notification_time=user.notification_time,
                        streak=user.streak,
                        total_questions=user.total_questions,
                        last_active=user.last_active,
                    )
                )
                await session.commit()
        except Exception as e:
            logger.warning(f"Database update failed: {e}")
        
        cache_key = f"user_{user.telegram_id}"
        await cache.set(cache_key, json.dumps(user.model_dump(), cls=DateTimeEncoder))
        self._users_cache[user.telegram_id] = user
        return user

    async def update_preferences(
        self,
        telegram_id: str,
        difficulty: Optional[Difficulty] = None,
        timezone: Optional[str] = None,
        notification_time: Optional[str] = None,
    ) -> Optional[User]:
        """Update user preferences."""
        user = await self.get_user(telegram_id)
        if user is None:
            return None

        if difficulty is not None:
            user.difficulty = difficulty
        if timezone is not None:
            user.timezone = timezone
        if notification_time is not None:
            user.notification_time = notification_time

        return await self.update_user(user)

    async def record_question_completion(self, telegram_id: str, question_id: int) -> Optional[User]:
        """Record that a user completed a question and update streak."""
        user = await self.get_user(telegram_id)
        if user is None:
            return None

        today = datetime.utcnow().date()
        
        try:
            async with AsyncSessionLocal() as session:
                # Check if already completed today
                result = await session.execute(
                    select(UserQuestionDB).where(
                        UserQuestionDB.telegram_id == telegram_id,
                        UserQuestionDB.question_id == question_id
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    if not existing.completed:
                        existing.completed = 1
                        existing.completed_at = datetime.utcnow()
                        await session.commit()
                else:
                    user_question = UserQuestionDB(
                        telegram_id=telegram_id,
                        question_id=question_id,
                        completed=1,
                        completed_at=datetime.utcnow(),
                    )
                    session.add(user_question)
                    await session.commit()
                
                # Update streak
                db_user = await session.execute(
                    select(UserDB).where(UserDB.telegram_id == telegram_id)
                )
                db_user = db_user.scalar_one_or_none()
                
                if db_user:
                    last_date = db_user.last_completion_date.date() if db_user.last_completion_date else None
                    
                    if last_date == today:
                        # Already completed today
                        pass
                    elif last_date == today - timedelta(days=1):
                        # Consecutive day - increment streak
                        db_user.streak += 1
                        db_user.longest_streak = max(db_user.longest_streak, db_user.streak)
                    else:
                        # Streak broken - reset to 1
                        db_user.streak = 1
                    
                    db_user.total_questions += 1
                    db_user.last_completion_date = datetime.utcnow()
                    await session.commit()
                    
                    user = self._db_to_model(db_user)
                    
        except Exception as e:
            logger.error(f"Failed to record completion: {e}")
            # Fallback: just update in memory
            user.total_questions += 1
        
        # Update cache
        cache_key = f"user_{telegram_id}"
        await cache.set(cache_key, json.dumps(user.model_dump(), cls=DateTimeEncoder))
        self._users_cache[telegram_id] = user
        
        return user

    async def get_streak_info(self, telegram_id: str) -> Optional[dict]:
        """Get detailed streak information for a user."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(UserDB).where(UserDB.telegram_id == telegram_id)
                )
                db_user = result.scalar_one_or_none()
                
                if db_user:
                    return {
                        "current_streak": db_user.streak,
                        "longest_streak": db_user.longest_streak,
                        "total_questions": db_user.total_questions,
                        "last_completion": db_user.last_completion_date,
                    }
        except Exception as e:
            logger.warning(f"Failed to get streak info: {e}")
        
        # Fallback to cache
        user = await self.get_user(telegram_id)
        if user:
            return {
                "current_streak": user.streak,
                "longest_streak": user.streak,
                "total_questions": user.total_questions,
                "last_completion": None,
            }
        return None

    async def get_history(self, telegram_id: str, limit: int = 10) -> List[dict]:
        """Get user's question completion history."""
        try:
            async with AsyncSessionLocal() as session:
                from database.db_models import QuestionDB
                
                result = await session.execute(
                    select(UserQuestionDB, QuestionDB)
                    .join(QuestionDB, UserQuestionDB.question_id == QuestionDB.id)
                    .where(
                        UserQuestionDB.telegram_id == telegram_id,
                        UserQuestionDB.completed == 1
                    )
                    .order_by(UserQuestionDB.completed_at.desc())
                    .limit(limit)
                )
                
                history = []
                for uq, question in result.fetchall():
                    history.append({
                        "title": question.title,
                        "difficulty": question.difficulty,
                        "category": question.category,
                        "completed_at": uq.completed_at,
                        "external_link": question.external_link,
                    })
                return history
        except Exception as e:
            logger.warning(f"Failed to get history: {e}")
            return []

    async def get_stats(self, telegram_id: str) -> Optional[dict]:
        """Get user statistics."""
        user = await self.get_user(telegram_id)
        if user is None:
            return None

        streak_info = await self.get_streak_info(telegram_id)
        
        return {
            "streak": streak_info.get("current_streak", user.streak) if streak_info else user.streak,
            "longest_streak": streak_info.get("longest_streak", user.streak) if streak_info else user.streak,
            "total_questions": user.total_questions,
            "difficulty": user.difficulty.value,
            "timezone": user.timezone,
            "notification_time": user.notification_time,
        }

    async def get_users_by_notification_time(self, time_str: str) -> List[User]:
        """Get all users with notifications enabled at the specified time."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(UserDB).where(
                        UserDB.notification_time == time_str,
                        UserDB.notifications_enabled == 1
                    )
                )
                return [self._db_to_model(u) for u in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Failed to get users by notification time: {e}")
            return []


user_service = UserService()
