import json
import random
import logging
from typing import Optional, List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from models import Question, QuestionExample, Difficulty
from database.database import AsyncSessionLocal
from database.db_models import QuestionDB
from utils.cache import cache

logger = logging.getLogger(__name__)


class QuestionService:
    def __init__(self):
        self._fallback_questions: List[Question] = []
        self._use_database = True

    def _db_to_model(self, db_question: QuestionDB) -> Question:
        """Convert database model to Pydantic model."""
        examples = []
        if db_question.examples:
            for ex in db_question.examples:
                examples.append(
                    QuestionExample(
                        input=ex.get("input", ""),
                        output=ex.get("output", ""),
                        explanation=ex.get("explanation"),
                    )
                )

        return Question(
            id=db_question.id,
            leetcode_id=db_question.leetcode_id,
            title=db_question.title,
            difficulty=Difficulty(db_question.difficulty),
            category=db_question.category,
            description=db_question.description,
            solution=db_question.solution,
            solution_approach=db_question.solution_approach,
            examples=examples,
            external_link=db_question.external_link,
            created_at=db_question.created_at,
        )

    async def get_daily_question(
        self, difficulty: Optional[Difficulty] = None
    ) -> Question:
        """Get a random daily question, optionally filtered by difficulty."""
        cache_key = f"daily_question_{difficulty.value if difficulty else 'random'}"

        # Check cache first
        cached_data = await cache.get(cache_key)
        if cached_data:
            return Question(**json.loads(cached_data))

        try:
            async with AsyncSessionLocal() as session:
                # Build query
                query = select(QuestionDB)
                if difficulty:
                    query = query.where(QuestionDB.difficulty == difficulty.value)

                # Get random question
                query = query.order_by(func.random()).limit(1)
                result = await session.execute(query)
                db_question = result.scalar_one_or_none()

                if db_question:
                    question = self._db_to_model(db_question)
                    # Cache for 24 hours
                    await cache.set(
                        cache_key,
                        json.dumps(question.model_dump(), default=str),
                        ttl=86400,
                    )
                    return question

        except Exception as e:
            logger.warning(f"Database query failed, using fallback: {e}")

        # Fallback to hardcoded questions if DB unavailable
        return self._get_fallback_question(difficulty)

    def _get_fallback_question(
        self, difficulty: Optional[Difficulty] = None
    ) -> Question:
        """Return a fallback question when database is unavailable."""
        fallback = Question(
            id=1,
            leetcode_id=1,
            title="Two Sum",
            difficulty=Difficulty.EASY,
            category="Array",
            description="Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            solution="""def twoSum(nums, target):
    prevMap = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[n] = i
    return""",
            solution_approach="Hash Map",
            examples=[
                QuestionExample(
                    input="nums = [2,7,11,15], target = 9",
                    output="[0,1]",
                    explanation="Because nums[0] + nums[1] == 9, we return [0, 1].",
                )
            ],
            external_link="https://leetcode.com/problems/two-sum/",
        )
        return fallback

    def _load_sample_questions(self) -> List[Question]:
        """Load sample questions for testing."""
        return [self._get_fallback_question()]

    async def get_solution(self, question_id: int) -> Optional[Question]:
        """Get a question by ID to retrieve its solution."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(QuestionDB).where(QuestionDB.id == question_id)
                )
                db_question = result.scalar_one_or_none()
                if db_question:
                    return self._db_to_model(db_question)
        except Exception as e:
            logger.warning(f"Failed to get solution from database: {e}")

        return None

    async def search_questions(self, query: str) -> List[Question]:
        """Search questions by title, category, or description."""
        try:
            async with AsyncSessionLocal() as session:
                search_pattern = f"%{query.lower()}%"
                result = await session.execute(
                    select(QuestionDB)
                    .where(
                        (func.lower(QuestionDB.title).like(search_pattern))
                        | (func.lower(QuestionDB.category).like(search_pattern))
                        | (func.lower(QuestionDB.description).like(search_pattern))
                    )
                    .limit(10)
                )
                return [self._db_to_model(q) for q in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Failed to search questions: {e}")
            return []

    async def get_all_questions(self) -> List[Question]:
        """Get all questions from the database."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(QuestionDB))
                return [self._db_to_model(q) for q in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Failed to get all questions: {e}")
            return []


question_service = QuestionService()
