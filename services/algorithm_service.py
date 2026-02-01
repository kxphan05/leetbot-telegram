import logging
from typing import Optional, List
from sqlalchemy import select, func
from models import Algorithm
from database.database import AsyncSessionLocal
from database.db_models import AlgorithmDB

logger = logging.getLogger(__name__)


class AlgorithmService:
    def __init__(self):
        self._use_database = True

    def _db_to_model(self, db_algo: AlgorithmDB) -> Algorithm:
        """Convert database model to Pydantic model."""
        return Algorithm(
            id=db_algo.id,
            name=db_algo.name,
            category=db_algo.category,
            description=db_algo.description,
            implementation=db_algo.implementation,
            complexity=db_algo.complexity,
            use_cases=db_algo.use_cases or [],
            external_link=db_algo.external_link
        )

    async def get_all_algorithms(self) -> List[Algorithm]:
        """Get all algorithms from the database."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(AlgorithmDB))
                return [self._db_to_model(a) for a in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Database query failed, using fallback: {e}")
            return self._get_fallback_algorithms()

    async def search_algorithms(self, query: str) -> List[Algorithm]:
        """Search algorithms by name, category, or description."""
        try:
            async with AsyncSessionLocal() as session:
                search_pattern = f"%{query.lower()}%"
                result = await session.execute(
                    select(AlgorithmDB).where(
                        (func.lower(AlgorithmDB.name).like(search_pattern)) |
                        (func.lower(AlgorithmDB.category).like(search_pattern)) |
                        (func.lower(AlgorithmDB.description).like(search_pattern))
                    ).limit(10)
                )
                return [self._db_to_model(a) for a in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Failed to search algorithms: {e}")
            return []

    async def get_algorithm_by_category(self, category: str) -> List[Algorithm]:
        """Get algorithms filtered by category."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AlgorithmDB).where(
                        func.lower(AlgorithmDB.category) == category.lower()
                    )
                )
                return [self._db_to_model(a) for a in result.scalars().all()]
        except Exception as e:
            logger.warning(f"Failed to get algorithms by category: {e}")
            return []

    async def get_algorithm(self, algorithm_id: int) -> Optional[Algorithm]:
        """Get a single algorithm by ID."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AlgorithmDB).where(AlgorithmDB.id == algorithm_id)
                )
                db_algo = result.scalar_one_or_none()
                if db_algo:
                    return self._db_to_model(db_algo)
        except Exception as e:
            logger.warning(f"Failed to get algorithm: {e}")
        return None

    async def get_categories(self) -> List[str]:
        """Get all unique algorithm categories."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AlgorithmDB.category).distinct()
                )
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to get categories: {e}")
            return []

    def _get_fallback_algorithms(self) -> List[Algorithm]:
        """Return fallback algorithms when database is unavailable."""
        return [
            Algorithm(
                id=1,
                name="Binary Search",
                category="Search",
                description="A search algorithm that finds the position of a target value within a sorted array by repeatedly dividing the search interval in half.",
                implementation="""def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",
                complexity="O(log n)",
                use_cases=["Finding elements in sorted arrays", "Finding insertion points"],
                external_link="https://en.wikipedia.org/wiki/Binary_search_algorithm"
            ),
        ]


algorithm_service = AlgorithmService()
