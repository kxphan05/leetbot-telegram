import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.question_service import QuestionService
from services.user_service import UserService
from services.algorithm_service import AlgorithmService
from models import User, Question, Difficulty, Algorithm


class TestQuestionService:
    @pytest.fixture
    def question_service(self):
        return QuestionService()

    def test_load_sample_questions(self, question_service):
        questions = question_service._load_sample_questions()
        assert len(questions) > 0
        assert all(isinstance(q, Question) for q in questions)

    def test_sample_questions_have_required_fields(self, question_service):
        questions = question_service._load_sample_questions()
        for q in questions:
            assert q.id is not None
            assert q.title != ""
            assert q.description != ""
            assert q.difficulty is not None


class TestUserService:
    @pytest.fixture
    def user_service(self):
        return UserService()

    def test_user_service_initialization(self, user_service):
        assert user_service._users_cache == {}

    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, user_service):
        with patch("services.user_service.cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value=None)
            user = await user_service.get_user("nonexistent")
            assert user is None


class TestAlgorithmService:
    @pytest.fixture
    def algorithm_service(self):
        return AlgorithmService()

    def test_get_all_algorithms(self, algorithm_service):
        algorithms = algorithm_service.get_all_algorithms()
        assert len(algorithms) > 0
        assert all(isinstance(a, Algorithm) for a in algorithms)

    def test_algorithms_have_required_fields(self, algorithm_service):
        algorithms = algorithm_service.get_all_algorithms()
        for algo in algorithms:
            assert algo.id is not None
            assert algo.name != ""
            assert algo.description != ""
            assert algo.implementation != ""

    def test_search_algorithms(self, algorithm_service):
        results = algorithm_service.search_algorithms("binary")
        assert len(results) > 0
        assert any("binary" in a.name.lower() for a in results)

    def test_search_algorithms_case_insensitive(self, algorithm_service):
        results1 = algorithm_service.search_algorithms("BINARY")
        results2 = algorithm_service.search_algorithms("binary")
        assert len(results1) == len(results2)

    def test_search_nonexistent(self, algorithm_service):
        results = algorithm_service.search_algorithms("xyznonexistent123")
        assert len(results) == 0

    def test_get_algorithm_by_category(self, algorithm_service):
        results = algorithm_service.get_algorithm_by_category("Search")
        assert len(results) > 0
        assert all(a.category == "Search" for a in results)

    def test_get_algorithm_by_id(self, algorithm_service):
        algo = algorithm_service.get_algorithm(algorithm_id=1)
        assert algo is not None
        assert algo.id == 1

    def test_get_nonexistent_algorithm(self, algorithm_service):
        algo = algorithm_service.get_algorithm(algorithm_id=999)
        assert algo is None

    def test_get_categories(self, algorithm_service):
        categories = algorithm_service.get_categories()
        assert len(categories) > 0
        assert "Search" in categories
