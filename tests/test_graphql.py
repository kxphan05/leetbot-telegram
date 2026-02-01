import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from services.graphql_service import GraphQLClient, GraphQLService
from models import User, Question, UserQuestion, Difficulty


class TestGraphQLClient:
    @pytest.fixture
    def client(self):
        return GraphQLClient(
            graphql_url="http://localhost:8080/v1/graphql", token="test-token"
        )

    def test_client_initialization(self, client):
        assert client.graphql_url == "http://localhost:8080/v1/graphql"
        assert client.token == "test-token"
        assert "x-hasura-admin-secret" in client.headers


class TestGraphQLService:
    @pytest.fixture
    def service(self):
        mock_client = MagicMock()
        mock_client.execute = AsyncMock()
        return GraphQLService(client=mock_client)

    @pytest.mark.asyncio
    async def test_get_user_found(self, service):
        mock_client = service.client
        mock_client.execute.return_value = {
            "data": {
                "users": [
                    {
                        "id": 1,
                        "telegram_id": "12345",
                        "username": "testuser",
                        "timezone": "UTC",
                        "difficulty": "Easy",
                        "notification_time": "09:00",
                        "streak": 5,
                        "total_questions": 10,
                        "created_at": "2026-01-01T00:00:00Z",
                        "last_active": "2026-01-31T00:00:00Z",
                    }
                ]
            }
        }

        user = await service.get_user("12345")

        assert user is not None
        assert user.telegram_id == "12345"
        assert user.username == "testuser"
        assert user.difficulty == Difficulty.EASY
        assert user.streak == 5

    @pytest.mark.asyncio
    async def test_get_user_not_found(self, service):
        mock_client = service.client
        mock_client.execute.return_value = {"data": {"users": []}}

        user = await service.get_user("nonexistent")

        assert user is None

    @pytest.mark.asyncio
    async def test_create_user(self, service):
        mock_client = service.client
        mock_client.execute.return_value = {
            "data": {
                "insert_users_one": {
                    "id": 1,
                    "telegram_id": "12345",
                    "username": "testuser",
                    "timezone": "UTC",
                    "difficulty": "Easy",
                    "notification_time": "09:00",
                    "streak": 0,
                    "total_questions": 0,
                    "created_at": "2026-01-01T00:00:00Z",
                    "last_active": "2026-01-31T00:00:00Z",
                }
            }
        }

        user = User(telegram_id="12345", username="testuser")

        result = await service.create_user(user)

        assert result is not None
        assert result.telegram_id == "12345"

    @pytest.mark.asyncio
    async def test_update_user(self, service):
        mock_client = service.client
        mock_client.execute.return_value = {
            "data": {
                "update_users_by_pk": {
                    "id": 1,
                    "telegram_id": "12345",
                    "username": "testuser",
                    "timezone": "America/New_York",
                    "difficulty": "Medium",
                    "notification_time": "10:00",
                    "streak": 10,
                    "total_questions": 20,
                    "created_at": "2026-01-01T00:00:00Z",
                    "last_active": "2026-01-31T00:00:00Z",
                }
            }
        }

        user = User(
            id=1,
            telegram_id="12345",
            username="testuser",
            timezone="America/New_York",
            difficulty=Difficulty.MEDIUM,
            notification_time="10:00",
            streak=10,
            total_questions=20,
        )

        result = await service.update_user(user)

        assert result is not None
        assert result.streak == 10
        assert result.difficulty == Difficulty.MEDIUM

    @pytest.mark.asyncio
    async def test_get_question_found(self, service):
        mock_client = service.client
        mock_client.execute.return_value = {
            "data": {
                "questions_by_pk": {
                    "id": 1,
                    "leetcode_id": 1,
                    "title": "Two Sum",
                    "difficulty": "Easy",
                    "category": "Array",
                    "description": "Find two numbers",
                    "solution": "def solution(nums, target): pass",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            }
        }

        question = await service.get_question(1)

        assert question is not None
        assert question.title == "Two Sum"
        assert question.difficulty == Difficulty.EASY

    @pytest.mark.asyncio
    async def test_get_user_stats(self, service):
        mock_client = service.client
        mock_client.execute.return_value = {
            "data": {
                "users_by_pk": {
                    "streak": 5,
                    "total_questions": 10,
                    "difficulty": "Medium",
                    "timezone": "UTC",
                    "notification_time": "09:00",
                },
                "user_questions_aggregate": {"aggregate": {"count": 15}},
            }
        }

        stats = await service.get_user_stats(1)

        assert stats is not None
        assert stats["streak"] == 5
        assert stats["total_questions"] == 10
        assert stats["difficulty"] == "Medium"

    def test_parse_datetime_valid(self, service):
        result = service._parse_datetime("2026-01-31T12:00:00Z")
        assert result is not None

    def test_parse_datetime_none(self, service):
        result = service._parse_datetime(None)
        assert result is None

    def test_parse_datetime_invalid(self, service):
        result = service._parse_datetime("invalid-date")
        assert result is None
