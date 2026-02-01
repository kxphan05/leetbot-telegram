import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update
from telegram.ext import ContextTypes
from main import (
    start,
    help_command,
    daily_question,
    get_solution,
    gallery,
    search_algorithms,
    stats,
)


class TestBotCommands:
    @pytest.fixture
    def mock_update(self):
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock()
        update.effective_user.id = "12345"
        update.effective_user.first_name = "Test"
        update.effective_user.username = "testuser"
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        return MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    @pytest.mark.asyncio
    async def test_start_command(self, mock_update, mock_context):
        with patch("main.user_service") as mock_user_service:
            mock_user_service.get_user = AsyncMock(return_value=None)
            mock_user_service.create_user = AsyncMock(return_value=MagicMock())

            await start(mock_update, mock_context)

            mock_update.message.reply_text.assert_called_once()
            call_args = mock_update.message.reply_text.call_args[0][0]
            assert "Welcome to LeetCode Bot" in call_args

    @pytest.mark.asyncio
    async def test_help_command(self, mock_update, mock_context):
        await help_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "LeetCode Bot Commands" in call_args
        assert "/daily" in call_args
        assert "/help" in call_args

    @pytest.mark.asyncio
    async def test_daily_question_command(self, mock_update, mock_context):
        with (
            patch("main.question_service") as mock_qs,
            patch("main.user_service") as mock_us,
        ):
            mock_us.get_user = AsyncMock(return_value=MagicMock(difficulty="Easy"))
            mock_qs.get_daily_question = AsyncMock(
                return_value=MagicMock(
                    title="Two Sum",
                    difficulty=MagicMock(value="Easy"),
                    category="Array",
                    description="Find two numbers",
                )
            )

            await daily_question(mock_update, mock_context)

            mock_update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_solution_command(self, mock_update, mock_context):
        with patch("main.question_service") as mock_qs:
            mock_qs.get_daily_question = AsyncMock(
                return_value=MagicMock(
                    title="Two Sum", solution="def two_sum(nums, target): pass"
                )
            )

            await get_solution(mock_update, mock_context)

            mock_update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_gallery_command(self, mock_update, mock_context):
        with patch("main.algorithm_service") as mock_as:
            mock_as.get_all_algorithms = MagicMock(return_value=[])
            mock_as.get_categories = MagicMock(return_value=["Search", "Sorting"])

            await gallery(mock_update, mock_context)

            mock_update.message.reply_text.assert_called_once()
            call_args = mock_update.message.reply_text.call_args[0][0]
            assert "Algorithm Gallery" in call_args

    @pytest.mark.asyncio
    async def test_search_algorithms_command(self, mock_update, mock_context):
        mock_update.args = ["binary", "search"]

        with patch("main.algorithm_service") as mock_as:
            mock_as.search_algorithms = MagicMock(
                return_value=[
                    MagicMock(
                        name="Binary Search", category="Search", description="Test"
                    )
                ]
            )

            await search_algorithms(mock_update, mock_context)

            mock_update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_algorithms_empty_query(self, mock_update, mock_context):
        mock_update.args = []

        with patch("main.algorithm_service") as mock_as:
            mock_as.search_algorithms = MagicMock(return_value=[])

            await search_algorithms(mock_update, mock_context)

            mock_update.message.reply_text.assert_called_once()
            call_args = mock_update.message.reply_text.call_args[0][0]
            assert "No algorithms found" in call_args

    @pytest.mark.asyncio
    async def test_stats_command(self, mock_update, mock_context):
        with patch("main.user_service") as mock_us:
            mock_us.get_user = AsyncMock(
                return_value=MagicMock(
                    streak=5,
                    total_questions=10,
                    difficulty=MagicMock(value="Medium"),
                    timezone="UTC",
                    notification_time="09:00",
                )
            )
            mock_us.get_stats = AsyncMock(
                return_value={
                    "streak": 5,
                    "total_questions": 10,
                    "difficulty": "Medium",
                    "timezone": "UTC",
                    "notification_time": "09:00",
                }
            )

            await stats(mock_update, mock_context)

            mock_update.message.reply_text.assert_called_once()
            call_args = mock_update.message.reply_text.call_args[0][0]
            assert "Your Statistics" in call_args
            assert "5" in call_args
