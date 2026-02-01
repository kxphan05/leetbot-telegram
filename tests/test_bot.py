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
    set_difficulty,
    preferences,
    SET_DIFFICULTY,
)
from telegram.ext import ConversationHandler
from models.models import Difficulty


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

    @pytest.mark.asyncio
    async def test_set_difficulty_menu_selection(self, mock_update, mock_context):
        with patch("main.user_service") as mock_user_service:
            mock_user_service.get_user = AsyncMock(
                return_value=MagicMock(
                    telegram_id="12345",
                    difficulty=MagicMock(value="Easy"),
                    timezone="UTC",
                    notification_time="09:00",
                )
            )
            mock_user_service.update_preferences = AsyncMock()

            mock_update.message.text = "1"
            mock_context.chat_data = {}

            result = await set_difficulty(mock_update, mock_context)

            assert result == SET_DIFFICULTY
            mock_update.message.reply_text.assert_called_once()
            call_args = mock_update.message.reply_text.call_args[0][0]
            assert "Select difficulty" in call_args
            assert mock_context.chat_data["waiting_for"] == "difficulty_level"

    @pytest.mark.asyncio
    async def test_set_difficulty_update_difficulty(self, mock_update, mock_context):
        with patch("main.user_service") as mock_user_service:
            mock_user_service.get_user = AsyncMock(
                return_value=MagicMock(telegram_id="12345")
            )
            mock_user_service.update_preferences = AsyncMock()

            mock_update.message.text = "2"
            mock_context.chat_data = {"waiting_for": "difficulty_level"}

            result = await set_difficulty(mock_update, mock_context)

            assert result == ConversationHandler.END
            mock_user_service.update_preferences.assert_called_once()
            call_args = mock_user_service.update_preferences.call_args
            assert call_args[1]["difficulty"] == Difficulty.MEDIUM
            mock_update.message.reply_text.assert_called_once_with(
                "Difficulty updated to Medium!"
            )

    @pytest.mark.asyncio
    async def test_set_difficulty_update_timezone(self, mock_update, mock_context):
        with patch("main.user_service") as mock_user_service:
            mock_user_service.get_user = AsyncMock(
                return_value=MagicMock(telegram_id="12345")
            )
            mock_user_service.update_preferences = AsyncMock()

            mock_update.message.text = "America/New_York"
            mock_context.chat_data = {"waiting_for": "timezone"}

            result = await set_difficulty(mock_update, mock_context)

            assert result == ConversationHandler.END
            mock_user_service.update_preferences.assert_called_once_with(
                "12345", timezone="America/New_York"
            )
            mock_update.message.reply_text.assert_called_once_with(
                "Timezone updated to America/New_York!"
            )

    @pytest.mark.asyncio
    async def test_set_difficulty_update_notification_time(
        self, mock_update, mock_context
    ):
        with patch("main.user_service") as mock_user_service:
            mock_user_service.get_user = AsyncMock(
                return_value=MagicMock(telegram_id="12345")
            )
            mock_user_service.update_preferences = AsyncMock()

            mock_update.message.text = "18:30"
            mock_context.chat_data = {"waiting_for": "time"}

            result = await set_difficulty(mock_update, mock_context)

            assert result == ConversationHandler.END
            mock_user_service.update_preferences.assert_called_once_with(
                "12345", notification_time="18:30"
            )
            mock_update.message.reply_text.assert_called_once_with(
                "Notification time updated to 18:30!"
            )

    @pytest.mark.asyncio
    async def test_set_difficulty_invalid_choice(self, mock_update, mock_context):
        mock_update.message.text = "5"
        mock_context.chat_data = {}

        result = await set_difficulty(mock_update, mock_context)

        assert result == SET_DIFFICULTY
        mock_update.message.reply_text.assert_called_once_with(
            "Invalid choice. Please reply with 1, 2, or 3."
        )
