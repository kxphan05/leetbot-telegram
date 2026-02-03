# Agent Guidelines for Telegram LeetCode Bot

## Overview

This is a Python Telegram bot project that delivers daily LeetCode questions. The codebase uses async/await patterns, Pydantic models, SQLAlchemy for database operations, and pytest for testing.

## Build, Lint, and Test Commands

### Install Dependencies
```bash
uv sync
```

### Run the Application
```bash
uv run python main.py
```

### Linting (Ruff)
```bash
uvx ruff check .
uvx ruff check --fix .    # Auto-fix issues
```

### Type Checking (if mypy is added)
```bash
uv run mypy .
```

### Testing
```bash
uv run pytest              # Run all tests
uv run pytest -v          # Verbose output
uv run pytest tests/      # Run specific directory
uv run pytest tests/test_bot.py::TestBotCommands::test_start_command  # Single test
uv run pytest -k "test_start"  # Run tests matching pattern
uv run pytest --collect-only  # List all tests without running
```

## Code Style Guidelines

### Imports
- Group imports in this order: stdlib → third-party → local application
- Use absolute imports for internal modules (`from services.question_service import ...`)
- Use relative imports for same-package modules (`from .config import ...`)
- Remove unused imports (ruff F401)

### Type Hints
- Always use type hints for function parameters and return values
- Use `Optional[T]` for nullable types instead of `T | None`
- Use `List[T]`, `Dict[K, V]` from `typing` (not built-in generics)
- Example: `def get_all_algorithms(self) -> List[Algorithm]:`

### Naming Conventions
- **Classes**: PascalCase (`UserService`, `QuestionModel`)
- **Functions/Variables**: snake_case (`get_user`, `user_id`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`)
- **Private Methods/Attributes**: Prefix with underscore (`_cache`, `_get_data`)
- **Async Functions**: Prefix with `async_` for clarity where appropriate

### Async/Await Patterns
- All I/O operations must be async
- Use `async def` for bot command handlers
- Use `await` for all async calls
- Create async context managers for database sessions
- Example:
  ```python
  async with AsyncSessionLocal() as session:
      result = await session.execute(select(User))
      return result.scalars().all()
  ```

### Pydantic Models
- Extend `BaseModel` for all data models
- Use `Field(default_factory=list)` for mutable defaults
- Use Enums extending `str, Enum` for string-based enums
- Use Optional fields with `= None` default
- Example:
  ```python
  class Difficulty(str, Enum):
      EASY = "Easy"
      MEDIUM = "Medium"
      HARD = "Hard"

  class User(BaseModel):
      id: Optional[int] = None
      telegram_id: str
      username: Optional[str] = None
      difficulty: Difficulty = Difficulty.EASY
  ```

### Error Handling
- Use custom exceptions in `utils/exceptions.py`
- Log errors with appropriate levels (`logger.error`, `logger.warning`)
- Gracefully handle database failures with fallbacks
- Return user-friendly error messages in bot responses
- Example pattern:
  ```python
  try:
      async with AsyncSessionLocal() as session:
          # db operation
  except Exception as e:
      logger.warning(f"Database query failed: {e}")
      return self._get_fallback_data()
  ```

### Logging
- Use module-level logger: `logger = logging.getLogger(__name__)`
- Configure logging once at application startup
- Log levels: ERROR for failures, WARNING for recoverable issues, INFO for significant events
- Use `logger.info` for rate limiting and user actions

### File Organization
- **models/**: Pydantic models for data structures
- **services/**: Business logic and external integrations
- **utils/**: Utility functions, config, caching, rate limiting
- **database/**: SQLAlchemy models and database connection
- **tests/**: Unit tests with same directory structure as source

### Bot Command Handlers
- Use `@rate_limit(cooldown=X)` decorator to prevent spam
- Always check `update.effective_user` for None before access
- Return early if user context is missing
- Use inline keyboards for user interaction
- Example:
  ```python
  @rate_limit(cooldown=2)
  async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
      user = update.effective_user
      if user is None:
          return
      # ... rest of handler
  ```

### Configuration
- Use dataclasses for configuration (`@dataclass` decorator)
- Load config from environment variables via `Config.from_env()`
- Use `.env` file for local development (already in `.gitignore`)
- Access via singleton: `from utils.config import config`

### Testing
- Use pytest with `pytest.mark.asyncio` for async tests
- Mock external services with `unittest.mock`
- Use `MagicMock` with `spec=` for type safety
- Use `AsyncMock` for async method mocks
- Follow naming: `test_*.py` files, `Test*` classes, `test_*` functions
- Pattern:
  ```python
  @pytest.mark.asyncio
  async def test_start_command(self, mock_update, mock_context):
      with patch("main.user_service") as mock_user_service:
          mock_user_service.get_user = AsyncMock(return_value=None)
          await start(mock_update, mock_context)
          mock_update.message.reply_text.assert_called_once()
  ```

### Git Workflow
- Create feature branches for new functionality
- Write meaningful commit messages
- Run linting before committing: `uvx ruff check .`
- Ensure all tests pass: `uv run pytest`
