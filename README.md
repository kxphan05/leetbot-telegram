# Telegram LeetCode Bot

A Telegram bot that delivers daily LeetCode questions and provides comprehensive algorithm learning resources to help users improve their coding skills.

## Features

- **Daily Questions**: Get a new LeetCode question every day based on your preferred difficulty
- **Solutions**: Request solutions with detailed explanations and multiple approaches
- **Algorithm Gallery**: Browse algorithms organized by category with implementations
- **Search**: Search for specific algorithms by name or keywords
- **Progress Tracking**: Track your learning progress and streaks
- **Personalization**: Set your preferred difficulty, timezone, and notification times

## Commands

- `/start` - Initialize the bot and set up your profile
- `/daily` - Get today's LeetCode question
- `/solution` - Get the solution for the current question
- `/gallery` - Browse the algorithm gallery
- `/search <query>` - Search algorithms by name or category
- `/stats` - View your learning statistics
- `/preferences` - Update your settings
- `/help` - Show all available commands

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Copy `.env.example` to `.env` and configure your environment variables:
   ```bash
   cp .env.example .env
   ```
4. Add your Telegram bot token from @BotFather
5. (Optional) Set up Redis and Hasura for persistent storage

## Running the Bot

```bash
uv run python main.py
```

## Project Structure

```
├── main.py                 # Bot entry point and command handlers
├── models/
│   └── models.py           # Pydantic models for data structures
├── services/
│   ├── question_service.py # Question fetching and management
│   ├── user_service.py     # User management and preferences
│   ├── algorithm_service.py # Algorithm gallery and search
│   └── graphql_service.py  # GraphQL database integration
├── utils/
│   ├── cache.py           # Redis caching layer
│   └── config.py          # Configuration management
└── tests/                 # Unit tests
```

## Technology Stack

- **python-telegram-bot**: Telegram bot framework
- **Pydantic**: Data validation and serialization
- **Redis**: Caching and session storage
- **aiohttp**: Async HTTP client for GraphQL
- **UV**: Fast Python package manager

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Adding New Algorithms

Edit `services/algorithm_service.py` to add new algorithms to the gallery.

### Adding New Commands

1. Create a new async function in `main.py`
2. Add the command handler to the application
3. Write unit tests in `tests/test_bot.py`

## License

MIT
