import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from models import User, Difficulty
from services.question_service import question_service
from services.user_service import user_service
from services.algorithm_service import algorithm_service
from services.notification_service import notification_service
from utils.config import config
from utils.rate_limiter import rate_limit
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

SET_DIFFICULTY, SET_TIMEZONE, SET_TIME = range(3)


# Inline keyboard for main menu
def get_main_menu_keyboard():
    """Create inline keyboard for main menu."""
    keyboard = [
        [
            InlineKeyboardButton("📅 Daily Question", callback_data="daily"),
            InlineKeyboardButton("📚 Algorithm Gallery", callback_data="gallery"),
        ],
        [
            InlineKeyboardButton("📊 My Stats", callback_data="stats"),
            InlineKeyboardButton("🔥 Streak", callback_data="streak"),
        ],
        [
            InlineKeyboardButton("⚙️ Preferences", callback_data="preferences"),
            InlineKeyboardButton("❓ Help", callback_data="help"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def get_question_keyboard(question_id: int):
    """Create inline keyboard for question actions."""
    keyboard = [
        [
            InlineKeyboardButton("💡 Show Solution", callback_data=f"solution_{question_id}"),
            InlineKeyboardButton("✅ Mark Complete", callback_data=f"complete_{question_id}"),
        ],
        [
            InlineKeyboardButton("📅 Next Question", callback_data="daily"),
            InlineKeyboardButton("🏠 Main Menu", callback_data="menu"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


@rate_limit(cooldown=2)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None:
        return

    existing_user = await user_service.get_user(str(user.id))
    if existing_user is None:
        await user_service.create_user(str(user.id), user.username)

    welcome_message = (
        f"Welcome to LeetCode Bot, {user.first_name}! 🎯\n\n"
        "I'm here to help you improve your coding skills with daily LeetCode questions.\n\n"
        "Use the buttons below or type a command to get started!"
    )
    await update.message.reply_text(
        welcome_message, 
        reply_markup=get_main_menu_keyboard()
    )


@rate_limit(cooldown=2)
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "📚 *LeetCode Bot - Complete Command Reference*\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "📋 *DAILY PRACTICE*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "*/daily* - Get today's LeetCode question\n"
        "  _Receive a random question based on your difficulty preference_\n\n"
        "*/solution* - View the solution\n"
        "  _Shows Python solution with approach explanation_\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 *PROGRESS TRACKING*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "*/stats* - View your learning statistics\n"
        "  _Shows streak, total questions, and preferences_\n\n"
        "*/streak* - Display detailed streak information\n"
        "  _Current streak, longest streak, and last completion_\n\n"
        "*/history* - Show recently completed questions\n"
        "  _List of your last 10 completed questions_\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "📚 *ALGORITHM GALLERY*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "*/gallery* - Browse all algorithms\n"
        "  _View algorithms organized by category_\n\n"
        "*/search* `<query>` - Search algorithms\n"
        "  _Example: /search binary search_\n\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "⚙️ *SETTINGS*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "*/preferences* - Update your settings\n"
        "  _Difficulty, timezone, notification time_\n\n"
        "*/help* - Show this help message\n"
    )
    keyboard = [
        [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
    ]
    await update.message.reply_text(
        help_text, 
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


@rate_limit(cooldown=3)
async def daily_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None:
        return

    user_data = await user_service.get_user(str(user.id))
    difficulty = user_data.difficulty if user_data else None

    question = await question_service.get_daily_question(difficulty)
    
    # Store current question in context for /solution command
    context.user_data["current_question_id"] = question.id

    message = (
        f"📅 *Today's Question*\n\n"
        f"*{question.title}* ({question.difficulty.value})\n"
        f"Category: {question.category}\n"
    )
    if question.solution_approach:
        message += f"Solution Approach: {question.solution_approach}\n"
    message += f"\n{question.description}\n\n"
    if question.external_link:
        message += f"🔗 [View on LeetCode]({question.external_link})\n\n"
    
    await update.message.reply_text(
        message, 
        parse_mode="Markdown", 
        disable_web_page_preview=True,
        reply_markup=get_question_keyboard(question.id)
    )


@rate_limit(cooldown=2)
async def get_solution(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None:
        return

    question = await question_service.get_daily_question()

    if question.solution:
        message = (
            f"💡 *Solution for {question.title}*\n\n"
        )
        if question.solution_approach:
            message += f"*Approach:* {question.solution_approach}\n\n"
        message += f"```python\n{question.solution}\n```"
        
        if question.external_link:
            message += f"\n\n🔗 [View on LeetCode]({question.external_link})"
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Mark Complete", callback_data=f"complete_{question.id}"),
                InlineKeyboardButton("📅 Next Question", callback_data="daily"),
            ]
        ]
        await update.message.reply_text(
            message, 
            parse_mode="Markdown",
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.message.reply_text("No solution available yet for this question.")


@rate_limit(cooldown=3)
async def gallery(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    algorithms = await algorithm_service.get_all_algorithms()
    categories = await algorithm_service.get_categories()

    message = "📚 *Algorithm Gallery*\n\n*Categories:*\n"
    for cat in categories:
        message += f"• {cat}\n"

    message += "\n*Available Algorithms:*\n"
    for algo in algorithms:
        link_text = f" - [Learn more]({algo.external_link})" if algo.external_link else ""
        message += f"• {algo.name} ({algo.category}){link_text}\n"

    message += "\nUse /search <query> to find a specific algorithm."

    keyboard = [
        [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
    ]
    await update.message.reply_text(
        message, 
        parse_mode="Markdown", 
        disable_web_page_preview=True,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


@rate_limit(cooldown=2)
async def search_algorithms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Please provide a search query. Example: /search binary search"
        )
        return

    query = " ".join(context.args)
    results = await algorithm_service.search_algorithms(query)

    if not results:
        await update.message.reply_text(
            f"No algorithms found for '{query}'. Try a different search term."
        )
        return

    message = f"🔍 *Search Results for '{query}'*\n\n"
    for algo in results:
        message += f"*{algo.name}* ({algo.category})\n{algo.description}\n"
        if algo.external_link:
            message += f"🔗 [Learn more]({algo.external_link})\n"
        message += "\n"

    await update.message.reply_text(
        message, 
        parse_mode="Markdown", 
        disable_web_page_preview=True
    )


@rate_limit(cooldown=2)
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None:
        return

    user_data = await user_service.get_user(str(user.id))
    if user_data is None:
        await update.message.reply_text("Use /start first to initialize your profile.")
        return

    stats = await user_service.get_stats(str(user.id))
    if stats is None:
        return

    message = (
        f"📊 *Your Statistics*\n\n"
        f"🔥 Current Streak: {stats['streak']} days\n"
        f"🏆 Longest Streak: {stats.get('longest_streak', stats['streak'])} days\n"
        f"✅ Total Questions: {stats['total_questions']}\n"
        f"📈 Preferred Difficulty: {stats['difficulty']}\n"
        f"🌍 Timezone: {stats['timezone']}\n"
        f"⏰ Notification Time: {stats['notification_time']}"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🔥 Streak Details", callback_data="streak"),
            InlineKeyboardButton("📜 History", callback_data="history"),
        ],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
    ]
    await update.message.reply_text(
        message, 
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


@rate_limit(cooldown=2)
async def streak_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show detailed streak information."""
    user = update.effective_user
    if user is None:
        return

    user_data = await user_service.get_user(str(user.id))
    if user_data is None:
        await update.message.reply_text("Use /start first to initialize your profile.")
        return

    streak_info = await user_service.get_streak_info(str(user.id))
    if streak_info is None:
        await update.message.reply_text("Could not retrieve streak information.")
        return

    # Create streak visualization
    streak_count = streak_info.get("current_streak", 0)
    fire_icons = "🔥" * min(streak_count, 10)
    if streak_count > 10:
        fire_icons += f" +{streak_count - 10}"

    last_completion = streak_info.get("last_completion")
    last_completion_str = last_completion.strftime("%Y-%m-%d %H:%M UTC") if last_completion else "Never"

    message = (
        f"🔥 *Streak Information*\n\n"
        f"*Current Streak:* {streak_count} days\n"
        f"{fire_icons}\n\n"
        f"*Longest Streak:* {streak_info.get('longest_streak', 0)} days\n"
        f"*Total Questions:* {streak_info.get('total_questions', 0)}\n"
        f"*Last Completion:* {last_completion_str}\n\n"
    )
    
    if streak_count == 0:
        message += "💪 Start your streak today with /daily!"
    elif streak_count < 7:
        message += f"📈 Keep going! {7 - streak_count} more days to reach a week!"
    elif streak_count < 30:
        message += f"🌟 Great progress! {30 - streak_count} more days to reach a month!"
    else:
        message += "🏆 Amazing dedication! You're a coding champion!"

    keyboard = [
        [
            InlineKeyboardButton("📅 Daily Question", callback_data="daily"),
            InlineKeyboardButton("📜 History", callback_data="history"),
        ],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
    ]
    await update.message.reply_text(
        message, 
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


@rate_limit(cooldown=2)
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user's question completion history."""
    user = update.effective_user
    if user is None:
        return

    user_data = await user_service.get_user(str(user.id))
    if user_data is None:
        await update.message.reply_text("Use /start first to initialize your profile.")
        return

    history = await user_service.get_history(str(user.id), limit=10)
    
    if not history:
        message = (
            "📜 *Question History*\n\n"
            "You haven't completed any questions yet.\n\n"
            "Start with /daily to get your first question!"
        )
    else:
        message = "📜 *Recently Completed Questions*\n\n"
        for i, item in enumerate(history, 1):
            completed_at = item["completed_at"].strftime("%m/%d") if item["completed_at"] else ""
            difficulty_emoji = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(item["difficulty"], "⚪")
            
            message += f"{i}. {difficulty_emoji} *{item['title']}*\n"
            message += f"   {item['category']} • {completed_at}\n"
            if item.get("external_link"):
                message += f"   [View on LeetCode]({item['external_link']})\n"
            message += "\n"

    keyboard = [
        [
            InlineKeyboardButton("📅 Daily Question", callback_data="daily"),
            InlineKeyboardButton("🔥 Streak", callback_data="streak"),
        ],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
    ]
    await update.message.reply_text(
        message, 
        parse_mode="Markdown",
        disable_web_page_preview=True,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button callbacks."""
    query = update.callback_query
    await query.answer()
    
    user = update.effective_user
    if user is None:
        return
    
    data = query.data
    
    if data == "menu":
        await query.edit_message_text(
            "🏠 *Main Menu*\n\nWhat would you like to do?",
            parse_mode="Markdown",
            reply_markup=get_main_menu_keyboard()
        )
    
    elif data == "daily":
        user_data = await user_service.get_user(str(user.id))
        difficulty = user_data.difficulty if user_data else None
        question = await question_service.get_daily_question(difficulty)
        context.user_data["current_question_id"] = question.id
        
        message = (
            f"📅 *Today's Question*\n\n"
            f"*{question.title}* ({question.difficulty.value})\n"
            f"Category: {question.category}\n"
        )
        if question.solution_approach:
            message += f"Solution Approach: {question.solution_approach}\n"
        message += f"\n{question.description}\n\n"
        if question.external_link:
            message += f"🔗 [View on LeetCode]({question.external_link})"
        
        await query.edit_message_text(
            message,
            parse_mode="Markdown",
            disable_web_page_preview=True,
            reply_markup=get_question_keyboard(question.id)
        )
    
    elif data == "gallery":
        algorithms = await algorithm_service.get_all_algorithms()
        categories = await algorithm_service.get_categories()
        
        message = "📚 *Algorithm Gallery*\n\n*Categories:*\n"
        for cat in categories:
            message += f"• {cat}\n"
        message += "\n*Available Algorithms:*\n"
        for algo in algorithms[:8]:  # Limit to avoid message too long
            message += f"• {algo.name} ({algo.category})\n"
        message += "\nUse /search <query> to find more."
        
        keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]]
        await query.edit_message_text(
            message,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "stats":
        stats = await user_service.get_stats(str(user.id))
        if stats:
            message = (
                f"📊 *Your Statistics*\n\n"
                f"🔥 Streak: {stats['streak']} days\n"
                f"✅ Total: {stats['total_questions']} questions\n"
                f"📈 Difficulty: {stats['difficulty']}"
            )
        else:
            message = "Use /start first to initialize your profile."
        
        keyboard = [
            [InlineKeyboardButton("🔥 Streak", callback_data="streak")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
        ]
        await query.edit_message_text(
            message,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "streak":
        streak_info = await user_service.get_streak_info(str(user.id))
        if streak_info:
            streak_count = streak_info.get("current_streak", 0)
            fire_icons = "🔥" * min(streak_count, 7)
            message = (
                f"🔥 *Streak Info*\n\n"
                f"Current: {streak_count} days {fire_icons}\n"
                f"Longest: {streak_info.get('longest_streak', 0)} days\n"
                f"Total: {streak_info.get('total_questions', 0)} questions"
            )
        else:
            message = "Use /start first to initialize your profile."
        
        keyboard = [
            [InlineKeyboardButton("📅 Daily", callback_data="daily")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
        ]
        await query.edit_message_text(
            message,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "history":
        history = await user_service.get_history(str(user.id), limit=5)
        if history:
            message = "📜 *Recent Questions*\n\n"
            for item in history:
                emoji = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}.get(item["difficulty"], "⚪")
                message += f"{emoji} {item['title']}\n"
        else:
            message = "📜 No completed questions yet.\nStart with /daily!"
        
        keyboard = [
            [InlineKeyboardButton("📅 Daily", callback_data="daily")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
        ]
        await query.edit_message_text(
            message,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "preferences":
        await query.edit_message_text(
            "⚙️ *Preferences*\n\nUse /preferences command to update your settings.",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]])
        )
    
    elif data == "help":
        help_text = (
            "📚 *Quick Help*\n\n"
            "/daily - Get a question\n"
            "/solution - View solution\n"
            "/streak - Your streak\n"
            "/history - Past questions\n"
            "/gallery - Algorithms\n"
            "/preferences - Settings"
        )
        keyboard = [[InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]]
        await query.edit_message_text(
            help_text,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data.startswith("solution_"):
        question_id = int(data.split("_")[1])
        question = await question_service.get_solution(question_id)
        if question and question.solution:
            message = f"💡 *Solution*\n\n```python\n{question.solution[:500]}...\n```"
            if question.external_link:
                message += f"\n\n[Full solution on LeetCode]({question.external_link})"
        else:
            message = "Solution not available."
        
        keyboard = [
            [InlineKeyboardButton("✅ Complete", callback_data=f"complete_{question_id}")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
        ]
        await query.edit_message_text(
            message,
            parse_mode="Markdown",
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data.startswith("complete_"):
        question_id = int(data.split("_")[1])
        await user_service.record_question_completion(str(user.id), question_id)
        
        streak_info = await user_service.get_streak_info(str(user.id))
        streak = streak_info.get("current_streak", 1) if streak_info else 1
        
        message = (
            f"✅ *Question Completed!*\n\n"
            f"🔥 Current Streak: {streak} days\n\n"
            "Great work! Keep it up! 💪"
        )
        
        keyboard = [
            [InlineKeyboardButton("📅 Next Question", callback_data="daily")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="menu")]
        ]
        await query.edit_message_text(
            message,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )


async def preferences(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    if user is None:
        return ConversationHandler.END

    user_data = await user_service.get_user(str(user.id))
    if user_data is None:
        await update.message.reply_text("Use /start first to initialize your profile.")
        return ConversationHandler.END

    message = (
        f"⚙️ *Your Preferences*\n\n"
        f"Current settings:\n"
        f"Difficulty: {user_data.difficulty.value}\n"
        f"Timezone: {user_data.timezone}\n"
        f"Notification Time: {user_data.notification_time}\n\n"
        "What would you like to update?\n"
        "1. Difficulty (Easy/Medium/Hard)\n"
        "2. Timezone\n"
        "3. Notification Time\n\n"
        "Reply with the number of your choice, or /cancel to go back."
    )
    await update.message.reply_text(message, parse_mode="Markdown")
    return SET_DIFFICULTY


async def set_difficulty(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    choice = update.message.text.strip()

    if choice == "1":
        await update.message.reply_text(
            "Select difficulty:\n1. Easy\n2. Medium\n3. Hard"
        )
        return SET_DIFFICULTY
    elif choice == "2":
        await update.message.reply_text(
            "Enter your timezone (e.g., America/New_York, Europe/London):"
        )
        return SET_TIMEZONE
    elif choice == "3":
        await update.message.reply_text("Enter notification time (e.g., 09:00, 18:30):")
        return SET_TIME
    elif choice in ["1", "2", "3"]:
        difficulty_map = {
            "1": Difficulty.EASY,
            "2": Difficulty.MEDIUM,
            "3": Difficulty.HARD,
        }
        if user := await user_service.get_user(str(update.effective_user.id)):
            await user_service.update_preferences(
                str(update.effective_user.id), difficulty=difficulty_map.get(choice)
            )
            await update.message.reply_text(
                f"Difficulty updated to {difficulty_map[choice].value}!"
            )
        return ConversationHandler.END
    else:
        await update.message.reply_text("Invalid choice. Please reply with 1, 2, or 3.")
        return SET_DIFFICULTY


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Preferences update cancelled.")
    return ConversationHandler.END


async def post_init(application) -> None:
    """Initialize database and notification service on startup."""
    from database.seed_data import seed_database
    
    try:
        await seed_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed (will use fallback): {e}")
    
    # Initialize notification service
    notification_service.initialize(application.bot)
    notification_service.start()
    logger.info("Notification service started")


async def post_shutdown(application) -> None:
    """Cleanup on shutdown."""
    notification_service.stop()
    logger.info("Notification service stopped")


def main() -> None:
    application = (
        ApplicationBuilder()
        .token(os.environ.get("BOT_TOKEN"))
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("daily", daily_question))
    application.add_handler(CommandHandler("solution", get_solution))
    application.add_handler(CommandHandler("gallery", gallery))
    application.add_handler(CommandHandler("search", search_algorithms))
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("streak", streak_command))
    application.add_handler(CommandHandler("history", history_command))

    # Callback query handler for inline buttons
    application.add_handler(CallbackQueryHandler(button_callback))

    # Conversation handler for preferences
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("preferences", preferences)],
        states={
            SET_DIFFICULTY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_difficulty),
            ],
            SET_TIMEZONE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_difficulty),
            ],
            SET_TIME: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_difficulty),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    application.add_handler(conv_handler)

    application.run_polling()


if __name__ == "__main__":
    main()
