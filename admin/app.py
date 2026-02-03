import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqladmin import Admin, ModelView
from sqlalchemy import create_engine

from database.db_models import UserDB, QuestionDB, AlgorithmDB, UserQuestionDB
from utils.config import config

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://bot:botpassword@localhost:5432/leetcode_bot"
)

sync_database_url = DATABASE_URL.replace("+asyncpg", "+psycopg2")

sync_engine = create_engine(sync_database_url, echo=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Admin API started")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

admin = Admin(app, sync_engine)


class UserAdmin(ModelView, model=UserDB):
    column_list = [
        UserDB.id,
        UserDB.telegram_id,
        UserDB.username,
        UserDB.streak,
        UserDB.longest_streak,
        UserDB.total_questions,
        UserDB.difficulty,
        UserDB.notifications_enabled,
        UserDB.created_at,
        UserDB.last_active,
    ]
    column_searchable_list = [UserDB.telegram_id, UserDB.username]
    column_sortable_list = [
        UserDB.id,
        UserDB.streak,
        UserDB.longest_streak,
        UserDB.total_questions,
        UserDB.created_at,
        UserDB.last_active,
    ]
    column_filters = [UserDB.difficulty, UserDB.notifications_enabled]
    can_create = False
    can_delete = False
    can_edit = True
    name = "User"
    name_plural = "Users"


class QuestionAdmin(ModelView, model=QuestionDB):
    column_list = [
        QuestionDB.id,
        QuestionDB.leetcode_id,
        QuestionDB.title,
        QuestionDB.difficulty,
        QuestionDB.category,
        QuestionDB.created_at,
    ]
    column_searchable_list = [QuestionDB.title, QuestionDB.category]
    column_sortable_list = [
        QuestionDB.id,
        QuestionDB.leetcode_id,
        QuestionDB.difficulty,
    ]
    column_filters = [QuestionDB.difficulty, QuestionDB.category]
    can_create = False
    can_delete = False
    can_edit = True
    name = "Question"
    name_plural = "Questions"


class AlgorithmAdmin(ModelView, model=AlgorithmDB):
    column_list = [
        AlgorithmDB.id,
        AlgorithmDB.name,
        AlgorithmDB.category,
        AlgorithmDB.complexity,
        AlgorithmDB.created_at,
    ]
    column_searchable_list = [AlgorithmDB.name, AlgorithmDB.category]
    column_sortable_list = [AlgorithmDB.id, AlgorithmDB.category]
    column_filters = [AlgorithmDB.category]
    can_create = False
    can_delete = False
    can_edit = True
    name = "Algorithm"
    name_plural = "Algorithms"


class UserQuestionAdmin(ModelView, model=UserQuestionDB):
    column_list = [
        UserQuestionDB.id,
        UserQuestionDB.telegram_id,
        UserQuestionDB.question_id,
        UserQuestionDB.completed,
        UserQuestionDB.viewed_at,
        UserQuestionDB.completed_at,
    ]
    column_searchable_list = [UserQuestionDB.telegram_id]
    column_sortable_list = [
        UserQuestionDB.id,
        UserQuestionDB.question_id,
        UserQuestionDB.viewed_at,
    ]
    column_filters = [UserQuestionDB.completed]
    can_create = False
    can_delete = False
    can_edit = True
    name = "User Question"
    name_plural = "User Questions"


admin.add_view(UserAdmin)
admin.add_view(QuestionAdmin)
admin.add_view(AlgorithmAdmin)
admin.add_view(UserQuestionAdmin)
