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
        "id",
        "telegram_id",
        "username",
        "streak",
        "longest_streak",
        "total_questions",
        "difficulty",
        "notifications_enabled",
        "created_at",
        "last_active",
    ]
    column_searchable_list = ["telegram_id", "username"]
    column_sortable_list = [
        "id",
        "streak",
        "longest_streak",
        "total_questions",
        "created_at",
        "last_active",
    ]
    can_create = False
    can_delete = False
    can_edit = True
    name = "User"
    name_plural = "Users"


class QuestionAdmin(ModelView, model=QuestionDB):
    column_list = [
        "id",
        "leetcode_id",
        "title",
        "difficulty",
        "category",
        "created_at",
    ]
    column_searchable_list = ["title", "category"]
    column_sortable_list = [
        "id",
        "leetcode_id",
        "difficulty",
    ]
    can_create = False
    can_delete = False
    can_edit = True
    name = "Question"
    name_plural = "Questions"


class AlgorithmAdmin(ModelView, model=AlgorithmDB):
    column_list = [
        "id",
        "name",
        "category",
        "complexity",
        "created_at",
    ]
    column_searchable_list = ["name", "category"]
    column_sortable_list = ["id", "category"]
    can_create = False
    can_delete = False
    can_edit = True
    name = "Algorithm"
    name_plural = "Algorithms"


class UserQuestionAdmin(ModelView, model=UserQuestionDB):
    column_list = [
        "id",
        "telegram_id",
        "question_id",
        "completed",
        "viewed_at",
        "completed_at",
    ]
    column_searchable_list = ["telegram_id"]
    column_sortable_list = [
        "id",
        "question_id",
        "viewed_at",
    ]
    can_create = False
    can_delete = False
    can_edit = True
    name = "User Question"
    name_plural = "User Questions"


admin.add_view(UserAdmin)
admin.add_view(QuestionAdmin)
admin.add_view(AlgorithmAdmin)
admin.add_view(UserQuestionAdmin)
