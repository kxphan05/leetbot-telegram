from enum import Enum
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


class User(BaseModel):
    id: Optional[int] = None
    telegram_id: str
    username: Optional[str] = None
    timezone: str = "UTC"
    difficulty: Difficulty = Difficulty.EASY
    notification_time: str = "09:00"
    streak: int = 0
    total_questions: int = 0
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None


class QuestionExample(BaseModel):
    input: str
    output: str
    explanation: Optional[str] = None


class Question(BaseModel):
    id: Optional[int] = None
    leetcode_id: int = 0
    title: str = ""
    difficulty: Difficulty = Difficulty.EASY
    category: str = ""
    description: str = ""
    solution: Optional[str] = None
    solution_approach: Optional[str] = None  # Algorithm type used (e.g., "Hash Map", "Two Pointers")
    examples: List[QuestionExample] = Field(default_factory=list)
    external_link: Optional[str] = None  # Link to LeetCode problem
    created_at: Optional[datetime] = None


class UserQuestion(BaseModel):
    id: Optional[int] = None
    user_id: int = 0
    question_id: int = 0
    completed: bool = False
    date_completed: Optional[datetime] = None
    time_taken: Optional[int] = None


class Algorithm(BaseModel):
    id: Optional[int] = None
    name: str = ""
    category: str = ""
    description: str = ""
    implementation: str = ""
    complexity: str = ""
    use_cases: List[str] = Field(default_factory=list)
    external_link: Optional[str] = None  # Link to more information

