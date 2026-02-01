from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from database.database import Base


class QuestionDB(Base):
    """SQLAlchemy model for LeetCode questions."""
    
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    leetcode_id = Column(Integer, nullable=False, unique=True)
    title = Column(String(255), nullable=False)
    difficulty = Column(String(20), nullable=False)  # Easy, Medium, Hard
    category = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    solution = Column(Text, nullable=True)
    solution_approach = Column(String(255), nullable=True)  # Algorithm type used (e.g., "Hash Map", "Two Pointers")
    examples = Column(JSONB, default=list)
    external_link = Column(String(500), nullable=True)  # Link to LeetCode problem
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Question(id={self.id}, title='{self.title}', difficulty='{self.difficulty}')>"


class AlgorithmDB(Base):
    """SQLAlchemy model for algorithm implementations."""
    
    __tablename__ = "algorithms"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    category = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    implementation = Column(Text, nullable=False)
    complexity = Column(String(100), nullable=False)
    use_cases = Column(JSONB, default=list)
    external_link = Column(String(500), nullable=True)  # Link to more information
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Algorithm(id={self.id}, name='{self.name}', category='{self.category}')>"


class UserDB(Base):
    """SQLAlchemy model for users."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(String(50), nullable=False, unique=True)
    username = Column(String(255), nullable=True)
    timezone = Column(String(100), default="UTC")
    difficulty = Column(String(20), default="Easy")
    notification_time = Column(String(10), default="09:00")
    notifications_enabled = Column(Integer, default=1)  # 1=enabled, 0=disabled
    streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    last_completion_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<User(id={self.id}, telegram_id='{self.telegram_id}', streak={self.streak})>"


class UserQuestionDB(Base):
    """SQLAlchemy model for tracking user question completions."""
    
    __tablename__ = "user_questions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    telegram_id = Column(String(50), nullable=False)
    question_id = Column(Integer, nullable=False)
    completed = Column(Integer, default=0)  # 1=completed, 0=viewed
    completed_at = Column(DateTime, nullable=True)
    viewed_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserQuestion(telegram_id='{self.telegram_id}', question_id={self.question_id})>"
