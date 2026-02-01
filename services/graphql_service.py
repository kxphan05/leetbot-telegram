import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import aiohttp
from models import User, Question, UserQuestion
from utils.config import config


class GraphQLClient:
    def __init__(self, graphql_url: str = None, token: str = None):
        self.graphql_url = graphql_url or config.graphql_url
        self.token = token
        self.headers = {
            "Content-Type": "application/json",
            "x-hasura-admin-secret": self.token if self.token else "",
        }

    async def execute(
        self, query: str, variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.graphql_url,
                json=payload,
                headers={k: v for k, v in self.headers.items() if v},
            ) as response:
                return await response.json()


class GraphQLService:
    def __init__(self, client: GraphQLClient = None):
        self.client = client or GraphQLClient()

    async def create_user(self, user: User) -> User:
        query = """
        mutation CreateUser($telegramId: String!, $username: String, $timezone: String!, $difficulty: String!, $notificationTime: String!) {
            insert_users_one(object: {
                telegram_id: $telegramId,
                username: $username,
                timezone: $timezone,
                difficulty: $difficulty,
                notification_time: $notificationTime
            }) {
                id
                telegram_id
                username
                timezone
                difficulty
                notification_time
                streak
                total_questions
                created_at
                last_active
            }
        }
        """
        variables = {
            "telegramId": user.telegram_id,
            "username": user.username,
            "timezone": user.timezone,
            "difficulty": user.difficulty.value,
            "notificationTime": user.notification_time,
        }
        result = await self.client.execute(query, variables)
        return self._parse_user(result.get("data", {}).get("insert_users_one"))

    async def get_user(self, telegram_id: str) -> Optional[User]:
        query = """
        query GetUser($telegramId: String!) {
            users(where: {telegram_id: {_eq: $telegramId}}) {
                id
                telegram_id
                username
                timezone
                difficulty
                notification_time
                streak
                total_questions
                created_at
                last_active
            }
        }
        """
        variables = {"telegramId": telegram_id}
        result = await self.client.execute(query, variables)
        users = result.get("data", {}).get("users", [])
        if users:
            return self._parse_user(users[0])
        return None

    async def update_user(self, user: User) -> User:
        query = """
        mutation UpdateUser($id: ID!, $streak: Int, $totalQuestions: Int, $timezone: String, $difficulty: String, $notificationTime: String) {
            update_users_by_pk(pk_columns: {id: $id}, _set: {
                streak: $streak,
                total_questions: $totalQuestions,
                timezone: $timezone,
                difficulty: $difficulty,
                notification_time: $notificationTime,
                last_active: now()
            }) {
                id
                telegram_id
                username
                timezone
                difficulty
                notification_time
                streak
                total_questions
                created_at
                last_active
            }
        }
        """
        variables = {
            "id": user.id,
            "streak": user.streak,
            "totalQuestions": user.total_questions,
            "timezone": user.timezone,
            "difficulty": user.difficulty.value,
            "notificationTime": user.notification_time,
        }
        result = await self.client.execute(query, variables)
        return self._parse_user(result.get("data", {}).get("update_users_by_pk"))

    async def create_user_question(
        self, user_id: int, question_id: int, completed: bool = False
    ) -> UserQuestion:
        query = """
        mutation CreateUserQuestion($userId: ID!, $questionId: ID!, $completed: Boolean) {
            insert_user_questions_one(object: {
                user_id: $userId,
                question_id: $questionId,
                completed: $completed
            }) {
                id
                user_id
                question_id
                completed
                date_completed
                time_taken
            }
        }
        """
        variables = {
            "userId": user_id,
            "questionId": question_id,
            "completed": completed,
        }
        result = await self.client.execute(query, variables)
        return self._parse_user_question(
            result.get("data", {}).get("insert_user_questions_one")
        )

    async def get_user_history(self, user_id: int) -> List[UserQuestion]:
        query = """
        query GetUserHistory($userId: ID!) {
            user_questions(where: {user_id: {_eq: $userId}}) {
                id
                user_id
                question_id
                completed
                date_completed
                time_taken
            }
        }
        """
        variables = {"userId": user_id}
        result = await self.client.execute(query, variables)
        questions = result.get("data", {}).get("user_questions", [])
        return [self._parse_user_question(q) for q in questions]

    async def create_question(self, question: Question) -> Question:
        query = """
        mutation CreateQuestion($leetcodeId: Int!, $title: String!, $difficulty: String!, $category: String!, $description: String!, $solution: String) {
            insert_questions_one(object: {
                leetcode_id: $leetcodeId,
                title: $title,
                difficulty: $difficulty,
                category: $category,
                description: $description,
                solution: $solution
            }) {
                id
                leetcode_id
                title
                difficulty
                category
                description
                solution
                created_at
            }
        }
        """
        variables = {
            "leetcodeId": question.leetcode_id,
            "title": question.title,
            "difficulty": question.difficulty.value,
            "category": question.category,
            "description": question.description,
            "solution": question.solution,
        }
        result = await self.client.execute(query, variables)
        return self._parse_question(result.get("data", {}).get("insert_questions_one"))

    async def get_question(self, question_id: int) -> Optional[Question]:
        query = """
        query GetQuestion($id: ID!) {
            questions_by_pk(id: $id) {
                id
                leetcode_id
                title
                difficulty
                category
                description
                solution
                created_at
            }
        }
        """
        variables = {"id": question_id}
        result = await self.client.execute(query, variables)
        question = result.get("data", {}).get("questions_by_pk")
        if question:
            return self._parse_question(question)
        return None

    async def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        query = """
        query GetUserStats($userId: ID!) {
            users_by_pk(id: $userId) {
                streak
                total_questions
                difficulty
                timezone
                notification_time
            }
            user_questions_aggregate(where: {user_id: {_eq: $userId}}) {
                aggregate {
                    count
                }
            }
        }
        """
        variables = {"userId": user_id}
        result = await self.client.execute(query, variables)
        user_data = result.get("data", {}).get("users_by_pk")
        if user_data:
            return {
                "streak": user_data.get("streak", 0),
                "total_questions": user_data.get("total_questions", 0),
                "difficulty": user_data.get("difficulty"),
                "timezone": user_data.get("timezone"),
                "notification_time": user_data.get("notification_time"),
            }
        return {}

    def _parse_user(self, data: Dict[str, Any]) -> Optional[User]:
        if not data:
            return None
        from models import Difficulty

        return User(
            id=data.get("id"),
            telegram_id=data.get("telegram_id"),
            username=data.get("username"),
            timezone=data.get("timezone", "UTC"),
            difficulty=Difficulty(data.get("difficulty", "Easy")),
            notification_time=data.get("notification_time", "09:00"),
            streak=data.get("streak", 0),
            total_questions=data.get("total_questions", 0),
            created_at=self._parse_datetime(data.get("created_at")),
            last_active=self._parse_datetime(data.get("last_active")),
        )

    def _parse_question(self, data: Dict[str, Any]) -> Optional[Question]:
        if not data:
            return None
        from models import Difficulty

        return Question(
            id=data.get("id"),
            leetcode_id=data.get("leetcode_id"),
            title=data.get("title"),
            difficulty=Difficulty(data.get("difficulty", "Easy")),
            category=data.get("category"),
            description=data.get("description"),
            solution=data.get("solution"),
            created_at=self._parse_datetime(data.get("created_at")),
        )

    def _parse_user_question(self, data: Dict[str, Any]) -> Optional[UserQuestion]:
        if not data:
            return None
        return UserQuestion(
            id=data.get("id"),
            user_id=data.get("user_id"),
            question_id=data.get("question_id"),
            completed=data.get("completed", False),
            date_completed=self._parse_datetime(data.get("date_completed")),
            time_taken=data.get("time_taken"),
        )

    def _parse_datetime(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None


graphql_service = GraphQLService()
