import pytest
from datetime import datetime
from models import User, Question, QuestionExample, Algorithm, Difficulty


class TestDifficulty:
    def test_difficulty_values(self):
        assert Difficulty.EASY.value == "Easy"
        assert Difficulty.MEDIUM.value == "Medium"
        assert Difficulty.HARD.value == "Hard"


class TestUser:
    def test_user_creation(self):
        user = User(
            id=1,
            telegram_id="12345",
            username="testuser",
            timezone="UTC",
            difficulty=Difficulty.EASY,
            notification_time="09:00",
        )
        assert user.telegram_id == "12345"
        assert user.username == "testuser"
        assert user.timezone == "UTC"
        assert user.difficulty == Difficulty.EASY
        assert user.streak == 0
        assert user.total_questions == 0

    def test_user_defaults(self):
        user = User(telegram_id="12345")
        assert user.timezone == "UTC"
        assert user.difficulty == Difficulty.EASY
        assert user.notification_time == "09:00"
        assert user.streak == 0
        assert user.total_questions == 0
        assert user.username is None


class TestQuestionExample:
    def test_question_example_creation(self):
        example = QuestionExample(
            input="[1,2,3]", output="6", explanation="Sum of array"
        )
        assert example.input == "[1,2,3]"
        assert example.output == "6"
        assert example.explanation == "Sum of array"


class TestQuestion:
    def test_question_creation(self):
        question = Question(
            id=1,
            leetcode_id=1,
            title="Two Sum",
            difficulty=Difficulty.EASY,
            category="Array",
            description="Find two numbers that add up to target",
        )
        assert question.id == 1
        assert question.leetcode_id == 1
        assert question.title == "Two Sum"
        assert question.difficulty == Difficulty.EASY
        assert question.examples == []

    def test_question_with_examples(self):
        example = QuestionExample(input="test", output="result")
        question = Question(id=2, title="Test Question", examples=[example])
        assert len(question.examples) == 1
        assert question.examples[0].input == "test"


class TestAlgorithm:
    def test_algorithm_creation(self):
        algorithm = Algorithm(
            id=1,
            name="Binary Search",
            category="Search",
            description="Search in sorted array",
            implementation="def binary_search(arr, target): pass",
            complexity="O(log n)",
            use_cases=["Finding elements"],
        )
        assert algorithm.id == 1
        assert algorithm.name == "Binary Search"
        assert algorithm.category == "Search"
        assert algorithm.use_cases == ["Finding elements"]

    def test_algorithm_defaults(self):
        algorithm = Algorithm(name="Test")
        assert algorithm.use_cases == []
