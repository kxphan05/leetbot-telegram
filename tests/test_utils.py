import pytest
from utils.config import Config


class TestConfig:
    def test_config_from_env_defaults(self):
        config = Config()
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.graphql_url == "http://localhost:8080/v1/graphql"
        assert config.default_timezone == "UTC"
        assert config.default_notification_time == "09:00"
        assert config.cache_ttl == 86400
