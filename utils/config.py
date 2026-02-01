import os
from dataclasses import dataclass


@dataclass
class Config:
    bot_token: str = ""
    redis_url: str = "redis://localhost:6379/0"
    graphql_url: str = "http://localhost:8080/v1/graphql"
    default_timezone: str = "UTC"
    default_notification_time: str = "09:00"
    cache_ttl: int = 86400

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            bot_token=os.getenv("BOT_TOKEN", ""),
            redis_url=os.getenv("REDIS_URL", cls.redis_url),
            graphql_url=os.getenv("GRAPHQL_URL", cls.graphql_url),
            default_timezone=os.getenv("DEFAULT_TIMEZONE", cls.default_timezone),
            default_notification_time=os.getenv("DEFAULT_NOTIFICATION_TIME", cls.default_notification_time),
            cache_ttl=int(os.getenv("CACHE_TTL", str(cls.cache_ttl))),
        )


config = Config.from_env()
