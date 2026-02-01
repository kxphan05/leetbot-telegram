import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class MemoryCache:
    """In-memory fallback cache when Redis is unavailable."""
    
    def __init__(self):
        self._data: dict[str, str] = {}
    
    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 86400) -> None:
        self._data[key] = value
    
    async def delete(self, key: str) -> None:
        self._data.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        return key in self._data


class Cache:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client = None
        self._use_memory = False
        self._memory_cache = MemoryCache()

    async def connect(self) -> None:
        if self._client is None and not self._use_memory:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url)
                # Test connection
                await self._client.ping()
                logger.info("Connected to Redis successfully")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory cache: {e}")
                self._use_memory = True
                self._client = None

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def get(self, key: str) -> Optional[str]:
        if self._use_memory:
            return await self._memory_cache.get(key)
        
        if self._client is None:
            await self.connect()
        
        if self._use_memory:
            return await self._memory_cache.get(key)
            
        try:
            return await self._client.get(key)
        except Exception as e:
            logger.warning(f"Redis get failed, using memory cache: {e}")
            self._use_memory = True
            return await self._memory_cache.get(key)

    async def set(self, key: str, value: Any, ttl: int = 86400) -> None:
        if self._use_memory:
            await self._memory_cache.set(key, value, ttl)
            return
            
        if self._client is None:
            await self.connect()
        
        if self._use_memory:
            await self._memory_cache.set(key, value, ttl)
            return
            
        try:
            await self._client.set(key, value, ex=ttl)
        except Exception as e:
            logger.warning(f"Redis set failed, using memory cache: {e}")
            self._use_memory = True
            await self._memory_cache.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        if self._use_memory:
            await self._memory_cache.delete(key)
            return
            
        if self._client is None:
            await self.connect()
        
        if self._use_memory:
            await self._memory_cache.delete(key)
            return
            
        try:
            await self._client.delete(key)
        except Exception as e:
            logger.warning(f"Redis delete failed, using memory cache: {e}")
            self._use_memory = True
            await self._memory_cache.delete(key)

    async def exists(self, key: str) -> bool:
        if self._use_memory:
            return await self._memory_cache.exists(key)
            
        if self._client is None:
            await self.connect()
        
        if self._use_memory:
            return await self._memory_cache.exists(key)
            
        try:
            return await self._client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Redis exists failed, using memory cache: {e}")
            self._use_memory = True
            return await self._memory_cache.exists(key)


cache = Cache()
