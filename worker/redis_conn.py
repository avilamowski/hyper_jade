import os
import redis
from typing import Optional

# Redis connection configuration (required)
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is required but not set")

class RedisConnection:
    """Redis connection management for the worker"""
    
    _instance: Optional[redis.Redis] = None
    _rq_instance: Optional[redis.Redis] = None
    
    @classmethod
    def get_redis(cls) -> redis.Redis:
        """Get Redis connection instance for general use (with decode_responses)"""
        if cls._instance is None:
            cls._instance = redis.from_url(REDIS_URL, decode_responses=True)
        return cls._instance
    
    @classmethod
    def get_rq_redis(cls) -> redis.Redis:
        """Get Redis connection instance for RQ (without decode_responses)"""
        if cls._rq_instance is None:
            cls._rq_instance = redis.from_url(REDIS_URL, decode_responses=False)
        return cls._rq_instance
    
    @classmethod
    def test_connection(cls) -> bool:
        """Test Redis connection"""
        try:
            redis_client = cls.get_redis()
            redis_client.ping()
            return True
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False
