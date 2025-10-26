import json
from typing import Dict, Any, Optional
from datetime import datetime
# Handle imports for both script execution and module import
try:
    from .redis_conn import RedisConnection
    from .models import StatusEnum, JobData
except ImportError:
    # When running as script, use absolute imports
    from redis_conn import RedisConnection
    from models import StatusEnum, JobData

class JobManager:
    """Redis-based job management for the worker"""
    
    # Redis key prefixes (must match main server)
    JOB_KEY_PREFIX = "job:"
    JOB_RESULT_PREFIX = "job_result:"
    JOB_STATUS_PREFIX = "job_status:"
    
    @classmethod
    def get_job(cls, job_id: str) -> Optional[JobData]:
        """Get job data by ID"""
        redis_client = RedisConnection.get_redis()
        
        job_data = redis_client.hgetall(f"{cls.JOB_KEY_PREFIX}{job_id}")
        if not job_data:
            return None
        
        # Parse JSON fields
        parsed_job = {}
        for key, value in job_data.items():
            if key == "payload":
                try:
                    parsed_job[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed_job[key] = {}
            else:
                parsed_job[key] = value
        
        return JobData(**parsed_job)
    
    @classmethod
    def get_job_status(cls, job_id: str) -> Optional[str]:
        """Get job status"""
        redis_client = RedisConnection.get_redis()
        status = redis_client.get(f"{cls.JOB_STATUS_PREFIX}{job_id}")
        return status
    
    @classmethod
    def update_job_status(cls, job_id: str, status: StatusEnum, error_message: Optional[str] = None):
        """Update job status in Redis"""
        redis_client = RedisConnection.get_redis()
        
        # Update status in Redis
        redis_client.set(f"{cls.JOB_STATUS_PREFIX}{job_id}", status.value)
        redis_client.hset(f"{cls.JOB_KEY_PREFIX}{job_id}", "status", status.value)
        
        # Update timestamps
        now = datetime.utcnow().isoformat()
        if status == StatusEnum.PROCESSING:
            redis_client.hset(f"{cls.JOB_KEY_PREFIX}{job_id}", "started_at", now)
        elif status in [StatusEnum.COMPLETED, StatusEnum.FAILED]:
            redis_client.hset(f"{cls.JOB_KEY_PREFIX}{job_id}", "completed_at", now)
        
        # Update error message if provided
        if error_message:
            redis_client.hset(f"{cls.JOB_KEY_PREFIX}{job_id}", "error_message", error_message)
    
    @classmethod
    def set_job_result(cls, job_id: str, result: Dict[str, Any]):
        """Set job result in Redis"""
        redis_client = RedisConnection.get_redis()
        
        # Store result
        redis_client.set(f"{cls.JOB_RESULT_PREFIX}{job_id}", json.dumps(result))
        redis_client.expire(f"{cls.JOB_RESULT_PREFIX}{job_id}", 86400)  # 24 hour TTL
