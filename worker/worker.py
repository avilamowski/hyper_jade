#!/usr/bin/env python3
"""
Independent RQ Worker for JADE Server
Communicates with main server via REST API
"""

import sys
import os
import logging
from rq import Queue, Worker
from redis_conn import RedisConnection
from job_manager import JobManager
from job_handlers import JobHandlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global job handlers instance
job_handlers = JobHandlers()

def process_job(job_id: str, job_type: str, payload: dict):
    """
    Job processing function for RQ worker
    This function signature must match what the main server enqueues
    """
    return job_handlers.process_job(job_id, job_type, payload)

class JadeWorker:
    """Independent JADE worker using REST API"""
    
    def __init__(self):
        self.redis_client = RedisConnection.get_rq_redis()
        self.queue = Queue(connection=self.redis_client)
    
    def start_worker(self):
        """Start the RQ worker"""
        logger.info("🚀 Starting JADE independent worker...")
        
        # Create RQ worker
        worker = Worker([self.queue], connection=self.redis_client)
        
        logger.info("👷 Worker is ready to process jobs...")
        logger.info(f"📡 Connected to Redis: {os.getenv('REDIS_URL', 'redis://localhost:6379')}")
        logger.info(f"🌐 API Server: {os.getenv('JADE_SERVER_URL', 'http://localhost:8000')}")
        
        # Start processing (this blocks)
        worker.work()

def main():
    """Main entry point"""
    print("🔧 JADE Independent Worker v1.0")
    print("=" * 40)
    
    # Test Redis connection
    if not RedisConnection.test_connection():
        print("❌ Failed to connect to Redis. Please ensure Redis is running.")
        print("   Default: redis://localhost:6379")
        print(f"   Current: {os.getenv('REDIS_URL', 'redis://localhost:6379')}")
        sys.exit(1)
    
    print("✅ Connected to Redis successfully")
    
    # Test API connection
    try:
        from api_client import APIClient
        api_client = APIClient()
        # Try a simple request to test connectivity
        print(f"🌐 API Server: {api_client.base_url}")
        print("✅ API client initialized")
    except Exception as e:
        print(f"⚠️  API client warning: {e}")
        print("   Worker will still start, but jobs may fail if API is unavailable")
    
    # Start worker
    worker = JadeWorker()
    
    try:
        worker.start_worker()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down worker...")
        logger.info("Worker shutdown by user")
    except Exception as e:
        print(f"❌ Worker error: {e}")
        logger.error(f"Worker error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()