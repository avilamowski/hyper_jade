#!/usr/bin/env python3
"""
Independent RQ Worker for JADE Server
Communicates with main server via REST API
"""

import sys
import os
import logging
from rq import Queue, Worker

# Import worker modules - works both as module and direct execution
try:
    from .redis_conn import RedisConnection
    from .job_manager import JobManager
    from .job_handlers import JobHandlers
except ImportError:
    from redis_conn import RedisConnection
    from job_manager import JobManager
    from job_handlers import JobHandlers

import json
import subprocess
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _spawn_job_process(job_id: str, job_type: str, payload: dict, timeout: int | None = None) -> dict:
    """Spawn a short-lived Python process to run the actual job.

    Communication is done via temporary JSON files. The child process will
    read the input file, execute the job, write a JSON result to the output
    file, and exit. This keeps the parent (worker/orchestrator) process
    alive while allowing tracing libraries to finish when the child exits.
    """
    # Prepare temp files
    in_f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    out_f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    try:
        json.dump({"job_id": job_id, "job_type": job_type, "payload": payload}, in_f)
        in_f.flush()
        in_f.close()
        out_f.close()

        # Call the same module as a child process with flags to run the job
        cmd = [sys.executable, os.path.abspath(__file__), '--run-job', in_f.name, out_f.name]
        logger.info(f"📣 Spawning child process for job {job_id}: {cmd}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if proc.returncode != 0:
            logger.error(f"Child process failed (rc={proc.returncode}): stdout={proc.stdout} stderr={proc.stderr}")
            # Try to read out file if exists to give more context
        try:
            with open(out_f.name, 'r') as f:
                result = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read child output file: {e}")
            raise RuntimeError(f"Child process failed: {proc.stderr or proc.stdout}") from e

        return result
    finally:
        # Cleanup temp files if they still exist
        try:
            os.unlink(in_f.name)
        except Exception:
            pass
        try:
            os.unlink(out_f.name)
        except Exception:
            pass


# Note: process_job function is now defined in job_queue/worker.py
# and imported via __init__.py to avoid RQ ambiguity

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
        try:
            from .api_client import APIClient
        except ImportError:
            from api_client import APIClient
        api_client = APIClient()
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

def _run_job_child(input_path: str, output_path: str) -> None:
    """Child process entrypoint: read input JSON, run job, write output JSON."""
    import traceback
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        job_id = data.get('job_id')
        job_type = data.get('job_type')
        payload = data.get('payload')

        logger.info(f"🔧 Child process: running job {job_id} type={job_type}")

        job_handlers = JobHandlers()
        result = job_handlers.process_job(job_id, job_type, payload)

        out = {"status": "ok", "result": result}
        with open(output_path, 'w') as f:
            json.dump(out, f)
        logger.info(f"✅ Child process completed job {job_id}")
        sys.exit(0)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Child job runner exception: {e}\n{tb}")
        out = {"status": "error", "error": str(e), "traceback": tb}
        try:
            with open(output_path, 'w') as f:
                json.dump(out, f)
        except Exception:
            pass
        sys.exit(2)


if __name__ == "__main__":
    # If invoked with --run-job <in> <out>, run the child job runner and exit.
    if len(sys.argv) >= 4 and sys.argv[1] == '--run-job':
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        _run_job_child(input_path, output_path)
    else:
        main()