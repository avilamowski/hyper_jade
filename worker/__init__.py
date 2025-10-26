# Import the process_job function to make it available at module level
from .main import _spawn_job_process

# Create a process_job function that RQ can import
def process_job(job_id: str, job_type: str, payload: dict):
    """
    Job processing function that matches the signature expected by RQ.
    This function will be called by RQ when processing jobs enqueued by the main server.
    """
    return _spawn_job_process(job_id, job_type, payload)

# Make it available for RQ
__all__ = ['process_job']
