# Import the process_job function to make it available at module level
from .job_queue.worker import process_job

# Make it available for RQ
__all__ = ['process_job']
