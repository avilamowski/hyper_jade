"""
Compatibility worker module for the independent worker.

This module provides the same function interface that the main server
expects when enqueuing jobs, but routes them to our independent worker
implementation.
"""

from job_handlers import JobHandlers

# Create global job handlers instance
_job_handlers = JobHandlers()

def process_job(job_id: str, job_type: str, payload: dict):
    """
    Job processing function that matches the signature expected by the main server.
    This function will be called by RQ when processing jobs enqueued by the main server.
    """
    return _job_handlers.process_job(job_id, job_type, payload)
