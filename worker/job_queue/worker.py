"""
Compatibility worker module for the independent worker.

This module provides the same function interface that the main server
expects when enqueuing jobs, but routes them to our independent worker
implementation.
"""

import sys
import os
import json
import subprocess
import tempfile
import logging

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

        # Call the main worker module as a child process with flags to run the job
        worker_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.py')
        cmd = [sys.executable, worker_path, '--run-job', in_f.name, out_f.name]
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

def process_job(job_id: str, job_type: str, payload: dict):
    """
    Job processing function that matches the signature expected by the main server.
    This function will be called by RQ when processing jobs enqueued by the main server.
    It spawns a child Python process that runs the actual job and returns the child's result.
    This keeps the parent worker long-running but makes the job execution ephemeral so traces finish.
    """
    return _spawn_job_process(job_id, job_type, payload)
