import os
import requests
from typing import Dict, Any, List, Optional
# Handle imports for both script execution and module import
try:
    from .models import (
        RequirementCreate, RequirementUpdate, CorrectionCreate,
        RequirementResponse, SubmissionResponse, CorrectionResponse,
        AssignmentCreate, AssignmentUpdate, AssignmentResponse,
        AssignmentCreatedResponse, RequirementCreatedResponse, SubmissionCreatedResponse, MessageResponse, SubmissionCreate
    )
except ImportError:
    # When running as script, use absolute imports
    from models import (
        RequirementCreate, RequirementUpdate, CorrectionCreate,
        RequirementResponse, SubmissionResponse, CorrectionResponse,
        AssignmentCreate, AssignmentUpdate, AssignmentResponse,
        AssignmentCreatedResponse, RequirementCreatedResponse, SubmissionCreatedResponse, MessageResponse, SubmissionCreate
    )
class APIClient:
    """REST API client for communicating with the JADE server"""
    
    def __init__(self, base_url: str = None):
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = os.getenv("JADE_SERVER_URL")
            if not self.base_url:
                raise ValueError("base_url parameter or JADE_SERVER_URL environment variable is required but not set")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "JADE-Worker/1.0"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {method} {url} - {e}")
            raise

    # Assignment Endpoints
    def get_assignment(self, assignment_id: int) -> AssignmentResponse:
        """Get assignment by ID"""
        response = self._make_request("GET", f"/assignment/{assignment_id}")
        return AssignmentResponse(**response.json())

    # Requirement Endpoints  
    def get_requirements(self, assignment_id: int) -> List[RequirementResponse]:
        """Get all requirements for an assignment"""
        response = self._make_request("GET", f"/assignment/{assignment_id}/requirement")
        return [RequirementResponse(**req) for req in response.json()]

    def get_requirement(self, assignment_id: int, requirement_id: int) -> RequirementResponse:
        """Get specific requirement"""
        response = self._make_request("GET", f"/assignment/{assignment_id}/requirement/{requirement_id}")
        return RequirementResponse(**response.json())

    def create_requirement(self, assignment_id: int, requirement: RequirementCreate) -> RequirementCreatedResponse:
        """Create a new requirement (RESTful, returns id and url)"""
        response = self._make_request(
            "POST", 
            f"/assignment/{assignment_id}/requirement",
            json=requirement.model_dump()
        )
        return RequirementCreatedResponse(**response.json())

    def create_submission(self, assignment_id: int, submission: SubmissionCreate) -> SubmissionCreatedResponse:
        """Create a new submission (RESTful, returns id and url)"""
        response = self._make_request(
            "POST",
            f"/assignment/{assignment_id}/submission",
            json=submission.model_dump()
        )
        return SubmissionCreatedResponse(**response.json())

    def delete_requirement(self, assignment_id: int, requirement_id: int) -> MessageResponse:
        """Delete a requirement (RESTful, returns message)"""
        response = self._make_request("DELETE", f"/assignment/{assignment_id}/requirement/{requirement_id}")
        return MessageResponse(**response.json())

    def delete_submission(self, assignment_id: int, submission_id: int) -> MessageResponse:
        """Delete a submission (RESTful, returns message)"""
        response = self._make_request("DELETE", f"/assignment/{assignment_id}/submission/{submission_id}")
        return MessageResponse(**response.json())

    def update_requirement(self, assignment_id: int, requirement_id: int, requirement: RequirementUpdate) -> RequirementResponse:
        """Update a requirement"""
        response = self._make_request(
            "PUT",
            f"/assignment/{assignment_id}/requirement/{requirement_id}",
            json=requirement.model_dump()
        )
        return RequirementResponse(**response.json())

    # Submission Endpoints
    def get_submission(self, assignment_id: int, submission_id: int) -> SubmissionResponse:
        """Get specific submission"""
        response = self._make_request("GET", f"/assignment/{assignment_id}/submission/{submission_id}")
        return SubmissionResponse(**response.json())

    # Correction Endpoints
    def get_corrections(self, assignment_id: int, submission_id: int) -> List[CorrectionResponse]:
        """Get corrections for a submission"""
        response = self._make_request("GET", f"/assignment/{assignment_id}/submission/{submission_id}/correction")
        return [CorrectionResponse(**corr) for corr in response.json()]

    def create_correction(self, assignment_id: int, submission_id: int, correction: CorrectionCreate) -> CorrectionResponse:
        """Create a correction"""
        response = self._make_request(
            "POST",
            f"/assignment/{assignment_id}/submission/{submission_id}/correction",
            json=correction.model_dump()
        )
        return CorrectionResponse(**response.json())

    # Worker Callback Endpoints
    def notify_job_completed(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Notify server that job completed"""
        try:
            response = self._make_request(
                "POST",
                "/internal/job-completed",
                json={"job_id": job_id, "result": result}
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to notify job completion: {e}")
            return False

    def notify_job_failed(self, job_id: str, error: str) -> bool:
        """Notify server that job failed"""
        try:
            response = self._make_request(
                "POST",
                "/internal/job-failed", 
                json={"job_id": job_id, "error": error}
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Failed to notify job failure: {e}")
            return False
