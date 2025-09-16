from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class MessageResponse(BaseModel):
    message: str

# Standard create response for RESTful POSTs
class CreatedResponse(BaseModel):
    id: int
    url: str

# Assignment create response (for clarity, can alias to CreatedResponse)
class AssignmentCreatedResponse(CreatedResponse):
    pass

# Requirement create response (for clarity, can alias to CreatedResponse)
class RequirementCreatedResponse(CreatedResponse):
    pass

# Submission create response (for clarity, can alias to CreatedResponse)
class SubmissionCreatedResponse(CreatedResponse):
    pass

# Correction create response (for clarity, can alias to CreatedResponse)
class CorrectionCreatedResponse(CreatedResponse):
    pass


class StatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Job Models
class JobData(BaseModel):
    job_id: str
    type: str
    payload: Dict[str, Any]
    session_id: str
    status: StatusEnum


# Assignment Models
class AssignmentCreate(BaseModel):
    name: str
    content: str

class AssignmentUpdate(BaseModel):
    name: str
    content: str

class AssignmentResponse(BaseModel):
    assignment_id: int
    name: str
    content: str

# Requirement Models
class RequirementCreate(BaseModel):
    requirement: str
    function: Optional[str] = None
    type: str
    prompt_template: Optional[str] = None

class RequirementUpdate(BaseModel):
    requirement: str
    function: Optional[str] = None
    type: str
    prompt_template: Optional[str] = None

class RequirementResponse(BaseModel):
    requirement_id: int
    requirement: str
    function: Optional[str] = None
    type: str
    prompt_template: Optional[str] = None

# Submission Models
class SubmissionCreate(BaseModel):
    name: str
    code: str

class SubmissionUpdate(BaseModel):
    name: str
    code: str

class SubmissionResponse(BaseModel):
    submission_id: int
    assignment_id: int
    name: str
    code: str

# Correction Models
class CorrectionCreate(BaseModel):
    requirement_id: int
    result: str

class CorrectionUpdate(BaseModel):
    result: str

class CorrectionDeleteResponse(BaseModel):
    message: str

class CorrectionResponse(BaseModel):
    correction_id: int
    requirement_id: int
    result: str

# Worker Internal Models
class JobCompletedRequest(BaseModel):
    job_id: str
    result: Dict[str, Any]

class JobFailedRequest(BaseModel):
    job_id: str
    error: str
