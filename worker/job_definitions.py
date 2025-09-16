"""
Shared job type definitions and payload structures for worker and server.
This file is copied from server/src/job_queue/job_definitions.py for type agreement.
"""

from enum import Enum
from typing import Dict, Any, List
from pydantic import BaseModel


class JobType(str, Enum):
    """Enumeration of all supported job types"""
    GENERATE_REQUIREMENTS = "generate_requirements"
    GENERATE_PROMPTS = "generate_prompts"
    CREATE_CORRECTION = "create_correction"


class GenerateRequirementsPayload(BaseModel):
    """Payload structure for requirements generation jobs"""
    assignment_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class GeneratePromptsPayload(BaseModel):
    """Payload structure for prompt generation jobs"""
    assignment_id: int
    requirement_ids: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class CreateCorrectionPayload(BaseModel):
    """Payload structure for correction creation jobs"""
    submission_id: int
    requirement_ids: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class JobDefinition(BaseModel):
    """Complete job definition with type and payload"""
    job_type: JobType
    payload: Dict[str, Any]
    
    @classmethod
    def create_generate_requirements(cls, assignment_id: int) -> 'JobDefinition':
        """Create a requirements generation job definition"""
        payload = GenerateRequirementsPayload(assignment_id=assignment_id)
        return cls(
            job_type=JobType.GENERATE_REQUIREMENTS,
            payload=payload.to_dict()
        )
    
    @classmethod
    def create_generate_prompts(cls, assignment_id: int, requirement_ids: List[int]) -> 'JobDefinition':
        """Create a prompts generation job definition"""
        payload = GeneratePromptsPayload(
            assignment_id=assignment_id,
            requirement_ids=requirement_ids
        )
        return cls(
            job_type=JobType.GENERATE_PROMPTS,
            payload=payload.to_dict()
        )
    
    @classmethod
    def create_correction(cls, submission_id: int, requirement_ids: List[int]) -> 'JobDefinition':
        """Create a correction creation job definition"""
        payload = CreateCorrectionPayload(
            submission_id=submission_id,
            requirement_ids=requirement_ids
        )
        return cls(
            job_type=JobType.CREATE_CORRECTION,
            payload=payload.to_dict()
        )


# Job validation functions
def validate_generate_requirements_payload(payload: Dict[str, Any]) -> GenerateRequirementsPayload:
    """Validate and parse requirements generation payload"""
    return GenerateRequirementsPayload(**payload)


def validate_generate_prompts_payload(payload: Dict[str, Any]) -> GeneratePromptsPayload:
    """Validate and parse prompts generation payload"""
    return GeneratePromptsPayload(**payload)


def validate_create_correction_payload(payload: Dict[str, Any]) -> CreateCorrectionPayload:
    """Validate and parse correction creation payload"""
    return CreateCorrectionPayload(**payload)


# Payload validation mapping
PAYLOAD_VALIDATORS = {
    JobType.GENERATE_REQUIREMENTS: validate_generate_requirements_payload,
    JobType.GENERATE_PROMPTS: validate_generate_prompts_payload,
    JobType.CREATE_CORRECTION: validate_create_correction_payload,
}
