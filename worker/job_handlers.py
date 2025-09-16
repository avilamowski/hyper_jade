
import time
import logging
from typing import Dict, Any, List
from api_client import APIClient
from job_manager import JobManager
from models import StatusEnum, RequirementCreate, RequirementUpdate, CorrectionCreate
from job_definitions import (
    JobType,
    GenerateRequirementsPayload,
    GeneratePromptsPayload,
    CreateCorrectionPayload,
    PAYLOAD_VALIDATORS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobHandlers:
    """Job processing handlers using REST API calls"""
    
    def __init__(self):
        self.api_client = APIClient()
    
    def process_job(self, job_id: str, job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a job based on its type, using validated models from job_definitions."""
        logger.info(f"🔄 Processing job {job_id} of type {job_type}")
        JobManager.update_job_status(job_id, StatusEnum.PROCESSING)
        try:
            # Validate and parse payload using shared models
            if job_type == JobType.GENERATE_REQUIREMENTS or job_type == "generate_requirements":
                parsed_payload = PAYLOAD_VALIDATORS[JobType.GENERATE_REQUIREMENTS](payload)
                result = self.handle_generate_requirements(parsed_payload)
            elif job_type == JobType.GENERATE_PROMPTS or job_type == "generate_prompts":
                parsed_payload = PAYLOAD_VALIDATORS[JobType.GENERATE_PROMPTS](payload)
                result = self.handle_generate_prompts(parsed_payload)
            elif job_type == JobType.CREATE_CORRECTION or job_type == "create_correction":
                parsed_payload = PAYLOAD_VALIDATORS[JobType.CREATE_CORRECTION](payload)
                result = self.handle_create_correction(parsed_payload)
            else:
                raise ValueError(f"Unknown job type: {job_type}")
            JobManager.update_job_status(job_id, StatusEnum.COMPLETED)
            JobManager.set_job_result(job_id, result)
            self.api_client.notify_job_completed(job_id, result)
            logger.info(f"✅ Job {job_id} completed successfully")
            return result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Job {job_id} failed: {error_msg}")
            JobManager.update_job_status(job_id, StatusEnum.FAILED, error_msg)
            self.api_client.notify_job_failed(job_id, error_msg)
            raise
    
    def handle_generate_requirements(self, payload: GenerateRequirementsPayload) -> Dict[str, Any]:
        """Generate requirements for an assignment using the RequirementGeneratorAgent"""
        import sys, os
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from src.config import load_config, get_agent_config, load_langsmith_config
        from src.agents.requirement_generator.requirement_generator import RequirementGeneratorAgent
        from src.models import Requirement
        import traceback
        assignment_id = payload.assignment_id
        logger.info(f"🔧 Generating requirements for assignment {assignment_id}")
        try:
            # Fetch assignment description from API
            assignment = self.api_client.get_assignment(assignment_id)
            assignment_description = assignment.content
            # Load config (use default path or env var)
            config_path = os.getenv("JADE_AGENT_CONFIG", "src/config/assignment_config.yaml")
            config = load_config(config_path)
            load_langsmith_config()
            agent = RequirementGeneratorAgent(config)
            requirements = agent.generate_requirements(assignment_description)
            created_count = 0
            for req in requirements:
                try:
                    # Convert PromptType enum to string for REST API model
                    req_data = {
                        "requirement": req["requirement"],
                        "function": req.get("function", None),
                        "type": req["type"].value if hasattr(req["type"], "value") else str(req["type"])
                    }
                    requirement = RequirementCreate(**req_data)
                    response = self.api_client.create_requirement(assignment_id, requirement)
                    created_count += 1
                    logger.info(f"📝 Created requirement: {response.id}")
                except Exception as e:
                    logger.error(f"❌ Failed to create requirement: {e}")
            result = {
                "message": f"Generated {created_count} requirements for assignment {assignment_id}",
                "assignment_id": assignment_id,
                "job_type": "generate_requirements",
                "count": created_count
            }
            logger.info(f"✅ Requirements generation completed: {created_count} created")
            return result
        except Exception as e:
            logger.error(f"❌ Exception in handle_generate_requirements: {e}\n{traceback.format_exc()}")
            raise
    
    def handle_generate_prompts(self, payload: GeneratePromptsPayload) -> Dict[str, Any]:
        import sys, os
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from src.config import load_config, get_agent_config, load_langsmith_config
        from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent
        from src.models import Requirement
        import traceback
        assignment_id = payload.assignment_id
        requirement_ids = payload.requirement_ids
        logger.info(f"🔧 Generating prompts for {len(requirement_ids)} requirements in assignment {assignment_id}")
        try:
            # Fetch assignment description from API
            assignment = self.api_client.get_assignment(assignment_id)
            assignment_description = assignment.content
            # Fetch requirements from API
            requirements = []
            for req_id in requirement_ids:
                req = self.api_client.get_requirement(assignment_id, req_id)
                # Convert to Requirement dict for agent
                req_dict = {
                    "requirement": req.requirement,
                    "function": req.function,
                    "type": req.type
                }
                requirements.append(req_dict)
            # Load config
            config_path = os.getenv("JADE_AGENT_CONFIG", "src/config/assignment_config.yaml")
            config = load_config(config_path)
            load_langsmith_config()
            agent = PromptGeneratorAgent(config)
            # Convert type string to PromptType if needed
            from src.models import PromptType
            for req in requirements:
                if isinstance(req["type"], str):
                    for pt in PromptType:
                        if pt.value == req["type"]:
                            req["type"] = pt
                            break
            # Generate prompts
            results = agent.generate_prompts_batch(requirements, assignment_description)
            updated_count = 0
            for i, result in enumerate(results):
                try:
                    req_id = requirement_ids[i]
                    prompt_template = result["jinja_template"]
                    req_obj = requirements[i]
                    req_update = RequirementUpdate(
                        requirement=req_obj["requirement"],
                        function=req_obj["function"],
                        type=req_obj["type"].value if hasattr(req_obj["type"], "value") else str(req_obj["type"]),
                        prompt_template=prompt_template
                    )
                    self.api_client.update_requirement(assignment_id, req_id, req_update)
                    updated_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to update requirement {requirement_ids[i]}: {e}")
            result = {
                "message": f"Generated and saved prompt templates for {updated_count} requirements",
                "assignment_id": assignment_id,
                "requirement_ids": requirement_ids,
                "job_type": "generate_prompts",
                "count": updated_count
            }
            logger.info(f"✅ Prompt generation completed: {updated_count} updated")
            return result
        except Exception as e:
            logger.error(f"❌ Exception in handle_generate_prompts: {e}\n{traceback.format_exc()}")
            raise
    
    def handle_create_correction(self, payload: CreateCorrectionPayload) -> Dict[str, Any]:
        """Create corrections for a submission using REST API"""
        import sys, os
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from src.config import load_config, get_agent_config, load_langsmith_config
        from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
        from src.models import Requirement, GeneratedPrompt, Submission
        import traceback
        submission_id = payload.submission_id
        requirement_ids = payload.requirement_ids
        logger.info(f"🔧 Creating corrections for submission {submission_id} against {len(requirement_ids)} requirements")
        try:
            # Find assignment_id by searching requirements
            assignment_id = None
            for test_assignment_id in range(1, 100):  # Try a wider range for robustness
                try:
                    requirements = self.api_client.get_requirements(test_assignment_id)
                    req_ids_in_assignment = [req.requirement_id for req in requirements]
                    if requirement_ids[0] in req_ids_in_assignment:
                        assignment_id = test_assignment_id
                        break
                except Exception:
                    continue
            if not assignment_id:
                raise ValueError("Could not determine assignment_id for the given requirements")
            logger.info(f"📍 Determined assignment_id: {assignment_id}")
            # Fetch assignment description
            assignment = self.api_client.get_assignment(assignment_id)
            assignment_description = assignment.content
            # Fetch requirements and prompt_templates
            requirements = []
            generated_prompts = []
            for req_id in requirement_ids:
                req = self.api_client.get_requirement(assignment_id, req_id)
                requirements.append(req)
                # Use prompt_template if available, else skip
                if not req.prompt_template:
                    logger.error(f"❌ Requirement {req_id} has no prompt_template. Skipping.")
                    continue
                # Convert to Requirement and GeneratedPrompt dicts
                req_dict = {
                    "requirement": req.requirement,
                    "function": req.function,
                    "type": req.type
                }
                # Convert type string to PromptType if needed
                from src.models import PromptType
                if isinstance(req_dict["type"], str):
                    for pt in PromptType:
                        if pt.value == req_dict["type"]:
                            req_dict["type"] = pt
                            break
                generated_prompt = {
                    "requirement": req_dict,
                    "examples": "",  # Not used in correction
                    "jinja_template": req.prompt_template,
                    "index": 0
                }
                generated_prompts.append(generated_prompt)
            # Fetch submission
            submission_obj = self.api_client.get_submission(assignment_id, submission_id)
            submission = {"code": submission_obj.code}
            # Load config
            config_path = os.getenv("JADE_AGENT_CONFIG", "src/config/assignment_config.yaml")
            config = load_config(config_path)
            load_langsmith_config()
            agent = CodeCorrectorAgent(config)
            # Run correction agent
            corrections = agent.correct_code_batch(generated_prompts, submission, assignment_description)
            created_count = 0
            for i, correction in enumerate(corrections):
                try:
                    req_id = requirement_ids[i]
                    correction_create = CorrectionCreate(
                        requirement_id=req_id,
                        result=correction["result"]
                    )
                    created_corr = self.api_client.create_correction(assignment_id, submission_id, correction_create)
                    created_count += 1
                    logger.info(f"📝 Created correction for requirement {req_id}: {created_corr.correction_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to create correction for requirement {requirement_ids[i]}: {e}")
            result = {
                "message": f"Created {created_count} corrections for submission {submission_id}",
                "submission_id": submission_id,
                "assignment_id": assignment_id,
                "requirement_ids": requirement_ids,
                "job_type": "create_correction",
                "count": created_count
            }
            logger.info(f"✅ Correction creation completed: {created_count} created")
            return result
        except Exception as e:
            logger.error(f"❌ Exception in handle_create_correction: {e}\n{traceback.format_exc()}")
            raise
