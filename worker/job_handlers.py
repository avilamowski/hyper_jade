
import time
import logging
import sys
import os
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to find src module
current_file = Path(__file__)
parent_dir = current_file.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Handle imports for both script execution and module import
try:
    from .api_client import APIClient
    from .job_manager import JobManager
    from .models import StatusEnum, RequirementCreate, RequirementUpdate, CorrectionCreate
    from .job_definitions import (
        JobType,
        GenerateRequirementsPayload,
        GeneratePromptsPayload,
        CreateCorrectionPayload,
        PAYLOAD_VALIDATORS
    )
except ImportError:
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

# Import src modules
from src.config import load_config, load_langsmith_config
from src.agents.requirement_generator.requirement_generator import RequirementGeneratorAgent
from src.agents.prompt_generator.prompt_generator import PromptGeneratorAgent
from src.agents.code_corrector.code_corrector import CodeCorrectorAgent
from src.models import PromptType
from src.agents.rag_prompt_generator.config import USE_RAG
from src.agents.rag_prompt_generator.rag_prompt_generator import RAGPromptGeneratorAgent

from langsmith import tracing_context, Client as LangSmithClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobHandlers:
    """Job processing handlers using REST API calls"""
    
    def __init__(self):
        self.api_client = APIClient()
        # Load LangSmith configuration once during initialization
        load_langsmith_config()
        # Create a single LangSmith client instance to reuse
        self.langsmith_client = LangSmithClient()
        logger.info("🔗 LangSmith configuration loaded")
    
    def _finalize_langsmith_traces(self):
        """Simple LangSmith trace cleanup - just flush the client."""
        self.langsmith_client.flush()
        logger.debug("🔗 Flushed LangSmith traces")
        
        # Log RAG status on initialization
        if USE_RAG:
            logger.info("🧠 Worker initialized with RAG Mode: ENABLED")
            logger.info("📚 Course theory integration: ACTIVE")
        else:
            logger.info("📝 Worker initialized with Standard Mode")
            logger.info("📚 Course theory integration: INACTIVE")
    
    def process_job(self, job_id: str, job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a job based on its type, using validated models from job_definitions."""
        logger.info(f"🔄 Processing job {job_id} of type {job_type}")
        JobManager.update_job_status(job_id, StatusEnum.PROCESSING)
        result = {} 
        
        # Wrap job execution in LangSmith tracing context for proper trace lifecycle
        trace_name = f"worker_job_{job_type}"
        trace_tags = [f"job_id:{job_id}", f"type:{job_type}"]
        
        logger.info(f"🔗 Starting LangSmith trace: {trace_name} with tags: {trace_tags}")
        
        try:
            with tracing_context(name=trace_name, tags=trace_tags):
                logger.info(f"🔗 Inside trace context for job ")
                result = self._execute_job_internal(job_id, job_type, payload) 
                logger.info(f"🔗 About to exit trace context for job {job_id}")

        except Exception:
            self._finalize_langsmith_traces() 
            raise
        
        # FINAL STEP: Ensure all nested and parent trace data is sent before returning.
        self._finalize_langsmith_traces()

        logger.info(f"🔗 Exited trace context for job {job_id}")
        logger.info(f"📝 Job {job_id} result: {result}")
        return result
        
    def _execute_job_internal(self, job_id: str, job_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Internal job execution logic separated for cleaner tracing."""
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
        assignment_id = payload.assignment_id
        logger.info(f"🔧 Generating requirements for assignment {assignment_id}")
        try:
            # Fetch assignment description from API
            assignment = self.api_client.get_assignment(assignment_id)
            assignment_description = assignment.content
            # Load config (use default path or env var)
            config_path = os.getenv("JADE_AGENT_CONFIG", "src/config/assignment_config.yaml")
            config = load_config(config_path)
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
        assignment_id = payload.assignment_id
        requirement_ids = payload.requirement_ids
        logger.info(f"🔧 Generating prompts for {len(requirement_ids)} requirements in assignment {assignment_id}")
        
        # Check RAG mode
        if USE_RAG:
            logger.info("🧠 RAG Mode: ENABLED - Using RAG-enhanced prompt generation")
            logger.info("📚 Course theory integration: ACTIVE")
        else:
            logger.info("📝 Standard Mode: Using standard prompt generation")
            logger.info("📚 Course theory integration: INACTIVE")
            
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
            
            # Choose agent based on RAG mode
            if USE_RAG:
                logger.info("🧠 Initializing RAG-enhanced prompt generator...")
                agent = RAGPromptGeneratorAgent(config)
                # Initialize RAG system
                import asyncio
                try:
                    asyncio.run(agent.initialize())
                    logger.info("✅ RAG agent initialized successfully")
                except Exception as init_error:
                    error_msg = str(init_error)
                    # Check if this is a Weaviate connection error
                    error_lower = error_msg.lower()
                    # Detect Weaviate-specific errors
                    is_weaviate_error = (
                        "weaviate" in error_lower or
                        ("not ready" in error_lower and "weaviate" in error_lower) or
                        ("connection" in error_lower and ("weaviate" in error_lower or "8080" in error_msg)) or
                        "failed to start" in error_lower
                    )
                    if is_weaviate_error:
                        logger.error(f"❌ Weaviate connection failed: {error_msg}")
                        raise Exception("Weaviate failed to start. Please ensure Weaviate is running.")
                    else:
                        # Re-raise other initialization errors as-is
                        raise
            else:
                logger.info("📝 Using standard prompt generator...")
                agent = PromptGeneratorAgent(config)
                
            for req in requirements:
                if isinstance(req["type"], str):
                    for pt in PromptType:
                        if pt.value == req["type"]:
                            req["type"] = pt
                            break
            # Generate prompts
            logger.info(f"🔧 Generating prompts for {len(requirements)} requirements")
            import asyncio
            import inspect
            # Check if generate_prompts_batch is async
            if inspect.iscoroutinefunction(agent.generate_prompts_batch):
                # RAG agent uses async method
                results = asyncio.run(agent.generate_prompts_batch(requirements, assignment_description))
            else:
                # Standard agent uses sync method
                results = agent.generate_prompts_batch(requirements, assignment_description)
            logger.info(f"📝 Generated {len(results)} prompt results")
            updated_count = 0
            
            # Ensure we don't exceed bounds
            max_results = min(len(results), len(requirement_ids), len(requirements))
            logger.info(f"📊 Processing {max_results} results (results: {len(results)}, requirement_ids: {len(requirement_ids)}, requirements: {len(requirements)})")
            
            if len(results) != len(requirement_ids):
                logger.warning(f"⚠️  Mismatch: {len(results)} results vs {len(requirement_ids)} requirement IDs")
            
            for i in range(max_results):
                try:
                    req_id = requirement_ids[i]
                    result = results[i]
                    req_obj = requirements[i]
                    prompt_template = result["jinja_template"]
                    
                    req_update = RequirementUpdate(
                        requirement=req_obj["requirement"],
                        function=req_obj["function"],
                        type=req_obj["type"].value if hasattr(req_obj["type"], "value") else str(req_obj["type"]),
                        prompt_template=prompt_template
                    )
                    self.api_client.update_requirement(assignment_id, req_id, req_update)
                    updated_count += 1
                    logger.info(f"✅ Updated requirement {req_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to update requirement {requirement_ids[i] if i < len(requirement_ids) else 'unknown'}: {e}")
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
            requirement_mapping = {}  # Map requirement content to requirement_id for correct ordering
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
                if isinstance(req_dict["type"], str):
                    for pt in PromptType:
                        if pt.value == req_dict["type"]:
                            req_dict["type"] = pt
                            break
                
                # Store the mapping from requirement content to requirement_id
                requirement_mapping[req.requirement] = req_id
                
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
            agent = CodeCorrectorAgent(config)
            # Run correction agent
            corrections = agent.correct_code_batch(generated_prompts, submission, assignment_description)
            
            # Map corrections back to correct requirement_ids using requirement content
            logger.info(f"📊 Mapping {len(corrections)} corrections to correct requirement IDs")
            created_count = 0
            processed_requirements = set()  # Track which requirements we've processed
            
            for correction in corrections:
                try:
                    # Find the correct requirement_id by matching the requirement content
                    requirement_content = correction["requirement"]["requirement"]
                    req_id = requirement_mapping.get(requirement_content)
                    
                    if not req_id:
                        logger.error(f"❌ Could not find requirement_id for requirement: {requirement_content[:50]}...")
                        continue
                    
                    if req_id in processed_requirements:
                        logger.warning(f"⚠️ Duplicate correction for requirement {req_id}, skipping")
                        continue
                    
                    processed_requirements.add(req_id)
                    
                    correction_create = CorrectionCreate(
                        requirement_id=req_id,
                        result=correction["result"]
                    )
                    created_corr = self.api_client.create_correction(assignment_id, submission_id, correction_create)
                    created_count += 1
                    logger.info(f"📝 Created correction for requirement {req_id}: {created_corr.correction_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to create correction for requirement: {e}")
            
            # Log any missing corrections
            expected_req_ids = set(requirement_mapping.values())
            missing_req_ids = expected_req_ids - processed_requirements
            if missing_req_ids:
                logger.warning(f"⚠️ Missing corrections for requirements: {missing_req_ids}")
            
            result = {
                "message": f"Created {created_count} corrections for submission {submission_id}",
                "submission_id": submission_id,
                "assignment_id": assignment_id,
                "requirement_ids": list(processed_requirements),  # Return actually processed requirement IDs
                "job_type": "create_correction",
                "count": created_count
            }
            logger.info(f"✅ Correction creation completed: {created_count} created")
            return result
        except Exception as e:
            logger.error(f"❌ Exception in handle_create_correction: {e}\n{traceback.format_exc()}")
            raise
