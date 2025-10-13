# LangSmith Tracing Integration for Supervised Evaluator

## Overview

The supervised evaluator pipeline has been updated to provide better LangSmith tracing structure. Instead of having separate traces for each operation, the pipeline now creates:

1. **One LangSmith trace for prompt generation** - covers the entire batch prompt generation process
2. **One LangSmith trace per submission** - covers both code correction and evaluation for that specific submission

## Changes Made

### 1. Enhanced Imports
- Added proper LangSmith imports with fallback to no-op decorators if LangSmith is not available
- Imports both `trace` (context manager) and `traceable` (decorator) from `langsmith`

### 2. New Traced Functions

#### `generate_prompts_traced()`
- Wraps the prompt generation process with `@traceable(name="generate_prompts", run_type="chain")`
- Creates a single trace for generating all prompts from requirements

#### `process_submission_traced()`
- Wraps both code correction and evaluation for a single submission
- Uses nested trace contexts:
  - Top-level: `submission_N_processing` (chain type)
  - Nested: `generate_corrections` (llm type) 
  - Nested: `evaluate_corrections` (llm type)
- Adds relevant tags for tracking (submission index, operation counts, etc.)

### 3. Pipeline Structure

The new pipeline flow creates the following LangSmith trace structure:

```
📊 Project: [Your LangSmith Project]
├── 🔗 generate_prompts (chain)
│   └── [All prompt generation LLM calls]
├── 🔗 submission_1_processing (chain) 
│   ├── 🤖 generate_corrections (llm)
│   │   └── [Code correction LLM calls]
│   └── 🤖 evaluate_corrections (llm)
│       └── [Evaluation LLM calls]
├── 🔗 submission_2_processing (chain)
│   ├── 🤖 generate_corrections (llm)
│   └── 🤖 evaluate_corrections (llm)
└── ... (more submissions)
```

### 4. Benefits

1. **Better Organization**: Each submission's correction and evaluation are grouped under one trace
2. **Easier Debugging**: You can see the complete flow for each submission in one place
3. **Better Performance Analysis**: Clear separation between prompt generation (done once) and per-submission processing
4. **Rich Tagging**: Traces include metadata like submission index, prompt counts, correction counts

## Usage

The pipeline works exactly the same as before:

```bash
python run_supervised_evaluator.py \
  --assignment ejemplos/3p/consigna.txt \
  --requirements ejemplos/3p/requirements/*.json \
  --submissions ejemplos/3p/alu*.py \
  --reference-corrections ejemplos/3p/alu*.txt \
  --output-dir outputs/supervised_evaluation
```

## LangSmith Configuration

Ensure your LangSmith configuration is properly set up in:
- `src/config/langsmith_config.yaml`
- Environment variables (`.env` file):
  - `LANGSMITH_API_KEY`
  - `LANGSMITH_PROJECT`

The pipeline will automatically detect if LangSmith is available and fall back gracefully if it's not configured.

## Verification

You can verify the tracing is working by:

1. Running the pipeline with a small test case
2. Checking your LangSmith dashboard for the new trace structure
3. Looking for the log message: `🔗 LangSmith tracing enabled for project: [project-id]`

## Backward Compatibility

All existing functionality remains unchanged:
- MLflow logging still works as before
- All output files are generated in the same format
- Command-line interface is identical
- Agent configurations remain the same

The only change is the improved LangSmith trace organization.