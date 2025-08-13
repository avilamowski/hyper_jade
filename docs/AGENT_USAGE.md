# Agent Usage Guide

This document explains how to use the individual agents in the Hyper Jade assignment evaluation system, both independently and as part of the complete pipeline.

## Overview

The system consists of three main agents that can be run independently or as a complete pipeline:

1. **Requirement Generator Agent** - Generates rubrics from assignment descriptions
2. **Prompt Generator Agent** - Creates correction prompts from rubrics
3. **Code Corrector Agent** - Evaluates student code using the prompts

## Quick Start

### Running Individual Agents

Each agent can be run independently using dedicated scripts:

```bash
# Generate a rubric from an assignment
python run_requirement_generator.py --assignment assignment.txt --language python

# Generate prompts from a rubric
python run_prompt_generator.py --assignment assignment.txt --rubric outputs/requirement_generator/rubric_assignment_2024-01-15T10:30:45.json

# Evaluate student code using prompts
python run_code_corrector.py --code student_code.py --assignment assignment.txt --prompts outputs/prompt_generator/prompts_assignment_2024-01-15T10:30:45.json
```

### Running the Complete Pipeline

```bash
# Run all agents in sequence
python main.py --code student_code.py --assignment assignment.txt --language python

# Use stored outputs from previous runs
python main.py --code student_code.py --assignment assignment.txt --use-stored --assignment-id my_assignment
```

## Detailed Usage

### 1. Requirement Generator Agent

Generates comprehensive rubrics from assignment descriptions.

**Basic Usage:**
```bash
python run_requirement_generator.py --assignment assignment.txt
```

**Full Options:**
```bash
python run_requirement_generator.py \
  --assignment assignment.txt \
  --language python \
  --config src/config/assignment_config.yaml \
  --output my_rubric.json \
  --assignment-id my_assignment \
  --verbose \
  --storage-dir outputs
```

**Output:**
- Saves rubric to storage directory
- Creates JSON file with rubric structure
- Prints summary of generated rubric items

### 2. Prompt Generator Agent

Creates specialized correction prompts for each rubric item.

**Basic Usage:**
```bash
python run_prompt_generator.py --assignment assignment.txt --rubric rubric_file.json
```

**Using Assignment ID (finds latest rubric):**
```bash
python run_prompt_generator.py --assignment assignment.txt --rubric my_assignment
```

**Full Options:**
```bash
python run_prompt_generator.py \
  --assignment assignment.txt \
  --rubric rubric_file.json \
  --config src/config/assignment_config.yaml \
  --output my_prompts.json \
  --assignment-id my_assignment \
  --verbose \
  --storage-dir outputs
```

**Output:**
- Saves prompts to storage directory
- Creates JSON file with prompt set
- Prints summary of generated prompts

### 3. Code Corrector Agent

Evaluates student code using the generated prompts.

**Basic Usage:**
```bash
python run_code_corrector.py --code student_code.py --assignment assignment.txt --prompts prompts_file.json
```

**Using Assignment ID (finds latest prompts):**
```bash
python run_code_corrector.py --code student_code.py --assignment assignment.txt --prompts my_assignment
```

**Full Options:**
```bash
python run_code_corrector.py \
  --code student_code.py \
  --assignment assignment.txt \
  --prompts prompts_file.json \
  --language python \
  --config src/config/assignment_config.yaml \
  --output evaluation_result.json \
  --assignment-id my_assignment \
  --verbose \
  --storage-dir outputs
```

**Output:**
- Saves evaluation results to storage directory
- Creates JSON file with detailed evaluation
- Prints error analysis summary

## Output Management

### Listing Stored Outputs

```bash
# List all outputs
python list_outputs.py

# List outputs for specific agent
python list_outputs.py --agent requirement_generator

# Show latest outputs for assignment
python list_outputs.py --latest my_assignment

# View specific output file
python list_outputs.py --view outputs/requirement_generator/rubric_my_assignment_2024-01-15T10:30:45.json

# Clean old outputs (keep only latest per assignment)
python list_outputs.py --clean
```

### Output Storage Structure

```
outputs/
├── requirement_generator/
│   ├── rubric_assignment1_2024-01-15T10:30:45.json
│   └── rubric_assignment2_2024-01-16T14:20:30.json
├── prompt_generator/
│   ├── prompts_assignment1_2024-01-15T10:31:15.json
│   └── prompts_assignment2_2024-01-16T14:21:00.json
├── code_corrector/
│   ├── correction_assignment1_2024-01-15T10:32:00.json
│   └── correction_assignment2_2024-01-16T14:22:00.json
└── metadata/
    ├── requirement_generator_assignment1_2024-01-15T10:30:45.json
    ├── prompt_generator_assignment1_2024-01-15T10:31:15.json
    └── code_corrector_assignment1_2024-01-15T10:32:00.json
```

## Workflow Examples

### Example 1: Step-by-Step Evaluation

```bash
# Step 1: Generate rubric
python run_requirement_generator.py \
  --assignment ejemplos/consigna.txt \
  --assignment-id alu_example \
  --verbose

# Step 2: Generate prompts
python run_prompt_generator.py \
  --assignment ejemplos/consigna.txt \
  --rubric alu_example \
  --assignment-id alu_example \
  --verbose

# Step 3: Evaluate student code
python run_code_corrector.py \
  --code ejemplos/alu1.py \
  --assignment ejemplos/consigna.txt \
  --prompts alu_example \
  --assignment-id alu_example \
  --verbose
```

### Example 2: Using Stored Outputs

```bash
# First run (generates and stores outputs)
python main.py \
  --code ejemplos/alu1.py \
  --assignment ejemplos/consigna.txt \
  --assignment-id alu_example

# Subsequent runs (uses stored outputs)
python main.py \
  --code ejemplos/alu2.py \
  --assignment ejemplos/consigna.txt \
  --use-stored \
  --assignment-id alu_example
```

### Example 3: Batch Processing

```bash
# Generate rubric once
python run_requirement_generator.py \
  --assignment ejemplos/consigna.txt \
  --assignment-id alu_batch

# Generate prompts once
python run_prompt_generator.py \
  --assignment ejemplos/consigna.txt \
  --rubric alu_batch \
  --assignment-id alu_batch

# Evaluate multiple student submissions
for i in {1..10}; do
  python run_code_corrector.py \
    --code "ejemplos/alu${i}.py" \
    --assignment ejemplos/consigna.txt \
    --prompts alu_batch \
    --assignment-id alu_batch \
    --output "results/alu${i}_evaluation.json"
done
```

## Configuration

### Model Configuration

Edit `src/config/assignment_config.yaml` to configure the LLM:

```yaml
# OpenAI configuration
provider: "openai"
model_name: "gpt-4"
api_key: "your-openai-api-key"

# Or Ollama configuration
provider: "ollama"
model_name: "qwen2.5:7b"
temperature: 0.1
```

### Output Storage Configuration

```bash
# Use custom storage directory
python run_requirement_generator.py \
  --assignment assignment.txt \
  --storage-dir /path/to/custom/outputs

# Default storage is ./outputs
```

### MLflow Logging

All agent runs are automatically logged to MLflow with:

- **Parameters**: Assignment file, programming language, model configuration
- **Metrics**: Execution time, output counts, error counts
- **Artifacts**: Generated rubrics, prompts, and evaluation results
- **Run IDs**: Stored in metadata for traceability

```bash
# View MLflow runs
mlflow ui

# Access run information programmatically
import mlflow
runs = mlflow.search_runs(experiment_names=["Default"])
```

## Troubleshooting

### Common Issues

1. **Missing stored outputs:**
   ```bash
   # Check what's available
   python list_outputs.py --latest my_assignment
   ```

2. **File not found errors:**
   - Ensure assignment and code files exist
   - Check file paths are correct
   - Verify file permissions

3. **Configuration errors:**
   - Check `src/config/assignment_config.yaml` exists
   - Verify API keys are set (for OpenAI)
   - Ensure Ollama is running (for Ollama provider)

### Debug Mode

Use `--verbose` flag for detailed output:

```bash
python run_requirement_generator.py \
  --assignment assignment.txt \
  --verbose
```

### Logging

Check logs for detailed error information:

```bash
# Set log level
export PYTHONPATH=src
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

## Best Practices

1. **Use consistent assignment IDs** across all agents for the same assignment
2. **Store outputs** to avoid regenerating rubrics and prompts
3. **Clean old outputs** periodically to save disk space
4. **Use verbose mode** when debugging or first-time setup
5. **Backup important outputs** before cleaning
6. **MLflow logging** for all agent runs with metrics and artifacts

## Integration with Existing Pipeline

The individual agents are fully compatible with the existing `main.py` pipeline:

- Individual agents save outputs that can be used by the main pipeline
- Main pipeline can use `--use-stored` to leverage previous agent runs
- All outputs are stored in the same format regardless of how they were generated

