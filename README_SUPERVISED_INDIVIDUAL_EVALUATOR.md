# Supervised Individual Evaluator

## Overview
`run_supervised_individual_evaluator.py` is a pipeline script that uses the **modular, config-driven** auxiliary and individual evaluators to assess student code corrections.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Flow                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. PromptGeneratorAgent                                         │
│     └─> Generate prompts from requirements                       │
│                                                                   │
│  2. CodeCorrectorAgent                                           │
│     └─> Generate corrections for each submission                 │
│                                                                   │
│  3. AuxiliaryMetricsEvaluator (LangGraph)                       │
│     ├─> compute_match (parallel)                                │
│     ├─> compute_missing (parallel)                              │
│     └─> compute_extra (parallel)                                │
│     Result: {"match": "...", "missing": "...", "extra": "..."}  │
│                                                                   │
│  4. IndividualMetricsEvaluator (LangGraph)                      │
│     ├─> evaluate_completeness (parallel)                        │
│     ├─> evaluate_restraint (parallel)                           │
│     ├─> evaluate_precision (parallel)                           │
│     ├─> evaluate_content_similarity (parallel)                  │
│     └─> evaluate_correctness (parallel)                         │
│     Result: {                                                    │
│       "scores": {...},                                           │
│       "explanations": {...},                                     │
│       "overall_score": 4.2                                       │
│     }                                                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Differences from `run_supervised_evaluator.py`

| Aspect | `run_supervised_evaluator.py` | `run_supervised_individual_evaluator.py` |
|--------|-------------------------------|------------------------------------------|
| **Evaluator** | `SupervisedEvaluator2Step` | `AuxiliaryMetricsEvaluator` + `IndividualMetricsEvaluator` |
| **Architecture** | Single evaluator class | Two separate LangGraph-based evaluators |
| **Auxiliary Metrics** | Computed internally | Explicit first step with dedicated evaluator |
| **Individual Metrics** | Computed internally | Explicit second step with dedicated evaluator |
| **Config-Driven** | Partially | Fully - all metrics defined in config |
| **Extensibility** | Requires code changes | Add metrics via config + templates |
| **Outputs** | `evaluation_results.json` | `auxiliary_metrics.json` + `evaluation_results.json` |
| **LangGraph** | Not used | Both evaluators use LangGraph for parallel execution |

## Usage

### Basic Command
```bash
python run_supervised_individual_evaluator.py \
  --assignment ejemplos/3p/consigna.txt \
  --requirements ejemplos/3p/requirements/*.json \
  --submissions ejemplos/3p/alu1.py \
  --reference-corrections ejemplos/3p/alu1.txt \
  --output-dir outputs/supervised_individual_evaluation
```

### With Custom Configs
```bash
python run_supervised_individual_evaluator.py \
  --assignment ejemplos/3p/consigna.txt \
  --requirements ejemplos/3p/requirements/*.json \
  --submissions ejemplos/3p/alu*.py \
  --reference-corrections ejemplos/3p/alu*.txt \
  --output-dir outputs/supervised_individual_evaluation \
  --config src/config/assignment_config.yaml \
  --evaluator-config src/config/evaluator_config.yaml
```

### Multiple Submissions
```bash
python run_supervised_individual_evaluator.py \
  --assignment ejemplos/3p/consigna.txt \
  --requirements ejemplos/3p/requirements/*.json \
  --submissions ejemplos/3p/alu1.py ejemplos/3p/alu2.py ejemplos/3p/alu3.py \
  --reference-corrections ejemplos/3p/alu1.txt ejemplos/3p/alu2.txt ejemplos/3p/alu3.txt \
  --output-dir outputs/supervised_individual_evaluation
```

## Output Structure

For each submission, the pipeline creates:

```
outputs/supervised_individual_evaluation/
├── submission_1/
│   ├── generated_corrections.json      # AI-generated corrections
│   ├── auxiliary_metrics.json          # MATCH, MISSING, EXTRA metrics
│   └── evaluation_results.json         # Scores, explanations, overall score
├── submission_2/
│   ├── generated_corrections.json
│   ├── auxiliary_metrics.json
│   └── evaluation_results.json
└── ...
```

### File Contents

#### `auxiliary_metrics.json`
```json
{
  "auxiliary_metrics": {
    "match": "1. Requirement X - High match quality\n2. Requirement Y - Medium match quality",
    "missing": "1. Requirement Z - Critical impact\n2. Requirement W - Minor impact",
    "extra": "1. Extra comment about syntax - Low relevance"
  },
  "timestamp": 1699999999.123
}
```

#### `evaluation_results.json`
```json
{
  "scores": {
    "completeness": 4,
    "restraint": 5,
    "precision": 3,
    "content_similarity": 4,
    "correctness": 5
  },
  "explanations": {
    "completeness": "The correction covers most requirements...",
    "restraint": "No unnecessary information...",
    "precision": "Some details could be more specific...",
    "content_similarity": "Good alignment with reference...",
    "correctness": "All technical details are accurate..."
  },
  "overall_score": 4.2,
  "timings": {
    "match": 2.34,
    "missing": 1.87,
    "extra": 2.01,
    "completeness": 3.12,
    "restraint": 2.98,
    "precision": 3.45,
    "content_similarity": 2.76,
    "correctness": 3.21
  },
  "timestamp": 1699999999.456
}
```

#### `generated_corrections.json`
```json
{
  "corrections": [
    {
      "requirement": {
        "requirement": "Check if function handles edge cases",
        "function": "validate_input",
        "type": "syntax"
      },
      "result": "The function correctly validates input..."
    }
  ],
  "timestamp": 1699999999.789,
  "metadata": {
    "total_corrections": 5,
    "generation_method": "supervised_individual_evaluation"
  }
}
```

## MLflow Integration

The pipeline logs to MLflow:

### Metrics Logged
- `overall_score`: Average of all individual metric scores
- `metric_completeness`: Completeness score (0-5)
- `metric_restraint`: Restraint score (0-5)
- `metric_precision`: Precision score (0-5)
- `metric_content_similarity`: Content similarity score (0-5)
- `metric_correctness`: Correctness score (0-5)
- `aux_match_length`: Length of MATCH auxiliary metric text
- `aux_missing_length`: Length of MISSING auxiliary metric text
- `aux_extra_length`: Length of EXTRA auxiliary metric text
- `num_generated_corrections`: Number of corrections generated

### Artifacts Logged
- All JSON files from the output directory
- Organized by submission (e.g., `submission_1/`, `submission_2/`)

### Run Organization
```
MLflow Runs:
├── prompt_generation (phase: prompt_generation)
├── supervised_individual_evaluation_1 (submission_index: 0)
├── supervised_individual_evaluation_2 (submission_index: 1)
└── ...
```

## LangSmith Integration

When LangSmith is configured, the pipeline creates detailed traces:

```
LangSmith Trace Hierarchy:
└── submission_1_processing
    ├── generate_corrections (code_correction)
    ├── compute_auxiliary_metrics (auxiliary_metrics)
    │   ├── compute_match
    │   ├── compute_missing
    │   └── compute_extra
    └── evaluate_individual_metrics (individual_metrics)
        ├── evaluate_completeness
        ├── evaluate_restraint
        ├── evaluate_precision
        ├── evaluate_content_similarity
        └── evaluate_correctness
```

## Configuration

### Evaluator Config (`src/config/evaluator_config.yaml`)

The pipeline reads metric definitions from the evaluator config:

```yaml
supervised_evaluator:
  auxiliary_metrics:
    match:
      template: "evaluators/individual/aux_match.jinja"
    missing:
      template: "evaluators/individual/aux_missing.jinja"
      required_aux_metrics: []
    extra:
      template: "evaluators/individual/aux_extra.jinja"
      required_aux_metrics: []
  
  evaluation_metrics:
    completeness:
      template: "evaluators/individual/eval_completeness.jinja"
      required_aux_metrics: ["match", "missing"]
    restraint:
      template: "evaluators/individual/eval_restraint.jinja"
      required_aux_metrics: ["extra"]
    # ... more metrics
```

### Adding New Metrics

**To add a new auxiliary metric:**
1. Add to `evaluator_config.yaml`:
   ```yaml
   auxiliary_metrics:
     new_metric:
       template: "evaluators/individual/aux_new_metric.jinja"
   ```
2. Create template `templates/evaluators/individual/aux_new_metric.jinja`
3. **No code changes needed!**

**To add a new evaluation metric:**
1. Add to `evaluator_config.yaml`:
   ```yaml
   evaluation_metrics:
     new_metric:
       template: "evaluators/individual/eval_new_metric.jinja"
       required_aux_metrics: ["match", "missing"]
   ```
2. Create template `templates/evaluators/individual/eval_new_metric.jinja`
3. **No code changes needed!**

## Performance

### Parallel Execution
Both evaluators use LangGraph for parallel metric computation:
- **Auxiliary metrics**: All 3 metrics (match, missing, extra) run in parallel
- **Individual metrics**: All 5 metrics run in parallel (when dependencies are met)

### Typical Timing (per submission)
- Prompt generation: ~5-10s (for all prompts, done once)
- Code correction: ~10-20s (depends on number of requirements)
- Auxiliary metrics: ~5-8s (parallel execution)
- Individual metrics: ~8-12s (parallel execution)
- **Total per submission**: ~25-40s

## Summary Output

After processing all submissions, the pipeline prints:

```
============================================================
SUMMARY
============================================================
Total submissions processed: 3
Total corrections generated: 15
Average overall evaluation score: 4.233

Metric Averages:
  completeness: 4.333
  content_similarity: 4.000
  correctness: 4.667
  precision: 3.667
  restraint: 4.333
============================================================
✓ Pipeline completed successfully
```

## Error Handling

The pipeline includes comprehensive error handling:
- Validates that number of submissions matches number of reference corrections
- Checks for missing configuration files
- Logs detailed error messages with stack traces
- Exits with code 1 on failure (for CI/CD integration)

## Comparison Table

| Feature | SupervisedEvaluator2Step | Auxiliary + Individual |
|---------|--------------------------|------------------------|
| **Parallel execution** | No | Yes (LangGraph) |
| **Modular architecture** | Monolithic | Two separate evaluators |
| **Config-driven** | Partial | Full |
| **Add metrics** | Code changes required | Config + template only |
| **Separate aux outputs** | No | Yes (`auxiliary_metrics.json`) |
| **State management** | Manual | LangGraph TypedDict |
| **Extensibility** | Low | High |
| **Testing** | Harder to isolate | Easy to test separately |

## When to Use

**Use `run_supervised_individual_evaluator.py` when:**
- ✅ You want to add custom metrics without changing code
- ✅ You need auxiliary metrics as separate outputs
- ✅ You want parallel execution for faster evaluation
- ✅ You prefer modular, testable architecture
- ✅ You plan to extend the evaluation system

**Use `run_supervised_evaluator.py` when:**
- ✅ You don't need to customize metrics
- ✅ You want a simpler, all-in-one evaluator
- ✅ You're maintaining existing workflows

## See Also

- `ARCHITECTURE_DIAGRAM.md` - Visual architecture of the evaluators
- `SCALABLE_STATE_REFACTORING.md` - Details on dictionary-based state management
- `evaluator_definition.md` - Specification for the evaluation system
- `src/config/evaluator_config.yaml` - Metric configuration
