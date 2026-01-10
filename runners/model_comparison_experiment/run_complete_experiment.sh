#!/bin/bash
# Script completo: 5 corridas por modelo (10 alumnos cada una)
set -e
cd /Users/abrilvilamowski/Desktop/PF/hyper_jade

ASSIGNMENT="ejemplos/3p/consigna.txt"
REQUIREMENTS="ejemplos/3p/requirements_es/requirement_*.json"
SUBMISSIONS="ejemplos/3p/alu1.py ejemplos/3p/alu2.py ejemplos/3p/alu3.py ejemplos/3p/alu4.py ejemplos/3p/alu5.py ejemplos/3p/alu6.py ejemplos/3p/alu7.py ejemplos/3p/alu8.py ejemplos/3p/alu9.py ejemplos/3p/alu10.py"
REFERENCE_CORRECTIONS="ejemplos/3p/alu1.json ejemplos/3p/alu2.json ejemplos/3p/alu3.json ejemplos/3p/alu4.json ejemplos/3p/alu5.json ejemplos/3p/alu6.json ejemplos/3p/alu7.json ejemplos/3p/alu8.json ejemplos/3p/alu9.json ejemplos/3p/alu10.json"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"

MODELS=("gemini_2_5_pro" "gemini_3_flash_preview")
RUNS_PER_MODEL=5

echo "🚀 Experimento completo: ${RUNS_PER_MODEL} corridas × 10 alumnos por modelo"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in "${MODELS[@]}"; do
    echo ""
    echo "🤖 Modelo: ${model}"
    for run in $(seq 1 $RUNS_PER_MODEL); do
        TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
        OUTPUT_DIR="outputs/model_comparison_experiment/${model}/${TIMESTAMP}"
        echo "  Run ${run}/${RUNS_PER_MODEL}: ${OUTPUT_DIR}"
        
        mkdir -p "$OUTPUT_DIR"
        
        uv run python runners/run_supervised_individual_evaluator.py \
            --assignment "$ASSIGNMENT" \
            --requirements $REQUIREMENTS \
            --submissions $SUBMISSIONS \
            --reference-corrections $REFERENCE_CORRECTIONS \
            --output-dir "$OUTPUT_DIR" \
            --config "runners/config/model_comparison_experiment/${model}.yaml" \
            --evaluator-config "$EVALUATOR_CONFIG" \
            --experiment-name "${model}_${TIMESTAMP}"
        
        [ $run -lt $RUNS_PER_MODEL ] && sleep 5
    done
    echo "✅ ${model} completo"
done

echo ""
echo "🎉 Experimento completo finalizado"
echo "📊 Generando plots..."
bash plots/model_comparison_experiment.sh
echo "✅ Listo!"
