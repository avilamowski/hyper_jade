#!/bin/bash
# Script completo: 5 corridas por modelo (15 alumnos cada una) - ej1-2025-s2-p2-ej1
# Models: gpt-4o-mini, gemini-2.0-flash, gemini-2.5-pro, gemma-3-12b-it
set -e
cd "$(dirname "$0")/../.."

ASSIGNMENT="ejemplos/ej1-2025-s2-p2-ej1/consigna.txt"
REQUIREMENTS="ejemplos/ej1-2025-s2-p2-ej1/requirements_es/requirement_*.json"
SUBMISSIONS="ejemplos/ej1-2025-s2-p2-ej1/alu1.py ejemplos/ej1-2025-s2-p2-ej1/alu3.py ejemplos/ej1-2025-s2-p2-ej1/alu4.py ejemplos/ej1-2025-s2-p2-ej1/alu5.py ejemplos/ej1-2025-s2-p2-ej1/alu6.py ejemplos/ej1-2025-s2-p2-ej1/alu7.py ejemplos/ej1-2025-s2-p2-ej1/alu8.py ejemplos/ej1-2025-s2-p2-ej1/alu9.py ejemplos/ej1-2025-s2-p2-ej1/alu11.py ejemplos/ej1-2025-s2-p2-ej1/alu12.py ejemplos/ej1-2025-s2-p2-ej1/alu13.py ejemplos/ej1-2025-s2-p2-ej1/alu14.py ejemplos/ej1-2025-s2-p2-ej1/alu15.py ejemplos/ej1-2025-s2-p2-ej1/alu17.py ejemplos/ej1-2025-s2-p2-ej1/alu19.py"
REFERENCE_CORRECTIONS="ejemplos/ej1-2025-s2-p2-ej1/alu1.json ejemplos/ej1-2025-s2-p2-ej1/alu3.json ejemplos/ej1-2025-s2-p2-ej1/alu4.json ejemplos/ej1-2025-s2-p2-ej1/alu5.json ejemplos/ej1-2025-s2-p2-ej1/alu6.json ejemplos/ej1-2025-s2-p2-ej1/alu7.json ejemplos/ej1-2025-s2-p2-ej1/alu8.json ejemplos/ej1-2025-s2-p2-ej1/alu9.json ejemplos/ej1-2025-s2-p2-ej1/alu11.json ejemplos/ej1-2025-s2-p2-ej1/alu12.json ejemplos/ej1-2025-s2-p2-ej1/alu13.json ejemplos/ej1-2025-s2-p2-ej1/alu14.json ejemplos/ej1-2025-s2-p2-ej1/alu15.json ejemplos/ej1-2025-s2-p2-ej1/alu17.json ejemplos/ej1-2025-s2-p2-ej1/alu19.json"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"

MODELS=("gpt_4o_mini" "gemini_2_0_flash" "gemini_2_5_pro")
RUNS_PER_MODEL=5

echo "🚀 Experimento completo: ${RUNS_PER_MODEL} corridas × 15 alumnos × ${#MODELS[@]} modelos (en paralelo)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run each model in parallel
for model in "${MODELS[@]}"; do
    (
        echo ""
        echo "🤖 Modelo: ${model} (iniciando en paralelo)"
        for run in $(seq 1 $RUNS_PER_MODEL); do
            TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
            OUTPUT_DIR="outputs/model_comparison_experiment/${model}/${TIMESTAMP}"
            echo "  [${model}] Run ${run}/${RUNS_PER_MODEL}: ${OUTPUT_DIR}"
            
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
    ) &
done

# Wait for all parallel processes to complete
echo ""
echo "⏳ Esperando que todos los modelos completen..."
wait
echo "✅ Todos los modelos completaron"

echo ""
echo "🎉 Experimento completo finalizado"
echo "📊 Generando plots..."
MPLBACKEND=Agg uv run python plots/plot_experiments.py \
    outputs/model_comparison_experiment \
    gpt_4o_mini gemini_2_0_flash gemini_2_5_pro \
    --title "Model Comparison - ej1-2025-s2-p2-ej1" \
    --output "model_comparison_ej1_2025_s2_p2_ej1"
echo "✅ Listo!"
