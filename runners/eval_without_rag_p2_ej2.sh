#!/bin/bash

# Simple supervised individual evaluator runner for ejemplos/ej1-2025-s2-p2-ej2
# Ejercicio 2 del Parcial 2: Gestión de Pedidos (costo y hacer_pedido)

# Define which files to use (modify these lists as needed)
ASSIGNMENT="ejemplos/ej1-2025-s2-p2-ej2/consigna.txt"
REQUIREMENTS="ejemplos/ej1-2025-s2-p2-ej2/requirements_es/*.json"
# REQUIREMENTS="ejemplos/ej1-2025-s2-p2-ej2/requirements/requirement_01.json ejemplos/ej1-2025-s2-p2-ej2/requirements/requirement_02.json"
SUBMISSIONS="ejemplos/ej1-2025-s2-p2-ej2/alu1.py ejemplos/ej1-2025-s2-p2-ej2/alu3.py ejemplos/ej1-2025-s2-p2-ej2/alu4.py ejemplos/ej1-2025-s2-p2-ej2/alu5.py ejemplos/ej1-2025-s2-p2-ej2/alu8.py ejemplos/ej1-2025-s2-p2-ej2/alu9.py ejemplos/ej1-2025-s2-p2-ej2/alu10.py ejemplos/ej1-2025-s2-p2-ej2/alu11.py ejemplos/ej1-2025-s2-p2-ej2/alu12.py ejemplos/ej1-2025-s2-p2-ej2/alu13.py ejemplos/ej1-2025-s2-p2-ej2/alu14.py ejemplos/ej1-2025-s2-p2-ej2/alu16.py ejemplos/ej1-2025-s2-p2-ej2/alu18.py"
REFERENCE_CORRECTIONS="ejemplos/ej1-2025-s2-p2-ej2/alu1.json ejemplos/ej1-2025-s2-p2-ej2/alu3.json ejemplos/ej1-2025-s2-p2-ej2/alu4.json ejemplos/ej1-2025-s2-p2-ej2/alu5.json ejemplos/ej1-2025-s2-p2-ej2/alu8.json ejemplos/ej1-2025-s2-p2-ej2/alu9.json ejemplos/ej1-2025-s2-p2-ej2/alu10.json ejemplos/ej1-2025-s2-p2-ej2/alu11.json ejemplos/ej1-2025-s2-p2-ej2/alu12.json ejemplos/ej1-2025-s2-p2-ej2/alu13.json ejemplos/ej1-2025-s2-p2-ej2/alu14.json ejemplos/ej1-2025-s2-p2-ej2/alu16.json ejemplos/ej1-2025-s2-p2-ej2/alu18.json"
# SUBMISSIONS="ejemplos/ej1-2025-s2-p2-ej2/alu1.py"
# REFERENCE_CORRECTIONS="ejemplos/ej1-2025-s2-p2-ej2/alu1.json"
# Add a timestamp so multiple runs don't overwrite each other. Format: YYYYMMDDTHHMMSS
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
OUTPUT_DIR="outputs/evaluation/p2_ej2_without_rag/${TIMESTAMP}"
SYSTEM_CONFIG="runners/config/plain.yaml"
EVALUATOR_CONFIG="runners/config/eval_config.yaml"
# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the evaluator
uv run runners/run_supervised_individual_evaluator.py \
    --assignment "$ASSIGNMENT" \
    --requirements $REQUIREMENTS \
    --submissions $SUBMISSIONS \
    --reference-corrections $REFERENCE_CORRECTIONS \
    --output-dir "$OUTPUT_DIR" \
    --config "$SYSTEM_CONFIG" \
    --evaluator-config "$EVALUATOR_CONFIG" \
    --experiment-name "p2_ej2_without_rag_${TIMESTAMP}" \
    --verbose
