# MLflow Logging en Hyper Jade

Este documento explica cómo usar el sistema de logging de MLflow implementado en el proyecto Hyper Jade para el seguimiento de métricas y artefactos durante la evaluación de asignaciones.

## 🚀 Características

- **Logging automático de métricas** en cada agente del pipeline
- **Artefactos generados** (requerimientos, prompts, análisis)
- **Métricas de rendimiento** (tiempo de ejecución, tasas de éxito)
- **Configuración flexible** para diferentes entornos
- **Visualización de métricas** con herramientas integradas

## 📊 Métricas Registradas

### Pipeline Principal
- `total_pipeline_time_seconds`: Tiempo total del pipeline
- `requirements_generated`: Número de requerimientos generados
- `prompts_generated`: Número de prompts generados
- `analyses_completed`: Número de análisis completados
- `pipeline_success_rate`: Tasa de éxito del pipeline

### Agente de Generación de Requerimientos
- `llm_generation_time_seconds`: Tiempo de generación del LLM
- `requirements_generated`: Número de requerimientos
- `total_generation_time_seconds`: Tiempo total de generación
- `requirements_per_second`: Requerimientos por segundo
- `requirement_X_length_chars`: Longitud de cada requerimiento
- `requirement_X_length_words`: Número de palabras por requerimiento

### Agente de Generación de Prompts
- `llm_generation_time_seconds`: Tiempo de generación del LLM
- `template_length_chars`: Longitud del template
- `template_length_lines`: Número de líneas del template
- `total_generation_time_seconds`: Tiempo total de generación
- `template_generation_rate`: Tasa de generación de templates
- `prompt_X_template_length_chars`: Longitud de cada prompt
- `prompt_X_has_code_variable`: Si el prompt tiene la variable `{{ code }}`

### Agente de Corrección de Código
- `template_render_time_seconds`: Tiempo de renderizado del template
- `rendered_prompt_length_chars`: Longitud del prompt renderizado
- `llm_analysis_time_seconds`: Tiempo de análisis del LLM
- `analysis_length_chars`: Longitud del análisis
- `total_analysis_time_seconds`: Tiempo total de análisis
- `analysis_rate`: Tasa de análisis
- `requirement_X_result`: Resultado del análisis (1.0 = YES, 0.0 = NO)

## 🛠️ Configuración

### Archivo de Configuración

El archivo `src/config/mlflow_config.yaml` permite configurar:

```yaml
mlflow:
  # Servidor de tracking
  tracking_uri: "file:./mlruns"  # Local (por defecto)
  # tracking_uri: "http://localhost:5000"  # Remoto
  
  # Nombre del experimento
  experiment_name: "assignment_evaluation"
  
  # Configuración de logging
  logging:
    level: "INFO"
    enable_console_logging: true
    enable_file_logging: false
    log_file_path: "./logs/mlflow.log"
  
  # Configuración de runs
  run:
    default_tags:
      project: "hyper_jade"
      version: "1.0.0"
```

### Configuración Local (Por Defecto)

```bash
# Los datos se guardan en ./mlruns/
mlflow ui --port 5000
```

### Configuración Remota

```yaml
mlflow:
  tracking_uri: "http://your-mlflow-server:5000"
  experiment_name: "assignment_evaluation"
```

## 📈 Uso

### 1. Ejecutar el Pipeline con Logging

```bash
# Ejecutar el pipeline normal (el logging es automático)
python main.py --assignment ejemplos/consigna.txt --code ejemplos/alu1.py

# Con contexto adicional
python main.py --assignment ejemplos/consigna.txt --code ejemplos/alu1.py --context "Análisis adicional"
```

### 2. Ver Métricas con MLflow UI

```bash
# Iniciar el servidor de MLflow UI
mlflow ui --port 5000

# Abrir en el navegador: http://localhost:5000
```

### 3. Usar el Script de Visualización

```bash
# Listar todos los runs
python view_mlflow_metrics.py --list-runs

# Ver detalles de un run específico
python view_mlflow_metrics.py --run-id <run_id>

# Comparar múltiples runs
python view_mlflow_metrics.py --compare <run_id1> <run_id2>

# Graficar una métrica específica
python view_mlflow_metrics.py --plot-metric "metrics.total_pipeline_time_seconds"

# Exportar datos de un run
python view_mlflow_metrics.py --export <run_id>
```

## 📁 Estructura de Artefactos

Cada run de MLflow incluye los siguientes artefactos:

```
artifacts/
├── assignment_description.txt          # Descripción de la asignación
├── requirements/                       # Requerimientos generados
│   ├── requirement_01.txt
│   ├── requirement_02.txt
│   └── ...
├── generated_templates/                # Templates de prompts
│   ├── requirement_01.jinja
│   ├── requirement_02.jinja
│   └── ...
├── input_prompt_template.jinja        # Template de entrada
├── input_student_code.py              # Código del estudiante
├── rendered_prompt.txt                # Prompt renderizado
├── generated_analysis.txt             # Análisis generado
└── output_analysis/                   # Análisis guardados
    ├── analysis_requirement_01.txt
    ├── analysis_requirement_02.txt
    └── ...
```

## 🔍 Análisis de Métricas

### Métricas de Rendimiento

- **Tiempo de Pipeline**: Monitorear el tiempo total de ejecución
- **Tiempo por Agente**: Identificar cuellos de botella
- **Tasa de Éxito**: Verificar la confiabilidad del sistema

### Métricas de Calidad

- **Longitud de Requerimientos**: Evaluar la complejidad
- **Longitud de Prompts**: Verificar la claridad de instrucciones
- **Resultados de Análisis**: Seguimiento de resultados YES/NO

### Métricas de LLM

- **Tiempo de Respuesta**: Rendimiento del modelo
- **Tamaño de Respuesta**: Complejidad de las respuestas
- **Tasa de Generación**: Eficiencia del proceso

## 🚨 Troubleshooting

### Problemas Comunes

1. **MLflow no puede conectarse al servidor**
   ```bash
   # Verificar que el servidor esté corriendo
   mlflow ui --port 5000
   ```

2. **No se ven métricas en la UI**
   - Verificar que el experimento esté configurado correctamente
   - Revisar los logs de la aplicación

3. **Errores de permisos**
   ```bash
   # Crear directorio de logs si no existe
   mkdir -p logs
   chmod 755 logs
   ```

### Logs de Debug

```bash
# Habilitar logging detallado
export MLFLOW_LOG_LEVEL=DEBUG

# Ver logs de MLflow
tail -f logs/mlflow.log
```

## 📚 Referencias

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow UI](https://mlflow.org/docs/latest/tracking.html#tracking-ui)

## 🤝 Contribución

Para agregar nuevas métricas o artefactos:

1. Modificar el agente correspondiente en `src/agents/`
2. Agregar logging en el método principal
3. Actualizar este documento
4. Probar con el script de visualización
