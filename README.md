# Hyper Jade - Assignment Evaluation System

Un sistema de evaluación de tareas de programación que utiliza tres agentes especializados para analizar código de estudiantes de manera automática y detallada.

## Estructura del Sistema

El sistema está compuesto por tres agentes que trabajan en secuencia:

### 1. RequirementGenerator
**Input:** Una consigna (archivo .txt)
**Output:** Múltiples requerimientos individuales (archivos .txt separados)

- Analiza la consigna de programación
- Identifica todos los aspectos que deben ser evaluados
- Genera requerimientos específicos y medibles
- Cada requerimiento se guarda como un archivo separado

### 2. PromptGenerator
**Input:** UN requerimiento (archivo .txt)
**Output:** UN prompt Jinja2 (archivo .jinja)

- Toma un requerimiento específico
- Genera una plantilla Jinja2 que servirá como prompt para el análisis
- La plantilla permite inyectar código y contexto adicional
- Específica para evaluar ese requerimiento en particular

### 3. CodeCorrector
**Input:** Un prompt Jinja2 (.jinja) y un archivo de código Python (.py o .txt)
**Output:** Un análisis de qué tan bien satisface el requerimiento el código

- Renderiza la plantilla Jinja2 con el código del estudiante
- Analiza el código contra el requerimiento específico
- Genera un análisis detallado con errores, sugerencias y evaluación conceptual

## Instalación

1. Clona el repositorio:
```bash
git clone <repository-url>
cd hyper_jade
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura el modelo de lenguaje en `src/config/assignment_config.yaml`:
```yaml
provider: "ollama"  # o "openai"
model_name: "qwen2.5:7b"  # o "gpt-4" para OpenAI
temperature: 0.1
```

## Uso

### Pipeline Completo

Para ejecutar todo el pipeline de evaluación:

```bash
python main.py --assignment ejemplos/consigna.txt --code ejemplos/alu1.py --output-dir outputs
```

### Agentes Individuales

#### 1. Generar Requerimientos
```bash
python run_requirement_generator.py --assignment ejemplos/consigna.txt --output-dir outputs/requirements
```

#### 2. Generar Prompts
```bash
python run_prompt_generator.py --requirement outputs/requirements/requirement_01.txt --output outputs/prompts/prompt_01.jinja
```

#### 3. Analizar Código
```bash
python run_code_corrector.py --prompt outputs/prompts/prompt_01.jinja --code ejemplos/alu1.py --output outputs/analyses/analysis_01.txt
```

### Modo Batch

Para analizar múltiples archivos de código:

```bash
python run_code_corrector.py --prompt outputs/prompts/prompt_01.jinja --batch --code-dir ejemplos --output-dir outputs/analyses
```

## Estructura de Archivos

```
hyper_jade/
├── src/
│   ├── agents/
│   │   ├── requirement_generator/
│   │   ├── prompt_generator/
│   │   └── code_corrector/
│   └── config/
├── ejemplos/
│   ├── consigna.txt
│   ├── alu1.py
│   └── ...
├── outputs/
│   ├── requirements/
│   │   ├── requirement_01.txt
│   │   └── ...
│   ├── prompts/
│   │   ├── prompt_01.jinja
│   │   └── ...
│   └── analyses/
│       ├── analysis_01.txt
│       └── ...
├── main.py
├── run_requirement_generator.py
├── run_prompt_generator.py
├── run_code_corrector.py
└── requirements.txt
```

## Configuración

### Modelos Soportados

- **Ollama:** Para uso local con modelos como qwen2.5:7b, llama3.2, etc.
- **OpenAI:** Para uso con GPT-4, GPT-3.5-turbo, etc.

### Configuración de Ollama

1. Instala Ollama: https://ollama.ai/
2. Descarga un modelo:
```bash
ollama pull qwen2.5:7b
```
3. Configura en `src/config/assignment_config.yaml`:
```yaml
provider: "ollama"
model_name: "qwen2.5:7b"
temperature: 0.1
```

### Configuración de OpenAI

1. Configura tu API key:
```bash
export OPENAI_API_KEY="tu-api-key"
```
2. Configura en `src/config/assignment_config.yaml`:
```yaml
provider: "openai"
model_name: "gpt-4"
temperature: 0.1
```

## Ejemplos de Uso

### Ejemplo 1: Evaluación Completa
```bash
# Evaluar un archivo de código contra una consigna
python main.py \
  --assignment ejemplos/consigna.txt \
  --code ejemplos/alu1.py \
  --output-dir outputs/evaluacion_alu1 \
  --verbose
```

### Ejemplo 2: Solo Generar Requerimientos
```bash
# Generar requerimientos para reutilizar después
python run_requirement_generator.py \
  --assignment ejemplos/consigna.txt \
  --output-dir outputs/requirements \
  --verbose
```

### Ejemplo 3: Analizar Múltiples Códigos
```bash
# Analizar todos los archivos en el directorio ejemplos
python run_code_corrector.py \
  --prompt outputs/prompts/prompt_01.jinja \
  --batch \
  --code-dir ejemplos \
  --output-dir outputs/analyses_batch
```

## Formato de Archivos

### Consigna (.txt)
```
Implementa una función que calcule el factorial de un número.
La función debe:
- Manejar números enteros positivos
- Retornar el resultado correcto
- Incluir validación de entrada
```

### Requerimiento (.txt)
```
Requerimiento 1: La función debe calcular correctamente el factorial de números enteros positivos
```

### Prompt Jinja2 (.jinja)
```jinja2
# Análisis de Código - Evaluación de Requerimiento

## Requerimiento a Evaluar:
{{ requirement }}

## Código del Estudiante:
```python
{{ student_code }}
```

## Instrucciones de Evaluación:
Evalúa el código contra el requerimiento específico...
```

### Análisis (.txt)
```
# Análisis de Código - Evaluación de Requerimiento

## Requerimiento a Evaluar:
La función debe calcular correctamente el factorial de números enteros positivos

## Código del Estudiante:
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
```

## Análisis:

### EVALUACIÓN DEL REQUERIMIENTO:
✅ El código satisface el requerimiento. La función factorial implementa correctamente el cálculo del factorial.

### ERRORES ENCONTRADOS:
❌ No hay validación de entrada para números negativos
❌ No hay manejo de casos edge (números muy grandes)

### UBICACIÓN DE PROBLEMAS:
- Línea 1: Falta validación de entrada
- No hay manejo de excepciones

### SUGERENCIAS DE MEJORA:
1. Agregar validación para números negativos
2. Implementar manejo de excepciones
3. Considerar límites de recursión para números grandes

### JUICIO CONCEPTUAL:
El estudiante demuestra comprensión del concepto de factorial y recursión, pero falta atención a la validación de entrada.
```

## Características

- **Modular:** Cada agente puede ejecutarse independientemente
- **Reutilizable:** Los requerimientos y prompts se pueden reutilizar
- **Escalable:** Soporte para análisis batch de múltiples archivos
- **Configurable:** Fácil cambio entre diferentes modelos de lenguaje
- **Detallado:** Análisis específico por requerimiento
- **Flexible:** Soporte para diferentes lenguajes de programación

## Contribución

1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.
