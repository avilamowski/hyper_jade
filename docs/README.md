# AI Agent for Generating Debug Prompts

An intelligent system that creates specialized prompts for AI agents designed to debug and correct student code. This system builds on the foundation of the [JADE_Scripts](https://github.com/david-wis/JADE_Scripts) repository and provides a comprehensive framework for generating targeted debugging prompts.

## Features

### 🎯 **Specialized Debug Agents**
- **Syntax Error Detector**: Identifies grammar, structure, and language rule violations
- **Logic Error Analyzer**: Finds algorithmic bugs, flow control issues, and data manipulation problems
- **Performance Optimizer**: Analyzes complexity, efficiency, and resource usage
- **Code Style Checker**: Evaluates readability, conventions, and documentation
- **Test Case Generator**: Creates comprehensive test scenarios and edge cases
- **Security Vulnerability Scanner**: Identifies input validation and data protection issues
- **Comprehensive Debug Agent**: Combines all specialized agents for complete analysis

### 🧠 **AI-Powered Prompt Generation**
- Uses LangGraph for structured prompt generation workflows
- Supports multiple LLM providers (Ollama, OpenAI)
- Configurable difficulty levels (beginner, intermediate, advanced)
- Language-specific patterns and best practices
- Template-based prompt generation with Jinja2

### 🔧 **Easy RAG Integration**
- Modular RAG extension for future enhancement
- Knowledge base management with relevance scoring
- Support for vector stores (Chroma, FAISS) - ready for implementation
- Mock implementation for development and testing

### 📊 **Comprehensive Output**
- Structured XML-like output format for consistent parsing
- Severity ratings for issues (Low/Medium/High/Critical)
- Detailed explanations and fix suggestions
- Learning resources and references
- Confidence scoring for generated prompts

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama (for local LLM support) or OpenAI API key
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd hyper_jade
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure the system**:
```bash
# Copy and edit the configuration file
cp debug_config.yaml debug_config_local.yaml
# Edit debug_config_local.yaml with your settings
```

4. **Run the example**:
```bash
python example_usage.py
```

## Usage

### Basic Usage

```python
from debug_prompt_generator import DebugPromptGenerator

# Initialize the generator
generator = DebugPromptGenerator("debug_config.yaml")

# Generate a debug prompt
result = generator.generate_prompt(
    student_code="def calculate_average(numbers):\n    return sum(numbers) / len(numbers)",
    assignment_description="Calculate the average of a list of numbers",
    programming_language="python",
    debug_agent_type="logic",
    difficulty_level="intermediate"
)

print(result["prompt"])
```

### Command Line Interface

```bash
# Generate a single debug prompt
python debug_prompt_generator.py \
    --code student_code.py \
    --assignment assignment.md \
    --language python \
    --agent-type logic \
    --difficulty intermediate \
    --output result.json

# Generate prompts for all agent types
python debug_prompt_generator.py \
    --code student_code.py \
    --assignment assignment.md \
    --all \
    --output all_prompts.json
```

### Advanced Usage with RAG

```python
from rag_extension import RAGEnhancer, RAGConfig

# Configure RAG
rag_config = RAGConfig(
    enable_rag=True,
    max_results=5,
    similarity_threshold=0.7
)

# Create RAG enhancer
enhancer = RAGEnhancer(rag_config)

# Enhance a prompt with relevant knowledge
enhanced_prompt, knowledge_items = enhancer.enhance_prompt(
    base_prompt=base_prompt,
    student_code=student_code,
    assignment_description=assignment_description,
    debug_agent_type="logic",
    programming_language="python",
    difficulty_level="intermediate"
)
```

## Configuration

### Debug Agent Configuration (`debug_config.yaml`)

```yaml
# LLM Configuration
model_name: "qwen2.5:7b"
provider: "ollama"  # or "openai"
temperature: 0.1
max_tokens: 2000

# RAG Configuration
enable_rag: false
rag_knowledge_base: null

# Language-specific settings
languages:
  python:
    syntax_patterns:
      - "IndentationError"
      - "SyntaxError"
    common_mistakes:
      - "forgetting self in methods"
    best_practices:
      - "PEP 8 style guide"

# Agent type configurations
agent_types:
  syntax:
    focus_areas: ["grammar", "structure", "language rules"]
    severity_levels: ["Low", "Medium", "High", "Critical"]
```

### Supported Programming Languages

- **Python**: Full support with PEP 8, type hints, and Python-specific patterns
- **JavaScript**: ES6+ features, async/await, and JavaScript-specific issues
- **Java**: SOLID principles, exception handling, and Java-specific patterns
- **Extensible**: Easy to add support for additional languages

## Architecture

### Core Components

1. **DebugPromptGenerator**: Main class for generating debug prompts
2. **StateGraph**: LangGraph-based workflow for prompt generation
3. **TemplateManager**: Jinja2-based template system
4. **RAGEnhancer**: Optional RAG integration for enhanced prompts

### Workflow

```
Student Code + Assignment → Context Analysis → Prompt Generation → Validation → Final Prompt
```

### State Management

The system uses a `DebugPromptState` that includes:
- Input: student code, assignment, language, agent type
- Context: similar examples, common mistakes, best practices
- Output: generated prompt, metadata, confidence score

## Output Format

The generated prompts produce structured output in XML-like format:

```xml
<ANALYSIS>
  <ISSUES>
    <ISSUE severity="High" type="logic">
      <DESCRIPTION>Off-by-one error in loop</DESCRIPTION>
      <LOCATION>Line 5: for i in range(len(numbers))</LOCATION>
      <EXPLANATION>Loop iterates one extra time</EXPLANATION>
      <FIX>Use range(len(numbers)-1) or adjust logic</FIX>
    </ISSUE>
  </ISSUES>
  
  <SUMMARY>
    <TOTAL_ISSUES>3</TOTAL_ISSUES>
    <CRITICAL_ISSUES>1</CRITICAL_ISSUES>
    <OVERALL_ASSESSMENT>Code has logic errors but good structure</OVERALL_ASSESSMENT>
  </SUMMARY>
  
  <LEARNING_RESOURCES>
    <RESOURCE type="documentation">
      <TITLE>Python Loops Guide</TITLE>
      <URL>https://docs.python.org/3/tutorial/controlflow.html</URL>
    </RESOURCE>
  </LEARNING_RESOURCES>
</ANALYSIS>
```

## Extending the System

### Adding New Debug Agent Types

1. **Update `DEBUG_AGENT_TYPES`** in `debug_prompt_generator.py`
2. **Add configuration** in `debug_config.yaml`
3. **Create template** in `templates/` directory
4. **Update helper functions** for language-specific patterns

### Adding RAG Support

1. **Enable RAG** in configuration
2. **Implement vector store** (Chroma, FAISS)
3. **Add embedding models** (OpenAI, Ollama)
4. **Create knowledge base** with relevant examples

### Adding New Languages

1. **Update language patterns** in `_get_language_patterns()`
2. **Add language-specific templates**
3. **Update configuration** with language-specific settings
4. **Test with sample code**

## Examples

### Sample Student Code with Issues

```python
def calculate_average(numbers):
    total = 0
    count = 0
    
    for i in range(len(numbers)):
        total += numbers[i]
        count += 1
    
    if count == 0:
        return 0
    else:
        return total / count

def find_maximum(values):
    max_val = values[0]  # Will crash on empty list
    
    for val in values:
        if val > max_val:
            max_val = val
    
    return max_val
```

### Generated Debug Prompt (Logic Agent)

```
You are a specialized Logic Error Analyzer for Python code.

TASK: Analyze the following student code and identify logic-related issues...

STUDENT CODE:
[student code here]

FOCUS AREAS:
- algorithms
- flow control
- data manipulation

COMMON LOGIC ISSUES TO CHECK:
- off-by-one errors
- wrong variable usage
- missing edge cases

INSTRUCTIONS:
1. Analyze the algorithm and logic flow
2. Identify off-by-one errors and boundary conditions
3. Check for incorrect variable usage and scope issues
...

OUTPUT FORMAT:
<LOGIC_ANALYSIS>
  [structured output format]
</LOGIC_ANALYSIS>
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-agent-type`
3. **Make your changes** and add tests
4. **Run tests**: `python -m pytest tests/`
5. **Submit a pull request**

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the [JADE_Scripts](https://github.com/david-wis/JADE_Scripts) repository
- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for workflow management
- Uses [Jinja2](https://jinja.palletsprojects.com/) for template rendering
- Supports [Ollama](https://ollama.ai/) for local LLM inference

## Roadmap

- [ ] **Vector Store Integration**: Full RAG implementation with Chroma/FAISS
- [ ] **Multi-language Support**: Add support for C++, Rust, Go, and more
- [ ] **Interactive Mode**: Web interface for prompt generation
- [ ] **Prompt Evaluation**: Metrics for prompt quality assessment
- [ ] **Knowledge Base Management**: Tools for managing and curating knowledge
- [ ] **Integration APIs**: REST API for external system integration
