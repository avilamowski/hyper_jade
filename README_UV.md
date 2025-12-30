# Migración a UV

Este proyecto ahora usa `uv` en lugar de `pip` y `requirements.txt`.

## Instalación de uv

```bash
# En Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# O usando pip
pip install uv
```

## Comandos básicos

### Instalar dependencias
```bash
uv sync
```

### Ejecutar scripts
```bash
uv run python main.py
uv run python setup_rag.py
```

### Agregar una nueva dependencia
```bash
uv add nombre-paquete
```

### Agregar una dependencia de desarrollo
```bash
uv add --dev nombre-paquete
```

### Actualizar dependencias
```bash
uv lock --upgrade
```

### Ejecutar con el script activado
```bash
./activate_and_run.sh python main.py
```

## Ventajas de UV

- **10-100x más rápido** que pip
- Resolución de dependencias determinista
- Lockfile automático (`uv.lock`)
- Compatible con `pyproject.toml`
- Gestión automática de entornos virtuales

## Archivos importantes

- `pyproject.toml`: Define las dependencias del proyecto
- `uv.lock`: Lockfile con versiones exactas (auto-generado)
- `.venv/`: Entorno virtual (auto-gestionado por uv)

## Notas

El archivo `requirements.txt` se mantiene por compatibilidad pero ya no es necesario para el desarrollo local.
