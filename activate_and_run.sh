#!/bin/bash

# Activate virtual environment and set Python path
source .venv/bin/activate
export PYTHONPATH=".venv/lib/python3.13/site-packages:$PYTHONPATH"

# Run the command passed as arguments
exec "$@"
