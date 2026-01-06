#!/bin/bash
# Run the Streamlit Comparator App

# Change to the script's directory
cd "$(dirname "$0")"

# Run streamlit using uv
uv run streamlit run app.py --server.headless true "$@"
