# Independent Worker - JADE Server

This is an independent worker implementation that communicates with the JADE server via REST API calls.

## Features

- **Independent Operation**: Works as a separate project without imports from the main server
- **REST API Communication**: Uses HTTP requests to interact with the main server
- **Redis Job Queue**: Processes jobs from the same Redis queue as the main server
- **Modular Design**: Clean separation of concerns with API client, job management, and handlers

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables (optional):
```bash
export JADE_SERVER_URL=http://localhost:8000
export REDIS_URL=redis://localhost:6379
```

3. Start the worker:
```bash
python worker.py
```

## Job Types Supported

- **generate_requirements**: Generates requirements for an assignment
- **generate_prompts**: Updates requirement prompt templates  
- **create_correction**: Creates corrections for submissions

## Architecture

```
worker.py           # Main entry point
├── models.py       # Pydantic models
├── api_client.py   # REST API communication
├── redis_conn.py   # Redis connection
├── job_manager.py  # Job lifecycle management
└── job_handlers.py # Job processing logic
```
