# ARGUS Quickstart

Get ARGUS running in under 10 minutes.

## Prerequisites

- Python 3.11+ (tested on 3.11, 3.12, 3.13)
- At least one LLM provider API key (Groq recommended — free tier available)
- Git

## Installation

```bash
# Clone the repository
git clone https://github.com/armanojha/ARGUS.git
cd ARGUS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate   # Windows PowerShell

# Install dependencies
pip install -e ".[core,retrieval,graph,multimodal,dev-test]"
```

## Configuration

```bash
# Copy the environment template
cp .env.example .env
```

Edit `.env` and add at least one API key:

```bash
# Option 1: Groq (free tier, recommended for getting started)
GROQ_API_KEY=gsk_your_key_here

# Option 2: Gemini
GEMINI_API_KEY=your_key_here
```

## Start the Server

```bash
uvicorn app.api.main:app --reload
```

Wait for the "Application startup complete" message.

## Open the Brain UI

Open your browser to:

```
http://localhost:8000/brain
```

You should see the ARGUS Brain with the Pipeline view.

## Try Demo Mode (No API Key)

If you don't have an API key yet:

1. Open `http://localhost:8000/brain`
2. Click **Load Demo** on the pipeline view
3. Explore the pre-recorded research result

## Ask a Question

1. Click **Research** in the sidebar
2. Type a question (e.g., "What is machine learning?")
3. Press Enter or click Send
4. Watch the pipeline build progressively
5. Click nodes to inspect each stage

## Ingest Documents

To query your own documents:

1. Place files (PDF, TXT, MD, CSV, XLSX) in `knowledge_base/`
2. Click **Knowledge Base** in the sidebar
3. Click **Re-ingest knowledge folder**
4. Or use the API:
   ```bash
   curl -X POST http://localhost:8000/api/v1/knowledge-base/ingest
   ```

## Verify It Works

```bash
# Health check
curl http://localhost:8000/health

# Quick query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is ARGUS?"}'
```

## Run Tests

```bash
python -m pytest tests/ -x -q --ignore=tests/llm_gateway/test_phase07e_recovery.py
```

Expected: 903 passed, 26 skipped, 5 pre-existing failures.

## Troubleshooting

### Server won't start
- Check Python version: `python --version` (needs 3.11+)
- Check dependencies: `pip list | grep fastapi`

### No answers from research
- Verify API key is set in `.env`
- Check provider status in Settings view
- Try Demo Mode to verify the UI works

### Brain UI shows "Could not reach the API"
- Ensure the server is running: `curl http://localhost:8000/health`
- Check for port conflicts: try `--port 8001`

### Tests fail
- Ensure you're in the project root directory
- Try: `pip install -e ".[core,retrieval,graph,multimodal,dev-test]"`

## Next Steps

- Read the [Architecture](ARCHITECTURE.md) guide
- Explore the [Brain UI](BRAIN_UI.md) features
- Check [Limitations](LIMITATIONS.md) for known issues
