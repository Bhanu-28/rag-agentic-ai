# Local Ollama Setup Guide

This guide explains how to run LLM and embedding models locally using Ollama in Docker.

## Architecture

```
Browser / Phone (LAN)
        |
        v
  Open WebUI (:3000)
        |
        v
    Ollama (:11434)
    ├── qwen2.5-coder:3b  (LLM)
    └── nomic-embed-text:v1.5 (Embeddings)
```

## Prerequisites

- Docker installed: `sudo apt install docker.io`
- Docker Compose: `sudo apt install docker-compose`
- Add user to docker group: `sudo usermod -aG docker $USER`

---

## Quick Start (Combined Setup)

Single Ollama container for both LLM and embeddings:

```bash
# Start services
docker-compose up -d

# Pull models into container
docker exec ollama ollama pull qwen2.5-coder:3b
docker exec ollama ollama pull nomic-embed-text:v1.5

# Verify models
docker exec ollama ollama list
```

Access Open WebUI at: http://localhost:3000

---

## Separate Containers Setup

Use this when running LLM and embeddings concurrently (better resource isolation):

```bash
# Start with separate config
docker-compose -f docker-compose.separate.yml up -d

# Pull models into respective containers
docker exec ollama-llm ollama pull qwen2.5-coder:3b
docker exec ollama-embed ollama pull nomic-embed-text:v1.5

# Verify
docker exec ollama-llm ollama list
docker exec ollama-embed ollama list
```

**Ports:**
- LLM: `http://localhost:11434`
- Embeddings: `http://localhost:11435`

---

## Manual Setup (Without Docker Compose)

If you prefer manual Docker commands:

```bash
# Create volume for models
docker volume create ollama_models

# Run Ollama container
docker run -d \
  -v ollama_models:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama

# Pull models
docker exec ollama ollama pull qwen2.5-coder:3b
docker exec ollama ollama pull nomic-embed-text:v1.5

# Run Open WebUI
docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

---

## Useful Commands

```bash
# List running models
docker exec ollama ollama ps

# Stop a model (free memory)
docker exec ollama ollama stop qwen2.5-coder:3b

# List downloaded models
docker exec ollama ollama list

# Check containers
docker ps | grep ollama

# View logs
docker logs ollama
docker logs open-webui

# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

## Access from Other Devices

Your local IP: Find using `ifconfig` or `ip addr`

From other devices on the same network:
- Open WebUI: `http://<your-ip>:3000`
- Ollama API: `http://<your-ip>:11434`

Example (based on your setup): `http://192.168.1.239:3000`

---

## Python Usage

See [ollama_utils.py](ollama_utils.py) for LangChain integration.

```python
from ollama_utils import get_ollama_llm, get_ollama_embeddings

# LLM
llm = get_ollama_llm()
response = llm.invoke("Hello!")

# Embeddings
embeddings = get_ollama_embeddings()
vectors = embeddings.embed_documents(["text1", "text2"])
```

---

## Environment Variables

Configure in `.env`:

```env
OLLAMA_LLM_URL=http://localhost:11434
OLLAMA_EMBED_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5-coder:3b
OLLAMA_EMBED_MODEL=nomic-embed-text:v1.5
```

For separate containers setup:
```env
OLLAMA_LLM_URL=http://localhost:11434
OLLAMA_EMBED_URL=http://localhost:11435
```
