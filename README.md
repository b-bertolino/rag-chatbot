# RAG Chatbot
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-phase_1-orange)


A production-oriented Retrieval-Augmented Generation (RAG) chatbot built with
free, open-source components. Runs fully local via Ollama, or in the cloud
via Groq's free API tier.

> **Status:** Phase 1 — project scaffolding. See [roadmap](#roadmap) below.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│   FastAPI    │────▶│   LangChain  │
│   Frontend   │     │   REST API   │     │   RAG Chain  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                     ┌────────────────────────────┼──────────────────┐
                     ▼                            ▼                  ▼
              ┌──────────────┐            ┌──────────────┐   ┌──────────────┐
              │   ChromaDB   │            │    Ollama    │   │   LangFuse   │
              │  (vectors)   │            │   or Groq    │   │  (tracing)   │
              └──────────────┘            └──────────────┘   └──────────────┘
```

## Tech stack

- **API:** FastAPI + Pydantic
- **Orchestration:** LangChain
- **Vector DB:** ChromaDB (local, persistent)
- **Embeddings:** sentence-transformers (local, free)
- **LLM (local):** Ollama (Llama 3.2)
- **LLM (cloud):** Groq API (free tier)
- **Frontend:** Streamlit
- **Observability:** LangFuse (free tier)
- **Container:** Docker + docker-compose
- **CI/CD:** GitHub Actions

## Quick start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed locally

### Setup

```bash
# Clone and enter
git clone <your-repo-url>
cd rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Pull the Ollama model (first time only)
ollama pull llama3.2:3b

# Run the API
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for interactive API docs.

### Run tests

```bash
pytest tests/ -v
```

## Roadmap

- [x] **Phase 1** — Project scaffolding, config system, health endpoint
- [ ] **Phase 2** — RAG core (embeddings, ChromaDB, retrieval chain)
- [ ] **Phase 3** — Full FastAPI endpoints (`/chat`, `/ingest/text`, `/ingest/pdf`)
- [ ] **Phase 4** — Streamlit frontend
- [ ] **Phase 5** — Tests with mocking + LangFuse observability
- [ ] **Phase 6** — Docker + GitHub Actions CI/CD
- [ ] **Phase 7** — Cloud deployment (Render / HuggingFace Spaces)
- [ ] **Phase 8** — Polish, docs, demo video

## Project structure

```
rag-chatbot/
├── app/
│   ├── core/          # Config, RAG engine, LLM/embedding factories
│   ├── api/           # FastAPI routes
│   ├── schemas/       # Pydantic models
│   └── main.py        # FastAPI entry point
├── data/documents/    # Source documents for ingestion
├── tests/             # Pytest test suite
├── .github/workflows/ # CI/CD pipelines
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## License

MIT
