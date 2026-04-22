# RAG Chatbot

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in_development-orange)

An end-to-end RAG stack built on free, open-source components.
Developed to deepen hands-on experience with modern retrieval pipelines,
vector databases, and free-tier cloud deployment.

> **Status**: currently in active development.

---

## Features

- **Chat with your documents.** Upload PDFs or plain text, then ask questions in natural language. Answers are grounded in the retrieved content with source citations.
- **Works in Italian and English.** Multilingual semantic search via `multilingual-e5-base` embeddings. Documents and queries can be in either language.
- **Run it free, anywhere.** Fully local dev with Ollama (no API costs, no data leaves your machine). Free-tier cloud deployment via Groq when you want to share it.
- **Keep context across turns.** Session-based conversation memory — follow-up questions use the prior exchange as context.
- **Swap the LLM with one env variable.** Provider-agnostic design: the same code runs on local Ollama or cloud Groq without modification.

---

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

The **FastAPI** layer exposes REST endpoints for ingestion and chat.
The **LangChain** core orchestrates retrieval and generation.
**ChromaDB** stores document embeddings persistently on disk.
**Ollama** runs an open-source LLM locally; **Groq** is the free-tier cloud alternative.
**LangFuse** traces every retrieval and generation call for debugging and cost tracking.

---

## Tech stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI + Pydantic |
| Orchestration | LangChain |
| Vector database | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (`multilingual-e5-base`) |
| LLM (local) | Ollama (llama3.2:3b) |
| LLM (cloud) | Groq free tier (llama-3.1-8b-instant) |
| Frontend | Streamlit |
| Observability | LangFuse |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |

---

## Quick start

### Prerequisites

- Python 3.12
- [Ollama](https://ollama.com) installed locally

### Setup

```bash
# Clone
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot

# Virtual environment
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env

# Pull the LLM model (first run only, ~2GB)
ollama pull llama3.2:3b

# Run the API
uvicorn app.main:app --reload
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API documentation.

### Run tests

```bash
pytest tests/ -v
```

---

## API endpoints

The REST API is being built incrementally. Current and planned endpoints:

| Method | Path | Purpose | Available |
|---|---|---|---|
| `GET` | `/health` | Service health check | ✓ |
| `POST` | `/ingest/text` | Ingest raw text | Soon |
| `POST` | `/ingest/pdf` | Upload and ingest a PDF | Soon |
| `POST` | `/chat` | Ask a question | Soon |
| `GET` | `/documents` | List ingested documents | Soon |

Full OpenAPI documentation auto-generated at `/docs` once the server is running.

---

## Project structure

```
rag-chatbot/
├── app/
│   ├── core/          # RAG engine, factories (embeddings, LLM, vectorstore)
│   ├── api/           # FastAPI routes
│   ├── schemas/       # Pydantic request/response models
│   └── main.py        # FastAPI entry point
├── data/documents/    # User-uploaded documents (gitignored)
├── tests/             # Pytest test suite
├── .github/workflows/ # CI/CD pipelines
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Production readiness

This project applies **production patterns** — typed configuration, factory-based dependency injection, container deployment, automated tests, CI/CD, observability — but is **not production-ready** for real traffic. The following are deliberately out of scope:

- Authentication and authorization
- Rate limiting and request throttling
- High availability, failover, horizontal scaling
- Centralized logging and alerting
- Security audit and penetration testing
- Multi-tenancy and workspace isolation

Each of these has a reasonable solution path (OAuth2/JWT, Redis-based rate limiters, load balancers, ELK/Datadog, etc.), discussed on request.

---

## License

MIT — see [LICENSE](./LICENSE).
