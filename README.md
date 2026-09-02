# BrAIn OS — Bi-directional Artificial Intelligence Network Operating System

> **An OS-inspired Agent Kernel that treats LLM agents as first-class processes with memory isolation, concurrent priority scheduling, hierarchical memory, and production-grade bi-directional guardrails.**

---

## 🌟 Overview

BrAIn OS is **not** another single-prompt chatbot wrapper. It is a **true operating system kernel designed for AI agents**:
* **Agents as Processes**: Every specialized agent is an isolated process with its own PID, priority, and capability whitelist.
* **Bi-Directional Guardrails**: Security and sanitization on both incoming user queries (prompt injection detection, input sanitization) and outgoing agent responses (PII redaction, hallucination checks, toxicity filtering).
* **Attention-Based MoE Router**: Gating mechanism routing queries to the top specialized expert agents.
* **CFS-Inspired Priority Scheduler**: Priority queue (P0–P3), concurrency semaphore, dynamic priority aging, and cooperative preemption.
* **Three-Tier Memory Hierarchy**: L1 Working Context (in-memory LRU/TTL) → L2 Episodic Store (Redis + semantic index) → L3 Persistent Store (ChromaDB vector database with cross-encoder reranking).
* **Production API Gateway**: FastAPI REST endpoints and real-time streaming WebSockets.

---

## 🏛 Architecture

```
User Request / WebSocket
         │
         ▼
┌──────────────────────────────┐
│  API Gateway (FastAPI)       │  ← REST + WebSocket, Rate Limiting, CORS
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Input Guardrails            │  ← Prompt Injection Detection, Sanitization
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  MoE Attention Router        │  ← Gating Network, Top-K Expert Selection
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Priority Scheduler          │  ← Async Concurrency Semaphore, Priority Aging
└──────────────┬───────────────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│Research │ │  Code   │ │Reasoning│  (5 Sandboxed Agent Processes)
│ Agent   │ │  Agent  │ │  Agent  │
└────┬────┘ └────┬────┘ └────┬────┘
     │           │           │
     ▼           ▼           ▼
┌──────────────────────────────┐
│  Hierarchical Memory         │  ← L1 Hot Cache → L2 Redis → L3 ChromaDB
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Output Guardrails           │  ← PII Redaction, Safety Filter, Validation
└──────────────┬───────────────┘
               │
               ▼
User Response / WebSocket Stream
```

---

## 🤖 The 5 Specialized Sub-Agents

| Agent | Priority | Allowed Tools | Role |
|---|---|---|---|
| **Research Agent** | P2 (Normal) | `web_search`, `file_read`, `llm_call` | Information discovery, web search, factual verification |
| **Code Agent** | P1 (High) | `file_read`, `file_write`, `code_exec`, `llm_call` | Code generation, debugging, refactoring |
| **Data Agent** | P1 (High) | `file_read`, `db_query`, `llm_call` | Data analysis, SQL queries, visualizations |
| **Reasoning Agent** | P0 (Critical) | `file_read`, `llm_call` | Chain-of-thought planning, task verification |
| **Conversation Agent** | P3 (Low) | `llm_call` | General Q&A, natural dialogue, summarization |

---

## ⚡ Quick Start

### 1. Prerequisites
* Python 3.11+
* [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
* Docker (for Redis and ChromaDB) or local Redis service

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/shreyas/BrAIn-OS-Bi-directional-Artificial-Intelligence-Network-Operating-System-.git
cd BrAIn-OS-Bi-directional-Artificial-Intelligence-Network-Operating-System-

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 3. Environment Setup
Copy the example environment file and configure your backends:
```bash
cp .env.example .env
```
Key settings in `.env`:
* **Local (Free)**: Set `LLM_DEFAULT_BACKEND=ollama` and ensure `ollama serve` is running.
* **Cloud (Fast)**: Set `LLM_DEFAULT_BACKEND=groq` and configure `GROQ_API_KEY`.

### 4. Running Supporting Infrastructure
Using Docker:
```bash
# Start Redis & ChromaDB
docker-compose up -d redis chromadb
```

Or run everything (including the BrAIn OS Gateway) with one command:
```bash
docker-compose up --build
```

### 5. Running the Kernel Gateway Locally
```bash
brain-os
```
The API is now live at `http://localhost:8080` (Interactive Swagger docs at `http://localhost:8080/docs`).

---

## 🧪 Testing

Run the full automated test suite:
```bash
pytest -v
```

Run test suite with coverage:
```bash
pytest --cov=brain_os tests/
```

---

## 📡 API Reference

### Submit Task
`POST /tasks`
```bash
curl -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Write a Python script to sort a list of dictionaries by key"
  }'
```

### System Health
`GET /health`
```bash
curl http://localhost:8080/health
```

### Runtime Metrics & Process State
`GET /status`
```bash
curl http://localhost:8080/status
```

### List Agents
`GET /agents`
```bash
curl http://localhost:8080/agents
```

### Interactive WebSocket Stream
`ws://localhost:8080/ws/{session_id}`
```json
// Send query
{ "query": "Analyze dataset sales.csv and summarize key trends" }
```

---

## 🔒 Security & Guardrails

1. **Least-Privilege Tool Sandboxing**: Agents attempting unauthorized tools (e.g. Research Agent trying to invoke `code_exec` or `file_write`) are immediately denied with a `PermissionDeniedError` and audited.
2. **Prompt Injection Protection**: Heuristic pattern filters reject jailbreak attempts, system prompt exfiltration, and adversarial instruction overrides.
3. **PII Masking**: Emails, phone numbers, SSNs, credit card numbers, and API tokens are automatically detected and masked with `[REDACTED_*]` tags.
4. **Structured Audit Trail**: All security events, tool calls, and input/output guardrail checks are saved as immutable JSON logs in `logs/audit.jsonl`.

---

## 📄 License
MIT License.