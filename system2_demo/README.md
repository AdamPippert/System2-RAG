# System2 Demo: "Analyst + Critic + Verified Citations + Uncertainty"

This is a small, runnable demo that illustrates the ideas in your client message:

1. **Adversarial loop** (Analyst drafts, Critic attacks, then revise)
2. **Calibrated uncertainty** via *self-consistency* sampling (N runs → disagreement → "Uncertain")
3. **Citations as a constraint** (LLM must provide exact quote spans; deterministic verifier rejects mismatches)

It’s intentionally lightweight and **open-source friendly**:
- Local LLM via **Ollama** (recommended for the demo)
- Embeddings via **sentence-transformers**
- Retrieval via **simple cosine over numpy** (no vector DB dependency)
- UI via **FastAPI + Jinja + HTMX**

## 1) Prereqs

- Python 3.10+ (3.11 recommended)
- (Recommended) Ollama installed and running: https://ollama.com

Pull a model (example):
```bash
ollama pull llama3.1:8b
```

## 2) Install

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt
```

## 3) Ingest docs (optional; sample docs are included)

Build the initial index from `./sample_docs`:

```bash
python -m app.ingest --docs ./sample_docs
```

## 4) Run the web app

```bash
uvicorn app.main:app --reload
```

Open:
- http://127.0.0.1:8000

## 5) Configure LLM provider

By default, it uses Ollama:

Environment variables:
- `LLM_PROVIDER` = `ollama` (default) or `openai`
- `OLLAMA_MODEL` = model name (default `llama3.1:8b`)
- `OLLAMA_HOST` = Ollama base URL (default `http://localhost:11434`)

Optional OpenAI:
- `OPENAI_API_KEY` required
- `OPENAI_MODEL` default `gpt-4o-mini` (edit as desired)

Example:
```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.1:8b
```

## 6) Suggested demo script for a client meeting

Try these questions (with the included synthetic “Acme” docs):

1) **Evidence-heavy:**  
   “What was Acme’s Q3 gross margin and what changed vs Q2? Quote the source.”

2) **Unsupported claim (forces refusal / critique):**  
   “What was Acme’s customer churn rate in Q3?”  
   (Not present → Critic flags missing support.)

3) **Uncertainty via sampling:**  
   “Project Q4 operating income given the trend in Q2→Q3. Provide a range and label uncertainty.”

What you can point at in the UI:
- “Critic notes” showing why an answer is weak
- “Verified citations” showing exact quoted spans
- “Uncertainty” showing distribution across multiple runs

## Notes

- This is a demo for architecture concepts. It’s not hardened for production.
- For real deployments you’d add: auth, audit logs, caching, document provenance, and stricter schema validation.
