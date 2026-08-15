# Chapter 12 — Production Engineering

> **Previous**: [Chapter 11 — MCP](Chapter-11-MCP.md) | **Next**: [Chapter 13 — Testing & Evaluation](Chapter-13-Testing-Evaluation.md)

---

## 12.1 Production Checklist

```
Observability
  [ ] LangSmith / OpenTelemetry tracing enabled
  [ ] Structured JSON logging
  [ ] Metrics: latency, error rate, token cost

Reliability
  [ ] Retries with exponential backoff
  [ ] Timeout on every LLM call
  [ ] Fallback model when primary fails
  [ ] Max iterations enforced on all agents

Security
  [ ] Input validation and sanitization
  [ ] Prompt injection detection
  [ ] SSRF protection on URL tools
  [ ] No hardcoded API keys
  [ ] Rate limiting on public endpoints

Cost
  [ ] Response caching
  [ ] Cheap model for routing; expensive for generation
  [ ] Token counting before large requests
  [ ] Budget alerts

Deployment
  [ ] Docker image with non-root user
  [ ] Health check endpoint
  [ ] Graceful shutdown
  [ ] Load testing done
```

---

## 12.2 Observability with LangSmith

```python
import os
from langsmith import traceable, Client

# Enable tracing (set these before any LangChain imports)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]    = "my-agent-production"

# @traceable instruments any function — all nested LLM calls are captured
@traceable(name="research_pipeline", tags=["rag", "production"])
def research(query: str, user_id: str) -> dict:
    docs   = retrieve_docs(query)       # auto-traced
    answer = generate_answer(docs, query)  # auto-traced
    return {"answer": answer, "sources": [d.metadata["source"] for d in docs]}

# Submit human feedback on runs
client = Client()

def record_feedback(run_id: str, score: float, comment: str = ""):
    client.create_feedback(
        run_id=run_id,
        key="user_rating",
        score=score,        # 0.0 to 1.0
        comment=comment
    )
```

---

## 12.3 Structured JSON Logging

```python
import logging, json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts":      datetime.utcnow().isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log.update(record.extra)
        return json.dumps(log)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("agent")

# Usage
logger.info("Query processed",
            extra={"query": "...", "tokens": 1200, "latency_ms": 450})
```

---

## 12.4 Retries with Exponential Backoff

```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from openai import RateLimitError, APITimeoutError
import logging

logger = logging.getLogger(__name__)

@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
async def call_llm(messages: list) -> str:
    """Resilient LLM call — retries on rate limits and timeouts."""
    response = await llm.ainvoke(messages)
    return response.content
```

---

## 12.5 Rate Limiting

```python
import asyncio
from datetime import datetime, timedelta
from collections import deque

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period    = timedelta(seconds=period_seconds)
        self.calls: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = datetime.now()
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                wait = (self.calls[0] + self.period - now).total_seconds()
                await asyncio.sleep(wait)

            self.calls.append(datetime.now())

# 60 LLM calls per minute
llm_limiter = RateLimiter(max_calls=60, period_seconds=60)

async def rate_limited_call(messages: list) -> str:
    await llm_limiter.acquire()
    return await call_llm(messages)
```

---

## 12.6 Security: Input Validation

```python
from pydantic import BaseModel, field_validator, Field
import re

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore.{0,20}(previous|above|all).{0,20}instruction",
    r"you are now",
    r"pretend.{0,20}(you are|to be)",
    r"disregard.{0,20}(all|previous).{0,20}instruction",
]

class SafeUserQuery(BaseModel):
    query:   str = Field(max_length=5000)
    user_id: str

    @field_validator("query")
    @classmethod
    def no_injection(cls, v: str) -> str:
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Query contains unsafe content")
        return v

class SafeToolInput(BaseModel):
    url:       str | None = None
    file_path: str | None = None

    @field_validator("url")
    @classmethod
    def no_ssrf(cls, v: str | None) -> str | None:
        """Block requests to internal/private IP ranges."""
        if v is None:
            return v
        import ipaddress
        from urllib.parse import urlparse

        parsed = urlparse(v)
        blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1",
                         "169.254.169.254"}  # AWS metadata endpoint

        host = parsed.hostname or ""
        if host in blocked_hosts:
            raise ValueError(f"Access to {v} is blocked")

        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback:
                raise ValueError("Access to private IPs is blocked")
        except ValueError:
            pass  # Not an IP — hostname is fine

        return v

    @field_validator("file_path")
    @classmethod
    def no_path_traversal(cls, v: str | None) -> str | None:
        """Prevent directory traversal attacks."""
        if v is None:
            return v
        import pathlib
        resolved = pathlib.Path(v).resolve()
        allowed  = pathlib.Path("./data").resolve()
        if not str(resolved).startswith(str(allowed)):
            raise ValueError("File access outside allowed directory")
        return str(resolved)
```

---

## 12.7 Cost Optimization

```python
import langchain
from langchain_community.cache import SQLiteCache

# Cache identical prompts — same input → same output, no API call
langchain.llm_cache = SQLiteCache(database_path=".llm_cache.db")

class SmartRouter:
    """Route to cheap model for simple tasks, expensive for complex."""

    CHEAP  = "gpt-4o-mini"
    QUALITY = "gpt-4o"

    SIMPLE_KEYWORDS = {"translate", "summarize", "classify", "extract", "list"}

    def pick_model(self, task: str) -> str:
        if any(kw in task.lower() for kw in self.SIMPLE_KEYWORDS):
            return self.CHEAP
        return self.QUALITY

class TokenBudget:
    """Track and alert on token spend."""

    PRICES = {
        "gpt-4o":      {"in": 2.50, "out": 10.00},
        "gpt-4o-mini": {"in": 0.15, "out":  0.60}
    }

    def __init__(self, budget_usd: float):
        self.budget = budget_usd
        self.spent  = 0.0

    def record(self, model: str, in_tokens: int, out_tokens: int):
        p = self.PRICES.get(model, {"in": 0, "out": 0})
        cost = (in_tokens * p["in"] + out_tokens * p["out"]) / 1_000_000
        self.spent += cost
        if self.spent > self.budget * 0.9:
            logger.warning(f"Token budget 90% consumed: ${self.spent:.4f}/${self.budget}")
```

---

## 12.8 FastAPI Production Endpoint

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import time, uuid, logging

app = FastAPI(title="Agent API", version="1.0.0")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"]
)

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None

class QueryResponse(BaseModel):
    answer:        str
    session_id:    str
    run_id:        str
    duration_ms:   float

logger = logging.getLogger("api")

@app.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    run_id     = str(uuid.uuid4())
    session_id = body.session_id or str(uuid.uuid4())
    t0         = time.time()

    try:
        # Validate
        safe = SafeUserQuery(query=body.query, user_id="anonymous")

        # Run agent
        result = await agent.ainvoke({"messages": [("user", safe.query)]})
        answer = result["messages"][-1].content

        ms = (time.time() - t0) * 1000
        logger.info("query_ok", extra={"run_id": run_id, "ms": ms})

        return QueryResponse(
            answer=answer, session_id=session_id,
            run_id=run_id, duration_ms=ms
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("query_error", extra={"run_id": run_id, "error": str(e)})
        raise HTTPException(status_code=500, detail="Agent error")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

---

## 12.9 Docker Deployment

```dockerfile
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## Summary

- LangSmith tracing is the easiest production observability for LangChain/LangGraph
- Always retry on `RateLimitError` with exponential backoff
- Validate all user input with Pydantic before it reaches the agent
- SSRF and path traversal are the two most common tool security vulnerabilities
- Cache responses + route cheap/expensive models to control costs

## Exercises

1. Add LangSmith tracing to your RAG chain. View a trace in the UI.
2. Wrap an LLM call with `@retry(stop=stop_after_attempt(3))`.
3. Write a `SafeUserQuery` validator that blocks prompt injection patterns.
4. Deploy your FastAPI agent with Docker and test the `/health` endpoint.

---

> **Next**: [Chapter 13 — Testing & Evaluation](Chapter-13-Testing-Evaluation.md)
