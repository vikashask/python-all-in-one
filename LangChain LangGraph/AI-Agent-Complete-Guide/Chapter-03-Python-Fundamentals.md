# Chapter 03 — Python Fundamentals for Agent Development

> **Previous**: [Chapter 02 — Roadmap](Chapter-02-Learning-Roadmap.md) | **Next**: [Chapter 04 — LLM Core Concepts](Chapter-04-LLM-Core-Concepts.md)

---

You need specific Python skills for agent work. This chapter covers exactly what matters — nothing more.

---

## 3.1 Type Hints and TypedDict

`TypedDict` is the foundation of LangGraph state. Type hints make agent code safe and readable.

```python
from typing import TypedDict, Annotated, Literal, Optional, List
import operator

# TypedDict: a dict with a fixed schema
class AgentState(TypedDict):
    query: str
    documents: list
    answer: str
    # Annotated + operator.add means "append new items, don't replace"
    messages: Annotated[list, operator.add]
    retry_count: int

# Literal restricts a value to specific strings
def route(state: AgentState) -> Literal["search", "generate", "end"]:
    if not state["documents"]:
        return "search"
    if state["retry_count"] > 3:
        return "end"
    return "generate"
```

---

## 3.2 Pydantic for Validation

Use Pydantic to validate tool inputs and structure LLM output. This is one of the most important skills in agent development.

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class ToolInput(BaseModel):
    location: str = Field(description="City name, e.g. 'Paris'")
    units: Literal["celsius", "fahrenheit"] = "celsius"

    @field_validator("location")
    @classmethod
    def location_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Location cannot be empty")
        return v.strip()

class AgentResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)  # must be between 0 and 1
    sources: List[str] = []
    reasoning: Optional[str] = None

# Pydantic raises a clear error if validation fails
try:
    bad = ToolInput(location="", units="celsius")
except ValueError as e:
    print(e)  # "Location cannot be empty"

good = AgentResponse(answer="Paris", confidence=0.95)
print(good.model_dump())
```

---

## 3.3 Async Python

Most production agent code is async because LLM calls and tool calls are I/O-bound. Async lets you run them in parallel.

```python
import asyncio

async def fetch_weather(city: str) -> str:
    """Simulates an async API call."""
    await asyncio.sleep(0.1)  # fake network delay
    return f"{city}: 22°C"

async def run_parallel_tools(cities: list[str]) -> list[str]:
    """Fetch weather for all cities at the same time."""
    tasks = [fetch_weather(city) for city in cities]
    results = await asyncio.gather(*tasks)
    return list(results)

# Run parallel requests
results = asyncio.run(run_parallel_tools(["Paris", "Tokyo", "NYC", "London"]))
# All 4 fetched simultaneously instead of sequentially
print(results)
```

### Async Agent Loop

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def agent_loop(goal: str) -> str:
    messages = [{"role": "user", "content": goal}]

    for _ in range(10):  # max iterations — always set a limit
        response = await llm.ainvoke(messages)

        if not getattr(response, "tool_calls", None):
            return response.content  # done

        # Execute tool calls in parallel
        tool_tasks = [
            execute_tool(tc.name, tc.args)
            for tc in response.tool_calls
        ]
        results = await asyncio.gather(*tool_tasks)
        messages.extend(format_tool_results(results))

    return "Max iterations reached"
```

---

## 3.4 Decorators for Tools

Decorators add cross-cutting concerns (logging, retries, timing) to tool functions without cluttering the tool logic.

```python
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def tool_logger(func):
    """Log tool calls and execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"Tool called: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Tool success: {func.__name__} ({(time.time()-start)*1000:.0f}ms)")
            return result
        except Exception as e:
            logger.error(f"Tool error: {func.__name__} — {e}")
            raise
    return wrapper

@tool_logger
def search_database(query: str, limit: int = 5) -> list:
    """Search the database."""
    return db.search(query, limit=limit)
```

---

## 3.5 Context Managers for Resource Safety

Always clean up sessions and connections, even if the agent crashes.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_session(session_id: str):
    """Guarantee cleanup even if agent raises an exception."""
    session = await create_session(session_id)
    try:
        yield session
    except Exception as e:
        await session.record_error(e)
        raise
    finally:
        await session.close()  # always runs

# Usage
async with managed_session("user-123") as session:
    result = await run_agent(session, query="...")
```

---

## 3.6 Environment Variables and Configuration

Never hardcode API keys. Use `pydantic-settings` for type-safe config from environment variables.

```python
# pip install pydantic-settings
from pydantic_settings import BaseSettings
from functools import lru_cache

class AgentConfig(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str = ""
    model_name: str = "gpt-4o"
    max_iterations: int = 10
    temperature: float = 0.0
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

@lru_cache
def get_config() -> AgentConfig:
    """Load config once, reuse everywhere."""
    return AgentConfig()
```

```bash
# .env file (never commit this)
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4o-mini
MAX_ITERATIONS=15
LANGSMITH_TRACING=true
```

---

## 3.7 Dataclasses for Agent Messages

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentMessage:
    role: str           # "user", "assistant", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_call_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}
```

---

## Python Checklist for Agent Development

```
[ ] Use TypedDict for all LangGraph state definitions
[ ] Use Pydantic BaseModel for tool input/output validation
[ ] Use async/await for LLM calls and tool calls in production
[ ] Load API keys from .env with pydantic-settings
[ ] Always set max_iterations in agent loops
[ ] Use @wraps(func) in decorators to preserve function metadata
[ ] Use asyncio.gather() for parallel tool execution
```

---

## Summary

- `TypedDict` + `Annotated[list, operator.add]` = LangGraph state
- `Pydantic` = validation, structured output, config
- `async/await` + `asyncio.gather()` = parallel tool execution
- `@wraps` decorator = add logging/timing without modifying tools
- `pydantic-settings` + `.env` = safe configuration management

## Exercises

1. Define an `AgentState` TypedDict with 5 fields, one using `Annotated[list, operator.add]`.
2. Create a `SearchInput` Pydantic model with a `field_validator` that blocks empty strings.
3. Write an async function that calls 3 fake "APIs" in parallel and times the total duration.
4. Create an `AgentConfig` with `pydantic-settings` that reads from a `.env` file.

---

> **Next**: [Chapter 04 — LLM Core Concepts](Chapter-04-LLM-Core-Concepts.md)
