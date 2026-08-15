# Chapter 15 — Project Architecture

> **Previous**: [Chapter 14 — Real-World Projects](Chapter-14-Real-World-Projects.md) | **Next**: [Chapter 16 — 90-Day Plan](Chapter-16-90-Day-Plan.md)

---

## 15.1 Recommended Folder Structure

```
my-agent-project/
├── agents/               ← Agent definitions and graph compilation
│   ├── __init__.py
│   ├── research.py       ← Research agent (StateGraph + nodes)
│   ├── support.py        ← Customer support agent
│   └── factory.py        ← AgentFactory class
├── tools/                ← All tool functions
│   ├── __init__.py
│   ├── search.py         ← Web search tools
│   ├── code.py           ← Code execution tools
│   ├── file_tools.py     ← File read/write tools
│   └── registry.py       ← ToolRegistry class
├── memory/               ← Memory and state management
│   ├── __init__.py
│   ├── checkpointer.py   ← LangGraph MemorySaver setup
│   └── vector_store.py   ← Long-term vector memory
├── graphs/               ← LangGraph StateGraph definitions
│   ├── __init__.py
│   ├── research_graph.py
│   └── support_graph.py
├── prompts/              ← Prompt templates and versions
│   ├── research_v1.yaml
│   ├── research_v2.yaml
│   └── system_prompts.py
├── api/                  ← FastAPI endpoints
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── query.py
│   │   └── feedback.py
│   └── middleware.py
├── evaluation/           ← Eval datasets and runners
│   ├── datasets/
│   │   └── rag_qa.json
│   └── runner.py
├── tests/                ← All tests
│   ├── unit/
│   │   ├── test_tools.py
│   │   └── test_validators.py
│   └── integration/
│       └── test_agent_e2e.py
├── config/               ← Settings and environment
│   ├── settings.py       ← Pydantic-settings config
│   └── logging.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 15.2 ToolRegistry Pattern

Centralize all tool definitions so agents can discover them by name:

```python
# tools/registry.py
from langchain_core.tools import BaseTool, tool
from typing import Callable

class ToolRegistry:
    """Central registry for all agent tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, func: Callable) -> Callable:
        """Decorator that registers a function as a LangChain tool."""
        t = tool(func)
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        return self._tools[name]

    def get_many(self, *names: str) -> list[BaseTool]:
        return [self.get(n) for n in names]

    def all(self) -> list[BaseTool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

# Singleton
registry = ToolRegistry()

# tools/search.py
from tools.registry import registry

@registry.register
def web_search(query: str) -> str:
    """Search the web for current information."""
    from langchain_community.tools import TavilySearchResults
    return TavilySearchResults(max_results=3).invoke({"query": query})

@registry.register
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "Error: unsafe expression"
    return str(eval(expression))

# Usage in agents
from tools.registry import registry

research_tools = registry.get_many("web_search", "calculator")
support_tools  = registry.get_many("web_search")
```

---

## 15.3 AgentFactory Pattern

Create agents consistently without repeating configuration:

```python
# agents/factory.py
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

@dataclass
class AgentConfig:
    name:          str
    model:         str        = "gpt-4o"
    temperature:   float      = 0.0
    max_iterations: int       = 10
    tools:         list[BaseTool] = field(default_factory=list)
    system_prompt: str        = ""
    with_memory:   bool       = False

class AgentFactory:
    """Creates configured agents consistently."""

    @staticmethod
    def create(config: AgentConfig):
        llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature
        )

        checkpointer = MemorySaver() if config.with_memory else None

        agent = create_react_agent(
            model=llm,
            tools=config.tools,
            state_modifier=config.system_prompt or None,
            checkpointer=checkpointer
        )

        return agent

# Usage
from tools.registry import registry
from agents.factory import AgentFactory, AgentConfig

config = AgentConfig(
    name="research-agent",
    model="gpt-4o",
    tools=registry.get_many("web_search", "calculator"),
    system_prompt="You are a thorough research assistant. Always cite sources.",
    with_memory=True
)

agent = AgentFactory.create(config)
```

---

## 15.4 Configuration with pydantic-settings

```python
# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # LLM
    openai_api_key:      str
    anthropic_api_key:   str = ""
    default_model:       str = "gpt-4o"
    fallback_model:      str = "gpt-4o-mini"

    # LangSmith
    langchain_api_key:   str = ""
    langchain_project:   str = "default"
    langchain_tracing_v2: bool = False

    # Vector DB
    chroma_persist_dir:  str = "./chroma_db"
    embedding_model:     str = "text-embedding-3-small"

    # API
    api_rate_limit:      int = 20        # requests per minute
    max_iterations:      int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

# Usage
settings = get_settings()
llm = ChatOpenAI(
    model=settings.default_model,
    api_key=settings.openai_api_key
)
```

---

## 15.5 Coding Standards Checklist

```
Tools
  [ ] Every tool validates its inputs with Pydantic
  [ ] Every tool has a clear docstring (the LLM reads it)
  [ ] No tool executes user-supplied shell commands
  [ ] File access tools are restricted to an allowed directory

Agents
  [ ] max_iterations is always set
  [ ] Every agent has a system prompt
  [ ] Async agents use ainvoke() / astream()
  [ ] Errors are caught and returned as tool error messages

State (LangGraph)
  [ ] State is a TypedDict — no plain dicts
  [ ] Reducers are explicit where needed (Annotated[list, operator.add])
  [ ] State is serializable (JSON-safe)

Config
  [ ] All secrets in .env, never hardcoded
  [ ] .env.example committed, .env in .gitignore
  [ ] pydantic-settings for config validation

Tests
  [ ] Every tool has at least 2 unit tests
  [ ] Agent routing logic is unit-tested with mocks
  [ ] Eval suite with min 10 Q&A pairs runs in CI
  [ ] Integration tests behind @pytest.mark.integration
```

---

## Summary

- The folder structure separates concerns: agents, tools, memory, graphs, API, evaluation
- ToolRegistry prevents duplicate definitions and makes tool discovery explicit
- AgentFactory ensures all agents are created with the same validated config
- pydantic-settings gives type-safe configuration from `.env`
- The coding standards checklist is your quality gate before shipping

## Exercises

1. Scaffold the full folder structure for one of the Chapter 14 projects.
2. Implement ToolRegistry and register 3 tools. Access them by name.
3. Create an AgentConfig and build an agent through AgentFactory.
4. Write a Settings class for your project with all needed keys.

---

> **Next**: [Chapter 16 — 90-Day Plan](Chapter-16-90-Day-Plan.md)
