# Chapter 16 — 90-Day Learning Plan

> **Previous**: [Chapter 15 — Project Architecture](Chapter-15-Project-Architecture.md) | **Next**: [Chapter 17 — Resources](Chapter-17-Resources.md)

---

## Month 1 (Days 1–30): Foundations

### Week 1: Python + LLM Basics (Days 1–7)

| Day | Topic | Task |
|-----|-------|------|
| 1 | Environment setup | Install Python 3.12, create venv, install openai + langchain |
| 2 | Python async | Rewrite 3 sync functions as async with `asyncio.gather()` |
| 3 | Pydantic v2 | Build 3 data models with validators |
| 4 | OpenAI API basics | Send prompts, count tokens, stream responses |
| 5 | Prompt engineering | Write zero-shot, few-shot, and CoT prompts |
| 6 | Structured output | Use `.with_structured_output()` to extract typed data |
| 7 | Review + mini-project | Build a structured data extractor from unstructured text |

**Week 1 Mini-Project**: Invoice extractor — parse raw invoice text into a Pydantic model with `invoice_number`, `date`, `total`, `line_items`.

---

### Week 2: LangChain Fundamentals (Days 8–14)

| Day | Topic | Task |
|-----|-------|------|
| 8  | LCEL basics | Build a `prompt | llm | parser` chain |
| 9  | ChatPromptTemplate | Create a chain with system + user templates |
| 10 | Output parsers | StrOutputParser, PydanticOutputParser, JsonOutputParser |
| 11 | RunnableParallel | Run two chains in parallel with `RunnableParallel` |
| 12 | @tool decorator | Create 3 custom tools with validated inputs |
| 13 | ReAct agent | Build a LangChain ReAct agent with 2 tools |
| 14 | Review + mini-project | Q&A bot that uses search and calculator |

**Week 2 Mini-Project**: Research bot — takes a question, searches the web, summarizes the answer.

---

### Week 3: RAG Pipeline (Days 15–21)

| Day | Topic | Task |
|-----|-------|------|
| 15 | Document loaders | Load PDFs and TXT files |
| 16 | Text splitting | Experiment with chunk sizes on a real document |
| 17 | Embeddings + Chroma | Index a document, run similarity search |
| 18 | Basic RAG chain | Retriever → prompt → LLM → answer |
| 19 | Citations | Return source filenames alongside answers |
| 20 | Retrieval strategies | Compare similarity, MMR, threshold filtering |
| 21 | Review + mini-project | FAQ bot for a PDF document |

**Week 3 Mini-Project**: Personal docs Q&A — index a folder of Markdown notes and answer questions with citations.

---

### Week 4: LangGraph Basics (Days 22–30)

| Day | Topic | Task |
|-----|-------|------|
| 22 | StateGraph concepts | Draw a 3-node graph on paper before coding it |
| 23 | TypedDict state | Build a state with 4 fields and update it through nodes |
| 24 | Conditional edges | Build a router that chooses between 2 paths |
| 25 | Loops | Add a retry loop with max_iterations guard |
| 26 | MemorySaver | Add multi-turn memory to an existing agent |
| 27 | Streaming | Stream node outputs with `.astream_events()` |
| 28 | Human-in-the-loop | Add an `interrupt()` approval step |
| 29 | Complex graph | Build a 5-node graph combining all the above |
| 30 | Month 1 project | Build the Research Assistant from Chapter 14 |

---

## Month 2 (Days 31–60): Intermediate

### Week 5: Agent Patterns (Days 31–37)

| Day | Topic | Task |
|-----|-------|------|
| 31 | ReAct deep dive | Trace a ReAct execution step-by-step in logs |
| 32 | Plan-and-Execute | Build planner + executor nodes |
| 33 | Reflection pattern | generate → critique → revise loop |
| 34 | Reflexion | Accumulate feedback across attempts |
| 35 | Corrective RAG | Grade retrieval quality, fall back to web search |
| 36 | Event-driven agents | Use a message queue to trigger agent runs |
| 37 | Review | Compare output quality across all patterns |

**Week 5 Mini-Project**: Coding assistant with reflection — generates code, runs it, revises.

---

### Week 6: Frameworks (Days 38–44)

| Day | Topic | Task |
|-----|-------|------|
| 38 | OpenAI Agents SDK | Build a single agent with function tools |
| 39 | Handoffs | Multi-agent handoff between specialists |
| 40 | CrewAI | 3-agent crew: researcher → writer → editor |
| 41 | AutoGen | Two-agent conversation: assistant + user-proxy |
| 42 | PydanticAI | Type-safe agent with dependency injection |
| 43 | LlamaIndex | ReAct agent over multiple data sources |
| 44 | Framework comparison | Build the same task in LangGraph + CrewAI, compare |

---

### Week 7: Multi-Agent Systems (Days 45–51)

| Day | Topic | Task |
|-----|-------|------|
| 45 | Supervisor pattern | Build a supervisor that routes to 3 specialists |
| 46 | Parallel agents | Run 3 agents in parallel with `asyncio.gather()` |
| 47 | A2A communication | Build a hub that passes messages between agents |
| 48 | Debate pattern | Two agents argue, a judge decides |
| 49 | Hierarchical teams | Supervisor → sub-supervisors → workers |
| 50 | State sharing | Pass structured state between agents |
| 51 | Review + project | Customer service system (routing + specialists) |

---

### Week 8: MCP + Tools (Days 52–60)

| Day | Topic | Task |
|-----|-------|------|
| 52 | MCP concepts | Read the MCP spec, understand primitives |
| 53 | Build MCP server | Resources + 2 tools + stdio transport |
| 54 | Connect to LangChain | Use `load_mcp_tools` to get tools from your server |
| 55 | Pre-built servers | Use filesystem MCP server to query a local directory |
| 56 | Tool security | Add SSRF + path traversal validation to all file tools |
| 57 | Tool testing | Write unit tests for every tool |
| 58 | ToolRegistry | Refactor your tools to use the ToolRegistry pattern |
| 59 | Tool documentation | Write clear docstrings — measure how they affect agent behavior |
| 60 | Month 2 project | Document Intelligence system from Chapter 14 |

---

## Month 3 (Days 61–90): Advanced + Production

### Week 9: Observability + Testing (Days 61–67)

| Day | Topic | Task |
|-----|-------|------|
| 61 | LangSmith setup | Enable tracing, view your first trace |
| 62 | Custom spans | Add `@traceable` to your key functions |
| 63 | Structured logging | Replace all `print()` with structured JSON logging |
| 64 | Unit tests | Achieve 80%+ coverage on all tool functions |
| 65 | LLM-as-judge | Build an eval suite with 10+ Q&A pairs |
| 66 | Prompt versioning | Create a PromptRegistry with 2 versions of your main prompt |
| 67 | A/B testing | Run the eval suite on both prompt versions |

---

### Week 10: Production Engineering (Days 68–74)

| Day | Topic | Task |
|-----|-------|------|
| 68 | Retries | Add `@retry` to all LLM calls |
| 69 | Rate limiting | Add a RateLimiter to your agent |
| 70 | Cost tracking | Add a TokenBudget to log per-request cost |
| 71 | FastAPI | Wrap your agent in a `/query` endpoint |
| 72 | Security | Add SafeUserQuery validation to the endpoint |
| 73 | Response caching | Add SQLiteCache, measure response time difference |
| 74 | Load testing | Run locust/k6 against your API |

---

### Week 11: Deployment (Days 75–81)

| Day | Topic | Task |
|-----|-------|------|
| 75 | Docker | Write a Dockerfile and run your API in a container |
| 76 | docker-compose | Add Chroma + your API as services |
| 77 | Health checks | Add `/health` endpoint, configure Docker HEALTHCHECK |
| 78 | Environment management | Move all secrets to `.env`, use pydantic-settings |
| 79 | Project architecture | Refactor project into the Chapter 15 folder structure |
| 80 | CI/CD basics | Add a GitHub Actions workflow that runs tests |
| 81 | Final review | Run the full eval suite, check all prod checklist items |

---

### Week 12: Capstone (Days 82–90)

Build a complete, production-quality AI agent system from scratch:

**Capstone Project**: Personal AI Assistant

Requirements:
- [ ] Accepts queries via FastAPI endpoint
- [ ] Uses RAG over your personal documents
- [ ] Can search the web for current info
- [ ] Has multi-turn memory (remembers context)
- [ ] Cites sources in every answer
- [ ] Deployed in Docker
- [ ] 80%+ eval suite pass rate
- [ ] Retries + rate limiting + structured logging

| Day | Task |
|-----|------|
| 82 | Scaffold project structure, set up settings |
| 83 | Build and register tools (search, RAG, calculator) |
| 84 | Build the LangGraph agent with memory |
| 85 | Add production engineering (retries, logging, caching) |
| 86 | Build FastAPI API with auth + rate limiting |
| 87 | Write unit tests and integration tests |
| 88 | Build eval suite, run LLM-as-judge |
| 89 | Dockerize and do a load test |
| 90 | Review, document, share |

---

## Summary

| Phase | Focus | Projects |
|---|---|---|
| Month 1 | Python, LangChain, RAG, LangGraph basics | 4 mini-projects |
| Month 2 | Agent patterns, frameworks, multi-agent | 3 mini-projects |
| Month 3 | Production, testing, deployment, capstone | Full production app |

---

> **Next**: [Chapter 17 — Resources](Chapter-17-Resources.md)
