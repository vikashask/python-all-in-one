# LangChain + LangGraph Study Guide

This file teaches you the core ideas in a simple, practical order.
Use it like a revision sheet while watching your Udemy course.

---

## 1) Big Picture

### What is LangChain?

LangChain helps you build LLM apps using reusable blocks:

- Prompt templates
- Chat models
- Tools
- Output parsers
- Retrieval pipelines

### What is LangGraph?

LangGraph is for workflow orchestration of agent systems:

- Graph nodes = steps
- Edges = transitions
- Shared state = memory between steps
- Loops/branching/retries = reliable agent behavior

### Simple rule

- Use **LangChain** for linear pipelines.
- Use **LangGraph** for multi-step, stateful, branching agents.

---

## 2) Learning Flow

```mermaid
flowchart LR
    A[Prompt + Model Basics] --> B[LangChain Chains]
    B --> C[Tool Calling Agents]
    C --> D[RAG Basics]
    D --> E[LangGraph State Machines]
    E --> F[Reflection and Reflexion Agents]
    F --> G[Agentic RAG]
    G --> H[MCP + Production + Security]
```

---

## 3) LangChain Core Architecture

```mermaid
flowchart TD
    U[User Input] --> P[Prompt Template]
    P --> M[Chat Model]
    M --> O[Output Parser]
    O --> R[Final Response]
```

### Minimal mental model

1. Format prompt.
2. Call model.
3. Parse output.
4. Return answer.

---

## 4) Agent Loop (ReAct idea)

ReAct means: **Reason -> Act -> Observe -> Repeat**.

```mermaid
flowchart TD
    Q[User Query] --> LLM[LLM Decides]
    LLM -->|Needs tool| T[Call Tool]
    T --> OBS[Tool Result / Observation]
    OBS --> LLM
    LLM -->|Enough info| ANS[Final Answer]
```

### Why this matters

Agents become useful when the model can:

- Search web/docs
- Query databases
- Run custom functions
- Decide when to stop

---

## 5) Function Calling Layers

```mermaid
flowchart LR
    A[Manual JSON Schema] --> B[Model Function Call]
    B --> C[Execute Tool in Python]
    C --> D[Return Observation to Model]
    D --> E[Final Structured Response]
```

### Three styles

- Raw SDK function calling (more control)
- LangChain tools abstraction (faster development)
- ReAct prompt only (manual, educational)

---

## 6) RAG Basics

RAG = Retrieval Augmented Generation.

```mermaid
flowchart TD
    D1[Raw Documents] --> D2[Chunking]
    D2 --> D3[Embeddings]
    D3 --> D4[Vector DB]

    Q[User Question] --> Q2[Embed Query]
    Q2 --> RET[Retrieve Top-K Chunks]
    RET --> CTX[Build Context]
    CTX --> LLM[Generate Answer]
    LLM --> OUT[Grounded Response]
```

### Important quality controls

- Better chunk size/overlap
- Metadata filters
- Reranking
- Relevance grading

---

## 7) LangGraph Agent Architecture

```mermaid
stateDiagram-v2
    [*] --> Router
    Router --> Retrieve: needs knowledge
    Router --> WebSearch: needs fresh info
    Router --> Answer: enough context

    Retrieve --> GradeDocs
    GradeDocs --> RewriteQuery: low relevance
    GradeDocs --> Answer: good relevance

    RewriteQuery --> Retrieve
    WebSearch --> Answer
    Answer --> [*]
```

### Why LangGraph here

This graph has branching and loops. A plain linear chain is not enough.

---

## 8) Reflection vs Reflexion

### Reflection Agent

- Generates answer
- Critiques answer
- Revises answer

### Reflexion Agent

- Maintains memory of failures/success
- Improves next attempts using feedback history
- Better for iterative tasks

```mermaid
flowchart TD
    A[Draft Answer] --> B[Critic Node]
    B --> C[Revision Node]
    C --> D{Good Enough?}
    D -->|No| B
    D -->|Yes| E[Final Output]
```

---

## 9) MCP in One View

MCP (Model Context Protocol) standardizes how AI clients use tools/resources.

```mermaid
flowchart LR
    Client[AI Client] --> MCP[MCP Protocol Layer]
    MCP --> S1[Server A: Docs]
    MCP --> S2[Server B: Database]
    MCP --> S3[Server C: Internal APIs]
```

### Why useful

- One standard interface
- Reusable tool servers
- Easier integration with multiple clients

---

## 10) Production Checklist

- Add tracing (LangSmith)
- Add guardrails (input/output validation)
- Add retries/timeouts/fallback model
- Add security (authz, rate limits, SSRF controls, prompt-injection defenses)
- Add evaluation datasets and regression tests

---

## 11) 14-Day Study Plan (from your course outline)

1. Day 1-2: LangChain fundamentals and simple chain.
2. Day 3-4: ReAct agent loop and tool calling.
3. Day 5-6: Raw function calling and structured output.
4. Day 7-8: RAG ingestion + retrieval.
5. Day 9-10: LangGraph basics and graph nodes/edges.
6. Day 11: Reflection/Reflexion patterns.
7. Day 12: Agentic RAG flows.
8. Day 13: MCP basics and pre-built server usage.
9. Day 14: Production concerns + security review.

---

## 12) Quick Interview Answers

### Difference between LangChain and LangGraph?

LangChain gives modular LLM components; LangGraph orchestrates complex, stateful, branching agent workflows.

### When should I use LangGraph?

Use it when you need loops, retries, branching, or multi-agent coordination.

### Why not only prompts?

Prompt-only systems are brittle. Tool calling + state + evaluation improves reliability.

### What is Agentic RAG?

RAG where an agent decides retrieval strategy, verifies relevance, and can rewrite/query/search iteratively.

---

## 13) Next Hands-On Steps

1. Build a tiny summarizer chain.
2. Convert it into a tool-calling ReAct agent.
3. Add a vector store and retrieval.
4. Move control flow into LangGraph with a relevance-check loop.
5. Add LangSmith tracing and basic evaluation.

If you want, I can create the next file: a **step-by-step coding lab** with folder structure and starter code (LangChain + LangGraph + Streamlit).
