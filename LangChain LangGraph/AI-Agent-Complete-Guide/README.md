# AI Agent Development — Complete Guide

> **A complete beginner-to-production roadmap for building AI agents with Python.**
> Each chapter is self-contained. Read sequentially or jump to any module.

---

## How to Use This Guide

| Level | Chapters | Time |
|---|---|---|
| Beginner | 01 → 05 | ~2 weeks |
| Intermediate | 06 → 09 | ~2 weeks |
| Advanced | 10 → 13 | ~2 weeks |
| Production | 14 → 17 | ~2 weeks |

---

## Table of Contents

### Foundations
| File | Topic |
|---|---|
| [Chapter 01](Chapter-01-What-Is-An-AI-Agent.md) | What Is an AI Agent? |
| [Chapter 02](Chapter-02-Learning-Roadmap.md) | Learning Roadmap |
| [Chapter 03](Chapter-03-Python-Fundamentals.md) | Python Fundamentals for Agents |
| [Chapter 04](Chapter-04-LLM-Core-Concepts.md) | LLM Core Concepts |
| [Chapter 05](Chapter-05-Prompting-Structured-Output.md) | Prompting, Structured Output, Tool Calling |

### Core Agent Skills
| File | Topic |
|---|---|
| [Chapter 06](Chapter-06-Memory-Context-State.md) | Memory, Context, and State Management |
| [Chapter 07](Chapter-07-Agent-Architecture-Patterns.md) | Agent Architecture Patterns |

### Frameworks
| File | Topic |
|---|---|
| [Framework Overview](frameworks/00-Framework-Overview.md) | Comparison Table + When to Use |
| [LangChain](frameworks/01-LangChain.md) | LCEL, Chains, Tools, RAG |
| [LangGraph](frameworks/02-LangGraph.md) | State Machines, Graphs, Checkpointing |
| [OpenAI Agents SDK](frameworks/03-OpenAI-Agents-SDK.md) | Handoffs, Tracing |
| [CrewAI](frameworks/04-CrewAI.md) | Role-based Multi-Agents |
| [AutoGen](frameworks/05-AutoGen.md) | Conversation Agents |
| [LlamaIndex](frameworks/06-LlamaIndex.md) | Data-Centric RAG |
| [PydanticAI](frameworks/07-PydanticAI.md) | Type-Safe Agents |
| [SmolAgents](frameworks/08-SmolAgents.md) | Minimal Code Agents |

### Advanced Topics
| File | Topic |
|---|---|
| [Chapter 09](Chapter-09-RAG.md) | Retrieval Augmented Generation |
| [Chapter 10](Chapter-10-Multi-Agent-Systems.md) | Multi-Agent Systems |
| [Chapter 11](Chapter-11-MCP.md) | Model Context Protocol |

### Production
| File | Topic |
|---|---|
| [Chapter 12](Chapter-12-Production-Engineering.md) | Production Engineering |
| [Chapter 13](Chapter-13-Testing-Evaluation.md) | Testing and Evaluation |
| [Chapter 14](Chapter-14-Real-World-Projects.md) | Real-World Projects |
| [Chapter 15](Chapter-15-Project-Architecture.md) | Project Architecture & Standards |

### Planning
| File | Topic |
|---|---|
| [Chapter 16](Chapter-16-90-Day-Plan.md) | 90-Day Learning Plan |
| [Chapter 17](Chapter-17-Resources.md) | Resources, Papers, Communities |

---

## Prerequisites

- Python 3.10+ installed
- Basic Python knowledge (functions, classes, loops)
- `pip install langchain langchain-openai langgraph openai pydantic`
- OpenAI API key (or use Ollama for local models)

## Quick Start

```bash
# Install core dependencies
pip install langchain langchain-openai langgraph openai pydantic tenacity

# Set environment variable
export OPENAI_API_KEY="your-key-here"

# Start from Chapter 01
```

---

*Guide version 1.0 | Created 2026-07-30*
