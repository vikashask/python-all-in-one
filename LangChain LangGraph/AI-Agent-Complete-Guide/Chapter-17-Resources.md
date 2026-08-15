# Chapter 17 — Resources

> **Previous**: [Chapter 16 — 90-Day Plan](Chapter-16-90-Day-Plan.md)

---

## 17.1 Official Documentation

| Tool / Framework | URL |
|---|---|
| LangChain | https://python.langchain.com/docs |
| LangGraph | https://langchain-ai.github.io/langgraph |
| LangSmith | https://docs.smith.langchain.com |
| OpenAI API | https://platform.openai.com/docs |
| OpenAI Agents SDK | https://openai.github.io/openai-agents-python |
| Pydantic v2 | https://docs.pydantic.dev/latest |
| FastAPI | https://fastapi.tiangolo.com |
| CrewAI | https://docs.crewai.com |
| AutoGen | https://microsoft.github.io/autogen |
| LlamaIndex | https://docs.llamaindex.ai |
| PydanticAI | https://ai.pydantic.dev |
| SmolAgents | https://huggingface.co/docs/smolagents |
| MCP Spec | https://spec.modelcontextprotocol.io |
| Chroma | https://docs.trychroma.com |
| Pinecone | https://docs.pinecone.io |

---

## 17.2 Essential Open-Source Repositories

| Repo | What it teaches |
|---|---|
| `langchain-ai/langchain` | Core LCEL, tools, chains |
| `langchain-ai/langgraph` | StateGraph, patterns, examples |
| `langchain-ai/langchain-academy` | Official LangGraph course notebooks |
| `openai/openai-agents-python` | Agents SDK examples |
| `microsoft/autogen` | Multi-agent conversation patterns |
| `crewAIInc/crewAI` | Role-based multi-agent systems |
| `modelcontextprotocol/servers` | Pre-built MCP server implementations |
| `huggingface/smolagents` | Minimal code-first agents |
| `Cinnamon/kotaemon` | Production RAG system reference |
| `run-llama/rags` | LlamaIndex RAG patterns |

---

## 17.3 Research Papers

Reading papers makes you understand *why* patterns work, not just *how*:

| Paper | Year | Key Idea |
|---|---|---|
| **ReAct: Synergizing Reasoning and Acting** | 2022 | Foundation of the ReAct agent loop |
| **Chain-of-Thought Prompting** (Wei et al.) | 2022 | Why step-by-step reasoning helps |
| **Reflexion** (Shinn et al.) | 2023 | Self-reflection to improve agent performance |
| **Self-RAG** | 2023 | Selective retrieval with self-critique |
| **Toolformer** | 2023 | LLMs learning to use tools self-supervised |
| **HuggingGPT** | 2023 | Using LLMs to orchestrate other models |
| **AutoGen** (Wu et al.) | 2023 | Multi-agent conversation framework |
| **LangGraph** (Chase et al.) | 2024 | Stateful agent workflows as graphs |

Search for any of these on https://arxiv.org

---

## 17.4 Blogs Worth Reading Regularly

| Blog | URL | Focus |
|---|---|---|
| LangChain Blog | https://blog.langchain.dev | New features, patterns |
| Lilian Weng (OpenAI) | https://lilianweng.github.io | Deep dives on agent architecture |
| Eugene Yan | https://eugeneyan.com | Applied ML + LLM systems |
| Simon Willison | https://simonwillison.net | LLM safety, practical experiments |
| Hamel Husain | https://hamel.dev | LLM evaluation, fine-tuning |
| Cameron Wolfe | https://cameronrwolfe.substack.com | Paper reviews, deep dives |

---

## 17.5 YouTube Channels

| Channel | Focus |
|---|---|
| LangChain Official | Tutorials for LangChain and LangGraph |
| AI Makerspace | Hands-on RAG and agent projects |
| Sam Witteveen | Practical LLM engineering |
| Andrej Karpathy | How LLMs work at the model level |
| DeepLearning.AI | Short courses including LLM-specific tracks |

---

## 17.6 Short Courses

| Course | Platform | What you learn |
|---|---|---|
| LangChain for LLM Apps | DeepLearning.AI | LangChain fundamentals |
| LangGraph: Build Agents | DeepLearning.AI | LangGraph from scratch |
| Building Agentic RAG | DeepLearning.AI | Advanced RAG with agents |
| AI Agents in LangGraph | DeepLearning.AI | Multi-agent + human-in-the-loop |
| Function Calling | DeepLearning.AI | Tool calling deep dive |

All are free at https://www.deeplearning.ai/short-courses

---

## 17.7 Communities

| Community | Where |
|---|---|
| LangChain Discord | https://discord.gg/langchain |
| Hugging Face Discord | https://huggingface.co/join/discord |
| OpenAI Developer Forum | https://community.openai.com |
| r/LocalLLaMA | Reddit — self-hosted models |
| r/MachineLearning | Reddit — research discussions |

---

## 17.8 Interview Preparation Checklist

### Conceptual Questions

- [ ] What is an AI agent? How is it different from a simple LLM call?
- [ ] Explain the ReAct loop. What problem does it solve?
- [ ] What is RAG? When would you use it vs fine-tuning?
- [ ] What are the 4 types of agent memory?
- [ ] What is LangGraph? Why use it instead of plain LangChain?
- [ ] What is MCP and why does it matter?
- [ ] Explain the Reflexion pattern. How does it differ from Reflection?
- [ ] What is Human-in-the-Loop? When is it necessary?
- [ ] How do you prevent prompt injection in a production agent?

### System Design Questions

- [ ] Design a customer support agent for a SaaS product.
- [ ] Design a RAG pipeline for 10M documents.
- [ ] How would you build a multi-agent research system with 5 agents?
- [ ] How do you scale an agent API to 1000 concurrent users?
- [ ] How would you evaluate whether an agent is answering correctly?
- [ ] Design a coding assistant that can run and test its own code.

### Coding Questions

- [ ] Build a LangGraph agent with a conditional retry loop.
- [ ] Write a tool with Pydantic input validation.
- [ ] Implement a sliding window memory with token limit.
- [ ] Write LLM-as-judge evaluation for a RAG system.
- [ ] Add retries with exponential backoff to an async LLM call.

### Production Questions

- [ ] How do you monitor an agent in production?
- [ ] How do you control LLM costs at scale?
- [ ] What security vulnerabilities are unique to LLM applications?
- [ ] How do you A/B test two versions of a system prompt?
- [ ] How do you handle failures when an external API (e.g., OpenAI) is down?

---

## 17.9 Key Python Packages Reference

```bash
# Core
pip install langchain langchain-openai langchain-community
pip install langgraph
pip install openai
pip install pydantic pydantic-settings
pip install python-dotenv

# Agents SDK
pip install openai-agents

# RAG
pip install langchain-chroma chromadb
pip install langchain-pinecone pinecone-client
pip install faiss-cpu
pip install tiktoken

# Tools
pip install tavily-python
pip install langchain-community[all]

# MCP
pip install mcp langchain-mcp-adapters

# Production
pip install fastapi uvicorn slowapi
pip install tenacity
pip install langsmith

# Testing
pip install pytest pytest-asyncio pytest-mock
pip install pyyaml

# Frameworks
pip install crewai
pip install pyautogen
pip install llama-index
pip install pydantic-ai
pip install smolagents
```

---

## Final Advice

1. **Build something real every week.** Reading without building doesn't stick.
2. **Read the source code.** When LangGraph does something unexpected, look at the code.
3. **Measure everything.** Don't claim your agent "works better" — prove it with eval scores.
4. **Start simple.** A 2-node LangGraph beats an over-engineered 15-node graph every time.
5. **Security is not optional.** Every tool that touches files, URLs, or code needs validation.
6. **Document your prompts.** Treat system prompts like code — version them, test them.

---

> You've reached the end of the guide. Return to the [README](README.md) for navigation.
