# Framework 07 — PydanticAI

> **Previous**: [LlamaIndex](06-LlamaIndex.md) | **Next**: [SmolAgents](08-SmolAgents.md)

---

## What Is PydanticAI?

PydanticAI is a **type-safe, model-agnostic** agent framework from the Pydantic team. Every input and output is validated. It works with OpenAI, Anthropic, Gemini, and local models.

**Best for**: Agents that produce structured, validated data. Data extraction, classification, analysis.

---

## Installation

```bash
pip install pydantic-ai
```

---

## Core Concepts

| Concept | Description |
|---|---|
| `Agent(model, result_type=Model)` | Agent that returns a validated Pydantic model |
| `@agent.tool` | Register a tool on the agent |
| `RunContext` | Injected context (dependencies, user data) |
| `result.data` | The validated Pydantic output |
| Model-agnostic | Works with OpenAI, Anthropic, Gemini, Groq |

---

## Basic Structured Agent

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from typing import List, Optional

class ProductAnalysis(BaseModel):
    product_name: str
    category: str
    strengths: List[str] = Field(min_length=1, max_length=5)
    weaknesses: List[str] = Field(max_length=5)
    target_audience: str
    price_range: str
    recommendation: str
    confidence_score: float = Field(ge=0.0, le=1.0)

model = OpenAIModel("gpt-4o")

analyst = Agent(
    model,
    result_type=ProductAnalysis,
    system_prompt="""You are a product analyst.
    Analyze products accurately and objectively.
    Base confidence score on information availability."""
)

import asyncio

async def analyze(product_description: str) -> ProductAnalysis:
    result = await analyst.run(
        f"Analyze this product:\n{product_description}"
    )
    return result.data   # already validated ProductAnalysis object

product = asyncio.run(analyze(
    "The iPhone 16 Pro is Apple's flagship smartphone with a 6.3-inch display, "
    "A18 Pro chip, 48MP camera system, and titanium build. Starts at $999."
))

print(f"Product: {product.product_name}")
print(f"Strengths: {product.strengths}")
print(f"Confidence: {product.confidence_score:.0%}")
```

---

## Tools with Type Safety

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List

# Define dependencies that are injected into tools
@dataclass
class SearchDeps:
    api_key: str
    max_results: int = 5

class ResearchSummary(BaseModel):
    topic: str
    key_points: List[str]
    confidence: float
    sources: List[str]

model = OpenAIModel("gpt-4o")

researcher = Agent(
    model,
    result_type=ResearchSummary,
    deps_type=SearchDeps,   # type of injected dependencies
    system_prompt="You are a thorough research assistant. Always cite sources."
)

@researcher.tool
async def search_web(ctx: RunContext[SearchDeps], query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query
    """
    # Access injected dependencies via ctx.deps
    results = await call_search_api(
        query,
        api_key=ctx.deps.api_key,
        max_results=ctx.deps.max_results
    )
    return str(results)

@researcher.tool
async def get_citation(ctx: RunContext[SearchDeps], url: str) -> str:
    """Fetch and format a citation for a URL.

    Args:
        url: The URL to cite
    """
    metadata = await fetch_metadata(url)
    return f"{metadata.get('title', url)} — {url}"

async def run_research(topic: str) -> ResearchSummary:
    deps = SearchDeps(api_key="your-search-api-key", max_results=10)
    result = await researcher.run(f"Research: {topic}", deps=deps)
    return result.data
```

---

## Multi-Model Support

PydanticAI works with multiple providers — swap the model without changing agent logic:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai    import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini    import GeminiModel

# Same agent, different models
openai_agent    = Agent(OpenAIModel("gpt-4o"),                  result_type=MyModel)
anthropic_agent = Agent(AnthropicModel("claude-3-5-sonnet-20241022"), result_type=MyModel)
gemini_agent    = Agent(GeminiModel("gemini-2.0-flash"),         result_type=MyModel)

# A/B test or use cheapest model that meets quality bar
async def run_with_fallback(prompt: str) -> MyModel:
    try:
        result = await openai_agent.run(prompt)
        return result.data
    except Exception:
        result = await anthropic_agent.run(prompt)
        return result.data
```

---

## Complete Example: Document Data Extractor

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

class ContractTerms(BaseModel):
    parties: List[str] = Field(description="All parties to the contract")
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    payment_terms: Optional[str] = None
    termination_notice_days: Optional[int] = None
    governing_law: Optional[str] = None
    key_obligations: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Any unusual or risky clauses"
    )
    confidence: float = Field(ge=0, le=1)

contract_extractor = Agent(
    OpenAIModel("gpt-4o"),
    result_type=ContractTerms,
    system_prompt="""You are a legal contract analyst.
    Extract key terms accurately.
    Flag any unusual clauses as risk flags.
    Set confidence based on how much information was explicitly stated vs. inferred."""
)

async def extract_contract_terms(contract_text: str) -> ContractTerms:
    result = await contract_extractor.run(
        f"Extract all key terms from this contract:\n\n{contract_text}"
    )
    return result.data

# Test
sample_contract = """
SERVICE AGREEMENT
Effective January 1, 2025 between TechCorp Inc. and ClientCo Ltd.

Services: Software development and consulting services.
Payment: $10,000/month, payable within 30 days of invoice.
Term: 12 months, renewing automatically unless 60 days written notice.
Governing Law: State of California.
Limitation of Liability: Total liability capped at 3 months of fees.
"""

terms = asyncio.run(extract_contract_terms(sample_contract))
print(f"Parties: {terms.parties}")
print(f"Payment: {terms.payment_terms}")
print(f"Termination notice: {terms.termination_notice_days} days")
print(f"Risk flags: {terms.risk_flags}")
print(f"Confidence: {terms.confidence:.0%}")
```

---

## Advantages

- Every output is a validated Pydantic object — no string parsing
- Type safety throughout — IDE autocomplete works
- Model-agnostic — swap providers with one line
- Dependency injection is clean and testable

## Limitations

- Newer framework — smaller ecosystem than LangChain
- Less built-in tooling for RAG workflows
- Streaming less mature than LangChain

---

> **Next**: [08 — SmolAgents](08-SmolAgents.md)
