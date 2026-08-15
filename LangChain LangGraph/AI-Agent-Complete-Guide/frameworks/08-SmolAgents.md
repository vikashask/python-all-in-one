# Framework 08 — SmolAgents

> **Previous**: [PydanticAI](07-PydanticAI.md) | **Next**: [Chapter 09 — RAG](../Chapter-09-RAG.md)

---

## What Is SmolAgents?

SmolAgents is a minimal agent framework from Hugging Face. Its key insight: instead of calling tools one at a time, the agent writes **Python code** that calls multiple tools in a single step. This is more efficient for complex tasks.

**Best for**: Data analysis, code generation, tasks requiring multi-tool coordination in a single step.

---

## Installation

```bash
pip install smolagents
```

---

## Two Agent Types

| Type | How it acts | Best for |
|---|---|---|
| `ToolCallingAgent` | Calls tools one at a time (like ReAct) | Standard tool use |
| `CodeAgent` | Writes and executes Python code | Complex multi-tool tasks |

---

## Basic ToolCallingAgent

```python
from smolagents import ToolCallingAgent, tool, HfApiModel, DuckDuckGoSearchTool

model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct")

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: City name
    """
    return f"{city}: 22°C, sunny"

agent = ToolCallingAgent(
    tools=[get_weather, DuckDuckGoSearchTool()],
    model=model
)

result = agent.run("What's the weather in Paris, and what's the latest news there?")
print(result)
```

---

## CodeAgent — Writing Python to Solve Tasks

The `CodeAgent` generates Python code, executes it, observes the output, and iterates.

```python
from smolagents import CodeAgent, tool, HfApiModel

model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct")

@tool
def read_csv(filepath: str) -> str:
    """Read a CSV file and return a statistical summary.

    Args:
        filepath: Path to the CSV file

    Returns:
        Statistical summary as string
    """
    import pandas as pd
    df = pd.read_csv(filepath)
    return df.describe().to_string()

@tool
def run_analysis(code: str) -> str:
    """Execute data analysis Python code and return output.

    Args:
        code: Valid Python code using pandas as pd, numpy as np

    Returns:
        Output of the code execution
    """
    import pandas as pd
    import numpy as np
    import io
    from contextlib import redirect_stdout

    output = io.StringIO()
    namespace = {"pd": pd, "np": np}
    with redirect_stdout(output):
        exec(code, namespace)  # noqa: S102 — tool is sandboxed
    return output.getvalue()

agent = CodeAgent(
    tools=[read_csv, run_analysis],
    model=model,
    max_steps=5
)

# Agent writes a multi-step Python script to answer this
result = agent.run(
    "Read data/sales_2024.csv, find the top 3 products by total revenue, "
    "and calculate what percentage of total revenue they represent."
)
print(result)
```

---

## How CodeAgent Works

Instead of one tool call per step, CodeAgent generates code like this:

```python
# Agent-generated code (step 1)
summary = read_csv("data/sales_2024.csv")
print(summary)

# Agent observes output, then generates step 2
analysis_code = """
df = pd.read_csv('data/sales_2024.csv')
top3 = df.groupby('product')['revenue'].sum().nlargest(3)
total = df['revenue'].sum()
for product, rev in top3.items():
    print(f"{product}: ${rev:,.0f} ({rev/total:.1%})")
"""
result = run_analysis(analysis_code)
print(result)
```

This is more efficient than calling `read_csv` + `groupby` + `filter` + `calculate` as separate tool calls.

---

## Using OpenAI with SmolAgents

```python
from smolagents import CodeAgent, tool, LiteLLMModel

# Use any model via LiteLLM
model = LiteLLMModel(model_id="gpt-4o")

@tool
def search_and_summarize(query: str) -> str:
    """Search the web and return a summary.

    Args:
        query: Search query string
    """
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return "\n".join([r["body"] for r in results])

agent = CodeAgent(
    tools=[search_and_summarize],
    model=model,
    max_steps=8
)

result = agent.run(
    "Research the top 5 Python AI agent frameworks released in 2024-2025. "
    "Compare them in a markdown table."
)
print(result)
```

---

## Complete Example: Automated Report Generator

```python
from smolagents import CodeAgent, tool, LiteLLMModel
import json

model = LiteLLMModel(model_id="gpt-4o")

@tool
def fetch_metrics(metric_name: str, period: str) -> str:
    """Fetch business metrics for a period.

    Args:
        metric_name: 'revenue', 'users', 'churn_rate', 'nps'
        period: 'last_week', 'last_month', 'last_quarter'

    Returns:
        JSON string with metric data
    """
    mock_data = {
        "revenue":    {"last_month": 125000, "last_quarter": 380000},
        "users":      {"last_month": 1250,   "last_quarter": 3800},
        "churn_rate": {"last_month": 0.023,  "last_quarter": 0.019},
        "nps":        {"last_month": 42,     "last_quarter": 39}
    }
    return json.dumps(mock_data.get(metric_name, {}).get(period, {}))

@tool
def format_report(sections: str) -> str:
    """Format report sections into a markdown document.

    Args:
        sections: JSON string with {title: content} pairs

    Returns:
        Formatted markdown report
    """
    data = json.loads(sections)
    lines = ["# Business Report\n"]
    for title, content in data.items():
        lines.append(f"## {title}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)

report_agent = CodeAgent(
    tools=[fetch_metrics, format_report],
    model=model,
    max_steps=6
)

report = report_agent.run(
    "Generate a complete monthly business report. "
    "Include: revenue summary, user growth, churn analysis, and NPS score. "
    "Compare to quarterly trends. Format as a professional markdown report."
)
print(report)
```

---

## Advantages

- Code generation is more efficient than sequential tool calls for complex tasks
- Minimal boilerplate — very fast to get started
- Supports local Hugging Face models (free, private)
- `CodeAgent` naturally handles multi-step reasoning in one generation

## Limitations

- Executing generated code requires careful sandboxing in production
- Less mature than LangChain/LangGraph
- Limited built-in observability

---

> **Next**: [Chapter 09 — RAG](../Chapter-09-RAG.md)
