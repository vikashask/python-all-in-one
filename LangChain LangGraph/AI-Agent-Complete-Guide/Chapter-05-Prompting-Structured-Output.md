# Chapter 05 — Prompting, Structured Output, and Tool Calling

> **Previous**: [Chapter 04 — LLM Core Concepts](Chapter-04-LLM-Core-Concepts.md) | **Next**: [Chapter 06 — Memory, Context, State](Chapter-06-Memory-Context-State.md)

---

## 5.1 Prompt Engineering for Agents

### Zero-Shot Prompting
No examples. Works for simple tasks:
```python
prompt = "Summarize this article in 3 bullet points: {article}"
```

### Few-Shot Prompting
Show examples to guide format and style:
```python
prompt = """
Classify sentiment. Examples:

Input: "I love this!"      → positive
Input: "Terrible service"  → negative
Input: "It's okay"         → neutral

Now classify: "{text}"
"""
```

### Chain-of-Thought (CoT)
Force step-by-step reasoning before the answer:
```python
prompt = """
Solve this step by step.

Problem: {problem}

Step 1: Identify what is being asked
Step 2: Identify the relevant information
Step 3: Solve
Step 4: Verify

Answer:
"""
```

### ReAct Prompt (for tool-using agents)
```
You have access to these tools:
{tools}

Use this format:
Thought: What do I need to do next?
Action: tool_name
Action Input: {{"param": "value"}}
Observation: [result from tool]
... (repeat as needed)
Thought: I have enough information.
Final Answer: [your final answer]

Begin!
Question: {input}
```

---

## 5.2 LangChain Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Simple template
summarize_prompt = PromptTemplate(
    template="Summarize this in one sentence: {text}",
    input_variables=["text"]
)

# Chat template (use this for chat models)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer using only the provided context."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# Format manually
formatted = rag_prompt.format_messages(
    context="Python was created in 1991.",
    question="When was Python created?"
)
```

---

## 5.3 Structured Output with Pydantic

Getting structured data from LLMs is critical. Use `.with_structured_output()` — the modern, reliable approach.

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

class ResearchFindings(BaseModel):
    topic: str = Field(description="The research topic")
    key_facts: List[str] = Field(description="3–5 key facts found")
    confidence: float = Field(ge=0, le=1, description="Confidence 0–1")
    sources: List[str] = Field(description="URLs or document titles")
    summary: str = Field(description="2–3 sentence summary")
    follow_up_questions: Optional[List[str]] = None

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(ResearchFindings)

# result is a ResearchFindings object — not a raw string
result = structured_llm.invoke("Research the latest advances in quantum computing")

print(result.topic)          # str
print(result.key_facts)      # list[str]
print(result.confidence)     # float
```

### When to Use Structured Output

| Task | Use structured output? |
|---|---|
| Simple Q&A | No |
| Extracting data fields | Yes |
| Routing/classification | Yes |
| Grading/scoring | Yes |
| Tool input validation | Yes (via Pydantic) |
| Writing/summarization | No |

---

## 5.4 Function Calling — How It Works Under the Hood

The model doesn't call Python functions directly. It outputs a structured JSON object describing *what* to call. Your code then executes it.

```
User: "What's the weather in Paris?"

Model output (raw):
{
  "tool_calls": [{
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "get_weather",
      "arguments": "{\"location\": \"Paris\"}"
    }
  }]
}

Your code: parse → execute → get_weather("Paris") → "18°C, cloudy"
Your code: send result back to model
Model: "The weather in Paris is 18°C and cloudy."
```

---

## 5.5 Raw Function Calling (No Framework)

Understanding the raw API helps you debug any framework:

```python
import json
from openai import OpenAI

client = OpenAI()

# Define tool schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

def get_weather(location: str) -> str:
    return f"Weather in {location}: 18°C, cloudy"

def agent_loop(user_query: str, max_iterations: int = 10) -> str:
    messages = [{"role": "user", "content": user_query}]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        # No tool call → agent is done
        if not msg.tool_calls:
            return msg.content

        # Append assistant message
        messages.append(msg.to_dict())

        # Execute each tool call
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = get_weather(**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)
            })

    return "Max iterations reached"

print(agent_loop("What is the weather in Rome?"))
```

---

## 5.6 LangChain Tools (Recommended for Development)

LangChain's `@tool` decorator auto-generates the JSON schema from your function signature and docstring:

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather for a city.

    Args:
        location: City name, e.g. 'Paris'
        units: 'celsius' or 'fahrenheit'

    Returns:
        Weather description string
    """
    return f"Weather in {location}: 22°{units[0].upper()}, sunny"

@tool
def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression.

    Args:
        expression: Python math expression, e.g. '(5 + 3) * 2'

    Returns:
        Numerical result as string
    """
    try:
        allowed = set("0123456789+-*/().** ")
        if not all(c in allowed for c in expression):
            return "Error: only numeric expressions allowed"
        return str(eval(expression))  # noqa: S307 — validated above
    except Exception as e:
        return f"Error: {e}"

# Bind tools to model
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([get_weather, calculate])

response = llm_with_tools.invoke("Weather in Rome? Also calculate 15 * 7")

for tc in response.tool_calls:
    print(f"Tool: {tc['name']}, Args: {tc['args']}")
```

---

## 5.7 Parallel Tool Calls

Modern models can call multiple tools in a single response:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

@tool
def translate(text: str, target_lang: str) -> str:
    """Translate text."""
    return f"[Translated to {target_lang}]: {text}"

llm = ChatOpenAI(model="gpt-4o").bind_tools([search, translate])

response = llm.invoke(
    "Search for 'quantum computing' AND translate 'Hello world' to French"
)

# response.tool_calls will have TWO entries — both run in parallel
for tc in response.tool_calls:
    print(tc["name"], tc["args"])
```

---

## Summary

- Zero-shot works for simple tasks; use few-shot for format-sensitive output
- `.with_structured_output(MyModel)` is the modern way to get structured data
- Tool calling = model outputs JSON → your code executes → result sent back
- Always validate tool inputs with Pydantic before execution
- `@tool` decorator generates JSON schema automatically from docstring

## Common Mistakes

- Not writing detailed tool docstrings → model calls wrong tool or wrong args
- Using `eval()` on unsanitized LLM output → security risk
- Forgetting to send tool results back to the model (the loop never completes)
- Using raw string parsing instead of `.with_structured_output()` → fragile

## Interview Questions

1. Explain the function calling cycle end-to-end.
2. What is the purpose of `tool_call_id` in the messages array?
3. When would you use `.with_structured_output()` vs. a custom parser?
4. How do you prevent prompt injection in tool inputs?

## Exercises

1. Write a `@tool` function for querying a SQLite database. Add Pydantic input validation.
2. Create a `MeetingSummary` Pydantic model with 5+ fields and extract it from a fake transcript.
3. Build a raw agent loop (no framework) that calls 2 tools and runs for max 5 iterations.
4. Write a few-shot prompt for classifying support tickets into 4 categories.

---

> **Next**: [Chapter 06 — Memory, Context, State Management](Chapter-06-Memory-Context-State.md)
