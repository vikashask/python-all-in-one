# Chapter 14 — Real-World Projects

> **Previous**: [Chapter 13 — Testing & Evaluation](Chapter-13-Testing-Evaluation.md) | **Next**: [Chapter 15 — Project Architecture](Chapter-15-Project-Architecture.md)

---

## Project 1: Personal Research Assistant

**Stack**: LangGraph + Tavily Search + Pydantic output
**What it does**: Takes a research question, searches the web, synthesizes findings into a structured report.

```python
# pip install langgraph langchain-openai langchain-community tavily-python pydantic

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
import operator, json

# ── Output schema ───────────────────────────────────────────────────────────

class ResearchReport(BaseModel):
    title:       str
    summary:     str = Field(description="2-3 paragraph executive summary")
    key_findings: list[str] = Field(description="5-7 bullet point findings")
    sources:     list[str]
    confidence:  float = Field(ge=0.0, le=1.0,
                               description="Confidence in report accuracy")

# ── Agent state ─────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query:       str
    search_results: Annotated[list, operator.add]
    report:      ResearchReport | None
    iterations:  int

# ── Tools ───────────────────────────────────────────────────────────────────

search = TavilySearchResults(max_results=5)
llm    = ChatOpenAI(model="gpt-4o", temperature=0)

# ── Nodes ───────────────────────────────────────────────────────────────────

def search_node(state: ResearchState) -> ResearchState:
    results = search.invoke({"query": state["query"]})
    return {"search_results": results, "iterations": state["iterations"] + 1}

def synthesize_node(state: ResearchState) -> ResearchState:
    context = "\n\n".join([
        f"Source: {r['url']}\n{r['content']}"
        for r in state["search_results"]
    ])

    structured_llm = llm.with_structured_output(ResearchReport)
    report = structured_llm.invoke(f"""
    Research question: {state["query"]}

    Sources found:
    {context}

    Write a comprehensive research report based on the sources above.
    """)

    return {"report": report}

def should_search_more(state: ResearchState) -> str:
    """Search again only if we got very few results."""
    if len(state["search_results"]) < 2 and state["iterations"] < 2:
        return "search"
    return "synthesize"

# ── Graph ────────────────────────────────────────────────────────────────────

builder = StateGraph(ResearchState)
builder.add_node("search",     search_node)
builder.add_node("synthesize", synthesize_node)
builder.set_entry_point("search")
builder.add_conditional_edges("search", should_search_more,
                               {"search": "search", "synthesize": "synthesize"})
builder.add_edge("synthesize", END)

research_agent = builder.compile()

# ── Run ───────────────────────────────────────────────────────────────────────

def research(question: str) -> ResearchReport:
    result = research_agent.invoke({
        "query":          question,
        "search_results": [],
        "report":         None,
        "iterations":     0
    })
    return result["report"]

report = research("What are the latest advances in AI agent memory systems?")
print(report.title)
print(report.summary)
for f in report.key_findings:
    print(f"  • {f}")
```

---

## Project 2: Coding Assistant with Reflection

**Stack**: LangGraph with critique loop
**What it does**: Generates code, runs it, critiques output, and improves iteratively.

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import TypedDict
import subprocess, tempfile, os

class CodingState(TypedDict):
    task:      str
    code:      str
    test_output: str
    critique:  str
    iteration: int
    final:     bool

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

def generate_code(state: CodingState) -> CodingState:
    history = ""
    if state.get("critique"):
        history = f"\nPrevious attempt:\n```python\n{state['code']}\n```\nCritique: {state['critique']}\n"

    code = llm.invoke(f"""
    Write Python code for: {state['task']}
    {history}
    Return ONLY the Python code, no explanations.
    """).content

    # Strip markdown fences if present
    code = code.replace("```python", "").replace("```", "").strip()
    return {"code": code, "iteration": state["iteration"] + 1}

def run_code(state: CodingState) -> CodingState:
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(state["code"])
        tmpfile = f.name

    try:
        result = subprocess.run(
            ["python", tmpfile],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = "ERROR: Code execution timed out"
    except Exception as e:
        output = f"ERROR: {e}"
    finally:
        os.unlink(tmpfile)

    return {"test_output": output}

def critique_code(state: CodingState) -> CodingState:
    critique = llm.invoke(f"""
    Task: {state['task']}
    Code:
    ```python
    {state['code']}
    ```
    Output: {state['test_output']}

    Is this code correct and complete? If not, what should be fixed?
    If it's perfect, respond with exactly: "APPROVED"
    """).content

    return {"critique": critique, "final": critique.strip() == "APPROVED"}

def should_continue(state: CodingState) -> str:
    if state.get("final") or state["iteration"] >= 3:
        return "done"
    return "generate"

builder = StateGraph(CodingState)
builder.add_node("generate",  generate_code)
builder.add_node("run",       run_code)
builder.add_node("critique",  critique_code)
builder.set_entry_point("generate")
builder.add_edge("generate", "run")
builder.add_edge("run",      "critique")
builder.add_conditional_edges("critique", should_continue,
                               {"generate": "generate", "done": END})

coding_agent = builder.compile()

result = coding_agent.invoke({
    "task":        "Write a function that finds all prime numbers up to N using the Sieve of Eratosthenes",
    "code":        "",
    "test_output": "",
    "critique":    "",
    "iteration":   0,
    "final":       False
})

print(f"Final code (after {result['iteration']} iterations):")
print(result["code"])
```

---

## Project 3: Document Intelligence System

**Stack**: LangChain + Chroma + multi-format loaders
**What it does**: Indexes multiple documents and answers questions citing sources.

```python
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

CHROMA_DIR = "./doc_intelligence_db"
CHUNK_SIZE  = 800
OVERLAP     = 100

def load_documents(directory: str) -> list:
    """Load PDF, TXT, and MD files from a directory."""
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md":  UnstructuredMarkdownLoader,
    }
    docs = []
    for path in Path(directory).rglob("*"):
        loader_cls = loaders.get(path.suffix.lower())
        if loader_cls:
            loaded = loader_cls(str(path)).load()
            # Tag each chunk with filename
            for doc in loaded:
                doc.metadata["filename"] = path.name
            docs.extend(loaded)
    return docs

def build_index(directory: str) -> Chroma:
    docs     = load_documents(directory)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
    )
    chunks  = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)

def build_qa_chain(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm       = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using only the context below.
    Always cite the source filenames.
    If the answer is not in the context, say "I don't have enough information."

    Context:
    {context}

    Question: {question}
    """)

    def format_context(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('filename', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        RunnableParallel(
            context=retriever | format_context,
            question=RunnablePassthrough()
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Build and query
vectorstore = build_index("./documents")
qa_chain    = build_qa_chain(vectorstore)

answer = qa_chain.invoke("What are the key differences between LangGraph and AutoGen?")
print(answer)
```

---

## Project 4: Data Analysis Agent

**Stack**: LangChain tools + pandas + matplotlib
**What it does**: Analyzes CSV datasets, produces statistics and charts on demand.

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import pandas as pd
import json

# Tools — each wraps pandas/matplotlib operations

@tool
def load_csv(filepath: str) -> str:
    """Load a CSV file and return a summary."""
    global _df
    _df = pd.read_csv(filepath)
    return json.dumps({
        "shape":   list(_df.shape),
        "columns": list(_df.columns),
        "dtypes":  _df.dtypes.astype(str).to_dict(),
        "sample":  _df.head(3).to_dict()
    })

@tool
def describe_data() -> str:
    """Return descriptive statistics for numeric columns."""
    return _df.describe().to_json()

@tool
def group_and_aggregate(group_by: str, agg_column: str, func: str = "mean") -> str:
    """Group by a column and aggregate another. func: mean, sum, count, max, min."""
    result = _df.groupby(group_by)[agg_column].agg(func)
    return result.to_json()

@tool
def filter_rows(column: str, operator: str, value: str) -> str:
    """Filter rows. operator: >, <, ==, >=, <=, !="""
    ops = {">": "__gt__", "<": "__lt__", "==": "__eq__",
           ">=": "__ge__", "<=": "__le__", "!=": "__ne__"}
    fn  = ops.get(operator, "__eq__")
    # Try numeric first
    try: val = float(value)
    except ValueError: val = value
    filtered = _df[getattr(_df[column], fn)(val)]
    return json.dumps({"rows": len(filtered), "data": filtered.head(10).to_dict()})

@tool
def plot_histogram(column: str, bins: int = 20, output: str = "histogram.png") -> str:
    """Plot a histogram of a numeric column."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    _df[column].dropna().plot.hist(ax=ax, bins=bins, color="#4472C4")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    return f"Histogram saved to {output}"

@tool
def correlation_matrix() -> str:
    """Return the correlation matrix for all numeric columns."""
    corr = _df.select_dtypes(include="number").corr()
    return corr.to_json()

_df: pd.DataFrame = None  # module-level dataframe store

llm   = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [load_csv, describe_data, group_and_aggregate,
         filter_rows, plot_histogram, correlation_matrix]

data_agent = create_react_agent(llm, tools)

def ask_data(question: str) -> str:
    result = data_agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content

# Example session
print(ask_data("Load sales.csv and give me a summary of the dataset"))
print(ask_data("Which product category has the highest average revenue?"))
print(ask_data("Show me the distribution of order values as a histogram"))
print(ask_data("What are the top 5 customers by total spend?"))
```

---

## Summary

| Project | Key Patterns | Skills Practiced |
|---|---|---|
| Research Assistant | Search + structured output | LangGraph, Pydantic, web search |
| Coding Assistant | Generation + reflection loop | Self-critique, iterative refinement |
| Document Intelligence | Multi-loader + RAG | Chunking, vector store, citations |
| Data Analysis | Tools wrapping libraries | Tool design, pandas, matplotlib |

## Exercises

1. Extend the Research Assistant to save reports to disk as Markdown files.
2. Add a "human approval" step to the Coding Assistant before it runs code.
3. Add metadata filtering to the Document Intelligence system (filter by date).
4. Add a `compare_columns` tool to the Data Analysis Agent.

---

> **Next**: [Chapter 15 — Project Architecture](Chapter-15-Project-Architecture.md)
