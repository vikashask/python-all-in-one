# Framework 05 — AutoGen

> **Previous**: [CrewAI](04-CrewAI.md) | **Next**: [LlamaIndex](06-LlamaIndex.md)

---

## What Is AutoGen?

AutoGen (Microsoft) is a framework for **multi-agent conversations**. Agents communicate through messages. It supports group chats where multiple agents collaborate, debate, and verify each other's work.

**Best for**: Research workflows, code review, debate/verification, iterative problem solving.

---

## Installation

```bash
pip install pyautogen
```

---

## Core Concepts

| Concept | Description |
|---|---|
| `AssistantAgent` | An LLM-backed agent with a system message |
| `UserProxyAgent` | Executes code and represents the human |
| `GroupChat` | Multi-agent conversation |
| `GroupChatManager` | Routes messages in group chat |
| `human_input_mode` | `"ALWAYS"`, `"NEVER"`, or `"TERMINATE"` |
| `is_termination_msg` | Lambda to detect when conversation ends |

---

## Two-Agent Conversation

```python
from autogen import AssistantAgent, UserProxyAgent
import os

llm_config = {
    "config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}],
    "temperature": 0
}

# AI assistant
assistant = AssistantAgent(
    name="AI_Assistant",
    system_message="""You are a helpful AI assistant.
    When writing code, always include type hints and docstrings.
    When done, reply with TERMINATE.""",
    llm_config=llm_config
)

# User proxy executes code on behalf of the user
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",              # fully autonomous
    code_execution_config={
        "work_dir": "agent_workspace",
        "use_docker": False
    },
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "")
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function that calculates compound interest. Test it."
)
```

---

## Group Chat (Multi-Agent)

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

llm_config = {"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]}

# Specialist agents
coder = AssistantAgent(
    name="Coder",
    system_message="""You write clean, well-tested Python code.
    Use type hints and docstrings.
    Reply TERMINATE when the code is complete and tested.""",
    llm_config=llm_config
)

reviewer = AssistantAgent(
    name="Code_Reviewer",
    system_message="""You review code for bugs, style, and best practices.
    Provide specific, actionable feedback.
    Approve with LGTM when satisfied.""",
    llm_config=llm_config
)

security = AssistantAgent(
    name="Security_Auditor",
    system_message="""You check code for security vulnerabilities.
    Look for: injection risks, unsafe eval, hardcoded secrets, OWASP issues.
    Rate severity: Critical / High / Medium / Low.""",
    llm_config=llm_config
)

user = UserProxyAgent(
    name="Developer",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "review_workspace", "use_docker": False},
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda msg: "LGTM" in msg.get("content", "")
)

# Set up group chat
group_chat = GroupChat(
    agents=[user, coder, reviewer, security],
    messages=[],
    max_round=12
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# Start the review
user.initiate_chat(
    manager,
    message="""Please review this code:

    def get_user(user_id):
        conn = sqlite3.connect('users.db')
        query = f'SELECT * FROM users WHERE id = {user_id}'
        result = conn.execute(query).fetchall()
        return result
    """
)
```

---

## Complete Example: Research Team with Debate

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

llm_config = {"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]}

researcher = AssistantAgent(
    name="Researcher",
    system_message="""You research topics thoroughly.
    Always provide evidence-based claims with sources.
    Challenge unsupported claims.""",
    llm_config=llm_config
)

critic = AssistantAgent(
    name="Critic",
    system_message="""You identify weaknesses in arguments.
    Point out missing evidence, logical flaws, and alternative viewpoints.
    Be constructive but rigorous.""",
    llm_config=llm_config
)

synthesizer = AssistantAgent(
    name="Synthesizer",
    system_message="""You combine research and critique into balanced conclusions.
    When a topic has been sufficiently debated, produce a final synthesis.
    End your synthesis with FINAL_SYNTHESIS.""",
    llm_config=llm_config
)

user = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    is_termination_msg=lambda msg: "FINAL_SYNTHESIS" in msg.get("content", "")
)

group_chat = GroupChat(
    agents=[user, researcher, critic, synthesizer],
    messages=[],
    max_round=8
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

user.initiate_chat(
    manager,
    message="Debate: Are large language models truly capable of reasoning?"
)
```

---

## Advantages

- Natural conversation metaphor — agents talk to each other like humans
- Group chat enables verification through disagreement
- Code execution built in via `UserProxyAgent`
- Good for research and educational workflows

## Limitations

- Conversations can drift off-topic
- Hard to enforce strict control flow
- Termination conditions require careful tuning
- Less deterministic than LangGraph

---

> **Next**: [06 — LlamaIndex](06-LlamaIndex.md)
