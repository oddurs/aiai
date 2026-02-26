# Python Agent Frameworks, SDKs, and Implementation Patterns (2024-2026)

> Research compiled February 2026 for the **aiai** project.
> Covers the Python ecosystem for building agent systems: SDKs, frameworks, patterns, testing, and infrastructure.

---

## Table of Contents

1. [Claude / Anthropic Python SDK](#1-claude--anthropic-python-sdk)
2. [OpenAI Agents SDK](#2-openai-agents-sdk)
3. [PydanticAI](#3-pydanticai)
4. [LangGraph](#4-langgraph-in-python)
5. [Lightweight Alternatives](#5-lightweight-alternatives)
6. [Structured Output and Tool Use](#6-structured-output-and-tool-use)
7. [Async Patterns for Agents](#7-async-patterns-for-agents)
8. [Testing Agent Systems](#8-testing-agent-systems)
9. [OpenRouter Python Integration](#9-openrouter-python-integration)
10. [Recommendations for aiai](#10-recommendations-for-aiai)

---

## 1. Claude / Anthropic Python SDK

### 1.1 Overview

The **anthropic** Python SDK (v0.84.0 as of Feb 2026, MIT license, Python 3.9+) provides access to the Claude Messages API with full support for tool use, streaming, prompt caching, batch processing, and structured outputs.

- **Repository**: https://github.com/anthropics/anthropic-sdk-python
- **Docs**: https://platform.claude.com/docs/en/api/sdks/python
- **Install**: `pip install anthropic`

### 1.2 Basic Usage

```python
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ],
)
print(response.content[0].text)
```

### 1.3 Tool Use (Function Calling)

Claude's tool use follows a multi-turn pattern: you define tools with JSON Schema, Claude returns `tool_use` blocks, you execute the tool and return `tool_result`, and Claude incorporates the result.

```python
import anthropic

client = anthropic.Anthropic()

# Step 1: Define tools and send initial request
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        }
    ],
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ],
)

# Step 2: Check for tool_use in response, execute tool, return result
# response.stop_reason == "tool_use" when Claude wants to call a tool
# Extract tool_use block, run your function, then continue conversation:
response2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[...],  # same tools
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {"role": "assistant", "content": response.content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                    "content": "59F, mostly cloudy",
                }
            ],
        },
    ],
)
```

**Agentic loop pattern**: In production agents, you wrap this in a loop that continues until `stop_reason != "tool_use"`:

```python
messages = [{"role": "user", "content": user_query}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        messages=messages,
    )

    # Append assistant response
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason != "tool_use":
        break  # Claude is done

    # Execute all tool calls and return results
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            })

    messages.append({"role": "user", "content": tool_results})
```

### 1.4 Streaming

The SDK provides both sync and async streaming with high-level helpers:

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

# Async streaming with text_stream helper
async with client.messages.stream(
    max_tokens=1024,
    messages=[{"role": "user", "content": "Say hello there!"}],
    model="claude-sonnet-4-20250514",
) as stream:
    async for text in stream.text_stream:
        print(text, end="", flush=True)

# Event-level streaming for fine-grained control
async with client.messages.stream(...) as stream:
    async for event in stream:
        if event.type == "text":
            print(event.text, end="", flush=True)
        elif event.type == "content_block_stop":
            print("\nBlock finished:", event.content_block)

# After consuming the stream, get the final accumulated message:
accumulated = await stream.get_final_message()
```

**Key stream methods:**
- `stream.text_stream` -- async iterator of text deltas
- `await stream.get_final_message()` -- full Message object
- `await stream.get_final_text()` -- concatenated text
- `await stream.close()` -- abort the request

The async client uses httpx by default but also supports aiohttp as a backend for improved concurrency.

### 1.5 Prompt Caching

Prompt caching reduces costs by up to 90% and latency by up to 85% for repeated prefixes. Cache has a 5-minute default lifetime (1-hour available at additional cost), refreshed on each use.

**Automatic caching** (simplest -- system applies cache to last cacheable block):

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    cache_control={"type": "ephemeral"},
    system="You are an expert analyst. <long context here>...",
    messages=[{"role": "user", "content": "Analyze this data."}],
)
# Check cache usage:
print(response.usage)  # includes cache_creation_input_tokens, cache_read_input_tokens
```

**Explicit cache breakpoints** (fine-grained control):

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "<very long reference document>",
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[{"role": "user", "content": "Question about the document"}],
)
```

**Pricing impact** (Claude Sonnet 4): Cache writes cost 1.25x base input, cache hits cost 0.1x base input.

### 1.6 Batch API (Message Batches)

For high-throughput, non-real-time workloads. Up to 50% cheaper. Processing takes up to 24 hours.

```python
import anthropic

client = anthropic.Anthropic()

# Create a batch
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": "request-1",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
            },
        },
        {
            "custom_id": "request-2",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Explain quantum computing."}
                ],
            },
        },
    ]
)

# Poll for completion
import time
while True:
    status = client.messages.batches.retrieve(batch.id)
    if status.processing_status == "completed":
        break
    time.sleep(10)

# Retrieve results
for result in client.messages.batches.results(batch.id):
    print(f"{result.custom_id}: {result.result.message.content[0].text}")
```

### 1.7 Structured Outputs

Available via beta header. Guarantees JSON Schema conformance.

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    extra_headers={"anthropic-beta": "structured-outputs-2025-11-13"},
    messages=[{"role": "user", "content": "Extract entities from this text..."}],
    output_format={
        "type": "json_schema",
        "json_schema": {
            "name": "entities",
            "schema": {
                "type": "object",
                "properties": {
                    "people": {"type": "array", "items": {"type": "string"}},
                    "places": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["people", "places"],
            },
        },
    },
)
```

### 1.8 Claude Agent SDK

The **Claude Agent SDK** (formerly "Claude Code SDK") is Anthropic's production-grade framework that exposes the same infrastructure powering Claude Code as a programmable library. Released May 2025, renamed September 2025.

- **Repository**: https://github.com/anthropics/claude-agent-sdk-python
- **PyPI**: `pip install claude-agent-sdk`
- **Requires**: Python 3.10+, Node.js 18+

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

async def main():
    async for message in query(
        prompt="Review utils.py for bugs. Fix any issues you find.",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob"],
            permission_mode="acceptEdits",
        ),
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text)
                elif hasattr(block, "name"):
                    print(f"Tool: {block.name}")
        elif isinstance(message, ResultMessage):
            print(f"Done: {message.subtype}")

asyncio.run(main())
```

**Key capabilities:**
- **Built-in tools**: Read, Edit, Glob, Grep, Bash, WebSearch
- **Permission modes**: `acceptEdits` (auto-approve file edits), `bypassPermissions` (full automation/CI), `default` (custom canUseTool callback)
- **Subagents**: Isolated context windows for parallel sub-tasks; spin up specialized subagents for security review, testing, etc.
- **Hooks**: Inject custom logic at PreToolUse, PostToolUse, PermissionRequest, Stop, etc.
- **MCP integration**: Connect to databases, browsers, APIs via Model Context Protocol
- **Sessions**: Multi-turn agents with persistent context
- **Provider flexibility**: Supports AWS Bedrock, Google Vertex AI, Azure AI Foundry

**Custom system prompt example:**
```python
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Glob", "Bash"],
    permission_mode="acceptEdits",
    system_prompt="You are a senior Python developer. Follow PEP 8.",
)
```

### 1.9 MCP (Model Context Protocol) Helpers

The SDK includes MCP integration helpers:

```python
# pip install anthropic[mcp]  (requires Python 3.10+)
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async with stdio_client(StdioServerParameters(command="mcp-server")) as (read, write):
    async with ClientSession(read, write) as mcp_client:
        await mcp_client.initialize()
        tools_result = await mcp_client.list_tools()

        runner = await client.beta.messages.tool_runner(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Use the available tools"}],
            tools=[async_mcp_tool(t, mcp_client) for t in tools_result.tools],
        )
        async for message in runner:
            print(message)
```

**References:**
- [Anthropic SDK Python GitHub](https://github.com/anthropics/anthropic-sdk-python)
- [Claude Agent SDK Python GitHub](https://github.com/anthropics/claude-agent-sdk-python)
- [Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Claude API Docs - Tool Use](https://platform.claude.com/docs/en/build-with-claude/tool-use)
- [Claude API Docs - Prompt Caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Claude API Docs - Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
- [Claude Agent SDK Overview](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Claude Agent SDK Quickstart](https://platform.claude.com/docs/en/agent-sdk/quickstart)
- [Claude Agent SDK Demos](https://github.com/anthropics/claude-agent-sdk-demos)

---

## 2. OpenAI Agents SDK

### 2.1 Overview

The **OpenAI Agents SDK** was released in March 2025 as the production-ready successor to the experimental Swarm project. It is a lightweight, Python-first open-source framework for multi-agent workflows.

- **Repository**: https://github.com/openai/openai-agents-python
- **Docs**: https://openai.github.io/openai-agents-python/
- **Install**: `pip install openai-agents`
- **License**: MIT

### 2.2 Core Primitives

The SDK has four core building blocks:

1. **Agents** -- LLM-powered entities with instructions, tools, and handoffs
2. **Tools** -- Python functions auto-converted to tool schemas
3. **Handoffs** -- Delegation mechanism between specialized agents
4. **Guardrails** -- Input/output validation that runs in parallel with agent execution

### 2.3 Basic Agent

```python
from agents import Agent, Runner
import asyncio

agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step.",
)

async def main():
    result = await Runner.run(agent, "What is 235 * 17?")
    print(result.final_output)

asyncio.run(main())
```

### 2.4 Handoffs Between Agents

Handoffs are the key multi-agent pattern. They appear as tools to the LLM, so the model can decide when to delegate.

```python
from agents import Agent, Runner
import asyncio

history_tutor = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries.",
)

math_tutor = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems.",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine which agent to use based on the user's question.",
    handoffs=[history_tutor, math_tutor],
)

async def main():
    result = await Runner.run(triage_agent, "Who was the first president?")
    print(result.final_output)

asyncio.run(main())
```

**Custom handoffs** with the `handoff()` function allow optional `tool_description_override` and `on_handoff` callbacks for data fetching during handoff.

### 2.5 Guardrails

Guardrails run input/output validation in parallel with agent execution and can short-circuit (fail fast) when checks fail.

```python
from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Homework Checker",
    instructions="Determine if the input is a homework question.",
    output_type=HomeworkOutput,
)

async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine which agent to use.",
    handoffs=[history_tutor, math_tutor],
    input_guardrails=[InputGuardrail(guardrail_function=homework_guardrail)],
)
```

### 2.6 Additional Features

- **Sessions**: Persistent memory layer for maintaining working context within an agent loop
- **Tracing**: Built-in visualization, debugging, and monitoring; integrates with OpenAI eval/fine-tuning/distillation tools
- **Provider-agnostic**: While optimized for OpenAI models, documented paths exist for 100+ other LLMs via Chat Completions API compatibility

### 2.7 When to Use

**Good for:**
- Multi-agent orchestration with clear handoff patterns
- Teams already using OpenAI models
- Need for built-in tracing and guardrails
- Production systems requiring sessions and persistence

**Less ideal for:**
- Anthropic-primary workflows
- Custom control flow beyond handoff patterns
- Highly dynamic graph-based workflows (LangGraph may be better)

**References:**
- [OpenAI Agents SDK Docs](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK GitHub](https://github.com/openai/openai-agents-python)
- [OpenAI Agents SDK Review (Dec 2025)](https://mem0.ai/blog/openai-agents-sdk-review)
- [Handoffs Documentation](https://openai.github.io/openai-agents-python/handoffs/)
- [Guardrails Documentation](https://openai.github.io/openai-agents-python/guardrails/)
- [OpenAI for Developers 2025](https://developers.openai.com/blog/openai-for-developers-2025/)

---

## 3. PydanticAI

### 3.1 Overview

**PydanticAI** is the Pydantic team's agent framework -- a type-safe, model-agnostic Python framework for building AI agents with validated structured outputs and dependency injection. Reached v1.0 in 2025.

- **Repository**: https://github.com/pydantic/pydantic-ai
- **Docs**: https://ai.pydantic.dev/
- **Install**: `pip install pydantic-ai`

### 3.2 Basic Agent with Structured Output

```python
from pydantic_ai import Agent
from pydantic import BaseModel, Field

class SupportOutput(BaseModel):
    support_advice: str = Field(description="Advice returned to the customer")
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description="Risk level of query", ge=0, le=10)

agent = Agent(
    "anthropic:claude-sonnet-4-6",
    output_type=SupportOutput,
    instructions="You are a support agent for a bank.",
)

result = agent.run_sync("I think my card was stolen!")
print(result.output)
# SupportOutput(support_advice='...', block_card=True, risk=9)
```

### 3.3 Tool Definitions with RunContext

Tool docstrings become descriptions passed to the LLM. The `RunContext` parameter provides type-safe access to dependencies.

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn

support_agent = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=SupportDependencies,
    output_type=SupportOutput,
)

@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
```

### 3.4 Dependency Injection

Dependencies are carried via `RunContext`, parameterized with the deps type. Static type checkers validate correct usage.

```python
async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run("What is my balance?", deps=deps)
    print(result.output)
```

This is especially powerful for testing -- inject mock dependencies:

```python
async def test_balance_query():
    mock_db = MockDatabaseConn(balance=1000.0)
    deps = SupportDependencies(customer_id=1, db=mock_db)
    result = await support_agent.run("What is my balance?", deps=deps)
    assert result.output.risk < 3
```

### 3.5 Structured Output Modes

PydanticAI supports three structured output modes:
1. **Tool calling** -- uses the LLM's function calling to return structured data
2. **Provider-managed structured outputs** -- native JSON Schema response format
3. **Manual prompting** -- prompts the LLM to return JSON, then validates

Streaming structured output with immediate validation is also supported.

### 3.6 Graph-Based Workflows

PydanticAI also provides a way to define **graphs using type hints** for complex applications that need explicit state machine control flow, going beyond simple agent loops.

### 3.7 Observability

```python
import logfire
logfire.configure()
logfire.instrument_pydantic_ai()
# Now all agent runs are traced in Pydantic Logfire
```

### 3.8 When to Use PydanticAI vs Rolling Your Own

**Use PydanticAI when:**
- You need validated, type-safe structured outputs
- You want dependency injection for clean testing
- You are model-agnostic (switch between OpenAI, Anthropic, Gemini, etc.)
- You already use Pydantic in your stack
- You want observability via Logfire

**Roll your own when:**
- You need maximum control over the agent loop
- Your tool use patterns are very custom
- You want to avoid framework lock-in entirely
- You are building a specialized single-model system

**References:**
- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [PydanticAI GitHub](https://github.com/pydantic/pydantic-ai)
- [PydanticAI v1 Announcement](https://pydantic.dev/articles/pydantic-ai-v1)
- [PydanticAI on PyPI](https://pypi.org/project/pydantic-ai/)
- [PydanticAI Tools Docs](https://ai.pydantic.dev/tools/)
- [PydanticAI Dependencies Docs](https://ai.pydantic.dev/dependencies/)

---

## 4. LangGraph in Python

### 4.1 Overview

**LangGraph** is an open-source framework by the LangChain team for building stateful, multi-agent applications using graph-based state machines. It provides durable execution, checkpointing, and human-in-the-loop patterns.

- **Repository**: https://github.com/langchain-ai/langgraph
- **Docs**: https://docs.langchain.com/oss/python/langgraph/overview
- **Install**: `pip install langgraph`

### 4.2 Core Concepts

- **State**: A shared TypedDict that flows through the graph; all nodes read from and write to it
- **Nodes**: Functions that process state (LLM calls, tool execution, business logic)
- **Edges**: Transitions between nodes (including conditional edges based on state)
- **Reducers**: Logic for merging state updates from nodes
- **Checkpointers**: Persistence layer that saves state at every super-step

### 4.3 Basic Graph Example

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

# Define a node function
def chat_node(state: MessagesState):
    # Call LLM with current messages
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Compile and run
app = graph.compile()
result = app.invoke({"messages": [("user", "Hello!")]})
```

### 4.4 Conditional Routing

```python
from typing import TypedDict, Literal

class AgentState(TypedDict):
    messages: list
    next_action: str

def route_decision(state: AgentState) -> Literal["tool_node", "respond"]:
    if state["next_action"] == "use_tool":
        return "tool_node"
    return "respond"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tool_node", tool_node)
graph.add_node("respond", respond_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route_decision)
graph.add_edge("tool_node", "agent")  # Loop back after tool use
graph.add_edge("respond", END)
```

### 4.5 Checkpointing and Persistence

Checkpointers save graph state at every super-step, enabling time-travel debugging, fault recovery, and human-in-the-loop.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
# For production: from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=checkpointer)

# Each invocation uses a thread_id for state persistence
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [("user", "Hello")]}, config=config)

# Later, resume the same conversation:
result2 = app.invoke({"messages": [("user", "Follow up")]}, config=config)
```

**Checkpointer options:**
- `langgraph-checkpoint-sqlite` -- local development/experimentation
- `langgraph-checkpoint-postgres` -- production (used by LangSmith)

### 4.6 Human-in-the-Loop

The `interrupt()` function pauses graph execution, surfaces a value to the client, and waits for human input. Requires a checkpointer.

```python
from langgraph.types import interrupt, Command

def human_review_node(state):
    # Pause and ask for human approval
    approval = interrupt({"question": "Approve this action?", "data": state["draft"]})

    if approval == "approved":
        return {"status": "approved"}
    else:
        return {"status": "rejected"}

# Client-side: resume with Command
app.invoke(Command(resume="approved"), config=config)
```

### 4.7 Performance Characteristics

**Strengths:**
- Durable execution -- agents survive failures and resume from checkpoints
- Complex control flow -- cycles, branches, conditional routing
- State isolation -- clear separation of concerns via graph structure
- Fault tolerance -- restart from last successful checkpoint on failure

**Considerations:**
- Higher overhead per agent than lightweight frameworks (Agno claims 10,000x faster agent creation)
- More complex mental model than simple agent loops
- LangChain dependency can be heavy if you only need the graph features
- Learning curve for graph-based thinking

### 4.8 When to Use LangGraph

**Good for:**
- Complex workflows with branching logic and cycles
- Long-running agents that need checkpointing/durability
- Human-in-the-loop approval flows
- Multi-agent systems with explicit state management
- Systems that need fault tolerance and recovery

**Less ideal for:**
- Simple single-agent tasks
- Prototyping (overhead is higher)
- Lightweight, stateless tool-calling agents

**References:**
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph Persistence Docs](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph State Machines Article](https://dev.to/jamesli/langgraph-state-machines-managing-complex-agent-task-flows-in-production-36f4)
- [LangGraph Human-in-the-Loop](https://blog.langchain.com/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt/)

---

## 5. Lightweight Alternatives

### 5.1 smolagents (Hugging Face)

A minimalist agent library (~1000 lines of core logic) where agents write Python code instead of JSON tool calls.

- **Repository**: https://github.com/huggingface/smolagents
- **Docs**: https://huggingface.co/docs/smolagents/en/index
- **Install**: `pip install smolagents`

**Key features:**
- **CodeAgents**: The primary agent type generates Python code to perform actions, rather than JSON/text
- **Model-agnostic**: Works with any LLM (local transformers, Ollama, OpenAI, Anthropic, etc.)
- **Minimal abstractions**: Logic fits in ~1000 lines of code
- **Multi-agent support**: Built-in orchestration

```python
from smolagents import CodeAgent, HfApiModel

model = HfApiModel()
agent = CodeAgent(tools=[], model=model)
result = agent.run("What is 2 + 2?")
```

**When to use:** Rapid prototyping, code-generation agents, small projects where you want minimal overhead. Good when you want agents that reason in Python code.

### 5.2 Agno (formerly Phidata)

A high-performance, model-agnostic framework for multi-modal AI agents with aggressive performance claims.

- **Repository**: https://github.com/agno-agi/agno
- **Website**: https://www.agno.com
- **Install**: `pip install agno`

**Key features:**
- Agent creation ~2 microseconds (claims ~10,000x faster than LangGraph)
- ~3.75 KiB memory per agent (~50x less than LangGraph)
- Native multimodal support (text, image, audio, video)
- Built-in Auto-RAG (agents search vector DBs for relevant info)
- Multi-agent team orchestration
- Model-agnostic (works with any provider)

**When to use:** Performance-critical agent systems, multimodal agents, scenarios requiring very low overhead per agent. Previously known as Phidata, so stable/mature codebase.

### 5.3 Marvin (by Prefect)

A lightweight toolkit for building "AI functions" -- small, focused AI-powered operations.

- **Repository**: https://github.com/PrefectHQ/marvin
- **Docs**: https://askmarvin.ai/welcome
- **Install**: `pip install marvin`

**Key features:**
- `marvin.cast()` -- transform data from one type to another using AI
- `marvin.classify()` -- classify inputs into categories
- `marvin.extract()` -- extract structured entities from text
- `marvin.fn()` -- create custom AI functions without source code
- As of Marvin 3.0, uses PydanticAI under the hood

```python
import marvin

# Classification
result = marvin.classify("I love this product!", labels=["positive", "negative", "neutral"])
# "positive"

# Extraction
entities = marvin.extract("I moved to New York from Paris last year.", target=str, instructions="cities")
# ["New York", "Paris"]

# AI functions
@marvin.fn
def summarize(text: str) -> str:
    """Return a one-sentence summary."""

summary = summarize("Long article text here...")
```

**When to use:** Quick AI-powered data transformations, classification, extraction. Not a full agent framework -- more like utility functions powered by LLMs.

### 5.4 Instructor

The most popular Python library for structured LLM output extraction (3M+ monthly downloads, 11k+ stars).

- **Repository**: https://github.com/567-labs/instructor
- **Docs**: https://python.useinstructor.com/
- **Install**: `pip install instructor`

**Key features:**
- Patches LLM client objects to add response_model, max_retries, and validation
- Automatic retry with validation feedback sent back to the LLM
- Works with 15+ providers (OpenAI, Anthropic, Google, Ollama, etc.)
- Streaming support for partial objects
- Pydantic-based validation

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI

# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

class User(BaseModel):
    name: str
    age: int

# Extract structured data with automatic validation and retry
user = client.chat.completions.create(
    model="gpt-4o",
    response_model=User,
    max_retries=3,
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
)
print(user)  # User(name='John', age=30)
```

**With Anthropic:**
```python
import instructor
import anthropic

client = instructor.from_anthropic(anthropic.Anthropic())

user = client.messages.create(
    model="claude-sonnet-4-20250514",
    response_model=User,
    max_retries=3,
    max_tokens=1024,
    messages=[{"role": "user", "content": "Extract: John is 30 years old"}],
)
```

**When to use:** When you specifically need reliable structured output extraction from LLMs. Lighter than PydanticAI (focused on output extraction rather than full agent framework). Excellent for data pipelines.

### 5.5 Outlines (by .txt)

Constrained generation library that guarantees structured outputs at the token level.

- **Repository**: https://github.com/dottxt-ai/outlines
- **Docs**: https://dottxt-ai.github.io/outlines/
- **Install**: `pip install outlines`

**Key features:**
- Works at the inference level (modifies token sampling)
- Supports Pydantic models, regex, JSON Schema, context-free grammars
- Integrates with transformers, llama.cpp, vLLM
- Guarantees valid output (not probabilistic)
- Most efficient method for structured output when you control inference

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

# Generate with Pydantic schema
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    weapon: str

generator = outlines.generate.json(model, Character)
result = generator("Give me a fantasy RPG character.")
print(result)  # Character(name='...', age=..., weapon='...')
```

**When to use:** When you run local/self-hosted models and need guaranteed structured output. Not applicable for API-based models (OpenAI, Anthropic, etc.) since you don't control token generation. Superior to prompt-based approaches for local inference.

### 5.6 Comparison Summary

| Library | Focus | Overhead | Best For |
|---------|-------|----------|----------|
| smolagents | Code-writing agents | Very low | Prototyping, code-gen agents |
| Agno | High-perf multi-agent | Very low | Performance-critical, multimodal |
| Marvin | AI utility functions | Minimal | Classification, extraction, transforms |
| Instructor | Structured output | Minimal | Data extraction from LLMs |
| Outlines | Constrained generation | Low | Local model structured output |

**References:**
- [smolagents GitHub](https://github.com/huggingface/smolagents)
- [smolagents Docs](https://huggingface.co/docs/smolagents/en/index)
- [Agno GitHub](https://github.com/agno-agi/agno)
- [Marvin GitHub](https://github.com/PrefectHQ/marvin)
- [Instructor Docs](https://python.useinstructor.com/)
- [Instructor GitHub](https://github.com/567-labs/instructor)
- [Outlines GitHub](https://github.com/dottxt-ai/outlines)
- [Structured Output Comparison](https://simmering.dev/blog/structured_output/)

---

## 6. Structured Output and Tool Use

### 6.1 The Three Main Approaches

Getting reliable structured output from LLMs is critical for agent tool use. There are three main approaches, ranked by reliability:

#### 1. Constrained Generation (Best for local models)

The model's token generation is constrained to only produce valid tokens matching a schema. Libraries like **Outlines** and **llguidance** implement this.

- **Pros**: Guaranteed valid output, most efficient, no retries needed
- **Cons**: Only works when you control inference (local models, vLLM, llama.cpp)
- **Libraries**: outlines, guidance, xgrammar

#### 2. Function Calling / Tool Use (Best for API models)

The LLM provider's native function calling feature. The model is trained to output structured tool calls.

- **Pros**: High reliability, supported by all major APIs, natural for agents
- **Cons**: Slight overhead from tool schema in context, occasional schema violations
- **Providers**: OpenAI, Anthropic, Google, all support this natively

#### 3. JSON Mode / Structured Outputs (Provider-managed)

The provider guarantees JSON Schema conformance in the response.

- **Pros**: Guaranteed valid JSON, no tool-calling overhead
- **Cons**: Not all providers support it, may add latency
- **Providers**: OpenAI Structured Outputs, Anthropic Structured Outputs (beta)

#### 4. Prompt-Based (Least reliable)

Simply instruct the model to return JSON in a specific format.

- **Pros**: Works with any model, no special API features needed
- **Cons**: Frequently produces invalid JSON, requires parsing/retry logic
- **Mitigation**: Instructor library's retry mechanism helps

### 6.2 Pydantic as the Universal Schema Language

All approaches converge on **Pydantic** as the Python-side schema definition:

```python
from pydantic import BaseModel, Field
from typing import Literal

class ToolCall(BaseModel):
    """Schema used across all approaches."""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict = Field(description="Arguments to pass to the tool")
    confidence: float = Field(ge=0, le=1, description="Confidence in this call")

class AgentOutput(BaseModel):
    reasoning: str = Field(description="Chain of thought reasoning")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: str | None = Field(default=None)
```

Pydantic models automatically generate JSON Schema, validate outputs, and provide clear error messages for retry loops.

### 6.3 Best Practices for Agent Tool Use

1. **Write detailed tool descriptions** -- treat them like prompt engineering
2. **Use `required` fields** -- reduce ambiguity for the model
3. **Keep schemas simple** -- deeply nested schemas cause more errors
4. **Provide examples in descriptions** -- e.g., `"The city and state, e.g. San Francisco, CA"`
5. **Use enums for constrained choices** -- `Literal["celsius", "fahrenheit"]`
6. **Add `strict: true`** when available (Anthropic, OpenAI) for guaranteed conformance
7. **Implement retry logic** -- Instructor's pattern of sending validation errors back to the LLM
8. **Limit tool count** -- More tools = more confusion; use tool search for large toolsets

### 6.4 Practical Pattern: Validated Tool Execution

```python
from pydantic import BaseModel, ValidationError
import json

class SearchQuery(BaseModel):
    query: str
    max_results: int = 10
    filter_by: str | None = None

def execute_tool(name: str, raw_input: dict) -> str:
    """Execute a tool with Pydantic validation."""
    tool_schemas = {
        "search": SearchQuery,
    }

    schema = tool_schemas.get(name)
    if not schema:
        return f"Error: Unknown tool '{name}'"

    try:
        validated = schema(**raw_input)
        # Execute with validated input
        return do_search(validated)
    except ValidationError as e:
        return f"Error: Invalid input for '{name}': {e}"
```

**References:**
- [Guide to Structured Outputs and Function Calling](https://agenta.ai/blog/the-guide-to-structured-outputs-and-function-calling-with-llms)
- [Best Library for Structured LLM Output](https://simmering.dev/blog/structured_output/)
- [Pydantic for LLMs](https://pydantic.dev/articles/llm-intro)
- [PydanticAI Output Docs](https://ai.pydantic.dev/output/)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [Anthropic Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)

---

## 7. Async Patterns for Agents

### 7.1 Why Async Matters for Agents

Agent systems are I/O bound -- they spend most of their time waiting for LLM API responses, database queries, and tool execution. Async Python (asyncio) allows concurrent handling of these operations, reducing total latency by 70%+ for parallel workloads.

### 7.2 Core Pattern: asyncio.gather for Parallel Agent Calls

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def call_llm(prompt: str) -> str:
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

async def parallel_analysis(texts: list[str]) -> list[str]:
    """Analyze multiple texts in parallel."""
    tasks = [call_llm(f"Analyze: {text}") for text in texts]
    return await asyncio.gather(*tasks)

# All calls execute concurrently -- total time ~= slowest single call
results = asyncio.run(parallel_analysis(["text1", "text2", "text3"]))
```

### 7.3 Rate Limiting with Semaphore

```python
import asyncio

class RateLimitedClient:
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = AsyncAnthropic()

    async def call(self, prompt: str) -> str:
        async with self.semaphore:
            response = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

# Only 5 concurrent API calls at a time
rate_limited = RateLimitedClient(max_concurrent=5)
tasks = [rate_limited.call(p) for p in prompts]
results = await asyncio.gather(*tasks)
```

### 7.4 Token Bucket Rate Limiter

For per-minute rate limits (e.g., OpenAI's RPM limits):

```python
import asyncio
import time

class TokenBucketRateLimiter:
    def __init__(self, rate: float, max_tokens: int):
        self.rate = rate  # tokens per second
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

# 60 requests per minute
limiter = TokenBucketRateLimiter(rate=1.0, max_tokens=60)

async def rate_limited_call(prompt: str):
    await limiter.acquire()
    return await client.messages.create(...)
```

### 7.5 Connection Pooling with httpx

```python
import httpx

# Configure connection limits
limits = httpx.Limits(
    max_connections=100,     # Total connection pool size
    max_keepalive_connections=20,  # Persistent connections
    keepalive_expiry=30,     # Seconds before closing idle connections
)

async with httpx.AsyncClient(limits=limits, timeout=60.0) as http_client:
    # Use this client for all API calls
    pass

# The Anthropic SDK manages its own httpx client internally,
# but you can configure it:
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    max_retries=3,
    timeout=60.0,
    # The SDK handles connection pooling internally
)
```

### 7.6 Streaming with Parallel Processing

Process streaming responses as they arrive while other tasks continue:

```python
async def stream_and_process(prompt: str):
    """Stream LLM response while processing chunks."""
    chunks = []
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            chunks.append(text)
            # Process incrementally (e.g., update UI, check for tool calls)

    return "".join(chunks)

# Stream multiple responses concurrently
results = await asyncio.gather(
    stream_and_process("Task 1"),
    stream_and_process("Task 2"),
)
```

### 7.7 anyio for Framework Compatibility

If your project needs to support both asyncio and trio (or you want framework-agnostic async):

```python
import anyio

async def portable_agent():
    """Works with both asyncio and trio."""
    async with anyio.create_task_group() as tg:
        tg.start_soon(subtask_1)
        tg.start_soon(subtask_2)

# Run with asyncio
anyio.run(portable_agent, backend="asyncio")
# Or with trio
anyio.run(portable_agent, backend="trio")
```

### 7.8 Hybrid Approach: Async I/O + Process Pool for CPU Work

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)

async def hybrid_agent_step(prompt: str):
    # I/O bound: async LLM call
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text

    # CPU bound: run in process pool
    loop = asyncio.get_event_loop()
    processed = await loop.run_in_executor(executor, cpu_heavy_processing, text)

    return processed
```

**References:**
- [Async LLM Pipelines Without Bottlenecks](https://dasroot.net/posts/2026/02/async-llm-pipelines-python-bottlenecks/)
- [Parallel API Requests for LLMs](https://medium.com/@ghaelen.m/how-to-run-multiple-parallel-api-requests-to-llm-apis-without-freezing-your-cpu-in-python-asyncio-af0da7e240e3)
- [Python Asyncio for LLM Concurrency](https://www.newline.co/@zaoyang/python-asyncio-for-llm-concurrency-best-practices--bc079176)
- [Concurrent vs Parallel LLM API Calls](https://towardsai.net/p/machine-learning/concurrent-vs-parallel-execution-in-llm-api-calls-from-an-ai-engineers-perspective)
- [How to Build Parallel Agentic Workflows](https://www.digitalocean.com/community/tutorials/how-to-build-parallel-agentic-workflows-with-python)

---

## 8. Testing Agent Systems

### 8.1 The Challenge

Agent systems are inherently non-deterministic. The same prompt can produce different outputs. Testing requires strategies that account for this while still providing confidence in system behavior.

### 8.2 Testing Pyramid for Agents

```
              /\
             /  \   Evals (LLM-as-judge, human review)
            /----\
           /      \  Integration tests (real API calls, cassettes)
          /--------\
         /          \ Unit tests (mocked LLM, deterministic)
        /------------\
```

**Unit tests** -- fast, deterministic, mock the LLM
**Integration tests** -- recorded API interactions, cassettes
**Evals** -- actual LLM calls, evaluated by criteria or another LLM

### 8.3 Unit Testing with Mocked LLMs

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

class MockMessage:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [MagicMock(type="text", text=text)]
        self.stop_reason = stop_reason
        self.usage = MagicMock(input_tokens=10, output_tokens=20)

@pytest.mark.asyncio
async def test_agent_processes_weather_query():
    """Test agent logic without calling the real API."""
    mock_client = AsyncMock()
    mock_client.messages.create.return_value = MockMessage(
        "The weather in SF is 65F and sunny."
    )

    agent = WeatherAgent(client=mock_client)
    result = await agent.handle("What's the weather in SF?")

    assert "65" in result
    mock_client.messages.create.assert_called_once()
    # Verify the right model/tools were passed
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "get_weather" in str(call_kwargs.get("tools", []))
```

**With PydanticAI's dependency injection:**
```python
@pytest.mark.asyncio
async def test_support_agent():
    """PydanticAI makes mocking natural via dependency injection."""
    mock_db = MockDatabaseConn(balance=500.0)
    deps = SupportDependencies(customer_id=42, db=mock_db)

    result = await support_agent.run("What is my balance?", deps=deps)
    assert result.output.risk < 5
    assert mock_db.balance_checked
```

### 8.4 Recording and Replaying with VCR.py

VCR.py (and its pytest plugin, pytest-recording) records HTTP interactions to "cassettes" and replays them in subsequent test runs. This eliminates flakiness from non-deterministic LLM responses.

```bash
pip install vcrpy pytest-recording
```

```python
import pytest
import vcr

# Method 1: VCR.py decorator
@vcr.use_cassette("tests/cassettes/weather_query.yaml")
def test_weather_agent():
    """First run: records API call. Subsequent runs: replays."""
    agent = WeatherAgent()
    result = agent.handle_sync("What's the weather in SF?")
    assert "temperature" in result.lower()

# Method 2: pytest-recording plugin (preferred)
@pytest.mark.vcr()
def test_weather_agent_v2():
    """Cassettes stored in tests/cassettes/ by default."""
    agent = WeatherAgent()
    result = agent.handle_sync("What's the weather in SF?")
    assert "temperature" in result.lower()
```

**Recording cassettes:**
```bash
# Record new cassettes
pytest --record-mode=once tests/

# Re-record all cassettes
pytest --record-mode=rewrite tests/

# Disable VCR (run against live API)
pytest --disable-recording tests/
```

**VCR configuration for LLM APIs:**
```python
# conftest.py
@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-api-key"],
        "record_mode": "none",  # Fail if cassette missing (CI mode)
        "match_on": ["method", "scheme", "host", "port", "path", "body"],
    }
```

### 8.5 Deterministic Testing Patterns

```python
import pytest
from unittest.mock import patch
import json

# Pattern 1: Fixed seed / temperature=0
def test_deterministic_output():
    """Use temperature=0 for more reproducible outputs."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        temperature=0,  # Most deterministic
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    assert "4" in response.content[0].text

# Pattern 2: Assert on structure, not exact text
def test_structured_output():
    """Validate output structure rather than exact wording."""
    result = agent.run_sync("Extract entities from: John lives in Paris")
    assert isinstance(result.output, EntityOutput)
    assert len(result.output.people) > 0
    assert "John" in result.output.people

# Pattern 3: Property-based testing for tool schemas
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_tool_handles_any_input(input_text):
    """Verify tool doesn't crash on arbitrary input."""
    try:
        result = search_tool.execute(query=input_text)
        assert isinstance(result, str)
    except ValueError:
        pass  # Expected for some inputs
```

### 8.6 Eval-Based Testing (LLM-as-Judge)

```python
# pytest-evals plugin approach
import pytest

@pytest.mark.eval(
    criteria="The response should be helpful and accurate",
    model="gpt-4o",
)
def test_agent_helpfulness():
    result = agent.run_sync("Explain quantum computing to a child")
    return result.output  # Evaluated by the judge model

# Manual LLM-as-judge pattern
async def eval_response(response: str, criteria: str) -> float:
    """Use a judge model to score a response."""
    judge_response = await judge_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": f"""Rate this response 0-10 based on: {criteria}

Response: {response}

Return only a number."""
        }],
    )
    return float(judge_response.content[0].text.strip())
```

### 8.7 Integration Testing Patterns

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_agent_loop():
    """Test the full agent loop with real API calls (or cassettes)."""
    agent = create_agent()

    result = await agent.run("Search for Python testing best practices")

    # Assert on observable behavior
    assert result.tool_calls_made > 0
    assert result.final_output is not None
    assert len(result.final_output) > 100

    # Assert on tool call structure
    for call in result.tool_history:
        assert call.tool_name in ALLOWED_TOOLS
        assert call.duration_ms < 30000  # No tool call took > 30s
```

### 8.8 Testing Recommendations for aiai

1. **Layer 1 (Fast, CI)**: Unit tests with mocked LLM responses. Test tool execution, routing logic, state management.
2. **Layer 2 (Recorded)**: VCR cassettes for integration tests. Record once with real API, replay in CI.
3. **Layer 3 (Periodic)**: Eval tests with real API calls. Run nightly or on release branches.
4. **Always**: Test tool schemas with property-based testing. Test error handling with edge cases.

**References:**
- [LLM Testing Practical Guide (Langfuse)](https://langfuse.com/blog/2025-10-21-testing-llm-applications)
- [VCR Tests for LLMs](https://anaynayak.medium.com/eliminating-flaky-tests-using-vcr-tests-for-llms-a3feabf90bc5)
- [pytest-recording GitHub](https://github.com/kiwicom/pytest-recording)
- [VCR.py Documentation](https://vcrpy.readthedocs.io/)
- [Mocking External APIs in Agent Tests](https://langwatch.ai/scenario/testing-guides/mocks/)
- [pytest-evals GitHub](https://github.com/AlmogBaku/pytest-evals)
- [Python Agent Testing Best Practices](https://dasroot.net/posts/2026/02/python-agent-testing-best-practices-tools/)

---

## 9. OpenRouter Python Integration

### 9.1 Overview

**OpenRouter** provides a unified API to 400+ AI models from dozens of providers through a single endpoint and API key. Its API is OpenAI-compatible, making integration trivial.

- **Docs**: https://openrouter.ai/docs/quickstart
- **API Base**: `https://openrouter.ai/api/v1`
- **Auth**: Bearer token via `OPENROUTER_API_KEY`

### 9.2 Using the OpenAI SDK (Recommended)

The simplest approach -- change base_url and api_key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<OPENROUTER_API_KEY>",
)

completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "https://your-app.com",    # Optional, for rankings
        "X-OpenRouter-Title": "Your App Name",      # Optional, for rankings
    },
    model="anthropic/claude-sonnet-4",
    messages=[
        {"role": "user", "content": "What is the meaning of life?"}
    ],
)
print(completion.choices[0].message.content)
```

### 9.3 Using httpx Directly

For more control, especially async:

```python
import httpx
import json

async def openrouter_call(prompt: str, model: str = "anthropic/claude-sonnet-4") -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app.com",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
```

### 9.4 Streaming

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<OPENROUTER_API_KEY>",
)

stream = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**Async streaming with httpx:**
```python
import httpx

async def stream_openrouter(prompt: str, model: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    content = data["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
```

### 9.5 Model Selection

```python
# Specific model
model = "anthropic/claude-sonnet-4"

# Auto-select best model for the prompt
model = "openrouter/auto"

# Some popular model identifiers:
# "anthropic/claude-sonnet-4"
# "anthropic/claude-haiku-4"
# "openai/gpt-4o"
# "google/gemini-2.0-flash"
# "meta-llama/llama-3.3-70b-instruct"
# "deepseek/deepseek-chat-v3"
```

### 9.6 Error Handling and Retries

```python
import httpx
import asyncio
from openai import OpenAI, APIError, RateLimitError

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="<OPENROUTER_API_KEY>",
)

async def robust_call(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="anthropic/claude-sonnet-4",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait = 2 ** attempt
            await asyncio.sleep(wait)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1)
    raise RuntimeError("Max retries exceeded")
```

### 9.7 Third-Party Python Client

An unofficial client with additional features:

```bash
pip install openrouter-python-client
```

Features: streaming support, automatic rate limiting, smart retries with exponential backoff, full type safety.

- **Repository**: https://github.com/dingo-actual/openrouter-python-client

### 9.8 Integration with Agent Frameworks

Most agent frameworks that support OpenAI can use OpenRouter by changing the base URL:

```python
# PydanticAI with OpenRouter
from pydantic_ai import Agent

agent = Agent(
    "openai:anthropic/claude-sonnet-4",  # Model via OpenRouter
    # Configure OpenAI client to use OpenRouter base_url
)

# OpenAI Agents SDK with OpenRouter
# The SDK supports custom model providers for non-OpenAI models
```

**References:**
- [OpenRouter Quickstart](https://openrouter.ai/docs/quickstart)
- [OpenRouter API Reference](https://openrouter.ai/docs/api/reference/overview)
- [OpenRouter Streaming](https://openrouter.ai/docs/api/reference/streaming)
- [OpenRouter Authentication](https://openrouter.ai/docs/api/reference/authentication)
- [OpenRouter Python SDK](https://openrouter.ai/docs/sdks/python)
- [OpenRouter OpenAI SDK Guide](https://openrouter.ai/docs/guides/community/openai-sdk)
- [OpenRouter in Python (Snyk)](https://snyk.io/articles/openrouter-in-python-use-any-llm-with-one-api-key/)
- [Unofficial Python Client](https://github.com/dingo-actual/openrouter-python-client)

---

## 10. Recommendations for aiai

### 10.1 Core Architecture Decision

For a self-improving AI infrastructure project, the recommended approach is a **layered architecture**:

```
Layer 4: Agent Orchestration (Claude Agent SDK / custom orchestrator)
Layer 3: Agent Framework   (PydanticAI for type-safe agents)
Layer 2: LLM Client        (anthropic SDK + OpenRouter for multi-model)
Layer 1: Async Runtime      (asyncio + httpx)
```

### 10.2 Specific Recommendations

| Concern | Recommendation | Rationale |
|---------|---------------|-----------|
| Primary LLM SDK | `anthropic` Python SDK | Direct, low-overhead, full feature access |
| Multi-model access | OpenRouter via OpenAI SDK | Single API for 400+ models, easy switching |
| Structured output | Instructor + Pydantic | Battle-tested, retry logic, multi-provider |
| Agent framework | PydanticAI for typed agents | Type safety, DI for testing, model-agnostic |
| Complex workflows | LangGraph (when needed) | Checkpointing, human-in-the-loop, durability |
| Autonomous agents | Claude Agent SDK | Full agent runtime, built-in tools, MCP |
| Async runtime | asyncio + Semaphore | Standard Python, rate limiting built-in |
| Testing | pytest + VCR.py + mocks | Three-layer testing pyramid |

### 10.3 Anti-Patterns to Avoid

1. **Over-abstracting early**: Do not build a framework before you have agent patterns to abstract
2. **Framework lock-in**: Use thin wrappers around LLM clients so you can swap providers
3. **Synchronous agents**: Always use async from the start; retrofitting is painful
4. **Ignoring structured output**: Raw text parsing is brittle; use Pydantic everywhere
5. **No testing strategy**: Set up VCR recording from day one
6. **Monolithic agent**: Prefer composable, single-purpose agents over one mega-agent

### 10.4 Quick-Start Stack for aiai

```python
# requirements.txt for initial aiai agent infrastructure
anthropic>=0.84.0
pydantic-ai>=1.0.0
instructor>=1.0.0
openai>=1.0.0          # For OpenRouter compatibility
httpx>=0.27.0
pydantic>=2.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
pytest-recording>=0.13.0
vcrpy>=6.0.0
```

This stack gives you:
- Direct Anthropic API access with streaming, caching, batching
- Type-safe agents via PydanticAI
- Structured output via Instructor
- Multi-model access via OpenRouter (through OpenAI SDK)
- Async-first architecture
- Comprehensive testing infrastructure
