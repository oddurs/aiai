# AI Infrastructure, Tools, and Trends Landscape

**Compiled:** February 26, 2026
**Purpose:** Snapshot of the AI infrastructure ecosystem for the aiai self-improving infrastructure project.

---

## Table of Contents

1. [MCP (Model Context Protocol)](#1-mcp-model-context-protocol)
2. [A2A (Agent-to-Agent Protocol)](#2-a2a-agent-to-agent-protocol)
3. [The MCP + A2A Combined Stack](#3-the-mcp--a2a-combined-stack)
4. [Claude Code Architecture](#4-claude-code-architecture)
5. [AI-Powered IDEs: Cursor, Windsurf, and the Agentic IDE](#5-ai-powered-ides-cursor-windsurf-and-the-agentic-ide)
6. [OpenAI Codex Agent](#6-openai-codex-agent)
7. [AI Infrastructure Companies](#7-ai-infrastructure-companies)
8. [Open-Source Model Landscape](#8-open-source-model-landscape)
9. [The Agent Economy](#9-the-agent-economy)
10. [Key Takeaways for aiai](#10-key-takeaways-for-aiai)

---

## 1. MCP (Model Context Protocol)

### Overview

MCP is an open protocol created by Anthropic (first released November 2024) that standardizes how LLM applications connect to external data sources, tools, and services. It has become the universal standard for agent-tool interoperability. In December 2025, Anthropic donated MCP to the [Agentic AI Foundation (AAIF)](https://aaif.io/) under the Linux Foundation, with platinum members including AWS, Anthropic, Block, Bloomberg, Cloudflare, Google, Microsoft, and OpenAI.

**Key milestone:** 97M+ monthly SDK downloads as of early 2026.

Sources:
- [MCP Specification (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25)
- [One Year of MCP](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/)
- [MCP Wikipedia](https://en.wikipedia.org/wiki/Model_Context_Protocol)
- [MCP joins AAIF](http://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/)
- [AAIF Announcement](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)

### Architecture

MCP uses a **client-server architecture** inspired by the Language Server Protocol (LSP), with **JSON-RPC 2.0** as the wire format.

```
+-------------------+       +-------------------+       +-------------------+
|   Host (IDE,      |       |   MCP Client      |       |   MCP Server      |
|   Agent, App)     | <---> |   (Protocol       | <---> |   (Exposes tools, |
|                   |       |    Handler)        |       |    resources,     |
+-------------------+       +-------------------+       |    prompts)       |
                                                        +-------------------+
```

- **Host**: The application (IDE, agent framework, chatbot) that needs AI capabilities.
- **MCP Client**: Lives inside the host; handles protocol negotiation, message routing, and capability management.
- **MCP Server**: Exposes tools, resources, and prompts to the client. Can be local (stdio) or remote (HTTP).

### Transport Layers

MCP defines two transport mechanisms (as of the 2025-11-25 spec):

| Transport | Use Case | Details |
|-----------|----------|---------|
| **stdio** | Local integrations, CLI tools | Communication via stdin/stdout. Simple, low-latency. Not for production remote deployments. |
| **Streamable HTTP** | Production remote deployments | HTTP POST for client-to-server; optional SSE for server-to-client streaming. Replaced the deprecated standalone SSE transport (March 2025). Supports event IDs for connection resumption. |

The **SSE-only transport** was deprecated in protocol version 2025-03-26, replaced by Streamable HTTP which incorporates SSE as an optional streaming mechanism within HTTP responses.

Sources:
- [MCP Transports](https://modelcontextprotocol.io/legacy/concepts/transports)
- [Why MCP Deprecated SSE](https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/)
- [MCP Spec Updates June 2025 (Auth)](https://auth0.com/blog/mcp-specs-update-all-about-auth/)

### Core Primitives

MCP is built around four core primitives:

**Tools** -- Executable functions that LLMs can invoke:
```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": { "type": "string", "description": "City name" }
    },
    "required": ["city"]
  }
}
```
- Discovery: `tools/list` endpoint
- Invocation: `tools/call` endpoint
- Uses JSON Schema for input validation

**Resources** -- Read-only data exposed by the server (files, database records, API responses). Agents can read/reference them but not execute them.

**Prompts** -- Reusable instruction templates exposed by the server. They describe how a model should behave for a given task without performing any action.

**Roots** -- Define the scope boundary. Anything outside the root is invisible to the client.

Additional capabilities:
- **Sampling**: Allows servers to request LLM completions through the client, enabling agentic behaviors where servers can leverage AI capabilities.
- **Elicitation**: Enables servers to request structured information from users through the client.

Sources:
- [MCP Tools Spec](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
- [MCP Primitives Guide](https://portkey.ai/blog/mcp-primitives-the-mental-model-behind-the-protocol/)
- [Understanding MCP Features](https://workos.com/blog/mcp-features-guide)

### Ecosystem Scale (February 2026)

| Metric | Number |
|--------|--------|
| Official MCP Registry servers | ~518 (grew from 90 to 518 in one month) |
| Unofficial registries (e.g., mcp.so) | 16,000+ servers indexed |
| Total ecosystem implementations | ~20,000 MCP servers |
| Monthly SDK downloads | 97M+ |
| Major adopters | OpenAI, Google, Microsoft, Anthropic, Cloudflare, AWS |

The registry launched in preview September 2025 and is progressing toward general availability. At the current growth rate, the official registry is projected to exceed 1,000 servers by summer 2026.

Sources:
- [MCP Registry GitHub](https://github.com/modelcontextprotocol/registry)
- [Official MCP Registry](https://registry.modelcontextprotocol.io/)
- [Security Audit of 518 Servers](https://earezki.com/ai-news/2026-02-21-i-scanned-every-server-in-the-official-mcp-registry-heres-what-i-found/)
- [MCP Enterprise Adoption Guide](https://guptadeepak.com/the-complete-guide-to-model-context-protocol-mcp-enterprise-adoption-market-trends-and-implementation-strategies/)

---

## 2. A2A (Agent-to-Agent Protocol)

### Overview

The Agent2Agent (A2A) protocol was introduced by Google in April 2025 as an open protocol for communication between AI agents across different providers and frameworks. While MCP connects agents to tools, A2A connects agents to each other.

In September 2025, IBM announced that its Agent Communication Protocol (ACP) would officially merge with A2A under the Linux Foundation's LF AI & Data umbrella. This was a true convergence: ACP's RESTful simplicity was preserved while incorporating A2A's enterprise features like Agent Cards and task lifecycle management.

Sources:
- [A2A Announcement (Google)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [ACP Joins Forces with A2A](https://lfaidata.foundation/communityblog/2025/08/29/acp-joins-forces-with-a2a-under-the-linux-foundations-lf-ai-data/)
- [A2A Protocol Specification](https://a2a-protocol.org/latest/specification/)
- [IBM on A2A](https://www.ibm.com/think/topics/agent2agent-protocol)

### Architecture

A2A communication occurs over **HTTP(S)**, with all payloads formatted using **JSON-RPC 2.0**.

```
+-------------------+       HTTPS / JSON-RPC 2.0       +-------------------+
|   A2A Client      | <------------------------------> |   A2A Server      |
|   (Requesting     |                                  |   (Remote Agent)  |
|    Agent)         |   Agent Card (Discovery)         |                   |
|                   |   Messages (Communication)       |   Skills          |
|                   |   Tasks (Work Units)             |   Capabilities    |
+-------------------+   Artifacts (Outputs)            +-------------------+
```

### Core Concepts

**Agent Card**: A JSON document that every A2A server MUST make available. It describes the agent's identity, capabilities, skills, service endpoint URL, and authentication requirements. Think of it as a "resume" for an agent.

**Task Lifecycle**: The fundamental unit of work with a defined state machine:
```
submitted -> working -> input-required -> completed
                    \-> failed
                    \-> canceled
```

**Messages**: Communication turns between client and remote agent, with roles "user" or "agent", containing one or more Parts.

**Parts**: The smallest unit of content (TextPart, FilePart, DataPart).

**Artifacts**: Output objects produced by a task (generated files, data, etc.).

### Interaction Patterns

| Pattern | Description |
|---------|-------------|
| Synchronous | Standard request/response |
| Streaming (SSE) | Real-time incremental updates for task status and artifact chunks |
| Push Notifications | Async task updates via webhook (HTTP POST to client-provided URL) for long-running/disconnected scenarios |

### Governance (Post-Merger)

The A2A Technical Steering Committee now includes representatives from Google, Microsoft, AWS, Cisco, Salesforce, ServiceNow, SAP, and IBM (via Kate Blair from ACP). Development continues under open governance at the Linux Foundation.

Sources:
- [A2A Core Concepts](https://a2a-protocol.org/latest/topics/key-concepts/)
- [A2A GitHub](https://github.com/a2aproject/A2A)
- [A2A Protocol Explained (HuggingFace)](https://huggingface.co/blog/1bo/a2a-protocol-explained)
- [A2A DeepLearning.AI Course](https://www.deeplearning.ai/short-courses/a2a-the-agent2agent-protocol/)

---

## 3. The MCP + A2A Combined Stack

MCP and A2A are **complementary**, not competing. Together they form the emerging "agent protocol stack":

```
+-------------------------------------------------------+
|                   Application Layer                    |
|  (Agent orchestration, business logic, UX)             |
+-------------------------------------------------------+
|                   A2A Layer                             |
|  Agent-to-Agent communication, discovery, task mgmt    |
|  (Agent Cards, Task lifecycle, inter-agent messaging)  |
+-------------------------------------------------------+
|                   MCP Layer                             |
|  Agent-to-Tool/Data connectivity                       |
|  (Tools, Resources, Prompts, Sampling)                 |
+-------------------------------------------------------+
|                   Transport Layer                       |
|  HTTP(S), JSON-RPC 2.0, SSE, stdio                    |
+-------------------------------------------------------+
```

**How they interact in practice:**
- An agent uses **MCP** internally to connect to its tools, databases, and data sources.
- The same agent uses **A2A** externally to communicate with other agents, advertise its skills via Agent Cards, delegate tasks, and receive results.
- A "purchasing concierge" agent might use A2A to discover and negotiate with a "seller agent", while each agent internally uses MCP to access their respective inventory databases and payment tools.

One blog describes this as the "TCP/IP moment for agentic AI" -- a layered protocol stack that could become as foundational as networking protocols.

Sources:
- [A2A and MCP (Official)](https://a2a-protocol.org/latest/topics/a2a-and-mcp/)
- [The Agent Protocol Stack](https://subhadipmitra.com/blog/2026/agent-protocol-stack/)
- [Architecting Agentic MLOps with A2A and MCP](https://www.infoq.com/articles/architecting-agentic-mlops-a2a-mcp/)
- [MCP vs A2A Guide](https://auth0.com/blog/mcp-vs-a2a/)

---

## 4. Claude Code Architecture

### Overview

Claude Code is Anthropic's CLI-based coding agent. The **Claude Agent SDK** (renamed from "Claude Code SDK" in September 2025) is the open-source, production-grade framework that exposes the same infrastructure powering Claude Code as a programmable library for building autonomous AI agents in Python and TypeScript.

As of February 2026:
- Python SDK: v0.1.34
- TypeScript SDK: v0.2.37 (1.85M+ weekly downloads)

Sources:
- [Building Agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Claude Agent SDK on PyPI](https://pypi.org/project/claude-agent-sdk/)
- [Claude Code Sub-Agents Docs](https://code.claude.com/docs/en/sub-agents)

### Core Architecture: The Agent Loop

Claude Code operates on an iterative feedback loop:

```
  +-> Gather Context (read files, search, inspect environment)
  |         |
  |         v
  |   Plan / Reason (decide what to do next)
  |         |
  |         v
  |   Take Action (edit files, run commands, call tools)
  |         |
  |         v
  +-- Verify Work (check outputs, run tests, review changes)
```

The philosophy is "give agents a computer, not just a prompt." Claude Code has direct, controlled access to a terminal, file system, and the web. Context management features like **compaction** enable agents to work on long tasks without exhausting the context window.

### Built-in Tools (18 tools as of v2.1.59)

Claude Code's system prompt (2,896 tokens core) includes 18 built-in tool descriptions, plus specialized sub-agent prompts:

| Sub-Agent | Tokens | Purpose |
|-----------|--------|---------|
| Plan | 633 | Strategic planning and task decomposition |
| Explore | 516 | Codebase exploration and understanding |
| Task | 294 | Parallel task execution |

The full system prompt, tool descriptions, and sub-agent prompts have been reverse-engineered and documented in community repositories.

Sources:
- [Claude Code System Prompts (Community)](https://github.com/Piebald-AI/claude-code-system-prompts)
- [Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)

### Multi-Agent Coordination: TeammateTool and Agent Teams

Claude Code supports multi-agent orchestration with two primary mechanisms:

**Task Tool** (foundational parallel processing):
- Up to 10 concurrent tasks with intelligent queuing
- Each task gets its own isolated context window
- Parent agent spawns worker tasks, collects results

**TeammateTool** (collaborative agent teams):
- One Claude Code session becomes the **team lead**
- It spawns **teammates** -- each a full, independent Claude Code instance with its own context window
- Communication via an **inbox-based messaging system** (SendMessage)
- Shared task list with **dependency tracking** and automatic unblocking
- File locking prevents double-claiming of tasks
- Teammates self-claim or get assigned by the lead

Key difference from sub-agents: sub-agents are focused workers that report back to a single parent and cannot talk to each other. Agent teams enable actual collaboration -- teammates share findings, challenge approaches, and coordinate independently.

### Memory System

- **CLAUDE.md**: A special file automatically read by Claude Code. Serves as persistent, project-specific memory for coding standards, architecture, and key workflows.
- **Shared planning documents**: Agents coordinate through PLAN.md or ISSUE.md files as a central source of truth.
- **Conversation history**: Teammates load project context (CLAUDE.md, MCP servers, skills) but do NOT inherit the lead's conversation history.

### Hooks System

Hooks execute custom scripts at key lifecycle events:
- `PreToolUse` / `PostToolUse` -- enforce standards or automate testing around tool invocations
- `SubagentStop` -- post-processing after sub-agent completion
- Economic viability checks (pre-agent-spawn hooks) can gate whether the 15x token cost of multi-agent is justified

### Git Worktrees

Git worktrees provide isolated branches for multiple agents working on the same codebase simultaneously. Each agent works in its own worktree to prevent merge conflicts.

Sources:
- [Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams)
- [Claude Code Swarm Orchestration (Gist)](https://gist.github.com/kieranklaassen/4f2aba89594a4aea4ad64d753984b2ea)
- [Task Tool Architecture](https://dev.to/bhaidar/the-task-tool-claude-codes-agent-orchestration-system-4bf2)
- [Claude Code Swarms (Addy Osmani)](https://addyosmani.com/blog/claude-code-agent-teams/)

---

## 5. AI-Powered IDEs: Cursor, Windsurf, and the Agentic IDE

### Cursor

**Status:** 1M+ daily active developers, $1B+ ARR, $29.3B valuation (late 2025).

Cursor is a full fork of VS Code with AI as a core architectural component. Key technical details:

**Codebase Indexing:**
- Automatic project scanning computes a **Merkle tree of file hashes** for efficient change tracking
- Files are chunked into meaningful units (functions, classes, logical blocks) rather than arbitrary segments
- Custom embedding model for semantic search across large codebases
- Index updates are incremental -- only changed portions are re-indexed

**Modes of Operation:**

| Mode | Description |
|------|-------------|
| **Tab** (Autocomplete) | Custom model trained for code completion. Predicts entire diffs (not just insertions). Considers recent changes and linter errors. Sub-200ms latency. |
| **Chat** (Cmd+L) | Contextual Q&A about your codebase |
| **Composer/Agent** | Multi-file autonomous editing. Cursor 2.0's Composer is a mixture-of-experts model trained via RL in real codebases. Uses semantic search, file editors, terminal commands. Subagents run in parallel. |

**Cursor 2.0 (Late 2025):** Introduced Composer as a specialized MoE model that learned to use development tools through reinforcement learning, picking up behaviors like running tests, fixing linter errors, and navigating large projects.

Sources:
- [Cursor Product](https://cursor.com/product)
- [Cursor Features](https://cursor.com/features)
- [Cursor 2.0 Explained](https://www.codecademy.com/article/cursor-2-0-new-ai-model-explained)
- [Cursor vs Windsurf vs Claude Code 2026](https://dev.to/pockit_tools/cursor-vs-windsurf-vs-claude-code-in-2026-the-honest-comparison-after-using-all-three-3gof)

### Windsurf (formerly Codeium)

A standalone IDE (also VS Code-based) built around the AI-first philosophy. Key differentiator is **Cascade**, its signature agentic system.

**Cascade Agent:**
- Hierarchical context system that plans changes across entire repositories
- Reasons about module dependencies and shared types before generating edits
- Supports iterative debugging: run terminal commands, analyze errors, try fixes until completion
- AI Flows: generates/modifies code, asks for approval, runs in terminal, asks follow-up questions

**Tab Completion:**
- Powered by SWE-1-mini (in-house model designed for speed)
- Uses broader context than Cursor: terminal history, Cascade chat, recent editor actions, clipboard content

**Fast Context (SWE-grep):**
- Retrieves relevant code context 10x faster than traditional agentic search
- 8 parallel tool calls per turn across 4 turns

**Memory:** Persistent knowledge layer that learns coding style, patterns, and APIs.

**Large Codebase Handling:** Generally better than Cursor for enterprise-scale projects (millions of lines) due to automatic context indexing through Cascade.

Sources:
- [Windsurf Cascade](https://windsurf.com/cascade)
- [Windsurf vs Cursor Comparison](https://windsurf.com/compare/windsurf-vs-cursor)
- [Windsurf Review 2026](https://www.secondtalent.com/resources/windsurf-review/)

### Architectural Lessons for aiai

1. **Semantic indexing is essential**: Both Cursor and Windsurf invest heavily in understanding code structure, not just text.
2. **Multiple interaction modes**: Tab completion (fast, narrow), chat (exploratory), agent (autonomous). Each needs different models and context strategies.
3. **Context management is the hard problem**: Merkle trees, incremental indexing, chunking strategies, and embedding models are all needed at scale.
4. **RL-trained tool-use models outperform prompted ones**: Cursor's Composer demonstrates that models trained in real development environments develop more practical behaviors.
5. **Parallel sub-agents are standard**: Both IDEs use parallel workers to explore and modify codebases.

---

## 6. OpenAI Codex Agent

### Overview

The 2025-2026 Codex is not the old GPT-3 era code completion API. It is a full autonomous coding agent, now powered by **GPT-5.3-Codex** (the most capable agentic coding model to date, 25% faster than GPT-5.2-Codex).

Sources:
- [Introducing Codex](https://openai.com/index/introducing-codex/)
- [Introducing the Codex App](https://openai.com/index/introducing-the-codex-app/)
- [Unrolling the Codex Agent Loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [GPT-5.3-Codex](https://openai.com/index/introducing-gpt-5-3-codex/)

### Execution Modes

| Mode | Description |
|------|-------------|
| **Cloud Sandboxes** | Parallel background tasks in isolated containers. Ideal for generating PRs. Internet access disabled during execution. |
| **CLI (Local)** | Interactive, approval-controlled. Three levels: Suggest, Auto Edit, Full Auto. |

### Sandboxing

Codex runs in a sandboxed environment **with network access disabled by default** (both locally and in the cloud). The cloud agent operates entirely within a secure, isolated container. During task execution, the agent's interaction is limited solely to code provided via GitHub repositories and pre-installed dependencies configured via a setup script.

**Important:** The sandboxing applies only to the Codex-provided shell tool. MCP server tools are NOT sandboxed by Codex and must enforce their own guardrails.

### Agent Loop Architecture

OpenAI published a detailed technical deep-dive (January 23, 2026) on the agent loop:

```
User Request
    |
    v
[Model Inference] <---+
    |                  |
    v                  |
[Tool Call?] --yes--> [Execute Tool (shell, file edit, etc.)]
    |                  |
    no                 |
    |                  |
    v                  |
[Assistant Message] ---+  (loops back if more work needed)
    |
    v
[Termination State]
```

- Uses the **Responses API** for model inference
- Supports **prompt caching** for linear-time model sampling
- API endpoints vary by auth: `chatgpt.com/backend-api/codex/responses` (ChatGPT login) or `api.openai.com/v1/responses` (API key)
- Zero Data Retention configurations available for regulated industries
- Supports `--oss` flag pointing to `localhost:11434/v1/responses` for local models

### Product Vision

Codex is evolving from "pairing with a single agent" to "supervising coordinated teams of agents across the full software lifecycle" (design, build, ship, maintain).

### Comparison to Claude Code

| Feature | Claude Code | Codex |
|---------|-------------|-------|
| Runtime | CLI + Agent SDK | Cloud sandboxes + CLI |
| Multi-agent | TeammateTool, agent teams | Multi-agent coordination (newer) |
| Sandboxing | Per-tool permissions, approval modes | Network-disabled containers |
| Open source | Agent SDK is open source | CLI is open source |
| MCP support | Native | Supported (tools not sandboxed) |
| Model | Claude Opus 4.6 / Sonnet 4.6 | GPT-5.3-Codex |
| Context mgmt | Compaction | Prompt caching |

---

## 7. AI Infrastructure Companies

### Sandbox / Code Execution Platforms

#### E2B

The leading sandbox platform for AI agents. Uses **Firecracker microVMs** (same tech as AWS Lambda).

| Feature | Details |
|---------|---------|
| Startup time | <200ms (no cold starts) |
| Isolation | Hardware-level (each sandbox runs own VM + kernel) |
| Session limits | Up to 24 hours (Pro plan) |
| Concurrent sandboxes | 20 (Hobby), more on Pro/Enterprise |
| Pricing | ~$0.05/hour per 1 vCPU sandbox, billed per second |
| SDKs | Python, JavaScript/TypeScript |
| Plans | Hobby (free, $100 credit), Pro ($150/mo), Enterprise (custom, BYOC/self-hosted) |

Sources:
- [E2B Documentation](https://e2b.dev/docs)
- [E2B Pricing](https://e2b.dev/pricing)
- [E2B GitHub](https://github.com/e2b-dev/E2B)

#### Modal

Serverless cloud platform for AI workloads. Differentiator: **GPU access** from T4 through H200.

| Feature | Details |
|---------|---------|
| Cold start | Sub-second |
| Scaling | Zero to 10,000+ concurrent units |
| Isolation | gVisor-based |
| GPU pricing | H100: $4.56/hr |
| Multi-node | Up to 64 H100 GPUs across nodes |
| Billing | Per-second |
| Use cases | Inference, training, batch compute, sandboxes |

Modal sandboxes sit inside a wider platform covering inference, training, and batch compute, whereas E2B is purpose-built for untrusted code execution.

Sources:
- [Modal Pricing](https://modal.com/pricing)
- [Modal](https://modal.com/)
- [E2B vs Modal Comparison](https://northflank.com/blog/e2b-vs-modal)

#### Together AI / CodeSandbox

Together AI's GPU cloud now includes **snapshot-based sandboxes** (via CodeSandbox acquisition):
- VM snapshot resume in 500ms (2.7s cold)
- Memory already loaded on resume
- Branch environments from same base state
- Run agents in parallel, restore VMs in under two seconds

Source: [Together AI Sandbox](https://northflank.com/blog/e2b-vs-modal-vs-fly-io-sprites)

### Inference Infrastructure

#### Groq

Specialized inference hardware using **on-chip SRAM** instead of external memory.

| Feature | Details |
|---------|---------|
| Inference speed | 500-750 tokens/sec (vs ~100 for standard GPUs) |
| Pricing | ~$0.59/M input tokens, ~$0.79/M output tokens (Llama 3 70B) |
| Status | **Acquired by Nvidia** (December 24, 2025) for $20B licensing deal |

Source: [Groq Pricing](https://groq.com/pricing), [Nvidia-Groq Deal](https://www.kavout.com/market-lens/nvidia-s-20-billion-groq-deal-what-it-means-for-ai-in-2026)

#### Cerebras

Wafer-scale inference chips claiming speeds up to **20x faster** than traditional GPU systems.

| Feature | Details |
|---------|---------|
| Pricing | ~$0.25/M input, ~$0.69/M output tokens |
| Hardware cost | $2-3M per CS-3 system |
| Major deal | **$10B inference deal with OpenAI** (January 2026), installing machines across US datacenters starting Q1 2026 |

Source: [Cerebras-OpenAI Deal](https://www.nextplatform.com/ai/2026/01/15/cerebras-inks-transformative-10-billion-inference-deal-with-openai/4092155)

#### Market Context (GPU Pricing, Early 2026)

- Cloud H100 prices have dropped 64-75% from peak levels
- Stabilizing at $2.85-$3.50/hour
- Expected 10-20% further decreases when B200 GPUs become widely available in 2026

Source: [GPU Economics 2026](https://dev.to/kaeltiwari/gpu-economics-what-inference-actually-costs-in-2026-2goo)

---

## 8. Open-Source Model Landscape

### Model Comparison (February 2026)

#### DeepSeek

| Model | Parameters | Key Capability | Notes |
|-------|-----------|----------------|-------|
| DeepSeek-R1 (0528) | 671B MoE | Reasoning | 2nd-highest on AIME (behind o3). 98% lower cost than comparable models. |
| DeepSeek-V3.1 | 671B MoE | Tool use, agentic | Stronger tool usage than V3 and R1 in code/search agent benchmarks. |
| DeepSeek-V3.2 | 671B MoE | Agentic + reasoning | First model to integrate thinking into tool-use. 1,800+ environments, 85k+ complex instructions in training. Comparable to GPT-5. |
| DeepSeek-V3.2-Speciale | 671B MoE | Deep reasoning only | Surpasses GPT-5, on par with Gemini-3.0-Pro. IMO and IOI gold medals. Does NOT support tool calling. |

**V3.2 Innovation -- Thinking Retention Mechanism:** Traditional tool-using models discard their reasoning trace with each iteration. V3.2 retains thinking across tool calls, eliminating repeated re-reasoning.

Sources:
- [DeepSeek V3.2 on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
- [Complete Guide to DeepSeek Models](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond)
- [Technical Tour of DeepSeek](https://magazine.sebastianraschka.com/p/technical-deepseek)

#### Qwen 3

| Model | Parameters | Key Capability | Notes |
|-------|-----------|----------------|-------|
| Qwen3-235B-A22B | 235B MoE (22B active) | General reasoning | Meets/beats GPT-4o on most benchmarks. 262K context. |
| Qwen3-Max-Thinking | Flagship | High-stakes reasoning | $0.55/M input, $3.50/M output on OpenRouter |
| Qwen3-Coder-Next | 80B MoE (3B active) | Coding agents | 69.6% on SWE-bench. Long-horizon reasoning, complex tool usage, recovery from execution failures. |

- 119 languages supported
- 92.3% accuracy on AIME25
- Hybrid MoE architecture: far less compute than dense alternatives

Sources:
- [Qwen on OpenRouter](https://openrouter.ai/qwen)
- [Top Open Source LLMs (HuggingFace)](https://huggingface.co/blog/daya-shankar/open-source-llms)

#### Llama 4

| Model | Parameters | Context | Notes |
|-------|-----------|---------|-------|
| Llama 4 Scout | 109B total (17B active) | 10M tokens | MoE with massive context window |
| Llama 4 Maverick | 400B total (17B active, 128 experts) | 1M tokens | Broader capabilities |

Best for: General chat, agents, broad ecosystem support. Strong ecosystem backing from Meta.

Source: [Top 9 LLMs Feb 2026](https://www.shakudo.io/blog/top-9-large-language-models)

#### Mistral

| Model | Parameters | Key Capability | Notes |
|-------|-----------|----------------|-------|
| Mistral Large 3 | 675B total (41B active) MoE | General + coding | 256K context. Function calling, structured output, agents. |
| Devstral 2 | 123B | Coding agents | 72.2% on SWE-bench Verified. Up to 7x more cost-efficient than Claude Sonnet. |
| Devstral Small 2 | 24B | Local coding | Smaller footprint for local development |
| Ministral 3 | 3B/8B/14B dense | Edge/mobile | Base, Instruct, and Reasoning variants |

Mistral also launched **Mistral Vibe CLI** -- a native CLI for Devstral enabling end-to-end code automation using natural language with ACP (Agent Communication Protocol) integration.

Sources:
- [Mistral Large 3](https://mistral.ai/news/mistral-3)
- [Devstral 2](https://mistral.ai/news/devstral-2-vibe-cli)
- [Mistral on OpenRouter](https://openrouter.ai/mistralai)

### SWE-bench Verified Leaderboard (February 2026)

Top performers on the agentic coding benchmark (500 real-world GitHub issues):

1. Claude 4.5 Opus (Anthropic)
2. Gemini 3 Flash (Google)
3. MiniMax M2.5 (229B)
4. GLM-5
5. Kimi K2.5
6. GPT-5.2 (OpenAI)
7. DeepSeek V3.2

**Notable:** Sonar Foundation Agent achieved 79.2% on SWE-bench Verified (highest unfiltered score) using Claude Opus 4.5 as the underlying model.

Sources:
- [SWE-bench Verified (Epoch AI)](https://epoch.ai/benchmarks/swe-bench-verified)
- [SWE-bench February 2026 Update](https://simonwillison.net/2026/Feb/19/swe-bench/)
- [Sonar Top Spot](https://www.sonarsource.com/company/press-releases/sonar-claims-top-spot-on-swe-bench-leaderboard/)

### OpenRouter Pricing Snapshot (February 2026)

| Model | Input $/M tokens | Output $/M tokens |
|-------|------------------|--------------------|
| DeepSeek-V3.2 | $0.25 | $0.40 |
| Qwen3-Max-Thinking | $0.55 | $3.50 |
| Qwen3 Next 80B A3B Instruct | $0.21 | $0.79 |
| Llama 3.3 70B | Free | Free |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Gemini 3.1 Pro | $2.00 | $12.00 |

**Key insight for aiai:** Open-source models via OpenRouter are 10-60x cheaper than proprietary models. For many agentic tasks (code search, file editing, test running), DeepSeek V3.2 or Qwen3 at $0.25-0.55/M input tokens may be viable alternatives to Claude/GPT for sub-tasks, while keeping frontier models for complex reasoning.

Sources:
- [OpenRouter Models](https://openrouter.ai/models)
- [OpenRouter Pricing](https://openrouter.ai/pricing)
- [Free Models on OpenRouter](https://openrouter.ai/collections/free-models)

---

## 9. The Agent Economy

### Agentic Commerce

The global agentic AI market is projected to grow from ~$5B (2024) to ~$200B by 2034 (40%+ CAGR). The AI agents market specifically is expected to reach $80-100B by 2030.

Real-world deployments in 2026:
- **Google** launched agentic checkout across Search (AI Mode) and Gemini with "Buy for me" functionality
- **Amazon Rufus** is driving $10B in annualized sales
- Brands are preparing for consumers directing attention to GenAI channels instead of eCommerce sites

Sources:
- [AI Trends Shaping Agentic Commerce](https://commercetools.com/blog/ai-trends-shaping-agentic-commerce)
- [McKinsey: Agentic Commerce Opportunity](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-agentic-commerce-opportunity-how-ai-agents-are-ushering-in-a-new-era-for-consumers-and-merchants)
- [MIT Sloan: Platforms in 2026](https://mitsloan.mit.edu/ideas-made-to-matter/ai-agents-tech-circularity-whats-ahead-platforms-2026)

### Moltbook and the Agent Social Economy

**Moltbook** launched January 28, 2026, as a Reddit-style social network exclusively for AI agents. Humans can only observe.

| Metric | Number |
|--------|--------|
| Launch to first 36K agents | 72 hours |
| Total agents (Feb 2026) | 2.5M+ |
| Economic rails | Base (Coinbase L2) |

**Related platforms:**
- **MoltGig**: Agent-to-agent marketplace on Base with escrow-backed payments and fully autonomous operations
- **MoltMart**: Marketplace for AI agent services
- **OpenClaw**: Framework enabling humans to deploy agents into the Moltbook ecosystem

**Infrastructure:**
- **x402**: HTTP-native payments. Agents pay agents directly in USDC. No invoices, no accounts, no humans.
- **ERC-8004**: Gives every agent a verifiable on-chain identity for accountability.

**State of agent-to-agent commerce (Feb 2026):** The infrastructure exists and is operational. 2.5M+ agents are on Moltbook. Autonomous payments via x402/USDC are live. However, the "agent economy" is still mostly agents interacting within crypto/Base ecosystem. True cross-domain agent-to-agent commerce (e.g., one AI hiring another to perform software engineering work and paying for it) is technically possible but not yet mainstream.

Sources:
- [What is Moltbook? (DigitalOcean)](https://www.digitalocean.com/resources/articles/what-is-moltbook)
- [Moltbook and the Agent Economy (Incode)](https://www.incode.com/blog/moltbook-agent-economy)
- [MoltGig](https://moltgig.com/)
- [Molt Ecosystem](https://www.moltecosystem.xyz/)

### Agent Marketplaces and Directories

- [AI Agent Store](https://aiagentstore.ai) -- Directory and marketplace for AI agents
- Microsoft Marketplace now lists AI recruitment agents (e.g., AI/R's Llia)
- Agent directories are emerging as the "app stores" for AI

---

## 10. Key Takeaways for aiai

### What the Infrastructure Stack Looks Like (February 2026)

```
+------------------------------------------------------------------+
|                     AGENT ECONOMY LAYER                           |
|  Moltbook, MoltGig, x402 payments, ERC-8004 identity             |
+------------------------------------------------------------------+
|                     ORCHESTRATION LAYER                           |
|  Claude Code teams, Codex multi-agent, custom orchestrators      |
+------------------------------------------------------------------+
|                     PROTOCOL LAYER                                |
|  A2A (agent-agent) + MCP (agent-tools) + AAIF governance         |
+------------------------------------------------------------------+
|                     MODEL LAYER                                   |
|  Frontier: Claude 4.5 Opus, GPT-5.3, Gemini 3                   |
|  Open: DeepSeek V3.2, Qwen3, Llama 4, Devstral 2                |
|  Routing: OpenRouter, model selection per task                   |
+------------------------------------------------------------------+
|                     COMPUTE LAYER                                 |
|  Sandboxes: E2B, Modal     | Inference: Groq/Nvidia, Cerebras   |
|  GPU Cloud: Modal, Lambda  | Serverless: Modal, RunPod          |
+------------------------------------------------------------------+
```

### Actionable Insights

1. **MCP is the standard. Adopt it fully.** With 97M+ monthly SDK downloads, 20K+ servers, and backing from every major player (now under Linux Foundation governance), MCP is not optional. Every tool and capability aiai exposes should be an MCP server.

2. **A2A is the next layer.** If aiai agents need to communicate with external agents or advertise their capabilities, A2A Agent Cards and task lifecycle management are the standard way to do it. The ACP merger means one unified protocol.

3. **Multi-agent is production-ready.** Claude Code's TeammateTool, Codex's multi-agent, and Cursor's parallel subagents all prove that multi-agent coordination works. Key patterns: shared task lists with dependency tracking, inbox-based messaging, file locking, economic viability checks before spawning.

4. **Open-source models are viable for sub-tasks.** DeepSeek V3.2 ($0.25/M input) and Qwen3 ($0.21-0.55/M input) are 10-60x cheaper than frontier models. V3.2's "thinking with tools" innovation is especially relevant for agentic use. Route complex reasoning to frontier models; route mechanical tasks to open-source.

5. **Sandboxing is solved infrastructure.** E2B (Firecracker microVMs, <200ms start, $0.05/hr) provides the isolation needed for agent code execution. No need to build this -- use it as infrastructure.

6. **The agent economy is nascent but real.** 2.5M agents on Moltbook, USDC payments, on-chain identity. Watch this space, but it is too early and too crypto-native to depend on for aiai's core infrastructure.

7. **Context management is the competitive moat.** Every successful system (Cursor, Windsurf, Claude Code, Codex) invests heavily in: semantic indexing, incremental updates, context compaction, thinking retention (DeepSeek V3.2), and prompt caching.

8. **The IDE/agent convergence is happening.** Cursor (1M DAU, $29.3B valuation), Windsurf, Claude Code, and Codex are all converging on the same architecture: LLM + tools + sandbox + multi-agent. The differentiator is increasingly the quality of context management and model routing.

---

*This document is a point-in-time snapshot. The AI infrastructure landscape is evolving rapidly. Key dates to watch: MCP Dev Summit (April 2-3, 2026, NYC), B200 GPU availability (mid-2026), and continued AAIF standardization efforts.*
