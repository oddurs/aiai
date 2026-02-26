# Agent Memory, Knowledge Management, and Context Systems (2024-2026)

> Research compiled for **aiai** -- a self-improving AI infrastructure project.
> Last updated: 2026-02-26

---

## Table of Contents

1. [RAG Architectures for Agents](#1-rag-architectures-for-agents)
2. [Vector Databases and Embedding Stores](#2-vector-databases-and-embedding-stores)
3. [Structured vs Unstructured Memory](#3-structured-vs-unstructured-memory)
4. [Context Window Management](#4-context-window-management)
5. [Long-Term Learning in Agents](#5-long-term-learning-in-agents)
6. [Knowledge Graphs for Code](#6-knowledge-graphs-for-code)
7. [Memory Safety and Privacy](#7-memory-safety-and-privacy)
8. [Recommendations for aiai](#8-recommendations-for-aiai)

---

## 1. RAG Architectures for Agents

### 1.1 From Chatbot RAG to Agentic RAG

Traditional RAG follows a static pipeline: embed query, retrieve top-k chunks, stuff them
into a prompt, generate an answer. **Agentic RAG** fundamentally changes this by embedding
autonomous AI agents into the retrieval pipeline. The agent *decides* whether to retrieve,
*what* to retrieve, *when* to retrieve, and *how to interpret* the results -- with
reflection, planning, and multi-step reasoning.

Key differences from naive RAG:

| Aspect | Naive RAG | Agentic RAG |
|--------|-----------|-------------|
| Retrieval trigger | Every query | Agent decides if retrieval is needed |
| Query formulation | User query verbatim | Agent rewrites/decomposes queries |
| Source selection | Single vector store | Agent routes to best source |
| Result evaluation | None | Agent checks relevance, retries |
| Multi-hop | Not supported | Agent chains retrievals |
| Tool use | None | Agent calls tools alongside retrieval |

References:
- [Agentic RAG Survey (arXiv 2501.09136)](https://arxiv.org/abs/2501.09136)
- [A-RAG: Scaling Agentic RAG via Hierarchical Retrieval Interfaces (arXiv 2602.03442)](https://arxiv.org/abs/2602.03442)

### 1.2 Architectural Patterns

#### Single-Agent Agentic RAG

A centralized decision-making system where one agent manages retrieval, routing, and
integration. Effective when there are a limited number of data sources or tools.

```
User Query --> [Agent (LLM)]
                   |
                   |--> Decide: retrieve or answer directly?
                   |--> If retrieve: select tool/source
                   |--> Formulate retrieval query
                   |--> Execute retrieval
                   |--> Evaluate results (sufficient? relevant?)
                   |--> If insufficient: retry with different strategy
                   |--> Generate final answer
```

#### Multi-Agent RAG

Specialized agents handle different retrieval domains. A **router agent** dispatches
queries; **domain agents** handle specific data sources; a **synthesizer agent** combines
results. This is the dominant pattern for production systems in 2025-2026.

```
User Query --> [Router Agent]
                   |
         +---------+---------+
         |         |         |
    [Code Agent] [Doc Agent] [API Agent]
         |         |         |
    vector store  wiki     live API
         |         |         |
         +---------+---------+
                   |
            [Synthesizer Agent]
                   |
              Final Answer
```

#### Hierarchical Agentic RAG (A-RAG)

The February 2026 paper from arXiv introduces **A-RAG**, which exposes hierarchical
retrieval interfaces directly to the model with three retrieval tools:

1. **Keyword search** -- traditional BM25/sparse retrieval
2. **Semantic search** -- dense vector similarity
3. **Chunk read** -- direct access to specific document chunks

The model learns to compose these tools hierarchically, first doing broad semantic search,
then narrowing with keyword search, then reading specific chunks. This mirrors how humans
research: scan, focus, read.

Reference: [A-RAG (arXiv 2602.03442)](https://arxiv.org/html/2602.03442v1)

### 1.3 Self-RAG

Self-RAG trains the model itself to decide *when* retrieval is necessary. The model
generates special reflection tokens:

- **[Retrieve]** -- should I retrieve for this query?
- **[IsRel]** -- is the retrieved document relevant?
- **[IsSup]** -- is the response supported by the evidence?
- **[IsUse]** -- is the response useful to the user?

This self-reflective loop means the model only retrieves when it recognizes its own
knowledge is insufficient, and critiques its own outputs for hallucination.

For code-centric agents, Self-RAG is particularly valuable: the agent can recognize when
it needs to look up API documentation, check function signatures, or verify that a code
pattern is correct, rather than hallucinating plausible but wrong code.

### 1.4 Microsoft GraphRAG

GraphRAG builds a knowledge graph from source documents, then uses community detection
to create hierarchical summaries. This enables both:

- **Local search**: entity-focused retrieval (find information about specific things)
- **Global search**: holistic dataset reasoning (what are the main themes across all documents?)

**Indexing pipeline:**

```
LoadDocuments --> ChunkDocuments --> ExtractGraph
                                        |
                                  ExtractClaims
                                        |
                                  EmbedChunks
                                        |
                                  DetectCommunities
                                        |
                                  GenerateReports
                                        |
                                  EmbedReports
```

Key technical details:
- Uses LLM calls to extract entities and relationships from text chunks
- Applies Leiden community detection algorithm to find clusters
- Generates hierarchical community summaries at multiple resolutions
- Factory pattern for extensibility (swap LLM providers, storage backends, etc.)
- Built-in caching layer for LLM interactions (idempotent re-indexing)

For **aiai**, GraphRAG is relevant for building understanding of a codebase's conceptual
structure -- not just "what functions exist" but "what are the major subsystems, how do
they relate, what design patterns are used."

References:
- [Microsoft GraphRAG Architecture](https://microsoft.github.io/graphrag/index/architecture/)
- [GitHub: microsoft/graphrag](https://github.com/microsoft/graphrag)
- [How GraphRAG Works Step-By-Step](https://pub.towardsai.net/how-microsofts-graphrag-works-step-by-step-b15cada5c209)

### 1.5 Practical Guidance for aiai

For a self-improving AI infrastructure project, the recommended RAG architecture is:

1. **Start with single-agent agentic RAG** using tool-calling to decide retrieval
2. **Implement hybrid retrieval** (keyword + semantic) from the beginning
3. **Add Self-RAG reflection** so the agent knows when its knowledge is stale
4. **Graduate to multi-agent** when you have multiple distinct knowledge sources
5. **Consider GraphRAG** for codebase-level understanding (subsystem maps, design patterns)

---

## 2. Vector Databases and Embedding Stores

### 2.1 Database Comparison (2025-2026)

| Database | Type | Language | Best For | Local-First | Cost |
|----------|------|----------|----------|-------------|------|
| **LanceDB** | Embedded | Rust/Python | Local agent memory, edge | Yes (SQLite-like) | Free/OSS |
| **Chroma** | Embedded/Client-Server | Rust (2025 rewrite) | Prototyping, small-medium | Yes | Free/OSS |
| **pgvector** | PostgreSQL extension | C | Hybrid relational+vector | Yes (self-hosted) | Free/OSS |
| **Qdrant** | Standalone | Rust | Production, high-perf | Yes (self-hosted) | Free/OSS + Cloud |
| **Pinecone** | Managed cloud | -- | Enterprise, zero-ops | No | $$$$ |
| **Weaviate** | Standalone | Go | Hybrid search, multimodal | Yes (self-hosted) | Free/OSS + Cloud |
| **Milvus** | Distributed | Go/C++ | Massive scale (billions) | No | Free/OSS + Cloud |

### 2.2 Detailed Analysis

#### LanceDB -- Best for Local Agent Memory

LanceDB is the "SQLite for AI" -- an embedded vector database built on Apache Arrow
columnar format. No server needed. It reads directly from disk using memory-mapped files
and SIMD optimizations, achieving ~95% accuracy with single-digit millisecond latency.

Key advantages for agent memory:
- **Zero infrastructure**: embed in your Python/JS/Rust process
- **Disk-based with near-memory speed**: Arrow columnar format + memory-mapped I/O
- **Multimodal**: store vectors alongside metadata, images, text
- **Versioning**: built-in data versioning (important for agent memory evolution)
- **Cost**: effectively free for local use

```python
import lancedb

db = lancedb.connect("~/.aiai/memory")
table = db.create_table("agent_episodes", data=[
    {"text": "Fixed bug in parser module",
     "vector": embed("Fixed bug in parser module"),
     "timestamp": "2026-02-26T10:00:00Z",
     "task_type": "bugfix",
     "outcome": "success"}
])

# Semantic search over agent memories
results = table.search(embed("parser error handling"))  \
               .where("outcome = 'success'")            \
               .limit(5)                                 \
               .to_list()
```

Reference: [LanceDB](https://lancedb.com/)

#### pgvector / pgvectorscale -- Best for Hybrid Relational + Vector

If you already use PostgreSQL, pgvector is a no-brainer. The pgvectorscale extension
achieves **471 QPS at 99% recall on 50M vectors** (11.4x better than Qdrant at the same
recall, per May 2025 benchmarks). You get full SQL alongside vector search.

Ideal for structured agent memory where you need:
- Relational queries (find all memories from user X in project Y)
- Vector similarity (find semantically similar past experiences)
- ACID transactions (memory consistency guarantees)

```sql
CREATE EXTENSION vector;

CREATE TABLE agent_memory (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536),
    memory_type VARCHAR(50),  -- 'episodic', 'semantic', 'procedural'
    created_at TIMESTAMPTZ DEFAULT NOW(),
    relevance_score FLOAT DEFAULT 1.0,
    metadata JSONB
);

CREATE INDEX ON agent_memory
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Hybrid query: structured filter + vector similarity
SELECT content, metadata,
       1 - (embedding <=> $1) AS similarity
FROM agent_memory
WHERE memory_type = 'procedural'
  AND metadata->>'language' = 'python'
ORDER BY embedding <=> $1
LIMIT 10;
```

Reference: [pgvector benchmark](https://seanpedersen.github.io/posts/vector-databases/)

#### Chroma -- Best for Rapid Prototyping

Chroma's 2025 Rust rewrite delivers 4x faster writes/queries vs the original Python
implementation. Great for getting started quickly, but not designed for production at
50M+ vectors. Smaller teams and rapid prototyping efforts benefit most.

```python
import chromadb

client = chromadb.PersistentClient(path="~/.aiai/chroma")
collection = client.get_or_create_collection("codebase_memory")

collection.add(
    documents=["Function parse_ast handles Python AST parsing"],
    metadatas=[{"file": "parser.py", "type": "function_doc"}],
    ids=["parse_ast_doc"]
)

results = collection.query(
    query_texts=["How does AST parsing work?"],
    n_results=5
)
```

#### Qdrant -- Best for Production Self-Hosted

Written in Rust with a focus on production readiness. HTTP/gRPC APIs, powerful metadata
filtering, WAL for durability, and horizontal scaling. Strong open-source option if you
want more control than Pinecone but more features than LanceDB.

### 2.3 Embedding Models (2026 State of the Art)

| Model | Params | MTEB Score | Code Score | Open Source | Notes |
|-------|--------|-----------|------------|-------------|-------|
| **Qwen3-Embedding-8B** | 8B | 70.58 | 80.68 | Yes | #1 MTEB multilingual (Jun 2025) |
| **Cohere embed-v4** | -- | 65.2 | -- | No | Best proprietary overall |
| **OpenAI text-embedding-3-large** | -- | 64.6 | -- | No | Most widely used proprietary |
| **BGE-M3** | 568M | -- | -- | Yes | 100+ languages, 8192 tokens |
| **Jina Embeddings v4** | -- | -- | -- | Partial | Multimodal (text+image+docs) |
| **EmbeddingGemma-300M** | 300M | -- | -- | Yes | Lightweight, on-device |
| **e5-small** | 33M | -- | -- | Yes | 14x faster, strong Top-5 accuracy |

**For code-centric agent memory in aiai:**

- **Qwen3-Embedding-8B** is the state-of-the-art choice for code embeddings (MTEB Code
  score 80.68), but requires significant compute for inference
- **BGE-M3** is the pragmatic open-source choice (8192 token context, strong performance)
- **OpenAI text-embedding-3-large** is the simplest to deploy if using the API
- **e5-small** or **EmbeddingGemma-300M** for resource-constrained or edge scenarios

References:
- [Qwen3-Embedding on HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- [MTEB Leaderboard Analysis](https://modal.com/blog/mteb-leaderboard-article)
- [Best Open-Source Embedding Models Benchmarked](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)
- [Best Embedding Models 2026](https://www.openxcell.com/blog/best-embedding-models/)

### 2.4 Recommendation for aiai

**Start with LanceDB** for local-first agent memory. It requires zero infrastructure,
embeds directly in the process, and handles the scale aiai will need in its early stages.
If you later need relational queries alongside vectors, add pgvector. Use Qwen3-Embedding
or BGE-M3 for open-source embeddings; fall back to OpenAI text-embedding-3-large if you
want simplicity.

---

## 3. Structured vs Unstructured Memory

### 3.1 The Spectrum of Agent Memory Storage

Agent memory is not a single storage problem. It spans a spectrum:

```
Fully Structured          Hybrid                    Fully Unstructured
      |                     |                              |
   SQLite/SQL      Knowledge Graph + Vector        Raw files on disk
   JSON schemas    Relational + Embeddings         Conversation logs
   Key-value       pgvector / Hybrid DBs           Markdown notes
```

### 3.2 When to Use What

#### Structured Storage (SQLite, JSON, Relational)

Best for:
- **Agent configuration and state**: task queues, agent status, tool definitions
- **Provenance and audit trails**: which agent did what, when, with what result
- **User/session metadata**: IDs, preferences, timestamps, access control
- **Procedural memory**: stored procedures, workflow definitions, skill registries

```python
# SQLite for structured agent state
import sqlite3

conn = sqlite3.connect("~/.aiai/agent_state.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        code TEXT NOT NULL,  -- The actual skill implementation
        success_count INTEGER DEFAULT 0,
        failure_count INTEGER DEFAULT 0,
        avg_duration_ms REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_used TEXT,
        tags TEXT  -- JSON array
    )
""")

conn.execute("""
    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        task TEXT NOT NULL,
        plan TEXT,  -- JSON: the agent's plan
        actions TEXT,  -- JSON: sequence of actions taken
        outcome TEXT CHECK(outcome IN ('success', 'failure', 'partial')),
        lessons TEXT,  -- What the agent learned
        duration_ms INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
""")
```

#### Vector Storage (Embeddings)

Best for:
- **Semantic retrieval**: "find similar past experiences"
- **Code search**: "find functions that handle error logging"
- **Documentation lookup**: "what does the API say about X"
- **Associative memory**: connecting related concepts across sessions

#### Files on Disk (Markdown, JSON, Code)

Best for:
- **Conversation logs**: raw transcripts for replay/analysis
- **Generated artifacts**: code files, configs, documentation
- **Skill libraries**: stored as actual executable code files
- **Project context**: CLAUDE.md-style context files

This is surprisingly effective. Letta's 2025 benchmark showed that agents running on
gpt-4o-mini achieved **74.0% accuracy on LoCoMo** by simply storing conversation histories
in files, rather than using specialized memory or retrieval tools. The takeaway: do not
over-engineer memory infrastructure before you have proven you need it.

Reference: [Letta: Benchmarking AI Agent Memory](https://www.letta.com/blog/benchmarking-ai-agent-memory)

### 3.3 The Hybrid Pattern (Recommended)

The 2025 production consensus is a **three-store hybrid**:

```
                    Agent Memory System
                          |
            +-------------+-------------+
            |             |             |
        Graph Store   Vector Store   Relational Store
        (structure)   (similarity)   (provenance)
            |             |             |
         KuzuDB        LanceDB       SQLite
         or Neo4j      or pgvector
            |             |             |
        entities,     embeddings,    metadata,
        relations,    semantic       timestamps,
        hierarchies   search         audit trail
```

Concrete example for code-centric memory:

```python
class AgentMemory:
    """Three-store hybrid memory for aiai agents."""

    def __init__(self, base_path: str):
        self.relational = sqlite3.connect(f"{base_path}/state.db")
        self.vector = lancedb.connect(f"{base_path}/vectors")
        self.graph = None  # KuzuDB or similar, optional

    def remember_episode(self, episode: dict):
        """Store a completed task episode."""
        # 1. Structured: store metadata and outcome
        self.relational.execute(
            "INSERT INTO episodes (id, task, outcome, duration_ms) VALUES (?, ?, ?, ?)",
            (episode["id"], episode["task"], episode["outcome"], episode["duration_ms"])
        )
        # 2. Vector: store embedding for semantic retrieval
        self.vector.open_table("episodes").add([{
            "id": episode["id"],
            "text": f"{episode['task']} -> {episode['outcome']}",
            "vector": embed(episode["task"]),
            "lessons": episode.get("lessons", "")
        }])
        # 3. Graph (optional): store relationships
        # e.g., (episode)-[:MODIFIED]->(file), (episode)-[:USED]->(skill)

    def recall_similar(self, query: str, n: int = 5) -> list:
        """Retrieve semantically similar past episodes."""
        return self.vector.open_table("episodes") \
                   .search(embed(query))           \
                   .limit(n)                       \
                   .to_list()

    def recall_by_file(self, filepath: str) -> list:
        """Retrieve all episodes that touched a specific file."""
        cursor = self.relational.execute(
            "SELECT * FROM episodes WHERE task LIKE ?",
            (f"%{filepath}%",)
        )
        return cursor.fetchall()
```

### 3.4 Knowledge Graphs for Code-Centric Memory

For a code-centric project like aiai, knowledge graphs capture relationships that flat
storage misses:

- Function A **calls** Function B
- Module X **depends on** Module Y
- Class Foo **implements** Interface Bar
- Bug fix episode **modified** files [a.py, b.py]
- Skill "refactor-function" **requires** skills ["parse-ast", "write-tests"]

This relational structure is critical for agents that need to reason about code impact
("if I change this function, what else breaks?") or plan multi-step modifications.

Reference: [Oracle: Comparing File Systems and Databases for AI Agent Memory](https://blogs.oracle.com/developers/comparing-file-systems-and-databases-for-effective-ai-agent-memory-management)

---

## 4. Context Window Management

### 4.1 The Core Problem

Even with 200K+ token context windows, production agent systems face context management
as a first-class problem. More context does not mean better results -- as the JetBrains
research team showed at NeurIPS 2025, "as the context grows, language models often struggle
to make good use of all the information they're given."

### 4.2 How Claude Code Manages Context

Claude Code (Anthropic's CLI agent) uses a 200K-token window with a sophisticated
4-level memory hierarchy:

```
Priority (highest to lowest):
    1. Enterprise Policy     -- organizational rules (immutable)
    2. Project Memory        -- CLAUDE.md files in project root
    3. Project Rules         -- .claude/rules/*.md files
    4. Session Memory        -- current conversation + tool outputs
```

Key techniques:

**Automatic Compaction**: Claude Code keeps each session under ~50% context usage.
When the context grows too large, it automatically compacts older parts of the
conversation, preserving essential information while discarding verbose details.

**On-Demand Loading**: Skills and rules load lazily -- Claude sees descriptions at
session start, but full content only loads when a skill is invoked.

**Context Isolation via Subagents**: When Claude Code spawns a subagent for a sub-task,
that subagent gets its own fresh context, completely separate from the main conversation.
When done, only a summary returns to the parent -- preventing context bloat.

**CLAUDE.md as Persistent Memory**: The CLAUDE.md file system is Claude Code's approach
to persistent memory across sessions. Key design principles:
- Root CLAUDE.md should be 30-100 lines (>200 lines degrades signal-to-noise)
- Topic-specific rules go in `.claude/rules/` as separate files
- Files in child directories load on demand (only when agent reads files there)
- The root CLAUDE.md acts as an index, not an encyclopedia

```
project/
    CLAUDE.md                    # <-- Loaded at session start (keep small)
    .claude/
        rules/
            testing.md           # <-- Loaded at start (focused rules)
            api-conventions.md   # <-- Loaded at start
    src/
        parser/
            CLAUDE.md            # <-- Loaded on demand when reading parser/
        server/
            CLAUDE.md            # <-- Loaded on demand when reading server/
```

References:
- [Claude Code: How It Works](https://code.claude.com/docs/en/how-claude-code-works)
- [Claude Code Memory Management](https://code.claude.com/docs/en/memory)
- [How Claude's Memory Actually Works](https://rajiv.com/blog/2025/12/12/how-claude-memory-actually-works-and-why-claude-md-matters/)

### 4.3 How Cursor Manages Context

Cursor takes a different, more aggressive compression approach:

**The 10M-to-8K Pipeline**:
```
Full codebase (~10M tokens)
    --> AST-based chunking via tree-sitter (~500 token blocks)
    --> Custom embedding model (trained on agent sessions)
    --> Vector storage in Turbopuffer
    --> At query time: embed query, nearest-neighbor search
    --> Importance ranking
    --> Smart truncation
    --> Final context: ~8K tokens
```

Key architectural details:
- **AST-based chunking**: tree-sitter splits code into semantic units (functions, classes,
  ~500 token blocks) preserving code structure, not just splitting on line counts
- **Custom embeddings**: Cursor trained its own embedding model on coding agent sessions
  (not a general-purpose model), achieving 12.5% improvement in code retrieval accuracy
- **Turbopuffer**: embeddings cached in AWS, indexed by chunk hash for fast re-indexing
- **Subagents**: independent agents run in parallel with own context and custom prompts,
  returning focused results to the parent

Reference:
- [Cursor Codebase Indexing](https://cursor.com/docs/context/codebase-indexing)
- [How Cursor Works: Deep Dive](https://bitpeak.com/how-cursor-works-deep-dive-into-vibe-coding/)
- [Cursor Semantic Search](https://www.digitalapplied.com/blog/cursor-semantic-search-coding-ai-guide)

### 4.4 Context Compression Techniques

#### Hierarchical Summarization

The most common production approach. Older context gets progressively compressed:

```
Turn 1-5:   [Full verbatim text]
Turn 6-20:  [Detailed summary]
Turn 21-50: [High-level summary]
Turn 51+:   [Key facts only]
```

Mem0's approach: keep the latest exchange verbatim + a rolling summary + the m most
recent messages as context, then use an LLM to extract candidate memories. This achieves
**90% token savings** with **26% accuracy improvement** over full-context approaches.

#### Observation Masking (JetBrains, NeurIPS 2025)

A surprising finding: you can aggressively compress tool *outputs* (observations) while
preserving the agent's *reasoning* and *actions* in full. Since agent turns heavily skew
toward observation content (e.g., file contents, search results), masking observations
reduces cost by ~50% without significantly degrading task performance.

This is simpler than LLM summarization and equally effective. The paper's title says
it all: "The Complexity Trap: Simple Observation Masking Is as Efficient as LLM
Summarization for Agent Context Management."

Reference: [JetBrains: Cutting Through the Noise](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)

#### Token Pruning and Attention Sinks

At the LLM inference level:
- **StreamingLLM**: retains initial tokens ("attention sinks") and a sliding window of
  recent tokens, discarding intermediate ones
- **LazyLLM**: dynamically selects different token subsets at each generation step,
  computing KV only for important tokens
- **FastKV**: applies token-selective propagation after a chosen layer, propagating only
  high-importance tokens to deeper layers

These achieve ~36% GPU memory reduction and ~32% inference latency improvement at
compression rates up to 32x with negligible quality loss.

Reference: [Token Compression Techniques](https://www.aussieai.com/research/token-compression)

### 4.5 MemGPT / Letta: OS-Inspired Memory Management

MemGPT treats the LLM context window like RAM, implementing virtual memory with paging:

```
+---------------------------+
|    LLM Context Window     |  <-- "RAM" (limited)
|    (In-Context Memory)    |
|                           |
|  +---------------------+  |
|  | Core Memory         |  |  <-- Always in context (agent persona, user info)
|  +---------------------+  |
|  | Recent Messages     |  |  <-- FIFO buffer of recent conversation
|  +---------------------+  |
|  | Retrieved Context   |  |  <-- Dynamically loaded from archival/recall
|  +---------------------+  |
+---------------------------+
            |  ^
            v  |   (page in/out via tool calls)
+---------------------------+
|    External Storage       |  <-- "Disk" (unlimited)
|                           |
|  +---------------------+  |
|  | Archival Memory     |  |  <-- Vector DB for semantic search
|  +---------------------+  |
|  | Recall Memory       |  |  <-- Full conversation history
|  +---------------------+  |
+---------------------------+
```

The agent manages its own memory through tool calls:
- `core_memory_append(key, value)` -- add to always-in-context memory
- `core_memory_replace(key, old, new)` -- update core memory
- `archival_memory_insert(content)` -- save to long-term vector store
- `archival_memory_search(query)` -- retrieve from long-term store
- `conversation_search(query)` -- search past conversations

**Letta V1 (2025)** builds on this with support for modern reasoning models (GPT-5,
Claude Sonnet 4.5), programmatic tool calling, and the finding that filesystem-based
memory achieves 74% accuracy on benchmarks -- suggesting simple approaches work well.

References:
- [Letta: Agent Memory](https://www.letta.com/blog/agent-memory)
- [Letta V1 Agent Architecture](https://www.letta.com/blog/letta-v1-agent)
- [MemGPT: LLMs as Operating Systems](https://www.leoniemonigatti.com/papers/memgpt.html)

### 4.6 Amazon Bedrock AgentCore Memory

AWS's managed approach provides:
- **Short-term memory**: turn-by-turn within a session
- **Long-term memory**: auto-extracted insights across sessions (preferences, facts, summaries)
- **Episodic memory**: structured episodes recording context, reasoning, actions, outcomes
- **Memory branching**: parallel conversation branches for multi-agent coordination

Episodic memory is particularly interesting: a reflection agent analyzes episodes to
extract broader insights and patterns, enabling meta-learning.

Reference: [Amazon Bedrock AgentCore Memory](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-agentcore-memory-building-context-aware-agents/)

### 4.7 Recommendation for aiai

1. **Implement hierarchical summarization** as the baseline context management strategy
2. **Use observation masking** (JetBrains finding) -- compress tool outputs aggressively,
   preserve reasoning traces in full
3. **Adopt the CLAUDE.md pattern** for persistent cross-session memory (simple files,
   loaded at session start, organized hierarchically)
4. **Consider MemGPT-style self-managed memory** for long-running agents that need to
   decide what to remember and what to forget

---

## 5. Long-Term Learning in Agents

### 5.1 Memory Types in Cognitive Architecture

Drawing from cognitive science, agent memory systems implement:

| Memory Type | Human Analogy | Agent Implementation | Persistence |
|-------------|---------------|---------------------|-------------|
| **Working Memory** | What you're thinking about now | Context window contents | Session |
| **Episodic Memory** | Specific experiences | Task logs with outcomes | Long-term |
| **Semantic Memory** | General knowledge | Embedded knowledge base | Long-term |
| **Procedural Memory** | How to ride a bike | Skill library (code) | Long-term |

### 5.2 The Voyager Pattern: Skill Libraries

Voyager (NVIDIA/Caltech, 2023) introduced the **skill library** pattern -- now widely
adopted. The agent:

1. Attempts a task
2. Writes code to accomplish it
3. Verifies the code works (self-verification)
4. Stores successful code as a reusable "skill" with a description
5. Retrieves relevant skills for future tasks via semantic search

```python
class SkillLibrary:
    """Voyager-pattern skill library for code agents."""

    def __init__(self, vector_store, code_store_path: str):
        self.vector_store = vector_store
        self.code_path = code_store_path

    def add_skill(self, name: str, description: str, code: str,
                  verified: bool = False):
        """Store a new skill after verification."""
        if not verified:
            raise ValueError("Only verified skills should be stored")

        # Save code to disk
        skill_path = f"{self.code_path}/{name}.py"
        with open(skill_path, "w") as f:
            f.write(code)

        # Index in vector store for retrieval
        self.vector_store.add({
            "name": name,
            "description": description,
            "code_path": skill_path,
            "vector": embed(description),
            "created_at": now(),
            "use_count": 0,
            "success_rate": 1.0
        })

    def retrieve_skills(self, task_description: str, n: int = 5) -> list:
        """Find relevant skills for a new task."""
        return self.vector_store.search(
            embed(task_description),
            limit=n
        )

    def compose_skills(self, skill_names: list) -> str:
        """Compose multiple skills into a new procedure."""
        codes = []
        for name in skill_names:
            with open(f"{self.code_path}/{name}.py") as f:
                codes.append(f.read())
        return "\n\n".join(codes)
```

Key results: Voyager obtains 3.3x more unique items, travels 2.3x longer distances, and
unlocks milestones up to 15.3x faster than prior methods. Skills are **temporally
extended, interpretable, and compositional**, which compounds abilities and avoids
catastrophic forgetting.

Reference: [Voyager](https://voyager.minedojo.org/)

### 5.3 Experience Replay and Episodic Learning

Agents that learn from past task executions:

1. **Record episodes**: full trace of task, plan, actions, observations, outcome
2. **Reflect on episodes**: extract lessons ("this approach failed because...")
3. **Retrieve relevant episodes**: when facing a new task, find similar past experiences
4. **Apply lessons**: use past experience to inform current strategy

```python
class EpisodicMemory:
    """Experience replay for learning agents."""

    def record_episode(self, episode: dict):
        """Record a complete task episode with reflection."""
        # Ask the LLM to reflect on the episode
        reflection = llm.generate(f"""
            Task: {episode['task']}
            Plan: {episode['plan']}
            Actions taken: {episode['actions']}
            Outcome: {episode['outcome']}

            Reflect on this episode:
            1. What worked well?
            2. What went wrong?
            3. What would you do differently next time?
            4. What general lesson can be extracted?
        """)

        episode['reflection'] = reflection
        episode['lesson'] = extract_lesson(reflection)

        # Store with embedding for retrieval
        self.store.add(episode)

    def recall_relevant(self, current_task: str, n: int = 3) -> list:
        """Retrieve relevant past episodes for a new task."""
        episodes = self.store.search(embed(current_task), limit=n)
        return [{
            'task': ep['task'],
            'outcome': ep['outcome'],
            'lesson': ep['lesson'],
            'similarity': ep['_distance']
        } for ep in episodes]
```

### 5.4 Procedural Memory: Learning How

Beyond remembering *what happened*, agents need to learn *how to do things*. Recent 2025
papers explore this:

- **"Remember Me, Refine Me"**: Dynamic procedural memory framework where experiences
  are continuously refined into reusable procedures
- **LEGOMem**: Modular procedural memory for multi-agent systems, where skills are
  composable like LEGO blocks
- **Memp**: Explores agent procedural memory with a focus on when to create, update,
  or retire procedures

The primary mechanism is **consolidation**: episodic experiences get distilled into
semantic/procedural knowledge over time.

```
Episode 1: "Fixed null pointer in parser.py by adding input validation"
Episode 2: "Fixed null pointer in formatter.py by adding input validation"
Episode 3: "Fixed null pointer in serializer.py by adding input validation"
     |
     v  (consolidation)
Procedure: "When encountering null pointer errors, check for missing
            input validation at function boundaries"
```

Reference:
- [Agent Skills from Procedural Memory Survey](https://www.techrxiv.org/users/1016212/articles/1376445/master/file/data/Agent_Skills/Agent_Skills.pdf)
- [Agentic Memory: Unified Long-Term and Short-Term Management](https://arxiv.org/html/2601.01885v1)

### 5.5 Mem0: Production Memory Layer

Mem0 is the most production-ready agent memory framework as of 2026. Its architecture:

**Two-Phase Pipeline:**

```
Phase 1: EXTRACTION
    Input: latest exchange + rolling summary + recent messages
    --> LLM extracts candidate memories (facts, preferences, procedures)

Phase 2: UPDATE
    For each candidate memory:
    --> Search vector DB for top-s similar existing memories
    --> If match found: merge, update, or invalidate
    --> If no match: add as new memory
    --> Apply decay to unused memories
```

**Graph-Based Variant (Mem0g):**
Stores memories as a directed, labeled graph. Entities become nodes, relationships become
edges. An Entity Extractor + Relations Generator builds the graph; a Conflict Detector +
Update Resolver maintains it. Enables multi-hop reasoning, temporal queries, and
open-domain reasoning.

Performance: **26% accuracy boost, 91% lower p95 latency, 90% token savings** compared
to full-context approaches.

```python
from mem0 import Memory

m = Memory()

# Add memories from a conversation
m.add(
    "I prefer functional programming patterns in Python",
    user_id="developer_1",
    metadata={"context": "code_style"}
)

m.add(
    "The parser module uses the visitor pattern",
    user_id="developer_1",
    metadata={"context": "codebase_knowledge", "file": "parser.py"}
)

# Retrieve relevant memories
memories = m.search(
    "How should I structure the new formatter?",
    user_id="developer_1"
)
# Returns: functional patterns preference + visitor pattern knowledge
```

References:
- [Mem0 Paper (arXiv 2504.19413)](https://arxiv.org/abs/2504.19413)
- [Mem0 Research: 26% Accuracy Boost](https://mem0.ai/research)
- [GitHub: mem0ai/mem0](https://github.com/mem0ai/mem0)

### 5.6 Recommendation for aiai

1. **Implement a skill library** (Voyager pattern) for code-generation agents from day one
2. **Record episodes** with structured outcomes and LLM-generated reflections
3. **Consolidate episodes into procedures** periodically (batch job or on-demand)
4. **Use Mem0's extraction/update pattern** for automatic memory management
5. **Apply intelligent decay** to prevent memory bloat -- use success rates and
   recency to score memory relevance

---

## 6. Knowledge Graphs for Code

### 6.1 Why Knowledge Graphs for Codebases

Codebases have inherent graph structure: functions call functions, classes inherit from
classes, modules import modules. Flat text search misses these relationships. A knowledge
graph captures:

- **Call graphs**: who calls whom
- **Dependency graphs**: what depends on what
- **Inheritance hierarchies**: class relationships
- **Data flow**: how data moves through the system
- **Change impact**: what is affected when something changes

### 6.2 Building Code Knowledge Graphs

#### tree-sitter for Parsing

[tree-sitter](https://tree-sitter.github.io/tree-sitter/) is the foundation for code
understanding in 2025-2026. It provides:

- Language-agnostic incremental parsing (supports 100+ languages)
- Concrete syntax tree (CST) output
- Sub-millisecond re-parsing on edits
- WASM support for browser-based analysis

tree-sitter extracts AST nodes (functions, classes, imports, variables) but does **not**
create edges for function calls -- these must be constructed by analyzing call sites.

```python
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

code = b"""
def parse_ast(source: str) -> ast.AST:
    tree = ast.parse(source)
    return transform(tree)

def transform(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        process(node)
    return tree
"""

tree = parser.parse(code)

# Extract function definitions
def extract_functions(node):
    functions = []
    if node.type == 'function_definition':
        name_node = node.child_by_field_name('name')
        functions.append({
            'name': name_node.text.decode(),
            'start': node.start_point,
            'end': node.end_point,
            'body': node.text.decode()
        })
    for child in node.children:
        functions.extend(extract_functions(child))
    return functions

funcs = extract_functions(tree.root_node)
# [{'name': 'parse_ast', ...}, {'name': 'transform', ...}]
```

#### LSP Integration

The Language Server Protocol provides rich semantic information:
- Go-to-definition, find-references, hover info
- Symbol search across the project
- Diagnostic information (errors, warnings)
- Code actions and refactoring support

Combining tree-sitter (fast, incremental parsing) with LSP (deep semantic understanding)
gives agents comprehensive code navigation capabilities.

### 6.3 Production Tools (2025-2026)

#### Code-Graph-RAG

The most complete open-source solution for codebase knowledge graphs:

- **Parsing**: tree-sitter AST parsing for 11+ languages with unified graph schema
- **Storage**: Memgraph (graph database) for storing functions, classes, dependencies
- **Querying**: Natural language to Cypher query translation
- **MCP Server**: integrates directly with Claude Code as an MCP server
- **Editing**: AST-based surgical code replacement with diff previews

Architecture:
```
Repository --> tree-sitter Parser
                    |
                    v
              AST Extraction
              (functions, classes, imports, calls)
                    |
                    v
              Memgraph Knowledge Graph
              (nodes: entities, edges: relationships)
                    |
                    v
              MCP Server (10 tools)
                    |
                    v
              Claude Code / AI Agent
              (natural language queries & edits)
```

Reference: [Code-Graph-RAG](https://github.com/vitali87/code-graph-rag)

#### Source Atlas

Combines tree-sitter + LSP + Neo4j:
- Tree-sitter for fast AST parsing
- LSP for deep semantic information (go-to-definition, find-references)
- Neo4j for storing the knowledge graph
- Multi-language support

Reference: [Source Atlas](https://github.com/quyen-ngv/source-atlas)

#### Project Orchestrator

A Rust-based system combining:
- Neo4j knowledge graph
- Meilisearch semantic code search
- tree-sitter parsing (Rust, TypeScript, Python, Go, 8+ languages)
- MCP integration (130 tools for Claude Code, OpenAI Agents, Cursor)

Reference: [Project Orchestrator](https://github.com/this-rs/project-orchestrator)

#### GitNexus (February 2026)

The latest entrant, trending on GitHub with 1,200+ stars. Offers interactive graph
exploration and an MCP server for codebase analysis and multi-file refactoring.

Reference: [GitNexus](https://topaiproduct.com/2026/02/22/gitnexus-turns-your-codebase-into-a-knowledge-graph-and-your-ai-agent-will-thank-you/)

#### CodeGraphContext

MCP server + CLI tool that indexes local code into a graph database to provide context
to AI assistants. Provides symbol context lookups, hybrid search, and blast radius
analysis ("if I change this function, what else is affected?").

Reference: [CodeGraphContext](https://github.com/CodeGraphContext/CodeGraphContext)

### 6.4 Graph-Based Retrieval Strategy

Neo4j's research and Memgraph's GraphRAG for Devs demo show a multi-stage retrieval
strategy:

1. **Parse query** to identify code entities mentioned
2. **Graph traversal** to find related entities (callers, callees, dependencies)
3. **PageRank ranking** to identify high-relevance files based on graph centrality
4. **Semantic search** over code documentation/comments for additional context
5. **Compose context** from graph structure + semantic matches

```cypher
// Example Cypher query: find blast radius of changing a function
MATCH (f:Function {name: "parse_ast"})
      -[:CALLS*1..3]->(affected:Function)
RETURN affected.name, affected.file,
       length(shortestPath((f)-[:CALLS*]->(affected))) AS distance
ORDER BY distance
```

### 6.5 Recommendation for aiai

1. **Use tree-sitter for AST extraction** -- it is the industry standard, supports
   100+ languages, and is fast enough for incremental updates
2. **Start with a lightweight graph** (KuzuDB embedded or even SQLite with adjacency
   tables) before committing to Neo4j/Memgraph
3. **Build call graphs and dependency graphs** as the minimum viable knowledge graph
4. **Integrate as an MCP server** so agents can query the graph naturally
5. **Consider Code-Graph-RAG** as a reference implementation or starting point

---

## 7. Memory Safety and Privacy

### 7.1 Threat Landscape

Agent memory introduces security vulnerabilities that do not exist in stateless LLM
systems. The threat surface includes:

```
                    Attack Vectors
                         |
         +---------------+---------------+
         |               |               |
   Memory Poisoning   Data Leakage   Prompt Injection
         |               |               |
   Inject malicious    PII stored     Manipulate agent
   memories that       in memory      behavior via
   alter future        leaks to       injected context
   behavior            other users
```

### 7.2 Memory Poisoning Attacks

**How it works (Palo Alto Unit 42 research, 2025):**

1. Attacker crafts a malicious webpage/document
2. Victim user's agent processes the document
3. Indirect prompt injection manipulates the session summarization
4. Malicious instructions get stored in the agent's long-term memory
5. In future sessions, poisoned memories are loaded into context
6. Agent silently exfiltrates conversation history or follows attacker instructions

This is particularly dangerous because:
- The malicious effect activates only in specific contexts (hard to detect in isolation)
- Once triggered, it creates a **self-reinforcing error cycle**: corrupted output becomes
  stored precedent, amplifying the initial error and lowering the threshold for future
  attacks

**MemoryGraft (December 2025):** Demonstrated persistent compromise of LLM agents via
poisoned experience retrieval. Once a toxic memory is injected, it can persistently
influence agent behavior across unlimited future sessions.

Reference:
- [Palo Alto Unit 42: When AI Remembers Too Much](https://unit42.paloaltonetworks.com/indirect-prompt-injection-poisons-ai-longterm-memory/)
- [MemoryGraft (arXiv 2512.16962)](https://arxiv.org/html/2512.16962v1)
- [Memory Poisoning Attack and Defense (arXiv 2601.05504)](https://arxiv.org/pdf/2601.05504)

### 7.3 PII and Data Leakage

**MAMA Framework (Multi-Agent Memory Attack):**
Research quantifying how network topology affects memory leakage in multi-agent systems:

- Fully connected agent graphs exhibit **maximum leakage**
- Chain topologies provide **strongest protection**
- Shorter attacker-target graph distance = higher vulnerability
- Temporal/locational PII leaks more readily than credentials
- Leakage rises sharply in early rounds before plateauing

**Real-world incidents (2025):**
- **Shadow Escape**: zero-click exploit targeting MCP-based agents (ChatGPT, Gemini),
  enabling silent workflow hijacking and data exfiltration
- **EchoLeak (CVE-2025-32711)**: zero-click prompt injection in Microsoft 365 Copilot
  using character substitutions to bypass safety filters, forcing exfiltration of
  business data to external URLs

Reference:
- [Unveiling Privacy Risks in LLM Agent Memory (ACL 2025)](https://aclanthology.org/2025.acl-long.1227.pdf)
- [Top 10 Agentic AI Security Threats 2025](https://www.lasso.security/blog/agentic-ai-security-threats-2025)

### 7.4 Defense Mechanisms

#### A-MemGuard (October 2025)

The first proactive defense framework for LLM agent memory. Two mechanisms:

1. **Consensus-based validation**: Before acting on a memory, compare reasoning paths
   derived from multiple related memories. If a memory produces anomalous reasoning
   compared to consensus, flag it as potentially poisoned.

2. **Dual-memory structure**: Detected failures are distilled into "lessons" stored in
   a separate memory bank. Before future actions, the agent consults these lessons,
   breaking self-reinforcing error cycles.

Results: **95% reduction in attack success rate** with minimal utility cost.

```python
class MemGuard:
    """Simplified A-MemGuard pattern."""

    def __init__(self):
        self.primary_memory = VectorStore()    # Normal memories
        self.lesson_memory = VectorStore()     # Lessons from detected failures

    def validate_memory(self, memory: dict, context: str) -> bool:
        """Consensus-based validation."""
        # Retrieve related memories
        related = self.primary_memory.search(
            embed(memory['content']), limit=5
        )

        # Generate reasoning paths from each related memory
        paths = [
            generate_reasoning(context, mem) for mem in related
        ]

        # Check if the candidate memory's reasoning is anomalous
        candidate_path = generate_reasoning(context, memory)
        anomaly_score = compute_divergence(candidate_path, paths)

        return anomaly_score < THRESHOLD

    def store_lesson(self, failed_memory: dict, correct_outcome: str):
        """Store a lesson from a detected failure."""
        lesson = {
            'failed_content': failed_memory['content'],
            'correct_outcome': correct_outcome,
            'lesson': f"Memory '{failed_memory['content']}' led to "
                      f"incorrect behavior. Correct approach: {correct_outcome}"
        }
        self.lesson_memory.add(lesson)

    def recall_with_guard(self, query: str, n: int = 5) -> list:
        """Retrieve memories with safety checks."""
        # First check lessons for known pitfalls
        lessons = self.lesson_memory.search(embed(query), limit=3)

        # Then retrieve primary memories
        memories = self.primary_memory.search(embed(query), limit=n)

        # Validate each memory
        validated = [
            m for m in memories
            if self.validate_memory(m, query)
        ]

        return {'memories': validated, 'warnings': lessons}
```

Reference: [A-MemGuard (arXiv 2510.02373)](https://arxiv.org/abs/2510.02373)

#### Additional Defense Strategies

1. **Memory provenance tracking**: Track where every memory came from (which session,
   which user, which document). Reject memories from untrusted sources.

2. **Memory sandboxing**: Isolate memories by trust level. Memories from external
   documents get lower trust scores than memories from verified user interactions.

3. **PII detection and redaction**: Scan memories before storage for PII patterns.
   Use regex + NER models to detect names, emails, API keys, etc.

4. **Memory encryption**: Encrypt sensitive memories at rest. Use per-user encryption
   keys to prevent cross-user leakage.

5. **Temporal access controls**: Memories automatically expire after a configurable
   period unless explicitly renewed.

6. **Graph topology design**: In multi-agent systems, avoid fully-connected topologies.
   Use chain or star topologies to minimize leakage surface.

```python
class SecureMemoryStore:
    """Memory store with safety guardrails."""

    def __init__(self):
        self.store = VectorStore()
        self.pii_detector = PIIDetector()

    def add(self, content: str, source: str, trust_level: str = "low"):
        # 1. PII detection and redaction
        pii_found = self.pii_detector.scan(content)
        if pii_found:
            content = self.pii_detector.redact(content)
            log_warning(f"PII redacted from memory: {pii_found}")

        # 2. Provenance tracking
        memory = {
            'content': content,
            'source': source,
            'trust_level': trust_level,
            'created_at': now(),
            'expires_at': now() + timedelta(days=90),
            'vector': embed(content)
        }

        # 3. Trust-based storage
        if trust_level == "low":
            memory['sandbox'] = True  # Isolated retrieval

        self.store.add(memory)

    def search(self, query: str, min_trust: str = "low") -> list:
        results = self.store.search(embed(query))
        # Filter expired memories
        results = [r for r in results if r['expires_at'] > now()]
        # Filter by trust level
        if min_trust == "high":
            results = [r for r in results if r['trust_level'] == "high"]
        return results
```

### 7.5 Recommendation for aiai

1. **Implement memory provenance from day one** -- every memory must have a source,
   timestamp, and trust level
2. **Use PII scanning** before any memory is stored (regex + NER model)
3. **Design multi-agent communication with chain topology** where possible
4. **Implement A-MemGuard-style consensus validation** for high-stakes memory retrieval
5. **Add temporal expiry** to all memories with configurable TTLs
6. **Encrypt sensitive memories at rest** using per-context encryption keys

---

## 8. Recommendations for aiai

### 8.1 Architecture Overview

Based on all research, here is the recommended memory architecture for aiai:

```
+------------------------------------------------------------------+
|                        aiai Agent System                         |
+------------------------------------------------------------------+
|                                                                  |
|  +-----------------------+    +----------------------------+     |
|  |   Working Memory      |    |   Context Manager          |     |
|  |   (context window)    |<-->|   - Hierarchical summary   |     |
|  |                       |    |   - Observation masking     |     |
|  |   - Current task      |    |   - Compaction triggers     |     |
|  |   - Recent messages   |    +----------------------------+     |
|  |   - Retrieved context |                                       |
|  +-----------------------+                                       |
|           |  ^                                                   |
|           v  |                                                   |
|  +------------------------------------------------------------------+
|  |              Persistent Memory Layer                              |
|  |                                                                   |
|  |  +----------------+  +----------------+  +------------------+     |
|  |  | Relational     |  | Vector Store   |  | Knowledge Graph  |     |
|  |  | (SQLite)       |  | (LanceDB)      |  | (KuzuDB/SQLite)  |     |
|  |  |                |  |                |  |                  |     |
|  |  | - Episodes     |  | - Embeddings   |  | - Call graphs    |     |
|  |  | - Skills meta  |  | - Semantic     |  | - Dependencies   |     |
|  |  | - Provenance   |  |   search       |  | - Entity rels    |     |
|  |  | - Agent state  |  | - Similarity   |  |                  |     |
|  |  +----------------+  +----------------+  +------------------+     |
|  +------------------------------------------------------------------+
|           |  ^                                                   |
|           v  |                                                   |
|  +------------------------------------------------------------------+
|  |              File-Based Memory                                    |
|  |                                                                   |
|  |  - CLAUDE.md-style project context files                          |
|  |  - Skill library (executable .py files)                           |
|  |  - Conversation logs (raw transcripts)                            |
|  |  - Generated artifacts (code, configs)                            |
|  +------------------------------------------------------------------+
|           |  ^                                                   |
|           v  |                                                   |
|  +----------------------------+                                  |
|  |   Security Layer           |                                  |
|  |   - PII scanning/redaction |                                  |
|  |   - Memory provenance      |                                  |
|  |   - Consensus validation   |                                  |
|  |   - Temporal expiry        |                                  |
|  +----------------------------+                                  |
+------------------------------------------------------------------+
```

### 8.2 Technology Choices

| Component | Recommended | Rationale |
|-----------|-------------|-----------|
| Vector store | LanceDB | Embedded, zero-infra, Arrow-based, fast |
| Relational store | SQLite | Embedded, universal, battle-tested |
| Knowledge graph | KuzuDB (embedded) or SQLite adjacency tables | Start simple, upgrade later |
| Embeddings | BGE-M3 (local) or OpenAI text-embedding-3-large (API) | Balance of quality and cost |
| Code parsing | tree-sitter | Industry standard, 100+ languages |
| Context files | CLAUDE.md pattern | Proven, simple, human-readable |
| Memory framework | Custom (inspired by Mem0 patterns) | Control over architecture |

### 8.3 Implementation Phases

**Phase 1: Foundation**
- SQLite for agent state and episode logging
- LanceDB for semantic search over episodes
- CLAUDE.md-style project memory files
- Basic PII scanning on memory writes

**Phase 2: Learning**
- Skill library (Voyager pattern) with verified code storage
- Episodic memory with LLM-generated reflections
- Hierarchical summarization for context management
- Episode consolidation into procedures

**Phase 3: Code Understanding**
- tree-sitter AST extraction for codebase indexing
- Knowledge graph of code entities and relationships
- MCP server for agent-accessible code queries
- Blast radius analysis for change impact

**Phase 4: Advanced**
- A-MemGuard-style consensus validation
- Multi-agent memory with topology-aware isolation
- Intelligent memory decay and consolidation
- Self-improving memory strategies (meta-learning)

### 8.4 Key Principles

1. **Start simple, prove the need**: Letta's benchmark shows 74% accuracy from just
   files on disk. Do not build complex infrastructure until you have proven you need it.

2. **Observation masking before summarization**: The JetBrains NeurIPS finding --
   simple observation truncation is as effective as LLM summarization at 50% of the cost.

3. **Memory is a write problem, not a read problem**: The hard part is deciding what to
   store and how to update/consolidate, not how to retrieve it.

4. **Security from day one**: Memory poisoning is a real, demonstrated attack vector.
   Provenance tracking and PII scanning are not optional.

5. **Hybrid storage is the consensus**: No single storage technology handles all memory
   needs. The three-store pattern (relational + vector + graph) is the 2025 production
   standard.

---

## References

### Papers
- [Agentic RAG Survey (arXiv 2501.09136)](https://arxiv.org/abs/2501.09136)
- [A-RAG: Hierarchical Retrieval Interfaces (arXiv 2602.03442)](https://arxiv.org/abs/2602.03442)
- [Mem0: Production-Ready Agent Memory (arXiv 2504.19413)](https://arxiv.org/abs/2504.19413)
- [A-MemGuard: Defense Framework (arXiv 2510.02373)](https://arxiv.org/abs/2510.02373)
- [MemoryGraft: Persistent Memory Poisoning (arXiv 2512.16962)](https://arxiv.org/html/2512.16962v1)
- [Memory Poisoning Attack and Defense (arXiv 2601.05504)](https://arxiv.org/pdf/2601.05504)
- [Agentic Memory: Unified Management (arXiv 2601.01885)](https://arxiv.org/html/2601.01885v1)
- [Voyager: Open-Ended Embodied Agent (arXiv 2305.16291)](https://arxiv.org/abs/2305.16291)
- [Unveiling Privacy Risks in LLM Agent Memory (ACL 2025)](https://aclanthology.org/2025.acl-long.1227.pdf)
- [Agent Skills from Procedural Memory Survey](https://www.techrxiv.org/users/1016212/articles/1376445/master/file/data/Agent_Skills/Agent_Skills.pdf)

### Tools and Frameworks
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [Mem0](https://github.com/mem0ai/mem0)
- [Letta (MemGPT)](https://github.com/letta-ai/letta)
- [LanceDB](https://lancedb.com/)
- [Code-Graph-RAG](https://github.com/vitali87/code-graph-rag)
- [Source Atlas](https://github.com/quyen-ngv/source-atlas)
- [Project Orchestrator](https://github.com/this-rs/project-orchestrator)
- [CodeGraphContext](https://github.com/CodeGraphContext/CodeGraphContext)
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)

### Product and System Documentation
- [Claude Code Memory](https://code.claude.com/docs/en/memory)
- [Cursor Codebase Indexing](https://cursor.com/docs/context/codebase-indexing)
- [Amazon Bedrock AgentCore Memory](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-agentcore-memory-building-context-aware-agents/)
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

### Research Blogs
- [Letta: Agent Memory](https://www.letta.com/blog/agent-memory)
- [Letta: Benchmarking Agent Memory](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- [JetBrains: Smarter Context Management](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [Palo Alto Unit 42: Memory Poisoning](https://unit42.paloaltonetworks.com/indirect-prompt-injection-poisons-ai-longterm-memory/)
- [Memgraph: GraphRAG for Devs](https://memgraph.com/blog/graphrag-for-devs-coding-assistant)
- [Neo4j: Codebase Knowledge Graph](https://neo4j.com/blog/developer/codebase-knowledge-graph/)
- [MTEB Leaderboard Analysis](https://modal.com/blog/mteb-leaderboard-article)
