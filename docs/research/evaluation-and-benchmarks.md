# Evaluating AI Agent Systems: Metrics, Benchmarks, and Quality Assurance (2024-2026)

> Research compiled February 2026 for the **aiai** self-improving AI infrastructure project.
> Goal: Understand how to measure whether the system is actually improving.

---

## Table of Contents

1. [SWE-bench Deep Dive](#1-swe-bench-deep-dive)
2. [Agent Benchmarks Beyond SWE-bench](#2-agent-benchmarks-beyond-swe-bench)
3. [Production Monitoring for Agents](#3-production-monitoring-for-agents)
4. [LLM-as-Judge Evaluation](#4-llm-as-judge-evaluation)
5. [Automated Code Quality Metrics](#5-automated-code-quality-metrics)
6. [A/B Testing for AI Systems](#6-ab-testing-for-ai-systems)
7. [Regression Detection](#7-regression-detection)
8. [Cost-Quality Pareto Analysis](#8-cost-quality-pareto-analysis)
9. [Practical Recommendations for aiai](#9-practical-recommendations-for-aiai)
10. [References](#10-references)

---

## 1. SWE-bench Deep Dive

### 1.1 What SWE-bench Is

[SWE-bench](https://www.swebench.com/) (published at ICLR 2024) is a benchmark that evaluates whether language models can resolve real-world GitHub issues. Each task instance consists of:

- A **problem statement** (the GitHub issue text)
- A **codebase snapshot** (the repository at the time the issue was filed)
- A **gold patch** (the human-written fix)
- A **test suite** that the patch must pass (including newly added tests that verify the fix)

The original dataset contains **2,294 task instances** drawn from **12 popular Python repositories** including Django, Flask, Scikit-learn, Matplotlib, Sympy, Requests, Sphinx, Astropy, Pylint, Pytest, Seaborn, and xarray.

### 1.2 Dataset Construction Pipeline

1. **Repository selection**: Popular, well-maintained Python projects with robust test suites.
2. **Pull request mining**: Identify merged PRs that (a) resolve a GitHub issue and (b) include test changes.
3. **Test isolation**: Extract the tests added/modified by the PR. These become the "pass criteria."
4. **Environment reproduction**: Create a Docker image that can check out the codebase at the relevant commit and run the test suite.
5. **Validation**: Verify that the gold patch passes the new tests, and the pre-patch code fails them.

### 1.3 Evaluation Harness

The evaluation harness (co-developed with OpenAI in mid-2024) uses **containerized Docker environments** for each task instance. This ensures:

- Reproducible dependency installation
- Isolated execution (no cross-task contamination)
- Consistent timeout and resource limits

Submission is now done via **sb-cli**, a cloud-based evaluation tool. The harness applies the model's proposed patch, runs the relevant test suite, and reports pass/fail.

**Key metric**: `% Resolved` -- the fraction of task instances where the model's patch causes all relevant tests to pass.

### 1.4 SWE-bench Variants

| Variant | Size | Description | Key Feature |
|---------|------|-------------|-------------|
| **SWE-bench (Full)** | 2,294 | Original dataset | Comprehensive but noisy |
| **SWE-bench Lite** | 300 | Curated subset | Easier, faster evaluation |
| **SWE-bench Verified** | 500 | Human-annotated subset | Non-problematic instances verified by annotators |
| **SWE-bench Multimodal** | 617 | JavaScript libraries with visual bugs | Tests visual reasoning + cross-language |
| **SWE-bench+** | 11,000+ | Multi-language, mutation-based | Contamination-resistant, 11 languages |
| **SWE-bench Live** | 1,890 | Continuously updated since Jan 2024 | Monthly updates, 223 repos, contamination-proof |
| **SWE-bench Pro** | 1,865 | Enterprise-level tasks, 41 repos | Long-horizon, multi-file, commercial codebases |
| **SWE-bench Java Verified** | Varies | Java-specific tasks | JVM ecosystem coverage |
| **SWE-bench Live/Windows** | Varies | Windows PowerShell tasks | Released Feb 2026 |

### 1.5 Current Leaderboard (February 2026) -- SWE-bench Verified

| Rank | System | Score (% Resolved) | Notes |
|------|--------|--------------------|-------|
| 1 | Claude Opus 4.5 (Anthropic) | 80.9% | Self-reported |
| 2 | Claude Opus 4.6 (Anthropic) | 80.8% | Self-reported |
| 3 | MiniMax M2.5 | 80.2% | |
| 4 | GPT-5.2 (OpenAI) | 80.0% | |
| 5 | Sonar Foundation Agent | 79.2% | Official evaluation, Feb 19 2026 |
| 6 | GLM-5 (Zhipu AI) | 77.8% | |
| 7 | Kimi K2.5 | 76.8% | |

**Note**: Scores are largely self-reported by model providers. Scaffold/harness differences significantly affect results. SWE-bench Pro scores are dramatically lower: top agents achieve only ~23.3% on the public set and ~17.8% on the commercial set.

### 1.6 What Makes Tasks Hard

Research by [Ganhotra (2025)](https://jatinganhotra.dev/blog/swe-agents/2025/04/15/swe-bench-verified-easy-medium-hard.html) analyzed task difficulty using OpenAI's human-annotated completion time estimates:

| Difficulty | Estimated Time | Characteristics |
|------------|---------------|-----------------|
| Easy | < 15 min | Trivial changes (adding assertions, simple fixes) |
| Medium | 15 min - 1 hr | Small changes requiring thought |
| Hard | 1 - 4 hrs | Substantial rewrites, multiple files |
| Very Hard | > 4 hrs | Esoteric issues, 100+ lines changed, deep research |

**Key findings on difficulty drivers**:

- **Lines changed**: 11x increase from Easy to Hard tasks
- **Files modified**: 2x increase from Easy to Hard
- **Hunks (separate code blocks)**: 5x increase from Easy to Hard
- **Multi-file tasks** require 4x more lines and 4x more hunks than single-file tasks
- **Repository-specific difficulty**: Some repos have <10% resolve rates across all models; others allow >50%
- **Failure modes differ by model size**: Large models fail on semantic/algorithmic correctness in multi-file edits; smaller models fail on syntax, formatting, tool use, and context management

### 1.7 Limitations and Criticisms

1. **Solution leakage**: Issue comments and linked PRs may contain hints or full solutions. SWE-bench+ found solution leakage impacted **32.67% of successes**.
2. **Weak test suites**: Original tests may not adequately verify correctness. SWE-bench+ found weak test suites caused **31.08% of patches** to be incorrectly labeled as passed. After rigorous revalidation, success rates dropped from 12.47% to 3.97% for SWE-Agent+GPT-4.
3. **Python-only bias**: The original benchmark covers only 12 Python repositories, missing the vast majority of real-world software engineering.
4. **Static dataset**: The original SWE-bench has not been updated since release, making it vulnerable to **data contamination** from LLM training corpora.
5. **Scaffolding confound**: Results depend heavily on the agent scaffold (file retrieval, tool use, search strategy), making it hard to isolate model capability from engineering.
6. **Visual context missing**: The original benchmark presents problems as text only. When images are introduced (SWE-bench Multimodal), performance drops by **73.2%**.
7. **No measure of code quality**: SWE-bench only checks "does it pass tests?" -- not readability, maintainability, efficiency, or idiomatic style.

### 1.8 SWE-bench Multimodal

[SWE-bench Multimodal](https://www.swebench.com/multimodal.html) evaluates AI on fixing bugs in **visual, user-facing JavaScript software**. It consists of 617 task instances from 17 JavaScript libraries.

**Key findings**:
- Top-performing SWE-bench systems struggle significantly on multimodal tasks
- **50% lower success** when images are omitted from tasks where visual context is necessary
- Methods deeply coupled to Python's AST or language-specific tooling cannot trivially adapt to JavaScript
- Reveals that current agents have poor cross-language generalization

### 1.9 SWE-bench Live

[SWE-bench Live](https://swe-bench-live.github.io/) (NeurIPS 2025 D&B, Microsoft) is designed to be **contamination-proof**:
- **Automated dataset curation pipeline** with monthly updates
- **1,890 tasks** from 223 repositories, restricted to issues created after January 1, 2024
- Automated **solution-leak detection** excludes issues that reveal patches in text/comments
- Each task has a dedicated Docker image for reproducible execution
- **SWE-bench-Live/Windows** (Feb 2026) and **SWE-bench-Live/Multi-Language** variants

---

## 2. Agent Benchmarks Beyond SWE-bench

### 2.1 GAIA (General AI Assistants)

**Paper**: [Mialon et al. (2023)](https://arxiv.org/abs/2311.12983) -- Meta-FAIR, Hugging Face, AutoGPT

**What it measures**: General-purpose AI assistant capability across reasoning, web browsing, multimodal understanding, and tool use.

**Dataset**: 466 human-curated questions with unambiguous, verifiable answers requiring multi-step reasoning and tool use (primarily web browsing).

**Methodology principles**:
- **Ungameability**: Hard to brute-force; requires reason traces
- **Unambiguity**: Single correct answer per question
- **Simplicity**: Easy for humans to verify answers

**Difficulty levels**:
- Level 1: Simple reasoning + basic tool use
- Level 2: Multi-step reasoning + multiple tools
- Level 3: Complex multi-step tasks requiring extensive tool orchestration

**Scores (as of late 2025)**:
- Human respondents: **92%**
- GPT-4 with plugins (2023): **15%**
- H2O GPTe Agent (2025): **75%** (first to achieve "C grade")
- Top agents (late 2025): **~90%**

**Relevance to aiai**: High. GAIA tests the kind of multi-step, tool-augmented reasoning that a self-improving coding agent needs. Useful for evaluating general reasoning capability improvements.

### 2.2 AgentBench

**Paper**: [Liu et al. (2023)](https://arxiv.org/abs/2308.03688) -- ICLR 2024

**What it measures**: Comprehensive autonomous agent evaluation across 8 diverse environments: operating systems, databases, knowledge graphs, digital card games, lateral thinking puzzles, house-holding tasks, web shopping, and web browsing.

**Key value**: Gives a **holistic picture** of agent capabilities. Reveals weak spots that domain-specific benchmarks miss.

**Relevance to aiai**: Medium. The OS and web browsing components are relevant; the card game and puzzle components less so.

### 2.3 WebArena

**Paper**: [Zhou et al. (2023)](https://webarena.dev/)

**What it measures**: Autonomous web navigation in realistic, self-hosted environments simulating e-commerce, social forums, collaborative development, and content management.

**Scores (2025-2026)**:
- Top agent: **71.2%**
- OpenAI Operator: **58%**
- Jace.AI: **57.1%**
- ORCHESTRA: **52.1%**
- Progress trajectory: From **14% to ~60%** in two years

**Relevance to aiai**: Medium-low for core coding tasks, but relevant if aiai agents need to interact with web-based development tools (GitHub, Jira, documentation sites).

### 2.4 OSWorld

**Paper**: [Xie et al. (2024)](https://os-world.github.io/)

**What it measures**: System-level GUI tasks in real computer environments. Includes extended 50-step evaluations for lengthy computer tasks.

**Scores (2025)**:
- OpenAI Operator: **38%**
- Simular Agent S2 (50-step): **34.5%** (state of the art)
- Open-source agents: **~24.5%**

**Relevance to aiai**: Medium. Tests the ability to operate in desktop environments, relevant if aiai needs to interact with IDEs, terminals, and desktop tools.

### 2.5 Terminal-Bench

**Paper**: [Laude Institute + Stanford (2025)](https://arxiv.org/abs/2601.11868)

**What it measures**: Agent performance on real-world terminal/CLI tasks: compiling code, training models, setting up servers, system administration, all in sandboxed environments.

**Terminal-Bench 2.0**: 89 tasks, manually verified by three human reviewers.

**Scores (early 2026)**:
- OpenAI Codex CLI (GPT-5 variant): **49.6%** (leading)
- Claude Opus 4.5: Competitive
- Claude Sonnet 4.5: Competitive
- Gemini 3 Pro: Competitive

**Relevance to aiai**: **Very high**. Terminal-Bench directly evaluates the kind of CLI-based development work that aiai agents perform. The benchmark tests real compilation, deployment, and system tasks.

### 2.6 MLAgentBench and MLE-bench

**MLAgentBench** ([Huang et al., 2023](https://arxiv.org/abs/2310.03302)): 13 ML experimentation tasks (improving CIFAR-10 performance, BabyLM, etc.). Agents perform actions like reading/writing files, executing code, inspecting outputs. Best result: Claude v3 Opus at **37.5% average success rate**.

**MLE-bench** ([OpenAI, 2024](https://arxiv.org/abs/2410.07095)): 75 ML engineering competitions from Kaggle testing real-world ML engineering skills.

**MLR-Bench** ([2025](https://arxiv.org/abs/2505.19955)): 201 research tasks from NeurIPS, ICLR, and ICML workshops for open-ended ML research evaluation.

**Relevance to aiai**: High for the self-improvement loop -- aiai needs to evaluate whether its ML experimentation capabilities are improving.

### 2.7 CUB (Computer Use Benchmark)

**Released**: Mid-2025 by Theta AI

**What it measures**: End-to-end computer and browser use workflows across 7 industries. 106 workflows testing real-world computer interaction skills.

**Relevance to aiai**: Medium. Relevant if aiai agents need general computer use capabilities beyond terminal and code editing.

### 2.8 Benchmark Comparison Matrix

| Benchmark | Domain | Tasks | Top Score | Contamination Risk | Relevance to aiai |
|-----------|--------|-------|-----------|--------------------|--------------------|
| SWE-bench Verified | Coding (Python) | 500 | ~81% | High (static) | Very High |
| SWE-bench Pro | Coding (enterprise) | 1,865 | ~23% | Low | Very High |
| SWE-bench Live | Coding (multi-lang) | 1,890 | Varies | Very Low | Very High |
| Terminal-Bench 2.0 | CLI tasks | 89 | ~50% | Low | Very High |
| GAIA | General assistant | 466 | ~90% | Medium | High |
| MLE-bench | ML engineering | 75 | Varies | Medium | High |
| WebArena | Web navigation | 812 | ~71% | Low | Medium |
| OSWorld | Desktop tasks | 369 | ~38% | Low | Medium |
| AgentBench | Multi-environment | 8 envs | Varies | Medium | Medium |
| CUB | Computer use | 106 | Varies | Low | Medium-Low |

---

## 3. Production Monitoring for Agents

### 3.1 Key Metrics for Agent Systems

#### Operational Metrics

| Metric | Definition | Target Range | How to Measure |
|--------|-----------|--------------|----------------|
| **Latency (P50/P95/P99)** | Time from request to final response | P95 < 30s for simple; < 5min for complex | Trace instrumentation |
| **Time to First Token (TTFT)** | Time until first output token | < 500ms | Proxy/gateway measurement |
| **Tokens per second (TPS)** | Output generation speed | > 50 TPS | Provider metrics |
| **Total token consumption** | Input + output tokens per task | Varies by task | Sum across all LLM calls in trace |
| **Cost per task** | Dollar cost for completing one task | Track trend, minimize | Token count x price per token |
| **Success rate** | Fraction of tasks completed correctly | > 90% for production | Automated + human validation |
| **Error rate** | Fraction of tasks that fail/crash | < 5% | Exception tracking |
| **Retry rate** | How often tasks need retries | < 10% | Agent loop monitoring |
| **Tool call success rate** | Fraction of tool invocations that succeed | > 95% | Per-tool tracking |
| **Steps per task** | Number of agent loop iterations | Monitor for regression | Agent loop counter |

#### Quality Metrics

| Metric | Definition | How to Measure |
|--------|-----------|----------------|
| **Task completion accuracy** | Did the agent produce the correct result? | Test suite, human review, LLM-as-judge |
| **Code correctness** | Does generated code pass tests? | Automated test execution |
| **Code quality score** | Maintainability, complexity, style | Static analysis (see Section 5) |
| **User satisfaction (CSAT)** | User-reported satisfaction | Thumbs up/down, surveys |
| **Hallucination rate** | Fraction of outputs containing fabricated information | LLM-as-judge + fact-checking |
| **Instruction following rate** | Did the agent follow the prompt correctly? | Rubric-based evaluation |

### 3.2 Observability Platforms

#### Langfuse

- **Type**: Open-source, vendor-neutral LLM observability
- **Architecture**: Self-hostable, OpenTelemetry-compatible
- **Key features**: Tracing, debugging, analytics, LLM-as-judge evaluations, custom scorer primitives
- **Integration**: Any LLM provider, framework-agnostic
- **Pricing**: Predictable unit-based model; free self-hosted tier
- **Strengths**: Full data control, no vendor lock-in, open-source
- **Weaknesses**: Teams must assemble orchestration layer themselves
- **URL**: [langfuse.com](https://langfuse.com)

#### LangSmith

- **Type**: Commercial LLM development/monitoring platform (by LangChain)
- **Architecture**: Cloud-hosted, deep LangChain integration
- **Key features**: Zero-config tracing for LangChain, full-stack tracing, live dashboards (latency, error rates, token consumption), evaluation framework
- **Integration**: Best with LangChain; supports other frameworks
- **Pricing**: Free tier (5K traces/month, 1 user); $39/user/month paid; custom enterprise
- **Strengths**: Seamless LangChain integration, production-ready monitoring, strong evaluation tooling
- **Weaknesses**: Vendor lock-in with LangChain, per-trace pricing scales expensively
- **URL**: [langchain.com/langsmith](https://www.langchain.com/langsmith)

#### Braintrust

- **Type**: Commercial evaluation and deployment platform
- **Architecture**: Cloud-hosted, CI/CD-integrated
- **Key features**: Automated evaluation with deployment blocking, CI/CD pipeline integration, statistical significance analysis, merge blocking on quality degradation
- **Integration**: Framework-agnostic
- **Pricing**: Multi-dimensional (data volume, scores, retention); free tier for small teams
- **Strengths**: Evaluation automation, non-technical reviewer UI, CI/CD native
- **Weaknesses**: Complex pricing model
- **URL**: [braintrust.dev](https://www.braintrust.dev)

#### Weights & Biases Weave

- **Type**: Commercial ML/LLM observability (extension of W&B platform)
- **Architecture**: Cloud-hosted, decorator-based instrumentation
- **Key features**: Automatic tracking via `@weave.op` decorator, token usage and cost tracking, latency monitoring, accuracy measurement, powerful visualizations, automatic versioning of datasets/code/scorers
- **Integration**: Any Python framework
- **Pricing**: Part of W&B pricing; can become expensive at scale
- **Strengths**: Mature ML platform, strong experiment tracking, purpose-built for agentic systems, visualization
- **Weaknesses**: Heavier setup, more expensive than alternatives
- **URL**: [docs.wandb.ai/weave](https://docs.wandb.ai/weave)

#### Helicone

- **Type**: Commercial LLM observability (proxy-based)
- **Architecture**: AI Gateway proxy -- single line URL change for integration
- **Key features**: Automatic request/response logging, cost tracking, latency monitoring, usage analytics
- **Integration**: Any LLM API (proxied)
- **Pricing**: Accessible, competitive
- **Strengths**: Fastest time-to-value (minutes to integrate), lightweight, purpose-built for LLMs
- **Weaknesses**: Less deep experiment tracking than W&B, proxy adds small latency
- **URL**: [helicone.ai](https://www.helicone.ai)

#### Other Notable Tools

- **Evidently AI**: Open-source ML monitoring with drift detection ([evidentlyai.com](https://www.evidentlyai.com))
- **Fiddler AI**: Enterprise model monitoring with explainability ([fiddler.ai](https://www.fiddler.ai))
- **Openlayer**: AI drift detection with automated alerts ([openlayer.com](https://www.openlayer.com))
- **Traceloop**: Automated prompt regression testing with CI/CD ([traceloop.com](https://www.traceloop.com))
- **Promptfoo**: Open-source LLM evaluation and red-teaming ([promptfoo.dev](https://www.promptfoo.dev))

### 3.3 Recommended Metrics Stack for aiai

For a self-improving AI system, the essential monitoring stack should track:

```
Layer 1: Infrastructure
  - API latency (P50, P95, P99)
  - Token consumption (input/output, per model)
  - Cost per task (broken down by model, step)
  - Error rates and retry counts

Layer 2: Agent Behavior
  - Steps per task (trend over time)
  - Tool call distribution and success rates
  - Context window utilization
  - Agent loop termination reasons

Layer 3: Quality
  - Task success rate (test-based)
  - Code quality metrics (static analysis)
  - LLM-as-judge scores (with calibration)
  - Regression rate vs. baseline

Layer 4: Improvement Signal
  - Benchmark scores over time (SWE-bench, Terminal-Bench)
  - Cost-quality ratio trend
  - Human review agreement rate
  - Self-improvement cycle effectiveness
```

---

## 4. LLM-as-Judge Evaluation

### 4.1 Methodology Overview

**LLM-as-judge** uses a strong LLM (typically GPT-4 class or above) to evaluate the outputs of other LLMs or agent systems. This approach was formalized in:

- **MT-Bench** ([Zheng et al., 2023](https://arxiv.org/abs/2306.05685)): 80 multi-turn questions across 8 categories (writing, roleplay, reasoning, math, coding, extraction, STEM, humanities). GPT-4 scores responses on a 1-10 scale.
- **AlpacaEval** ([Li et al., 2023](https://github.com/tatsu-lab/alpaca_eval)): Pairwise comparison of model outputs against a reference. Length-Controlled (LC) variant corrects for verbosity bias.
- **Chatbot Arena** ([lmsys.org](https://chat.lmsys.org/)): Human-preference-based Elo ratings via crowdsourced blind pairwise comparisons.

### 4.2 Evaluation Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Pointwise scoring** | Judge assigns a numeric score (1-10) to a single output | Absolute quality assessment |
| **Pairwise comparison** | Judge picks the better of two outputs | Relative ranking, A/B testing |
| **Reference-based** | Judge compares output to a gold reference | Tasks with known correct answers |
| **Rubric-based** | Judge evaluates against specific criteria | Structured, reproducible evaluation |
| **Multi-aspect** | Judge scores multiple dimensions independently | Detailed quality decomposition |

### 4.3 Known Biases

| Bias | Description | Severity | Mitigation |
|------|-------------|----------|------------|
| **Position bias** | Favors outputs in a specific position (usually first) | High | Randomize presentation order; evaluate both orderings |
| **Verbosity bias** | Prefers longer, more verbose responses | High | Length-Controlled evaluation (LC-AlpacaEval); normalize by length |
| **Self-enhancement bias** | GPT-4 rates GPT-4 outputs higher | Medium | Use judge from different model family |
| **Prompt sensitivity** | Results vary with evaluation prompt wording | Medium | Use validated, standardized prompts; test multiple phrasings |
| **Anchoring** | First exposure to quality level affects subsequent ratings | Low-Medium | Batch evaluation with randomized order |
| **Format bias** | Prefers markdown, bullet points, structured output | Low-Medium | Specify evaluation criteria that devalue formatting |

### 4.4 Reliability and Calibration

**Agreement with humans**: GPT-4 judge agrees with human evaluations at **>80%**, matching the rate of human-human agreement (from 3K expert votes and 3K crowdsourced votes in the MT-Bench study).

**For code quality specifically** (from [AXIOM, Dec 2025](https://arxiv.org/abs/2512.20159)):
- LLM-as-judge metrics often **hallucinate flaws** in functionality and code quality
- **Cannot reliably estimate refinement effort** (distinguishing tweaks from refactors)
- **Scoring consistency inversely correlates with procedural complexity**: complex agentic frameworks suffer more scoring variance
- Accuracy and reliability are **independent dimensions** -- different models excel in different aspects

**Best practices for reliable LLM-as-judge**:
1. **Randomize position** of model outputs in the prompt
2. Use **logprobs** instead of generated binary preferences
3. Use the **strongest available model** as judge (ideally from a different family than the model being judged)
4. **Simplify** the evaluator prompt
5. Use **explicit rubrics** with concrete criteria
6. **Ensemble** across multiple judge models or multiple runs
7. Apply **post-hoc quantitative calibration** to adjust scores
8. Validate judge agreement against **human annotations** on a held-out set

### 4.5 LLM-as-Judge for Code Quality

For evaluating code quality specifically, the recommended approach is multi-dimensional:

```python
# Example rubric for LLM-as-judge code evaluation
CODE_EVALUATION_RUBRIC = {
    "correctness": {
        "description": "Does the code solve the stated problem?",
        "scale": "1-5",
        "weight": 0.30
    },
    "readability": {
        "description": "Is the code clear, well-named, and easy to follow?",
        "scale": "1-5",
        "weight": 0.20
    },
    "maintainability": {
        "description": "Is the code modular, DRY, with clear separation of concerns?",
        "scale": "1-5",
        "weight": 0.20
    },
    "efficiency": {
        "description": "Is the algorithmic approach reasonable? No obvious performance issues?",
        "scale": "1-5",
        "weight": 0.15
    },
    "robustness": {
        "description": "Are edge cases handled? Is there appropriate error handling?",
        "scale": "1-5",
        "weight": 0.15
    }
}
```

**Caveat**: LLM-as-judge is useful as a **signal**, not as ground truth for code quality. It should be combined with deterministic metrics (static analysis, test results) for reliable evaluation.

---

## 5. Automated Code Quality Metrics

### 5.1 Complexity Metrics

#### Cyclomatic Complexity (CC)

**Definition**: The number of linearly independent execution paths through a code section. Each decision point (if/else, switch/case, loop, exception handler) adds one to the complexity.

**Formula**: `CC = E - N + 2P` where E = edges, N = nodes, P = connected components in the control flow graph.

**Interpretation**:
| CC Score | Risk Level | Interpretation |
|----------|-----------|----------------|
| 1-5 | Low | Simple, easy to test |
| 6-10 | Moderate | Reasonable complexity |
| 11-20 | High | Difficult to test, consider refactoring |
| 21-50 | Very High | Untestable, must refactor |
| 50+ | Extreme | Error-prone, unmaintainable |

**Tools**: `radon` (Python), ESLint complexity rule (JavaScript/TypeScript), `gocyclo` (Go)

#### Cognitive Complexity (SonarSource)

An improvement over cyclomatic complexity that accounts for **human comprehension difficulty**. It penalizes:
- Nested control flow (each nesting level increases the increment)
- Breaks in linear flow (e.g., `break`, `continue`, `goto`)
- Boolean operator sequences

**Key difference from CC**: A series of `if` statements at the same level has low cognitive complexity (linear reading), while deeply nested conditions have high cognitive complexity.

#### Halstead Metrics

**Components**:
- `n1` = number of distinct operators
- `n2` = number of distinct operands
- `N1` = total number of operators
- `N2` = total number of operands

**Derived metrics**:
- **Program Vocabulary**: `n = n1 + n2`
- **Program Length**: `N = N1 + N2`
- **Volume**: `V = N * log2(n)` -- represents the information content
- **Difficulty**: `D = (n1/2) * (N2/n2)` -- propensity for bugs
- **Effort**: `E = D * V` -- estimated effort to understand

### 5.2 Maintainability Index

**Original formula** (Oman & Hagemeister, 1992):

```
MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC)
```

Where V = Halstead Volume, CC = Cyclomatic Complexity, LOC = Lines of Code.

**Microsoft's normalized formula** (0-100 scale):

```
MI = MAX(0, (171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC)) * 100 / 171)
```

**SEI variant with comments**:

```
MI_SEI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC) + 50 * sin(sqrt(2.46 * perCM))
```

Where `perCM` = percentage of commented lines.

**Interpretation** (Microsoft scale):
| MI Score | Color | Interpretation |
|----------|-------|----------------|
| 20+ | Green | Good maintainability |
| 10-19 | Yellow | Moderate maintainability |
| 0-9 | Red | Low maintainability |

**Tools**: `radon` (Python `radon mi`), Visual Studio Code Metrics, SonarQube

### 5.3 Test Coverage Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Line coverage** | % of lines executed by tests | > 80% |
| **Branch coverage** | % of conditional branches taken | > 70% |
| **Function coverage** | % of functions called by tests | > 90% |
| **Mutation testing score** | % of code mutations caught by tests | > 60% |

**Tools**: `pytest-cov` (Python), `istanbul/nyc` (JavaScript), `go test -cover` (Go)

### 5.4 Type Coverage

| Metric | Definition | Target |
|--------|-----------|--------|
| **Type annotation coverage** | % of functions/parameters with type hints | > 90% for Python |
| **Type check pass rate** | % of files passing type checker | 100% |
| **Any-type usage** | Count of `Any` type annotations | Minimize |

**Tools**: `mypy` (Python), `pyright` (Python), `tsc --strict` (TypeScript)

### 5.5 Security Scanning

#### Bandit (Python SAST)

- Open-source Python static analysis for security issues
- Scans for common vulnerabilities: SQL injection, shell injection, hardcoded passwords, use of `eval()`, insecure crypto
- **Metrics**: Issue count by severity (LOW/MEDIUM/HIGH) and confidence
- Best for: Python-specific security scanning
- **URL**: [github.com/PyCQA/bandit](https://github.com/PyCQA/bandit)

#### Semgrep

- Multi-language SAST with custom rule engine
- **2,000+ community rules** + custom rule creation
- AI-powered detection (Nov 2025 beta) for business logic vulnerabilities (IDORs, broken authorization)
- Hybrid approach: deterministic rules + AI for logic flaws
- 80% of participants uncovered real IDORs that traditional scanners missed
- **Metrics**: Issues by severity, rule category, confidence
- **URL**: [semgrep.dev](https://semgrep.dev)

#### SonarQube

- Comprehensive code quality and security platform
- Automatic scanning on commits and PRs
- Industry compliance standards (OWASP, CWE)
- **Metrics**: Bugs, vulnerabilities, code smells, security hotspots, duplications, cognitive complexity
- **URL**: [sonarsource.com/products/sonarqube](https://www.sonarsource.com/products/sonarqube/)

### 5.6 AI-Generated Code Quality Findings (2025-2026)

Research from GitClear, SonarSource, and Second Talent reveals significant quality concerns:

- AI-generated code introduces **1.7x more overall issues** compared to human-written code
- **Maintainability and code quality errors are 1.64x higher** in AI-generated code
- 90% increase in AI adoption correlated with **9% climb in bug rates**
- AI adoption associated with **91% increase in code review time** and **154% increase in PR size**
- Up to **40% of AI-generated code contains security vulnerabilities** (SQL injection, XSS, weak authentication)

**Implication for aiai**: The self-improving system should track these metrics over time to ensure that AI-generated improvements are not degrading code quality even as they increase task completion rates.

### 5.7 Recommended Automated Quality Pipeline

```yaml
# Example CI/CD quality gate configuration
quality_gates:
  complexity:
    cyclomatic_max_per_function: 15
    cognitive_max_per_function: 20
    maintainability_index_min: 20

  coverage:
    line_coverage_min: 80
    branch_coverage_min: 70
    new_code_coverage_min: 90

  types:
    type_annotation_coverage_min: 90
    type_check_pass: true

  security:
    bandit_high_issues: 0
    semgrep_critical_issues: 0
    semgrep_high_issues: 0

  duplication:
    duplicate_lines_max_percent: 3

  style:
    linter_warnings: 0
    formatter_compliance: true
```

---

## 6. A/B Testing for AI Systems

### 6.1 Why A/B Testing is Different for AI Agents

Traditional A/B testing assumes deterministic systems -- same input always produces same output. AI agents introduce:

1. **Stochastic outputs**: Same prompt, different results each time (temperature > 0)
2. **High variance**: Much larger variance in output quality than traditional software
3. **Multi-step interactions**: A single "outcome" may involve 10+ LLM calls
4. **Confounding factors**: Time of day, API latency, context window effects
5. **Small sample sizes**: Complex tasks are expensive to evaluate; you may only have 50-100 test cases

### 6.2 Experimental Design

#### Comparison Dimensions

| What to Compare | Example | Key Metric |
|----------------|---------|------------|
| **Prompts** | System prompt v1 vs v2 | Task success rate, quality score |
| **Models** | GPT-4o vs Claude Opus | Success rate, cost, latency |
| **Scaffolds** | ReAct vs Plan-and-Execute | Steps to completion, success rate |
| **Parameters** | Temperature 0.0 vs 0.3 | Output diversity, correctness |
| **Tool configs** | Different retrieval strategies | Relevance, task completion |
| **Coordination** | Sequential vs parallel agents | Quality, cost, latency |

#### Sample Size Requirements

Due to high variance in LLM outputs, **power analysis is essential**:

```python
# Example power analysis for AI agent A/B test
from scipy.stats import norm
import numpy as np

def required_sample_size(baseline_rate, minimum_detectable_effect,
                          alpha=0.05, power=0.80):
    """Calculate required sample size per variant."""
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = norm.ppf(power)

    p_bar = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / \
         (p2 - p1) ** 2
    return int(np.ceil(n))

# Example: baseline 60% success, want to detect 10% improvement
n = required_sample_size(0.60, 0.10)
# Result: ~388 samples per variant
```

**Rule of thumb**: For detecting a 10% absolute improvement in success rate from a 60% baseline, you need roughly **400 samples per variant** with 80% power and 5% significance level. For smaller effects, sample sizes grow quadratically.

### 6.3 Statistical Methods

#### Frequentist Approaches

| Method | When to Use | Implementation |
|--------|------------|----------------|
| **Two-proportion z-test** | Binary outcomes (pass/fail) | `scipy.stats.proportions_ztest` |
| **Welch's t-test** | Continuous metrics (quality score) | `scipy.stats.ttest_ind` |
| **Mann-Whitney U test** | Non-normal continuous data | `scipy.stats.mannwhitneyu` |
| **Chi-square test** | Categorical outcomes | `scipy.stats.chi2_contingency` |
| **Bootstrap confidence intervals** | Any metric, small samples | Resampling with replacement |

#### Bayesian Approaches (Recommended for Small Samples)

[Parloa (Nov 2025)](https://www.parloa.com/labs/research/ai-agent-testing/) introduced a **hierarchical Bayesian model** for A/B testing AI agents:

**Key advantages over frequentist methods**:
- **75% reduction** in required sample size
- Continuous monitoring without multiple-comparison penalties
- Natural handling of **grouped/hierarchical data** (scenarios within conversations)
- **Partial pooling** shares statistical strength across groups without conflating them
- Posterior probability of improvement is directly interpretable ("82% probability that variant B is better")

**Implementation approach**:
1. Define **binary metrics** (success/failure) and **continuous metrics** (LLM-judge scores)
2. Model both in a single hierarchical framework
3. Use **Beta-Binomial** for binary outcomes, **Normal-Normal** for continuous
4. Group conversations by **scenario type** to reduce variance
5. Compute **posterior probability that B > A** for each metric

```python
# Simplified Bayesian A/B test for binary outcome
import numpy as np

def bayesian_ab_test(successes_a, trials_a, successes_b, trials_b,
                      prior_alpha=1, prior_beta=1, n_samples=100000):
    """
    Returns probability that B is better than A.
    Uses Beta-Binomial conjugate model.
    """
    # Posterior distributions
    posterior_a = np.random.beta(
        prior_alpha + successes_a,
        prior_beta + trials_a - successes_a,
        n_samples
    )
    posterior_b = np.random.beta(
        prior_alpha + successes_b,
        prior_beta + trials_b - successes_b,
        n_samples
    )

    prob_b_better = np.mean(posterior_b > posterior_a)
    expected_lift = np.mean(posterior_b - posterior_a)

    return {
        "prob_b_better": prob_b_better,
        "expected_lift": expected_lift,
        "credible_interval_95": np.percentile(posterior_b - posterior_a, [2.5, 97.5])
    }
```

### 6.4 Practical A/B Testing Workflow for aiai

```
1. Define hypothesis (e.g., "New prompt increases success rate by >5%")
2. Choose test set (golden dataset + random production samples)
3. Run power analysis to determine sample size
4. Execute both variants on same inputs (paired design reduces variance)
5. Collect metrics: success rate, quality score, cost, latency
6. Analyze with Bayesian model (probability of improvement)
7. Decision threshold: Deploy if P(B > A) > 0.90 AND no regression on safety metrics
8. Log experiment metadata for future reference
```

---

## 7. Regression Detection

### 7.1 The Regression Problem for AI Systems

AI systems can regress in subtle ways that traditional software testing may not catch:

- **Silent quality degradation**: The system still "works" but produces lower-quality outputs
- **Capability regression**: New improvements break previously working capabilities
- **Cost regression**: Changes that increase cost without proportional quality improvement
- **Latency regression**: Prompt changes that trigger longer reasoning chains
- **Safety regression**: Changes that increase hallucination or unsafe output rates

### 7.2 Golden Dataset Approach

**A golden dataset** is a curated collection of inputs and their ideal outputs or evaluation criteria.

**Composition** (start with 10-15 cases, grow over time):

| Category | Purpose | % of Dataset |
|----------|---------|--------------|
| **Happy path** | Common, straightforward tasks | 40% |
| **Edge cases** | Boundary conditions, unusual inputs | 25% |
| **Adversarial inputs** | Inputs designed to confuse the system | 15% |
| **Out-of-scope** | Tasks the system should gracefully decline | 10% |
| **Known failure cases** | Previously failed inputs (add after each bug) | 10% |

**Best practice**: Add new failures to the golden set after every production incident. This creates a "ratchet" -- the system can never regress on previously identified issues.

### 7.3 Regression Testing Pipeline

```
On every change (prompt, model, scaffold):

1. RUN golden dataset evaluation
   - Execute all golden dataset inputs against new version
   - Compute per-category scores

2. COMPARE against baseline
   - Compare scores to the last known-good version
   - Flag any category where score drops > threshold

3. GATE deployment
   - Block merge/deploy if regression detected
   - Require manual override for borderline cases

4. EXTEND dataset
   - If new failure found, add it to golden dataset
   - Re-baseline after confirming fix
```

**Tools for CI/CD integration**:
- **Promptfoo**: Open-source, supports CI/CD integration, custom scorers ([promptfoo.dev/docs/integrations/ci-cd](https://www.promptfoo.dev/docs/integrations/ci-cd/))
- **Braintrust**: Automated evaluation with merge blocking ([braintrust.dev](https://www.braintrust.dev))
- **Traceloop**: Automated prompt regression testing with LLM-as-judge ([traceloop.com](https://www.traceloop.com))
- **GitHub Actions**: Custom workflow triggered on PR with modified prompts

### 7.4 Canary Deployments for AI

**Strategy**: Gradually roll out changes to increasing traffic fractions while monitoring quality metrics.

```
Stage 1: Shadow deployment (0% live traffic)
  - Run new version alongside production on same inputs
  - Compare outputs without affecting users
  - Metric: output similarity, quality delta

Stage 2: Canary (5% traffic)
  - Route 5% of requests to new version
  - Monitor: error rate, latency, quality scores
  - Duration: 1-2 days minimum

Stage 3: Progressive rollout (5% -> 20% -> 50% -> 100%)
  - Increase traffic if metrics stable
  - Automatic rollback if regression detected
  - Monitor each stage for minimum duration

Rollback triggers:
  - Error rate increases by > 2x
  - P95 latency increases by > 50%
  - Success rate drops by > 5 percentage points
  - Any critical failure (security, safety)
```

### 7.5 Drift Detection

Drift occurs when the distribution of inputs, outputs, or quality metrics changes over time.

#### Statistical Methods

| Method | What It Detects | Sensitivity | Best For |
|--------|----------------|------------|----------|
| **Population Stability Index (PSI)** | Distribution shift in categorical/binned data | Moderate | Feature distributions, stable metric |
| **Kolmogorov-Smirnov (K-S) test** | Any distributional difference | High (too sensitive on large datasets) | Small-medium datasets |
| **Wasserstein Distance** | Distribution shift (considers magnitude) | Moderate-High | Good compromise sensitivity |
| **Earth Mover's Distance (EMD)** | Embedding space drift | Moderate | NLP/embedding monitoring |
| **CUSUM** | Mean shift in time series | Adjustable | Streaming metrics |

**PSI interpretation**:
| PSI Value | Interpretation |
|-----------|---------------|
| < 0.10 | No significant shift |
| 0.10 - 0.25 | Moderate shift, investigate |
| > 0.25 | Significant shift, action required |

**PSI formula**:
```
PSI = SUM((actual_% - expected_%) * ln(actual_% / expected_%))
```

#### Prompt Drift Detection

In 2025-2026, leading teams **treat prompts like production code**:

1. **Version control**: All prompts stored in version-controlled files
2. **Diff tracking**: Every prompt change generates a diff
3. **Impact assessment**: Automated evaluation on golden dataset for every prompt change
4. **Audit trail**: Log which prompt version was used for every production request
5. **Automated alerting**: Flag when output distribution shifts (even without prompt changes -- may indicate model provider updates)

### 7.6 Monitoring Dashboard Metrics

```
Dashboard: Agent Regression Monitor

[Rolling 7-day window]
- Success rate: 87.3% (baseline: 86.1%) [GREEN]
- P95 latency: 12.4s (baseline: 11.8s) [YELLOW - within tolerance]
- Cost per task: $0.42 (baseline: $0.38) [YELLOW - investigate]
- Error rate: 2.1% (baseline: 2.3%) [GREEN]
- Steps per task (avg): 6.2 (baseline: 5.8) [YELLOW]

[Per-category breakdown]
- Bug fixes: 91.2% (baseline: 89.0%) [GREEN]
- Feature additions: 78.4% (baseline: 80.1%) [YELLOW]
- Refactoring: 85.7% (baseline: 84.3%) [GREEN]
- Test writing: 93.1% (baseline: 92.5%) [GREEN]

[Drift alerts]
- Input length distribution: PSI = 0.08 [OK]
- Output length distribution: PSI = 0.15 [WARNING]
- Tool usage pattern: PSI = 0.04 [OK]
```

---

## 8. Cost-Quality Pareto Analysis

### 8.1 The Cost-Quality Tradeoff

Every AI system operates on a tradeoff between cost and quality. The goal is to find the **Pareto frontier** -- the set of configurations where you cannot improve one metric without degrading the other.

**Dimensions of cost**:
- Token cost (input + output, per model)
- Compute cost (inference time, GPU hours)
- API call count (per task)
- Human review cost (for quality validation)

**Dimensions of quality**:
- Task success rate
- Code correctness (test pass rate)
- Code quality (maintainability, readability)
- User satisfaction

### 8.2 Current LLM API Pricing (February 2026)

| Model | Input ($/M tokens) | Output ($/M tokens) | Relative Cost |
|-------|--------------------|--------------------|---------------|
| DeepSeek V3.2 | $0.14 | $0.28 | Lowest |
| Grok | $0.20 | ~$0.60 | Very Low |
| GPT-5 Nano | ~$0.50 | ~$1.50 | Low |
| Gemini 2.0 Flash | $1.25 | ~$3.75 | Low-Medium |
| GPT-4o | $5.00 | $15.00 | Medium |
| Claude Sonnet 4 | ~$3.00 | ~$15.00 | Medium |
| Claude Opus 4 | $15.00 | $75.00 | High |
| GPT-5 | ~$15.00 | ~$60.00 | High |

**Critical insight**: Output tokens cost **3-10x more** than input tokens (median ratio ~4x). Optimization should focus on reducing output verbosity.

### 8.3 Cost Optimization Strategies

| Strategy | Expected Savings | Tradeoff |
|----------|-----------------|----------|
| **Model routing** | 60-80% | Complexity of routing logic; occasional misrouting |
| **Prompt caching** | Up to 90% on cached tokens | Requires cache infrastructure; cache invalidation |
| **Batch APIs** | 50% discount | Higher latency (not real-time) |
| **Context truncation** | 30-50% on input tokens | May lose relevant context |
| **Output length limits** | 20-40% on output tokens | May truncate useful output |
| **Fine-tuned smaller models** | 70-90% | Training cost; domain specificity |
| **Prompt compression** | 10-30% | Minor quality impact |

### 8.4 Model Routing for Pareto Optimization

Route easy queries to cheap models and escalate to expensive models only when needed:

```python
# Conceptual model router
class ModelRouter:
    def __init__(self):
        self.tiers = {
            "simple": {
                "model": "gpt-5-nano",
                "cost_per_1k_tokens": 0.001,
                "max_complexity": "low"
            },
            "standard": {
                "model": "claude-sonnet-4",
                "cost_per_1k_tokens": 0.009,
                "max_complexity": "medium"
            },
            "premium": {
                "model": "claude-opus-4",
                "cost_per_1k_tokens": 0.045,
                "max_complexity": "high"
            }
        }

    def route(self, task):
        complexity = self.estimate_complexity(task)
        if complexity == "low":
            return self.tiers["simple"]
        elif complexity == "medium":
            return self.tiers["standard"]
        else:
            return self.tiers["premium"]

    def estimate_complexity(self, task):
        """
        Classify task complexity based on:
        - Number of files likely affected
        - Estimated lines of code to change
        - Whether multi-step reasoning is needed
        - Domain specificity
        """
        # Could use a small classifier model, heuristics, or
        # historical data on similar tasks
        pass
```

### 8.5 Pareto Frontier Construction

**Step 1**: Define objective functions
```
Objective 1: Maximize quality Q(x) -- e.g., success rate on benchmark
Objective 2: Minimize cost C(x) -- e.g., dollars per task
```

**Step 2**: Evaluate configurations
```
For each configuration x in {model, prompt, scaffold, parameters}:
    Run evaluation suite
    Record (Q(x), C(x))
```

**Step 3**: Identify Pareto-optimal points
```python
def pareto_frontier(points):
    """
    points: list of (quality, cost) tuples
    Returns indices of Pareto-optimal points.
    """
    pareto = []
    for i, (q_i, c_i) in enumerate(points):
        dominated = False
        for j, (q_j, c_j) in enumerate(points):
            if i != j and q_j >= q_i and c_j <= c_i and (q_j > q_i or c_j < c_i):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto
```

**Step 4**: Select operating point based on constraints
- If budget-constrained: Choose the highest-quality point within budget
- If quality-constrained: Choose the cheapest point meeting quality threshold
- If balanced: Use a utility function `U(Q, C) = w_q * Q - w_c * C`

### 8.6 Multi-Objective Optimization Methods

| Method | Description | When to Use |
|--------|-------------|------------|
| **Grid search** | Evaluate all combinations | Few dimensions, discrete choices |
| **Bayesian optimization (qLogNEHVI)** | Surrogate model + acquisition function for multi-objective | Many dimensions, expensive evaluations |
| **ParetoPrompt** | RL-based prompt optimization with preference pairs | Automated prompt optimization |
| **MOHOLLM** | LLM as surrogate + hierarchical search | Complex configuration spaces |
| **Weighted sum** | Combine objectives into single scalar | Simple tradeoffs, clear preferences |

**Practical approach** (recommended for aiai):

```
1. Start with 3-5 model configurations (cheap/medium/expensive)
2. Evaluate each on golden dataset (50-100 tasks)
3. Record: success_rate, quality_score, cost, latency
4. Plot Pareto frontier
5. Select configuration based on current priority (quality vs. cost)
6. Re-evaluate monthly as models improve and prices change
```

### 8.7 Cost-Quality Dashboard

```
                    Quality (% success)
                    |
               100% |                         * Opus-4 ($0.85/task)
                    |                    * GPT-5 ($0.72/task)
                90% |               * Sonnet-4 ($0.31/task)
                    |
                80% |          * GPT-4o ($0.22/task)
                    |
                70% |     * DeepSeek ($0.04/task)
                    |
                60% | * GPT-5-Nano ($0.02/task)
                    |___________________________________________
                    $0.01    $0.10     $0.50    $1.00   Cost/task

Pareto frontier: GPT-5-Nano -> DeepSeek -> Sonnet-4 -> Opus-4
                 (Each offers best quality at its price point)

Current selection: Sonnet-4 (best cost-quality balance)
Upgrade trigger: When task requires >90% accuracy or multi-file changes
```

---

## 9. Practical Recommendations for aiai

### 9.1 Evaluation Strategy (Phased)

#### Phase 1: Foundation (Week 1-2)
- Set up **golden dataset** with 20-30 representative tasks across difficulty levels
- Integrate **one observability tool** (recommend Langfuse for open-source control, or Helicone for speed)
- Add **basic static analysis** to CI: cyclomatic complexity, type coverage, Bandit/Semgrep
- Define baseline metrics on current system

#### Phase 2: Benchmarking (Week 3-4)
- Run system against **SWE-bench Lite** (300 tasks) to establish baseline
- Run against **Terminal-Bench** subset for CLI task evaluation
- Set up **automated regression testing** in CI/CD (Promptfoo or custom)
- Implement **cost tracking** per task

#### Phase 3: Optimization (Month 2)
- Implement **model routing** (cheap model for simple tasks, expensive for complex)
- Build **Pareto frontier** analysis for model configurations
- Add **LLM-as-judge** evaluation with validated rubric
- Set up **A/B testing framework** with Bayesian analysis

#### Phase 4: Continuous Improvement (Ongoing)
- Monitor **drift detection** on input/output distributions
- Run **SWE-bench Live** monthly for contamination-free evaluation
- Expand golden dataset with every production failure
- Re-evaluate Pareto frontier as models improve and prices change

### 9.2 Metric Hierarchy

```
Level 0 (Safety): Does it avoid harmful outputs?
  -> Gate: Must pass 100% of safety checks

Level 1 (Correctness): Does the code work?
  -> Metric: Test pass rate (automated)
  -> Target: > 85%

Level 2 (Quality): Is the code good?
  -> Metrics: Maintainability Index > 20, CC < 15, type coverage > 90%
  -> Target: No regression from baseline

Level 3 (Efficiency): Is it cost-effective?
  -> Metrics: Cost per task, tokens per task, latency
  -> Target: Pareto-optimal for current quality level

Level 4 (Improvement): Is it getting better?
  -> Metrics: Benchmark scores over time, golden dataset trend
  -> Target: Positive trend over 30-day rolling window
```

### 9.3 Key Tools Shortlist

| Need | Recommended Tool | Alternative |
|------|-----------------|-------------|
| Observability | Langfuse (self-hosted) | Helicone (SaaS) |
| Evaluation/CI | Promptfoo | Braintrust |
| Static analysis | Ruff + Semgrep | SonarQube |
| Type checking | mypy / pyright | -- |
| Test coverage | pytest-cov | -- |
| Benchmarking | SWE-bench Lite/Live | Terminal-Bench |
| Cost tracking | Helicone | Custom via API logs |
| A/B testing | Custom Bayesian (PyMC) | Braintrust |
| Drift detection | Evidently AI | Custom PSI monitoring |

---

## 10. References

### SWE-bench

- [SWE-bench: Can Language Models Resolve Real-world Github Issues?](https://arxiv.org/abs/2310.06770) -- Jimenez et al., ICLR 2024
- [SWE-bench Leaderboard](https://www.swebench.com/) -- Official leaderboard
- [SWE-bench Verified Leaderboard (Feb 2026)](https://www.marc0.dev/en/leaderboard)
- [SWE-bench Verified | Epoch AI](https://epoch.ai/benchmarks/swe-bench-verified)
- [Introducing SWE-bench Verified | OpenAI](https://openai.com/index/introducing-swe-bench-verified/)
- [SWE-bench GitHub Repository](https://github.com/SWE-bench/SWE-bench)
- [SWE-bench Multimodal](https://www.swebench.com/multimodal.html)
- [SWE-bench Goes Live! (NeurIPS 2025)](https://arxiv.org/abs/2505.23419) -- Microsoft
- [SWE-bench Live Leaderboard](https://swe-bench-live.github.io/)
- [SWE-bench Verified Task Difficulty Analysis](https://jatinganhotra.dev/blog/swe-agents/2025/04/15/swe-bench-verified-easy-medium-hard.html) -- Ganhotra, 2025
- [SWE-bench Pro: Long-Horizon Tasks](https://scale.com/research/swe_bench_pro) -- Scale AI
- [SWE-bench Pro Leaderboard](https://scale.com/leaderboard/swe_bench_pro_public)
- [Sonar Claims Top Spot (Feb 2026)](https://www.sonarsource.com/company/press-releases/sonar-claims-top-spot-on-swe-bench-leaderboard/)
- [SWE-bench February 2026 Leaderboard Update](https://simonwillison.net/2026/Feb/19/swe-bench/) -- Simon Willison

### Agent Benchmarks

- [GAIA: A Benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983) -- Mialon et al., 2023
- [GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard) -- Hugging Face
- [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688) -- ICLR 2024
- [WebArena](https://webarena.dev/) -- Zhou et al., 2023
- [OSWorld](https://os-world.github.io/) -- Xie et al., 2024
- [Terminal-Bench](https://arxiv.org/abs/2601.11868) -- Laude Institute + Stanford, 2025
- [Terminal-Bench 2.0 | VentureBeat](https://venturebeat.com/ai/terminal-bench-2-0-launches-alongside-harbor-a-new-framework-for-testing)
- [MLAgentBench](https://arxiv.org/abs/2310.03302) -- Huang et al., 2023
- [MLE-bench](https://arxiv.org/abs/2410.07095) -- OpenAI, 2024
- [MLR-Bench](https://arxiv.org/abs/2505.19955) -- 2025
- [Agent Benchmark Compendium](https://github.com/philschmid/ai-agent-benchmark-compendium) -- Phil Schmid
- [Best AI Agent Benchmarks 2025 Guide](https://o-mega.ai/articles/the-best-ai-agent-evals-and-benchmarks-full-2025-guide) -- O-mega

### LLM-as-Judge

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) -- Zheng et al., 2023
- [AXIOM: Benchmarking LLM-as-a-Judge for Code](https://arxiv.org/abs/2512.20159) -- Dec 2025
- [AlpacaEval GitHub](https://github.com/tatsu-lab/alpaca_eval)
- [LLMs-as-Judges: A Comprehensive Survey](https://arxiv.org/abs/2412.05579)
- [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594)
- [LLM-as-a-Judge Practical Guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) -- Evidently AI
- [LLM-as-a-Judge Wikipedia](https://en.wikipedia.org/wiki/LLM-as-a-Judge)

### Production Monitoring

- [Langfuse](https://langfuse.com)
- [LangSmith](https://www.langchain.com/evaluation)
- [Braintrust](https://www.braintrust.dev)
- [Weights & Biases Weave](https://docs.wandb.ai/weave)
- [Helicone](https://www.helicone.ai)
- [Helicone LLM Observability Comparison](https://www.helicone.ai/blog/the-complete-guide-to-LLM-observability-platforms)
- [Best LLM Observability Tools 2025](https://www.firecrawl.dev/blog/best-llm-observability-tools)
- [LangWatch Comparison](https://langwatch.ai/blog/langwatch-vs-langsmith-vs-braintrust-vs-langfuse-choosing-the-best-llm-evaluation-monitoring-tool-in-2025)

### Code Quality

- [Code Quality in 2025 | Qodo](https://www.qodo.ai/blog/code-quality/)
- [AI Code Quality 2025 Research | GitClear](https://www.gitclear.com/ai_assistant_code_quality_2025_research)
- [AI-Generated Code Quality Metrics 2026 | Second Talent](https://www.secondtalent.com/resources/ai-generated-code-quality-metrics-and-statistics-for-2026/)
- [Radon Metrics Documentation](https://radon.readthedocs.io/en/latest/intro.html)
- [Maintainability Index | Microsoft](https://learn.microsoft.com/en-us/visualstudio/code-quality/code-metrics-maintainability-index-range-and-meaning)
- [Semgrep](https://semgrep.dev)
- [Bandit](https://github.com/PyCQA/bandit)
- [SonarQube](https://www.sonarsource.com/products/sonarqube/)
- [AI Code Security Benchmark 2025](https://sanj.dev/post/ai-code-security-tools-comparison)

### A/B Testing

- [How to A/B Test AI Agents With a Bayesian Model | Parloa](https://www.parloa.com/labs/research/ai-agent-testing/)
- [A/B Testing Strategies for AI Agents | Maxim](https://www.getmaxim.ai/articles/a-b-testing-strategies-for-ai-agents-how-to-optimize-performance-and-quality/)
- [A/B Testing Prompts Guide | Maxim](https://www.getmaxim.ai/articles/how-to-perform-a-b-testing-with-prompts-a-comprehensive-guide-for-ai-teams/)
- [AgentA/B: Automated Web A/B Testing](https://arxiv.org/abs/2504.09723)
- [AI Model Versioning and A/B Testing | Dynatrace](https://www.dynatrace.com/news/blog/the-rise-of-agentic-ai-part-6-introducing-ai-model-versioning-and-a-b-testing-for-smarter-llm-services/)

### Regression and Drift Detection

- [Automated Prompt Regression Testing with CI/CD | Traceloop](https://www.traceloop.com/blog/automated-prompt-regression-testing-with-llm-as-a-judge-and-ci-cd)
- [Ship Prompts Like Software | anup.io](https://www.anup.io/ship-prompts-like-software-regression-testing-for-llms/)
- [CI/CD for Evals in GitHub Actions | Kinde](https://www.kinde.com/learn/ai-for-software-engineering/ai-devops/ci-cd-for-evals-running-prompt-and-agent-regression-tests-in-github-actions/)
- [LLM Testing Guide | Langfuse](https://langfuse.com/blog/2025-10-21-testing-llm-applications)
- [CI/CD Integration | Promptfoo](https://www.promptfoo.dev/docs/integrations/ci-cd/)
- [Drift Detection | Evidently AI](https://www.evidentlyai.com/blog/data-drift-detection-large-datasets)
- [Population Stability Index | Arize AI](https://arize.com/blog-course/population-stability-index-psi/)
- [MLOps Best Practices 2025](https://www.thirstysprout.com/post/mlops-best-practices)

### Cost-Quality Optimization

- [Cost-Aware Model Selection: Multi-Objective Trade-offs](https://arxiv.org/abs/2602.06370) -- Feb 2025
- [Navigating the Pareto Frontier | Stanford CS224R](https://cs224r.stanford.edu/projects/pdfs/CS224R_Project_Final_Report.pdf)
- [Pareto Prompt Optimization | OSTI](https://www.osti.gov/servlets/purl/2543057)
- [MOHOLLM: Multi-Objective Hierarchical Optimization](https://arxiv.org/abs/2601.13892)
- [LLM API Pricing Comparison 2026](https://pricepertoken.com/)
- [LLM Cost Comparison Guide 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
- [Helicone LLM Cost Calculator](https://www.helicone.ai/llm-cost)

---

*Research compiled February 26, 2026. Landscape is evolving rapidly -- recommend re-evaluating benchmarks and tools quarterly.*
