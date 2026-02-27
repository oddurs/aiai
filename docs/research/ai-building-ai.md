# AI Building AI: The Complete Landscape (2024-2026)

> Foundational research document for the **aiai** project -- AI that builds itself, fully autonomous, no human gates.
> Last updated: 2026-02-26

---

## Table of Contents

1. [AI Writing AI Code](#1-ai-writing-ai-code)
   - 1.1 [GitHub Copilot](#11-github-copilot)
   - 1.2 [Cursor](#12-cursor)
   - 1.3 [Claude Code](#13-claude-code)
   - 1.4 [OpenAI Codex](#14-openai-codex)
   - 1.5 [Devin](#15-devin)
   - 1.6 [Success Rates and Benchmarks](#16-success-rates-and-benchmarks)
   - 1.7 [Quality Metrics and Limitations](#17-quality-metrics-and-limitations)
2. [Automated Machine Learning (AutoML)](#2-automated-machine-learning-automl)
   - 2.1 [Neural Architecture Search (NAS)](#21-neural-architecture-search-nas)
   - 2.2 [Hyperparameter Optimization](#22-hyperparameter-optimization)
   - 2.3 [Automated Feature Engineering](#23-automated-feature-engineering)
   - 2.4 [LLM-Guided NAS: AI Designing AI Architectures](#24-llm-guided-nas-ai-designing-ai-architectures)
3. [Self-Bootstrapping AI Systems](#3-self-bootstrapping-ai-systems)
   - 3.1 [The Compiler Analogy](#31-the-compiler-analogy)
   - 3.2 [Capability Bootstrapping Patterns](#32-capability-bootstrapping-patterns)
   - 3.3 [AlphaEvolve: Evolving Code with LLMs as Mutation Operators](#33-alphaevolve-evolving-code-with-llms-as-mutation-operators)
   - 3.4 [FunSearch: Program Search for Mathematical Discovery](#34-funsearch-program-search-for-mathematical-discovery)
   - 3.5 [DeepSeek-R1: Bootstrapping Reasoning from Scratch](#35-deepseek-r1-bootstrapping-reasoning-from-scratch)
4. [The Acceleration Thesis](#4-the-acceleration-thesis)
   - 4.1 [Recursive Self-Improvement Theory](#41-recursive-self-improvement-theory)
   - 4.2 [Intelligence Explosion Concepts](#42-intelligence-explosion-concepts)
   - 4.3 [Empirical Evidence for Acceleration](#43-empirical-evidence-for-acceleration)
   - 4.4 [Current Practical Limits](#44-current-practical-limits)
   - 4.5 [Why Full Autonomy Matters for Velocity](#45-why-full-autonomy-matters-for-velocity)
5. [AI Research Agents](#5-ai-research-agents)
   - 5.1 [AlphaFold: AI Solving Scientific Problems](#51-alphafold-ai-solving-scientific-problems)
   - 5.2 [The AI Scientist](#52-the-ai-scientist)
   - 5.3 [FunSearch and AlphaEvolve as Research Tools](#53-funsearch-and-alphaevolve-as-research-tools)
   - 5.4 [ML Research Benchmarks](#54-ml-research-benchmarks)
6. [Synthetic Data and Self-Play](#6-synthetic-data-and-self-play)
   - 6.1 [AI Generating Training Data for AI](#61-ai-generating-training-data-for-ai)
   - 6.2 [Self-Play in Reinforcement Learning](#62-self-play-in-reinforcement-learning)
   - 6.3 [SWE-RL: Self-Play for Software Engineering](#63-swe-rl-self-play-for-software-engineering)
   - 6.4 [Constitutional AI's Self-Improvement](#64-constitutional-ais-self-improvement)
   - 6.5 [Model Collapse: The Central Risk](#65-model-collapse-the-central-risk)
7. [End-to-End Autonomous Development](#7-end-to-end-autonomous-development)
   - 7.1 [The Full Pipeline](#71-the-full-pipeline)
   - 7.2 [What Exists Today](#72-what-exists-today)
   - 7.3 [What Is Still Missing](#73-what-is-still-missing)
   - 7.4 [Reference Architecture for Autonomous Development](#74-reference-architecture-for-autonomous-development)
8. [Practical Patterns for aiai](#8-practical-patterns-for-aiai)
   - 8.1 [Test-Driven Autonomous Development](#81-test-driven-autonomous-development)
   - 8.2 [Git as Safety Net](#82-git-as-safety-net)
   - 8.3 [Cost-Optimized Model Cascading](#83-cost-optimized-model-cascading)
   - 8.4 [Metric-Driven Self-Improvement](#84-metric-driven-self-improvement)
   - 8.5 [The aiai Bootstrap Sequence](#85-the-aiai-bootstrap-sequence)

---

## 1. AI Writing AI Code

The most concrete evidence that AI is building AI comes from the coding agents that now write, test, and commit production code. As of early 2026, AI-generated code has moved from experiment to infrastructure. 41% of all code written globally is now AI-generated or AI-assisted, and the figure is rising fast.

### 1.1 GitHub Copilot

**What it is:** GitHub Copilot is an AI coding assistant embedded in VS Code, JetBrains, Eclipse, Xcode, and as a CLI tool. It started as autocomplete in 2022 and has evolved into a full autonomous coding agent.

**Scale:** Over 20 million users (July 2025), 1.3 million paid subscribers, used by 90% of Fortune 100 companies. Copilot generates an average of 46% of code for its users, with Java developers reaching 61%.

**Autonomous Agent Mode (2025-2026):**

The Copilot coding agent, announced at Microsoft Build 2025, operates asynchronously. When you assign a GitHub issue to Copilot, it:

1. Spins up a secure development environment powered by GitHub Actions.
2. Reads the repository and plans its approach.
3. Writes code, creates tests, and iterates until tests pass.
4. Pushes commits to a draft pull request.
5. Logs every step for human review in session logs.

As of February 2026, Copilot CLI added "autopilot mode" -- full autonomy where it executes tools, runs commands, and iterates without stopping for approval. It automatically delegates to specialized sub-agents:

| Sub-Agent | Role |
|-----------|------|
| **Explore** | Fast codebase analysis and navigation |
| **Task** | Running builds, test suites, and scripts |
| **Code Review** | High-signal review of changes |
| **Plan** | Implementation planning for complex tasks |

**Limitations:** Excels at low-to-medium complexity tasks in well-tested codebases. Struggles with novel architectural decisions, cross-repository changes, and ambiguous requirements. Only ~30% of suggestions are accepted by developers, though 88% of accepted suggestions are retained in final code.

**Sources:**
- [GitHub Introduces Coding Agent For GitHub Copilot](https://github.com/newsroom/press-releases/coding-agent-for-github-copilot)
- [Copilot CLI is now generally available](https://github.blog/changelog/2026-02-25-github-copilot-cli-is-now-generally-available/)
- [About GitHub Copilot coding agent](https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-coding-agent)

### 1.2 Cursor

**What it is:** Cursor is an AI-native IDE built as a fork of VS Code, designed from the ground up around agent-driven coding. It surpassed 7 million monthly active users in 2025, with 40,000+ paying teams and $500M+ ARR.

**Agent Mode:** Cursor's agent can read the entire codebase, plan changes across dozens of files, and execute tasks autonomously. The landmark 0.50 release (late 2025) introduced Background Agents -- agents that work on long-running tasks like refactoring, test monitoring, and PR reviews while the developer focuses on other work.

**Key features (2025-2026):**

- **Long-Running Agents**: Autonomous work over longer horizons for complex tasks. Plans first, then finishes difficult work without intervention.
- **Mission Control**: Grid-view interface (like macOS Expose) for monitoring multiple in-progress agent tasks, switching between architectural plans.
- **Agent Sub-skills and Custom Tools**: Web search, image generation, API calls, and custom tooling as part of coding workflows.
- **Visual Editor**: Drag-and-drop within rendered web apps, visual sliders for properties, "point and prompt" interactions.
- **Multi-model**: Supports Claude, GPT, Gemini, and open-source models. The developer picks the best model for each task type.

**Positioning:** Cursor occupies the "IDE-first" niche -- the developer stays in the editor and the AI works alongside them, in contrast to cloud-first agents like Codex or Devin. Its agent mode bridges the gap between interactive pair-programming and full autonomy.

**Sources:**
- [Cursor Agent](https://cursor.com/product)
- [Cursor AI Review 2025](https://skywork.ai/blog/cursor-ai-review-2025-agent-refactors-privacy/)
- [Opsera - Cursor Adoption](https://opsera.ai/blog/cursor-ai-adoption-trends-real-data-from-the-fastest-growing-coding-tool/)

### 1.3 Claude Code

**What it is:** Claude Code is Anthropic's agentic coding tool. It runs in the terminal, reads your entire codebase, edits files, runs commands, writes tests, creates commits, and manages git workflows through natural language.

**Scale:** Claude Code reached $1 billion annualized run rate within six months of launch. Boris Cherny (head of Claude Code) has stated he has not manually written a single line of code since November 2025.

**Autonomous capabilities:**

- **Multi-file editing**: Reads repository structure, plans an approach, writes across multiple files, runs tests, and can create pull requests.
- **Long-running autonomy**: Claude Sonnet 4.5 handles 30+ hours of autonomous coding while maintaining coherence across massive codebases. A Rakuten case study showed 7 hours of autonomous work achieving 99.9% numerical accuracy.
- **Agent Teams**: A "multi-agent" mode lets users spawn multiple Claude Code agents that work in parallel. A lead agent coordinates the work, assigns subtasks, and merges results. Designed for read-heavy tasks like codebase reviews.
- **Checkpoints**: Saves progress and allows instant rollback to any previous state -- critical for autonomous operation where mistakes compound.
- **Full-auto mode**: When given CLAUDE.md instructions (as in the aiai project), Claude Code operates with zero human gates -- writing, testing, committing, and pushing code directly.

**Claude writing Claude:** Anthropic CEO Dario Amodei confirmed in September 2025 that "Claude is playing this very active role in designing the next Claude." By January 2026, over 90% of the code for new Claude models and features is authored autonomously by AI agents. Anthropic's internal development underwent a "phase transition" from human-centric programming to AI-primary development with humans as architects and auditors.

**Relevance to aiai:** Claude Code is the primary agent runtime for aiai. Its full-auto mode, checkpoint system, and git integration map directly to the aiai architecture.

**Sources:**
- [Claude Code overview](https://code.claude.com/docs/en/overview)
- [Eight trends defining how software gets built in 2026](https://claude.com/blog/eight-trends-defining-how-software-gets-built-in-2026)
- [Anthropic - Effective Harnesses for Long-Running Agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Anthropic Releases Claude Opus 4.6](https://www.marktechpost.com/2026/02/05/anthropic-releases-claude-opus-4-6-with-1m-context-agentic-coding-adaptive-reasoning-controls-and-expanded-safety-tooling-capabilities/)

### 1.4 OpenAI Codex

**What it is:** The 2025 incarnation of OpenAI Codex is a cloud-based software engineering agent that can work on many tasks in parallel. Unlike the 2021 autocomplete model, the new Codex is a fully autonomous agent.

**How it works:**

1. You assign a task (write a feature, fix a bug, answer a question about the codebase).
2. Codex spins up a secure, isolated cloud sandbox preloaded with your repository.
3. It writes code, executes it, runs tests, and iteratively refines until tests pass.
4. It proposes a pull request for review.
5. Internet access is disabled during execution for security.

**The model underneath:** Codex is powered by codex-1, a specialized version of OpenAI's o3 model optimized for coding tasks.

**GPT-5.3-Codex -- "The First Model That Helped Create Itself":** On February 5, 2026, OpenAI made a landmark claim: GPT-5.3-Codex was instrumental in creating itself. The Codex team used early versions of the model to:

- Debug its own training pipeline (finding race conditions, memory leaks, performance bottlenecks).
- Manage its own deployment (dynamically scaling GPU clusters during launch).
- Diagnose test results and evaluations (root-causing low cache hit rates, finding context rendering bugs).

The resulting model runs 25% faster than its predecessor and achieves state-of-the-art on SWE-Bench Pro across four languages. The release cadence itself is evidence of acceleration: GPT-5 (August 2025) to GPT-5.3-Codex (February 2026) -- four major releases in six months.

**Sources:**
- [Introducing Codex -- OpenAI](https://openai.com/index/introducing-codex/)
- [Introducing GPT-5.3-Codex -- OpenAI](https://openai.com/index/introducing-gpt-5-3-codex/)
- [OpenAI Codex: From 2021 Code Model to 2025 Autonomous Agent](https://medium.com/@aliazimidarmian/openai-codex-from-2021-code-model-to-a-2025-autonomous-coding-agent-85ef0c48730a)

### 1.5 Devin

**What it is:** Devin, by Cognition Labs, was introduced in March 2024 as "the first AI software engineer." It was the first agent to significantly outperform baselines on SWE-bench, achieving 13.86% (vs. the prior state-of-the-art of 1.96%).

**Real-world performance (2025 testing):**

Results are mixed. From independent testing of 20 tasks:
- 3 succeeded fully.
- 14 failed.
- 3 showed unclear results.

In practice, Devin completes about 15% of complex tasks without human assistance. However, it excels at specific task types:

- **File migrations**: 3-4 hours vs. 30-40 for human engineers (10x improvement).
- **Java version migrations**: 14x faster than a human engineer per repository.
- **Web scraping and API integrations**: Consistently strong performance.

**Devin 2.0 (April 2025):** Reduced pricing from $500/month to $20/month for the Core plan. Goldman Sachs announced piloting Devin alongside 12,000 human developers in July 2025, targeting 20% efficiency gains.

**Limitations:** Struggles with complex recursive functions, ambiguous requirements, and scenarios requiring deep domain knowledge. The gap between benchmark performance and real-world utility remains significant.

**Sources:**
- [Devin's 2025 Performance Review -- Cognition](https://cognition.ai/blog/devin-annual-performance-review-2025)
- [Devin AI Review: The Good, Bad & Costly Truth](https://trickle.so/blog/devin-ai-review)
- [Introducing Devin -- Cognition](https://cognition.ai/blog/introducing-devin)

### 1.6 Success Rates and Benchmarks

**SWE-bench** is the primary benchmark for AI coding agents. It presents real GitHub issues from popular Python repositories and measures whether agents can produce correct patches.

| Model / Agent | Date | SWE-bench Verified |
|---------------|------|--------------------|
| GPT-4o | May 2024 | ~33% |
| Claude 3.5 Sonnet | June 2024 | ~49% |
| Devin (Cognition) | March 2024 | 13.86% (original) |
| Claude 3.7 Sonnet | Feb 2025 | 62.3% |
| GPT-5 | Aug 2025 | 74.9% |
| Verdent | 2025 | 76.1% |
| Claude Opus 4.5 (SWE-Bench Pro) | Jan 2026 | 45.89% |
| Gemini 3 Pro (SWE-Bench Pro) | Jan 2026 | 43.30% |

The trajectory: from 4.4% in early 2024 to 76%+ by late 2025 -- a 67 percentage point leap in one year. SWE-bench Verified shows a doubling time of under 3 months.

**METR Time Horizons:** METR (Model Evaluation and Threat Research) measures the duration of tasks AI agents can complete autonomously. The task duration doubles every 4.3 months (post-2023). If extrapolated, AI agents handling multi-day autonomous projects become realistic within 2-3 years.

**Sources:**
- [SWE-bench Verified -- Epoch AI](https://epoch.ai/benchmarks/swe-bench-verified)
- [SWE-bench Leaderboards](https://www.swebench.com/)
- [METR Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/)

### 1.7 Quality Metrics and Limitations

**The trust gap is real:** Despite high adoption (92% of developers use AI coding tools regularly), only 33% fully trust AI-generated code. Acceptance rates hover around 30% -- most suggestions are rejected. But 88% of accepted suggestions are retained in the final code, meaning when AI gets it right, it stays right.

**Known failure modes of AI coding agents:**

1. **Hallucinated APIs**: Agents call functions or use library methods that do not exist.
2. **Shallow fixes**: Agents fix the symptom rather than the root cause, especially for complex bugs.
3. **Test-fitting**: When writing both code and tests, agents can write tests that verify broken behavior.
4. **Context loss**: On long tasks, agents lose track of earlier decisions and introduce contradictions.
5. **Over-engineering**: Agents tend to add unnecessary abstraction when simpler solutions suffice.
6. **Security blindness**: AI-generated code is as likely to contain vulnerabilities as human code, but moves faster.

**The quality equation:** AI agents are not better than expert humans at writing code. They are faster. The value proposition is speed multiplied by "good enough" quality, with tests as the quality gate.

---

## 2. Automated Machine Learning (AutoML)

AutoML is where AI designs AI architectures, selects models, tunes hyperparameters, and engineers features -- reducing or eliminating human involvement in the ML pipeline.

### 2.1 Neural Architecture Search (NAS)

Neural Architecture Search automates the design of neural network architectures. The field has evolved through three generations:

**First Generation -- Reinforcement Learning (2016-2018):**

- **NASNet** (Google Brain, 2017): RL controller searched for cell architectures. Achieved 90.84% on CIFAR-10 but required 2,050 GPU days (~$50K+ in compute).
- **AmoebaNet** (Google Brain, 2018): Evolutionary approach, 3,150 GPU days.

**Second Generation -- Weight-Sharing and Differentiable (2018-2020):**

- **DARTS** (CMU, 2018): Continuous relaxation enabling gradient-based optimization. Reduced search cost to 1.5-4 GPU days -- three orders of magnitude cheaper. However, DARTS suffers from performance collapse due to skip connection aggregation.
- **EfficientNet** (Google, 2019): Compound scaling of depth, width, and resolution. EfficientNetV2 (2021) further improved training speed.

**Third Generation -- LLM-Guided NAS (2024-2026):**

This is where the "AI building AI" thesis becomes concrete. LLMs encode architectural knowledge from their pretraining corpus (papers, code, documentation), enabling them to propose plausible architectures without exhaustive search.

### 2.2 Hyperparameter Optimization

Hyperparameter optimization is the most mature subfield of AutoML. Key approaches:

**Bayesian Optimization:** Uses a probabilistic surrogate model (typically Gaussian Process or Tree-structured Parzen Estimator) to model the objective function. Optuna is the dominant open-source framework:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = build_model(lr=lr, n_layers=n_layers, dropout=dropout)
    accuracy = train_and_evaluate(model)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

**Population-Based Training (PBT):** Trains a population of models in parallel, periodically copying weights from top performers to underperformers while mutating hyperparameters. Combines the advantages of random search (diversity) with hand-tuning (exploitation).

**LLM-Guided Tuning:** The emerging approach uses LLMs as the meta-optimizer. The LLM reads trial results, understands the hyperparameter landscape from pretraining knowledge, and proposes next configurations. Early results show competitive performance with far fewer trials than Bayesian optimization.

### 2.3 Automated Feature Engineering

Feature engineering remains the weakest link in full ML automation. Current approaches:

| Approach | Tool | Capability |
|----------|------|------------|
| **Automated transforms** | Featuretools, tsfresh | Generate polynomial, aggregation, and time-based features |
| **Deep feature synthesis** | Featuretools | Automatically traverses relational tables to create features |
| **LLM-guided features** | Emerging research | LLMs propose domain-specific features from data descriptions |
| **End-to-end learned** | AutoGluon, H2O | Model handles feature learning internally |

A 2025 study in *Scientific Reports* benchmarked AutoML tools and found no single tool consistently outperformed others. TransmogrifAI excelled in binary classification, AutoGluon in multiclass, and AutoKeras led in deep learning multiclass.

### 2.4 LLM-Guided NAS: AI Designing AI Architectures

The most significant recent development: using LLMs to design neural architectures, replacing expensive search with knowledge-guided proposals.

| Method | Key Innovation | Search Cost | Year |
|--------|---------------|-------------|------|
| **RZ-NAS** | Reflective zero-cost strategy with "humanoid reflections" | Minutes | 2025 (ICML) |
| **LLM-NAS** | Hardware-aware, complexity-driven partitioning + LLM co-evolution | Minutes (vs. days) | 2025 |
| **CoLLM-NAS** | Two LLMs: Navigator (guides direction) + Generator (synthesizes) | Reduced | 2025 |
| **PhaseNAS** | Phase-aware dynamic scaling; smaller LM for exploration, larger for exploitation | Cost-efficient | 2025 |

Key advances:

- **Zero-cost proxies** replace actual training to evaluate candidates, reducing search from days to minutes.
- **Hardware-awareness** is built in -- architectures are optimized for specific hardware (latency, memory, throughput).
- **LLMs as architecture priors**: The LLM's pretraining on millions of ML papers and codebases gives it implicit knowledge of what architectures work, making the search far more efficient than random or evolutionary methods.

**Sources:**
- [RZ-NAS -- ICML 2025](https://proceedings.mlr.press/v267/ji25a.html)
- [LLM-NAS -- arXiv](https://arxiv.org/abs/2510.01472)
- [AutoML: A systematic review -- ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2949715923000604)
- [Advances in neural architecture search -- National Science Review](https://academic.oup.com/nsr/article/11/8/nwae282/7740455)

---

## 3. Self-Bootstrapping AI Systems

How does a system build itself from minimal foundations? This is the central question for aiai.

### 3.1 The Compiler Analogy

The problem of creating AI that creates AI mirrors one of the oldest problems in computer science: how do you write a compiler for a language using that same language?

Historical precedents:

- **1958 -- NELIAC**: First high-level language to bootstrap itself.
- **1961 -- Burroughs B5000 Algol**: First widely used self-compiling language.
- **1962 -- LISP at MIT**: Hart and Levin wrote a LISP compiler in LISP, tested inside an existing LISP interpreter. Once it could compile its own source code, it was self-hosting.
- **Niklaus Wirth** wrote the first Pascal compiler in Fortran.

The universal pattern:

1. **Stage 0**: Write a minimal version in a different, already-existing language/system.
2. **Stage 1**: Use the minimal version to compile a more capable version written in the target language.
3. **Stage 2+**: Iteratively improve, using each version to compile the next.

Applied to AI:

1. **Stage 0** (pre-2024): Humans write AI systems from scratch.
2. **Stage 1** (2024-2025): AI assists humans in writing AI code. Copilot, Cursor, Claude Code help with boilerplate, debugging, and implementation.
3. **Stage 2** (2025-2026): AI writes the majority of AI code. OpenAI's GPT-5.3-Codex debugged its own training pipeline. Anthropic reports 90%+ of Claude code written by Claude. AlphaEvolve optimizes kernels used in Gemini training.
4. **Stage 3** (projected): AI manages autonomous AI development -- architecture search, training, evaluation, deployment, monitoring.

### 3.2 Capability Bootstrapping Patterns

Several technical approaches to bootstrapping have emerged:

**Pattern 1: Iterative Self-Distillation**

Train a model, use it to generate training data, filter the data, retrain. Each iteration produces a slightly more capable model. DeepSeek-R1-Zero demonstrates this: pure RL from scratch, bootstrapping reasoning ability from its own outputs.

```
initial_model -> generate(tasks) -> evaluate(outputs) -> filter(best) -> retrain -> improved_model
     ^                                                                                    |
     |____________________________________________________________________________________|
```

**Pattern 2: Evolutionary Search with LLM Mutation**

AlphaEvolve's approach. Maintain a population of candidate solutions. Use LLMs to propose mutations. Evaluate with automated metrics. Select the best. Repeat.

```
population -> select(top_k) -> llm_mutate(selected) -> evaluate(mutants) -> update_population
     ^                                                                            |
     |____________________________________________________________________________|
```

**Pattern 3: Recursive Self-Aggregation (RSA)**

Aggregate populations of reasoning chains at each refinement step to leverage partial correctness. Multiple candidate solutions contribute their best parts to the next generation.

**Pattern 4: Bootstrapping Task Spaces**

Formalized by arXiv research in September 2025: self-improvement by bootstrapping increasingly difficult task distributions. The system creates its own curriculum, starting with tasks it can solve and gradually generating harder ones.

```python
# Conceptual task bootstrapping loop
def bootstrap_capability(agent, initial_tasks, generations=10):
    current_tasks = initial_tasks
    for gen in range(generations):
        # Solve current tasks
        solutions = agent.solve_batch(current_tasks)

        # Evaluate and filter
        successful = [(task, sol) for task, sol in zip(current_tasks, solutions)
                      if evaluate(task, sol).passed]

        # Train on successful solutions
        agent.train(successful)

        # Generate harder tasks based on current capability boundary
        current_tasks = agent.generate_harder_tasks(
            solved=successful,
            difficulty_increment=1.2
        )

    return agent
```

### 3.3 AlphaEvolve: Evolving Code with LLMs as Mutation Operators

AlphaEvolve, unveiled by Google DeepMind in May 2025, is the most significant demonstration of AI self-improvement via code evolution.

**Architecture:**

AlphaEvolve has four tightly coupled components:

1. **Prompt Sampler**: Constructs context-rich prompts combining problem specification, high-performing programs from the database, and their evaluation scores. Biases toward top performers but maintains diversity to avoid premature convergence.

2. **LLM Ensemble (Mutation Operators)**: Two Gemini models serve as complementary mutation operators:

   | Model | Role | Purpose |
   |-------|------|---------|
   | Gemini 2.0 Flash | Breadth | High throughput, diverse algorithmic variants |
   | Gemini 2.0 Pro | Depth | Fewer but more insightful mutations |

   The LLMs propose diff-based code changes, not full rewrites. This is more sample-efficient and produces smaller, more interpretable changes.

3. **Evaluation Pipeline**: Automated evaluator that verifies, executes, and scores proposed programs. This is the critical constraint: AlphaEvolve requires a programmatically evaluable fitness function.

4. **Programs Database**: Stores all evaluated programs with their scores. Functions as both an evolutionary population and a retrieval system for the prompt sampler.

**Key results:**

- Sped up a kernel in Gemini's architecture by 23%, reducing Gemini training time by 1%.
- Discovered a heuristic for Borg (Google's cluster orchestrator) that continuously recovers 0.7% of Google's worldwide compute. In production for over a year.
- Found a procedure to multiply 4x4 complex-valued matrices using 48 scalar multiplications -- the first improvement over Strassen's 1969 algorithm in 56 years.
- Across 50 open mathematical problems: rediscovered state-of-the-art 75% of the time, improved upon it 20% of the time.

**The recursive loop:** AlphaEvolve optimizes kernels used to train Gemini. Gemini is the LLM backbone of AlphaEvolve. Each improvement to Gemini training makes AlphaEvolve more capable, which makes the next improvement more likely.

**OpenEvolve:** An open-source reimplementation of AlphaEvolve's approach, available on Hugging Face, making this pattern accessible to projects like aiai.

**Sources:**
- [AlphaEvolve -- Google DeepMind](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [AlphaEvolve -- arXiv](https://arxiv.org/abs/2506.13131)
- [OpenEvolve -- Hugging Face](https://huggingface.co/blog/codelion/openevolve)

### 3.4 FunSearch: Program Search for Mathematical Discovery

FunSearch (Google DeepMind, December 2023/Nature 2024) preceded AlphaEvolve and demonstrated the core pattern: using LLMs to evolve functions.

**How it works:**

1. Start with a simple seed function.
2. LLM proposes modifications to the function.
3. An evaluator scores each variant against a mathematical objective.
4. Top-performing variants are fed back to the LLM as context.
5. Iterate.

**Key difference from AlphaEvolve:** FunSearch evolves single functions; AlphaEvolve evolves entire codebases.

**Results:** FunSearch discovered new constructions of large cap sets in the cap set problem, going beyond the best-known solutions in both finite dimensional and asymptotic cases. It also found more effective bin-packing algorithms with direct practical applications for data center efficiency.

**Sources:**
- [FunSearch -- Google DeepMind](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
- [Mathematical discoveries from program search with large language models -- Nature](https://www.nature.com/articles/s41586-023-06924-6)

### 3.5 DeepSeek-R1: Bootstrapping Reasoning from Scratch

DeepSeek-R1 (January 2025) is a key datapoint for AI bootstrapping: reasoning abilities incentivized through pure reinforcement learning, without supervised fine-tuning or human-labeled reasoning trajectories.

- **DeepSeek-R1-Zero**: Trained via large-scale RL without SFT. Demonstrated self-verification, reflection, and long chain-of-thought generation -- all emergent from RL alone.
- Pass@1 on AIME 2024 increased from 15.6% to 71.0% through RL; with majority voting, 86.7% (matching OpenAI o1).
- The model bootstraps from earlier checkpoints of itself, using previous versions as ground-truth judges.
- Fully open-sourced: DeepSeek-R1-Zero, DeepSeek-R1, and six distilled dense models.

This demonstrates that a core cognitive capability (reasoning) can be bootstrapped entirely from self-play without human demonstrations.

**Sources:**
- [DeepSeek-R1 -- arXiv](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 -- Nature](https://www.nature.com/articles/s41586-025-09422-z)

---

## 4. The Acceleration Thesis

Is AI development accelerating? Can recursive self-improvement lead to an intelligence explosion? What are the practical limits? And why does full autonomy matter?

### 4.1 Recursive Self-Improvement Theory

Recursive self-improvement (RSI) is the hypothesis that an AI system could improve its own capabilities, which would in turn improve its ability to improve itself, creating an accelerating feedback loop.

The formal argument:

1. An AI system with capability level C can improve itself to level C + delta.
2. At level C + delta, it can improve itself to C + delta + delta', where delta' >= delta.
3. If delta' > delta consistently, the process accelerates.
4. If there is no hard ceiling, the process runs away.

**Current practical state:** Modern systems show bounded self-improvement. Agentic systems log errors, identify patterns, and iterate -- but humans still set the boundaries. The improvement is real but constrained by:

- The need for automated evaluation functions (no eval = no improvement signal).
- Error propagation over recursive cycles that causes stability breakdowns.
- Diminishing returns as easy improvements are exhausted.

### 4.2 Intelligence Explosion Concepts

The intelligence explosion hypothesis, first articulated by I.J. Good in 1965, posits that a sufficiently intelligent machine could design an even more intelligent machine, leading to a rapid cascade of improvements that far surpasses human intelligence.

**Three scenarios:**

1. **Hard takeoff**: Self-improvement happens in days or weeks. A system goes from human-level to vastly superhuman before anyone can respond. Most AI safety researchers consider this the most dangerous scenario.

2. **Soft takeoff**: Self-improvement happens over months or years. Capability increases are noticeable and measurable. This appears more consistent with current evidence.

3. **No takeoff**: Self-improvement hits diminishing returns. Each generation is marginally better, not explosively better. Compute and energy constraints dominate.

**Current evidence favors soft takeoff:** The METR time horizon data shows steady exponential improvement (doubling every ~4.3 months), not sudden jumps. This is fast -- but it is predictable and measurable.

**Expert estimates:** Some researchers place the window for potential loss of control at 2027-2030, but this is contested. Independent evaluators describe current self-improvement results as "very preliminary" and "quite underwhelming in the actual details."

### 4.3 Empirical Evidence for Acceleration

The strongest empirical evidence for acceleration:

**METR Time Horizons:**

| Metric | Value |
|--------|-------|
| Original doubling estimate (March 2025) | ~7 months |
| Refined estimate (2024 acceleration) | ~4 months |
| Time Horizon 1.1 (January 2026) | 130.8 days (4.3 months) |
| SWE-bench Verified doubling | Under 3 months |

**SWE-bench Progress:**

- Early 2024: ~4.4%
- March 2024 (Devin): 13.86%
- February 2025 (Claude 3.7 Sonnet): 62.3%
- August 2025 (GPT-5): 74.9%
- Late 2025 (Verdent): 76.1%

A 67 percentage point leap in roughly one year.

**Release Cadence Compression:**

OpenAI's 2025-2026 cadence: GPT-5 (August 2025) -> GPT-5.2 (December 2025) -> GPT-5.2-Codex (January 2026) -> GPT-5.3-Codex (February 2026). Four major releases in six months, with each partially built by the previous one.

**AI writing AI code at scale:**

- Anthropic: 90%+ of Claude code written by Claude (January 2026).
- Google: 50%+ of production code passing review each week is AI-generated (early 2026).
- OpenAI: GPT-5.3-Codex debugged its own training pipeline.

### 4.4 Current Practical Limits

Despite the evidence for acceleration, several hard constraints exist:

**1. Compute bottlenecks:** A 2025 analysis showed that compute and labor may be strong complements. Gains in cognitive labor (AI writing code) may not suffice without proportional compute increases for training and inference. Chips take years to design and fabricate.

**2. Evaluation bottleneck:** Self-improvement requires automated evaluation. Many important capabilities (creativity, safety, alignment, real-world impact) resist automated evaluation. Without an eval function, there is no gradient to follow.

**3. Diminishing returns on scaling:** Pre-training scaling laws show diminishing returns. The frontier has shifted to post-training (RLHF, RLAIF) and inference-time compute (chain-of-thought, search). These are powerful but not infinitely scalable.

**4. Data wall:** High-quality human-generated data is finite. Synthetic data helps but risks model collapse. The 74% of web content that is now AI-generated threatens the foundation on which LLMs are trained.

**5. Energy and infrastructure:** Training frontier models requires hundreds of megawatts. Data center construction has a multi-year lead time. Physical constraints impose hard limits on acceleration.

**6. Verification complexity:** As AI systems grow more capable, verifying their outputs becomes harder. The asymmetry between generation (easy) and verification (hard) is a fundamental constraint.

### 4.5 Why Full Autonomy Matters for Velocity

This is the core thesis of aiai. Why no human gates?

**The human bottleneck:** In a traditional development workflow, every change passes through human review. For a system that wants to iterate rapidly, humans become the rate-limiting factor:

```
Traditional: write -> PR -> wait for review -> address feedback -> re-review -> merge -> deploy
Time: hours to days per change

Autonomous:  write -> test -> commit -> deploy
Time: minutes per change
```

**The math of removal:**

If an AI agent can make a meaningful improvement every 10 minutes, but human review takes 2 hours:
- With human gates: 4 improvements per 8-hour day.
- Without human gates: 48 improvements per 8-hour day (12x).
- Running 24/7 without breaks: 144 improvements per day (36x).

But the gains are even larger than this, because:

1. **Compounding improvements**: Each improvement makes subsequent improvements faster or more effective. A 12x speedup compounds over iterations.
2. **No context switching**: Humans switching between review and other work lose context. Agents do not.
3. **Parallelism**: Multiple agents can work simultaneously on different improvements. Humans reviewing in parallel requires multiple humans.
4. **Consistency**: Agents apply the same quality standards every time. Human review quality varies with fatigue, mood, and attention.

**The tradeoff:** Full autonomy sacrifices human judgment for speed. The thesis is that automated tests plus git-based rollback provide sufficient safety at much higher throughput. This is empirically testable -- if autonomous operation produces more regressions than the velocity gain is worth, re-introduce gates.

**Evidence from industry:** Companies that have adopted high-autonomy AI coding workflows report 10-14x speedups on migration tasks (Devin), 90% reduction in engineering time on certain task types (Spotify), and compressed release cadences (OpenAI). The evidence suggests that for well-defined, well-tested domains, full autonomy pays off.

---

## 5. AI Research Agents

AI systems that do scientific research -- reading papers, forming hypotheses, running experiments, and writing up results.

### 5.1 AlphaFold: AI Solving Scientific Problems

AlphaFold is the most successful application of AI to a scientific problem in history.

**AlphaFold 2 (2020):** Solved the protein folding problem -- predicting a protein's 3D structure from its amino acid sequence with experimental-level accuracy. This had been an open problem for 50 years.

**Impact by 2025:**
- Over 240 million protein structure predictions in the database.
- Used by 3+ million researchers across 190+ countries.
- Tackling antimicrobial resistance, crop resilience, and heart disease.
- Demis Hassabis and John Jumper awarded the Nobel Prize in Chemistry (2024).

**AlphaFold 3 (2024-2025):** Predicts entire biomolecular complexes, not just single proteins. Can jointly model proteins with DNA, RNA, small molecules, ions, and post-translational modifications.

**Boltz-2 (2025):** Open-source alternative from MIT and Recursion. Can co-fold a protein-ligand pair and output both the 3D complex and a binding affinity estimate in about 20 seconds on a single GPU.

**Relevance to aiai:** AlphaFold demonstrates that AI can make fundamental scientific discoveries when given:
1. A well-defined problem with a clear evaluation metric.
2. Massive computational resources.
3. Domain-specific training data.

The pattern is: define the problem precisely, build an evaluation function, let AI search the solution space.

**Sources:**
- [AlphaFold -- Google DeepMind](https://deepmind.google/science/alphafold/)
- [AlphaFold: Five Years of Impact](https://deepmind.google/blog/alphafold-five-years-of-impact/)
- [What's next for AlphaFold -- MIT Technology Review](https://www.technologyreview.com/2025/11/24/1128322/whats-next-for-alphafold-a-conversation-with-a-google-deepmind-nobel-laureate/)

### 5.2 The AI Scientist

Sakana AI's **The AI Scientist** is the most ambitious attempt at fully automated scientific discovery.

**v1 (August 2024):** First comprehensive system for fully automated scientific discovery. It automates:
1. Generating novel research ideas.
2. Writing necessary code.
3. Executing experiments.
4. Summarizing and visualizing results.
5. Writing full scientific manuscripts.
6. Automated peer review of generated papers.

Cost: approximately $15 per paper.

**v2 (April 2025):** Significant upgrade:
- Removed reliance on human-authored templates.
- Generalized across ML domains.
- Uses progressive agentic tree search guided by an experiment manager agent.
- Generated the first workshop paper written entirely by AI and accepted through peer review at ICLR 2025's "I Can't Believe It's Not Better" workshop (scores: 6, 7, 6 -- top 45%).

**Honest assessment of limitations:**

Independent evaluations reveal serious problems:

| Issue | Detail |
|-------|--------|
| Experiment failure rate | 42% fail due to coding errors |
| Loop behavior | Gets stuck trying the same broken code repeatedly |
| Literature review quality | Misclassifies established concepts as novel |
| Code changes | Average of just 8% modified from template |
| Result accuracy | In one case, claimed improvement but results showed the opposite |
| Citation quality | Median of 5 citations, most outdated |

A 2025 evaluation concluded: "current AI Scientist systems lack the execution capabilities needed to execute rigorous experiments and produce high-quality scientific papers." The system is promising as a research tool but not yet a replacement for human researchers.

**Sources:**
- [The AI Scientist -- Sakana AI](https://sakana.ai/ai-scientist/)
- [The AI Scientist-v2 -- GitHub](https://github.com/SakanaAI/AI-Scientist-v2)
- [Evaluating Sakana's AI Scientist -- arXiv](https://arxiv.org/abs/2502.14297)

### 5.3 FunSearch and AlphaEvolve as Research Tools

FunSearch and AlphaEvolve (covered in Section 3) function as AI research agents for mathematics and computer science:

- FunSearch discovered new solutions to the cap set problem -- a genuine mathematical contribution.
- AlphaEvolve improved on Strassen's algorithm after 56 years.
- AlphaEvolve solved or improved 95% of 50 open mathematical problems it was applied to.

The key difference from The AI Scientist: these systems focus on well-defined optimization problems with clear evaluation functions, rather than open-ended research.

### 5.4 ML Research Benchmarks

Several benchmarks now measure AI agents on ML research tasks:

**MLAgentBench** (Stanford, 2023): 13 end-to-end ML experimentation tasks. The agent must autonomously develop or improve an ML model given a dataset and task description.

**MLE-bench** (OpenAI, October 2024): 75 ML engineering competitions from Kaggle. Best setup (o1-preview + AIDE scaffolding) achieves at least Kaggle bronze medal in 16.9% of competitions.

**MLR-Bench** (2025): 201 research tasks from NeurIPS, ICLR, and ICML workshops. Includes MLR-Judge (automated evaluation) and MLR-Agent (modular scaffold).

**Sources:**
- [MLAgentBench -- arXiv](https://arxiv.org/abs/2310.03302)
- [MLE-bench -- arXiv](https://arxiv.org/abs/2410.07095)
- [MLR-Bench -- OpenReview](https://openreview.net/forum?id=JX9DE6colf)

---

## 6. Synthetic Data and Self-Play

AI generating training data for AI. Self-play in reinforcement learning. Constitutional AI's self-improvement. And the risks of model collapse.

### 6.1 AI Generating Training Data for AI

Synthetic data generation has become essential to AI development. The motivation: high-quality human-generated data is finite and expensive, while AI can generate unlimited training data at minimal cost.

**Current scale:** By April 2025, over 74% of newly created webpages contained AI-generated text. This is both an opportunity (more data) and a threat (model collapse).

**Where synthetic data works:**

- **Privacy-preserving training**: Generate synthetic medical, financial, or personal data that preserves statistical properties without exposing real individuals.
- **Rare event augmentation**: Oversample conditions and edge cases that are underrepresented in real data.
- **Reasoning bootstrapping**: DeepSeek-R1's pure RL approach demonstrates self-generated reasoning traces bootstrapping reasoning ability.
- **Cost reduction**: Synthetic data generation is far cheaper than human annotation. The AI Scientist generates entire research papers for $15.

**Where synthetic data fails:**

- **Distribution shift**: Synthetic data does not perfectly match real-world distributions. Models trained primarily on synthetic data develop blind spots.
- **Compounding errors**: Errors in the generator propagate to the training data, which propagates to the next model, which generates worse training data.
- **Loss of diversity**: Generative models have mode collapse tendencies -- they overrepresent common patterns and underrepresent rare ones. Over generations, the tail of the distribution vanishes.

### 6.2 Self-Play in Reinforcement Learning

Self-play is a training paradigm where an AI agent generates its own training data by acting as both protagonist and adversary (or curriculum designer).

**Historical milestones:**

- **TD-Gammon** (1992): Backgammon through self-play, reaching world-champion level.
- **AlphaGo** (2016): Combined human expert games with self-play. Defeated world champion Lee Sedol.
- **AlphaGo Zero** (2017): Pure self-play, no human games. Surpassed AlphaGo within 40 hours.
- **AlphaZero** (2017): Generalized to chess, shogi, and Go. All from self-play, all superhuman.
- **OpenAI Five** (2019): Dota 2 at professional level via self-play.

**The self-play insight:** When the task has a clear objective function (win/lose, score, pass/fail), the agent can generate an infinite curriculum of increasingly challenging opponents -- itself. No human labeling required. No data collection required. The agent's training data improves as the agent improves.

**Modern self-play beyond games:**

```python
# Conceptual self-play training loop
def self_play_training(agent, environment, iterations=1000):
    for i in range(iterations):
        # Agent plays against a copy of itself
        opponent = agent.clone()

        # Generate trajectory through interaction
        trajectory = environment.run_episode(agent, opponent)

        # Compute rewards based on outcome
        rewards = environment.compute_rewards(trajectory)

        # Update agent via RL
        agent.update(trajectory, rewards)

        # Periodically evaluate against fixed benchmarks
        if i % 100 == 0:
            score = evaluate_fixed_benchmark(agent)
            log(f"Iteration {i}: benchmark score = {score}")

    return agent
```

### 6.3 SWE-RL: Self-Play for Software Engineering

**SWE-RL** (Meta/Facebook Research, 2025) is the first approach to scale RL-based LLM reasoning for real-world software engineering.

The approach: leverage open-source software evolution data (commit histories, pull requests, issue resolutions) as a natural source of task-reward pairs. The reward signal is the similarity between the LLM's generated solution and the ground-truth human solution.

Results: Llama3-SWE-RL-70B achieves a 41.0% solve rate on SWE-bench Verified.

**Self-Play SWE-RL (SSR):** Extends SWE-RL by removing the reliance on human-curated training data entirely. A single LLM agent is trained in a dual-role self-play game:

1. **Bug Injector Role**: The agent injects bugs into real codebases.
2. **Bug Fixer Role**: The same agent tries to fix the bugs.
3. **Curriculum**: Bug complexity increases iteratively.

The only requirement is access to sandboxed repositories with source code and installed dependencies. No human-labeled issues or tests needed.

```
inject_bug(codebase) -> broken_codebase -> fix_bug(broken_codebase) -> evaluate(fix) -> reward
     ^                                                                                    |
     | (increase difficulty if fix succeeds)                                              |
     |____________________________________________________________________________________|
```

This is directly relevant to aiai: a system that generates its own training signal by breaking and fixing its own code.

**Sources:**
- [SWE-RL -- arXiv](https://arxiv.org/abs/2502.18449)
- [Self-Play SWE-RL -- arXiv](https://arxiv.org/abs/2512.18552)
- [SWE-RL -- GitHub (Facebook Research)](https://github.com/facebookresearch/swe-rl)

### 6.4 Constitutional AI's Self-Improvement

Anthropic's Constitutional AI (CAI), published in December 2022, is the foundational example of AI self-improvement in production.

**Supervised Phase (Self-Critique and Revision):**

1. Start with an instruction-tuned model.
2. Generate responses to harmful prompts.
3. The model critiques its own outputs against a "constitution" -- a set of natural language principles.
4. The model revises its own outputs based on its critique.
5. Fine-tune on the revised responses.

```python
# Simplified Constitutional AI self-critique loop
def constitutional_revision(model, prompt, constitution):
    # Generate initial response
    response = model.generate(prompt)

    for principle in constitution:
        # Self-critique
        critique = model.generate(
            f"Given the principle: '{principle}'\n"
            f"Critique this response: '{response}'\n"
            f"Identify any violations of the principle."
        )

        # Self-revision
        response = model.generate(
            f"Given the critique: '{critique}'\n"
            f"Revise the response to address the concerns: '{response}'"
        )

    return response
```

**RL Phase (RLAIF -- Reinforcement Learning from AI Feedback):**

1. Generate pairs of responses.
2. A model evaluates which response is better (replacing human preference labeling).
3. Train a preference/reward model from AI preferences.
4. Use RL (PPO or DPO) to optimize against the AI-generated reward signal.

Results: CAI improved harmlessness by 40% at a cost of ~9% decrease in helpfulness.

The term RLAIF was coined in this work. By 2025, RLHF/RLAIF became the default alignment strategy for LLMs, with 70% of enterprises adopting these methods.

**Why this matters for aiai:** Constitutional AI demonstrates that an AI system can improve itself along a specified dimension (harmlessness) using only AI feedback, no human labeling. The pattern generalizes: define the dimension of improvement, write a constitution (principles), and let the system self-critique and revise.

**Sources:**
- [Constitutional AI -- Anthropic](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [Constitutional AI -- arXiv](https://arxiv.org/abs/2212.08073)
- [RLHF Book -- Constitutional AI](https://rlhfbook.com/c/13-cai)

### 6.5 Model Collapse: The Central Risk

Model collapse is a degenerative feedback loop that arises when generative models are trained on outputs of earlier models.

**What happens:**
- Over successive generations, the system's view of reality narrows.
- Rare details vanish, outputs become repetitive.
- The model loses the variability that makes human-generated content rich.
- Eventually, the model converges to a degenerate distribution.

**Two scenarios tested:**
- **Replace** (real data swapped with synthetic): Collapse is nearly inevitable.
- **Accumulate** (synthetic added to real): Pipelines showed remarkable resilience across model types and domains.

**Solutions (2025 state of the art):**

1. **Synthetic Data Verification**: Use a verifier (human or better model) to filter low-quality synthetic samples. Verification eliminates collapse even in iterative retraining.
2. **Real-Data Anchoring**: Keep a fixed human-authored anchor set of 25-30% in every retrain. The anchor preserves tail distribution information.
3. **Data Accumulation**: Never replace real data with synthetic -- always accumulate. The absolute amount of real data never decreases.
4. **Provenance Tracking**: Track data origins; filter synthetic content before it enters training pipelines.
5. **Quality Filtering**: Grammar checkers, LLM-as-judge, pretrained discriminators, or human annotation to screen synthetic data.

**Implication for aiai:** A self-improving system must be vigilant about model collapse. Every self-improvement cycle should be validated against a fixed benchmark set derived from real data. Ratchet mechanisms (never degrade below current best) are essential.

**Sources:**
- [Model Collapse -- Wikipedia](https://en.wikipedia.org/wiki/Model_collapse)
- [Escaping Model Collapse via Synthetic Data Verification -- arXiv](https://arxiv.org/abs/2510.16657)
- [AI training in 2026: anchoring synthetic data in human truth](https://invisibletech.ai/blog/ai-training-in-2026-anchoring-synthetic-data-in-human-truth)

---

## 7. End-to-End Autonomous Development

The full pipeline: task -> design -> implement -> test -> deploy -> monitor -> improve. Where does each piece stand?

### 7.1 The Full Pipeline

An end-to-end autonomous development system would handle:

```
1. TASK INTAKE        Receive a task description (natural language, issue, metric target)
         |
2. DESIGN             Decompose into subtasks, select architecture, plan implementation
         |
3. IMPLEMENT          Write code across multiple files, handle dependencies
         |
4. TEST               Generate tests, run test suites, fix failures, iterate
         |
5. REVIEW             Automated code review, security scanning, style checking
         |
6. DEPLOY             Push to staging/production, manage infrastructure
         |
7. MONITOR            Watch metrics, detect regressions, alert on anomalies
         |
8. IMPROVE            Analyze performance data, identify improvements, loop back to step 1
```

### 7.2 What Exists Today

**Task Intake (mostly solved):**
- Natural language task descriptions work well with Claude Code, Copilot, Codex.
- GitHub issue assignment triggers autonomous agents (Copilot coding agent).
- Structured task specs (YAML, JSON) enable programmatic task submission.

**Design (partially solved):**
- AI agents can decompose tasks into subtasks for well-understood domains.
- Architecture planning for novel systems remains weak.
- Multi-agent coordination (Claude Code agent teams) enables parallel subtask execution.

**Implementation (largely solved for bounded tasks):**
- SWE-bench Verified scores above 75% demonstrate strong implementation capability.
- Multi-file editing, dependency management, and API integration work reliably.
- Novel algorithm design and complex architectural work remain human-dependent.

**Testing (partially solved):**
- AI agents write tests and run them. The 2025 DORA report finds that TDD amplifies AI effectiveness.
- The risk: test-fitting, where agents write tests that verify broken behavior.
- 72% of QA teams are exploring AI-driven testing workflows (2025 Test Guild report).

**Review (largely solved):**
- Automated linting, type checking, and security scanning are mature.
- AI-powered code review (Cursor, Copilot) provides high-signal feedback.
- The gap: AI review catches syntax and pattern violations but misses semantic bugs.

**Deployment (partially solved):**
- CI/CD pipelines are highly automated. AI can write and modify pipeline configs.
- Infrastructure as Code generation is increasingly AI-assisted.
- Rollback and canary deployment are well-established patterns.

**Monitoring (partially solved):**
- Observability platforms (Datadog, Grafana) collect metrics automatically.
- Anomaly detection is mature for known failure modes.
- Novel failure mode detection remains challenging.

**Self-Improvement (early stage):**
- AlphaEvolve demonstrates improvement on well-defined optimization targets.
- Constitutional AI demonstrates improvement along specified dimensions.
- General-purpose self-improvement remains aspirational.

### 7.3 What Is Still Missing

1. **Problem formulation**: AI cannot reliably translate vague business requirements into well-specified problems. "Make the app faster" is not actionable without decomposition into specific metrics and targets.

2. **Data pipeline automation**: Data preprocessing, cleaning, and feature engineering remain the weakest link in end-to-end ML automation.

3. **Safety and alignment verification**: No automated system can reliably evaluate whether a model is safe for deployment in high-stakes domains.

4. **Novel architecture design**: AI excels at implementing known patterns but struggles with genuinely novel architectural decisions.

5. **Cross-system coordination**: Autonomous agents work well within a single repository but coordinating across microservices, databases, and infrastructure remains largely manual.

6. **Feedback loop design**: Deciding what to optimize for, how to collect feedback, and how to iterate still requires human judgment for consequential systems.

### 7.4 Reference Architecture for Autonomous Development

A practical architecture for an autonomous development system like aiai:

```
+-------------------------------------------------------------------+
|                        ORCHESTRATOR                                |
|  Receives tasks, decomposes, assigns to agents, tracks progress   |
+-------------------------------------------------------------------+
         |              |              |              |
    +---------+   +---------+   +---------+   +---------+
    | PLANNER |   | CODER   |   | TESTER  |   | REVIEWER|
    | Agent   |   | Agent   |   | Agent   |   | Agent   |
    +---------+   +---------+   +---------+   +---------+
         |              |              |              |
+-------------------------------------------------------------------+
|                    SHARED INFRASTRUCTURE                           |
|  Git repo | CI/CD | Test runner | Metrics DB | Model router       |
+-------------------------------------------------------------------+
         |              |              |              |
+-------------------------------------------------------------------+
|                    EVALUATION ENGINE                               |
|  Benchmarks | Regression tests | Quality metrics | Cost tracking  |
+-------------------------------------------------------------------+
         |
+-------------------------------------------------------------------+
|                    EVOLUTION ENGINE                                |
|  Identifies improvements | Proposes changes | Validates results   |
+-------------------------------------------------------------------+
```

Each agent is a Claude Code instance (or equivalent) with specific responsibilities. The orchestrator manages task assignment and tracks progress. The evaluation engine provides the fitness function. The evolution engine closes the self-improvement loop.

---

## 8. Practical Patterns for aiai

Concrete patterns this project should use, drawn from the research above.

### 8.1 Test-Driven Autonomous Development

**The core principle:** In autonomous AI development, tests are the only quality gate. There are no human reviewers. If the tests pass, the code ships.

**Why TDD is critical for AI agents:**

The 2025 DORA report finds that AI acts as an amplifier -- it makes existing good practices more effective. TDD is the most critical practice to have in place before giving an AI agent autonomy because it prevents the most dangerous failure mode: **agents writing tests that verify broken behavior**.

When an agent writes both the code and the tests, it can inadvertently write tests that confirm its bugs. TDD prevents this by requiring tests to exist (and ideally fail) before the implementation:

```python
# Pattern: Test-First Autonomous Development
#
# 1. Agent receives task specification
# 2. Agent writes failing tests FIRST
# 3. Human (or separate agent) reviews tests for correctness
# 4. Agent writes implementation to make tests pass
# 5. Agent runs full test suite
# 6. If all pass -> commit. If any fail -> fix and retry.

class TestRouterSelectsModel:
    """Tests written BEFORE implementation. Define expected behavior."""

    def test_trivial_task_routes_to_cheapest_model(self):
        router = ModelRouter(config="config/models.yaml")
        result = router.select(task_complexity="trivial")
        assert result.model in TIER_1_MODELS  # cheap, fast models
        assert result.cost_per_token < 0.001

    def test_critical_task_routes_to_best_model(self):
        router = ModelRouter(config="config/models.yaml")
        result = router.select(task_complexity="critical")
        assert result.model in TIER_5_MODELS  # most capable models
        assert result.quality_score > 0.95

    def test_fallback_on_model_unavailability(self):
        router = ModelRouter(config="config/models.yaml")
        # Simulate primary model being down
        with mock_model_unavailable("claude-opus-4.6"):
            result = router.select(task_complexity="complex")
            assert result.model is not None  # fallback worked
            assert result.model != "claude-opus-4.6"

    def test_cost_tracking_updates_after_call(self):
        router = ModelRouter(config="config/models.yaml")
        initial_cost = router.total_cost
        router.select(task_complexity="simple")
        assert router.total_cost > initial_cost
```

**The two-agent pattern:** Use one agent to write tests and a separate agent to write implementation. This creates adversarial pressure -- the test agent is trying to define correct behavior, the implementation agent is trying to satisfy those definitions. Neither agent's bugs automatically confirm the other's.

### 8.2 Git as Safety Net

**The core principle:** In a system with no human gates, git is the safety net. Every change is committed. Every commit can be reverted. The git history is the audit trail.

**Concrete patterns:**

**1. Small, atomic commits:**
```bash
# Every logical change is a single commit
# BAD: "refactor router and add caching and fix tests"
# GOOD: Three separate commits:
#   "refactor(router): extract model selection logic into separate method"
#   "feat(router): add response caching with TTL"
#   "fix(tests): update router tests for new API"
```

**2. Checkpoint before risky changes:**
```bash
# Before any operation that might break things, create a checkpoint
git tag checkpoint/pre-router-refactor

# Make changes, run tests
# If tests fail and the fix is not obvious:
git revert HEAD  # or
git reset --hard checkpoint/pre-router-refactor
```

**3. Automated rollback on test failure:**
```python
# Pattern: commit-or-revert
import subprocess

def commit_or_revert(message: str, files: list[str]) -> bool:
    """Commit changes if tests pass, revert if they fail."""
    # Stage changes
    subprocess.run(["git", "add"] + files, check=True)

    # Run tests
    result = subprocess.run(["python", "-m", "pytest", "tests/"], capture_output=True)

    if result.returncode == 0:
        # Tests pass -- commit
        subprocess.run(["git", "commit", "-m", message], check=True)
        return True
    else:
        # Tests fail -- revert staged changes
        subprocess.run(["git", "checkout", "--"] + files, check=True)
        return False
```

**4. Branch-based experimentation:**
```bash
# For changes with uncertain outcomes, use branches
git checkout -b evolve/optimize-prompt-template

# Make changes, test, iterate
# If improvement is validated:
git checkout main && git merge evolve/optimize-prompt-template

# If improvement fails:
git checkout main && git branch -d evolve/optimize-prompt-template
```

**5. Secret scanning on every commit:**
```bash
# agent-git.sh pattern from aiai
# Before every commit, scan for secrets
grep -rn "sk-" --include="*.py" src/ && echo "BLOCKED: possible API key" && exit 1
grep -rn "password=" --include="*.py" src/ && echo "BLOCKED: possible password" && exit 1
```

### 8.3 Cost-Optimized Model Cascading

**The core principle:** Use cheap models for simple tasks, expensive models for hard ones. Most tasks are simple.

**The economics:**

Research shows that 60-70% of queries can be handled by small, efficient models without quality loss. The cost difference between tiers is 8-20x. A well-implemented cascade achieves 40-85% cost reduction with zero quality loss on the tasks that matter.

**The cascade pattern:**

```python
# Model cascading implementation pattern for aiai
from dataclasses import dataclass
from enum import Enum

class Complexity(Enum):
    TRIVIAL = "trivial"    # formatting, renaming, simple lookups
    SIMPLE = "simple"      # single-file edits, straightforward implementations
    MEDIUM = "medium"      # multi-file changes, moderate reasoning
    COMPLEX = "complex"    # architecture decisions, hard bugs
    CRITICAL = "critical"  # system-wide changes, core architecture

@dataclass
class ModelTier:
    name: str
    models: list[str]
    cost_per_1k_tokens: float
    max_context: int

# From config/models.yaml
TIERS = {
    Complexity.TRIVIAL: ModelTier(
        name="tier1",
        models=["meta-llama/llama-3.3-70b-instruct"],
        cost_per_1k_tokens=0.0003,
        max_context=131072,
    ),
    Complexity.SIMPLE: ModelTier(
        name="tier2",
        models=["anthropic/claude-sonnet-4", "google/gemini-2.5-flash"],
        cost_per_1k_tokens=0.003,
        max_context=200000,
    ),
    Complexity.MEDIUM: ModelTier(
        name="tier3",
        models=["anthropic/claude-sonnet-4", "openai/gpt-4o"],
        cost_per_1k_tokens=0.01,
        max_context=200000,
    ),
    Complexity.COMPLEX: ModelTier(
        name="tier4",
        models=["anthropic/claude-opus-4", "openai/o3"],
        cost_per_1k_tokens=0.06,
        max_context=200000,
    ),
    Complexity.CRITICAL: ModelTier(
        name="tier5",
        models=["anthropic/claude-opus-4.6"],
        cost_per_1k_tokens=0.075,
        max_context=1000000,
    ),
}

class ModelRouter:
    """Routes tasks to appropriate models based on complexity."""

    def __init__(self, config_path: str = "config/models.yaml"):
        self.config = self._load_config(config_path)
        self.total_cost = 0.0
        self.call_log = []

    def select(self, task_complexity: str) -> ModelTier:
        complexity = Complexity(task_complexity)
        tier = TIERS[complexity]

        self.call_log.append({
            "complexity": task_complexity,
            "tier": tier.name,
            "model": tier.models[0],
        })

        return tier

    def escalate(self, current_complexity: str) -> ModelTier:
        """Escalate to the next tier if current model fails."""
        complexities = list(Complexity)
        current_idx = complexities.index(Complexity(current_complexity))
        if current_idx < len(complexities) - 1:
            next_complexity = complexities[current_idx + 1]
            return self.select(next_complexity.value)
        raise ValueError("Already at maximum complexity tier")
```

**The unified routing-cascading approach:** Research from ETH Zurich (2024) proposes cascade routing -- a unified framework that integrates routing (choose one model per query) and cascading (sequentially try models until one succeeds). The system iteratively picks the best model, can skip models, reorder them, or run as few as needed.

**Cost impact example:**

| Scenario | Monthly Cost | Quality |
|----------|-------------|---------|
| All queries to Claude Opus 4.6 | $10,000 | Highest |
| Model cascading (60% tier 1, 25% tier 2-3, 15% tier 4-5) | $2,500 | Same for 85% of queries |
| Savings | $7,500 (75%) | Negligible quality loss |

**Sources:**
- [A Unified Approach to Routing and Cascading for LLMs -- arXiv](https://arxiv.org/abs/2410.10347)
- [LLM Cost Optimization Guide](https://ai.koombea.com/blog/llm-cost-optimization)
- [CascadeFlow -- GitHub](https://github.com/lemony-ai/cascadeflow)

### 8.4 Metric-Driven Self-Improvement

**The core principle:** Self-improvement without metrics is not improvement -- it is wandering. Every change must be measured. Every improvement must be demonstrated.

**The metrics stack:**

```python
@dataclass
class ImprovementMetrics:
    """Metrics tracked for every self-improvement cycle."""

    # Capability metrics
    test_pass_rate: float          # % of tests passing
    swe_bench_score: float         # Performance on SWE-bench-like tasks
    task_completion_rate: float    # % of assigned tasks completed successfully
    time_to_completion: float      # Average seconds per task

    # Quality metrics
    lint_score: float              # Code quality score (0-10)
    type_coverage: float           # % of code with type hints
    test_coverage: float           # % of code covered by tests
    cyclomatic_complexity: float   # Average complexity per function

    # Cost metrics
    total_api_cost: float          # Total spend on LLM API calls
    cost_per_task: float           # Average cost per completed task
    tokens_per_task: int           # Average tokens consumed per task
    model_tier_distribution: dict  # % of calls per tier

    # Safety metrics
    reverts: int                   # Number of git reverts in period
    secret_scan_blocks: int        # Number of commits blocked by secret scanning
    test_regressions: int          # Number of times passing tests started failing
```

**The improvement loop:**

```python
def self_improvement_cycle(system, baseline_metrics: ImprovementMetrics):
    """One cycle of metric-driven self-improvement."""

    # 1. Identify improvement target
    #    Pick the metric with the most room for improvement
    target = identify_weakest_metric(baseline_metrics)

    # 2. Propose improvement
    #    The system generates a hypothesis about how to improve
    proposal = system.propose_improvement(target)

    # 3. Implement on a branch
    branch = f"evolve/{target.name}-{timestamp()}"
    system.create_branch(branch)
    system.implement(proposal)

    # 4. Measure
    new_metrics = system.run_full_evaluation()

    # 5. Compare -- EVERY metric must be checked
    if new_metrics.is_improvement_over(baseline_metrics, target):
        # Target metric improved
        if new_metrics.no_regressions_vs(baseline_metrics):
            # No other metrics degraded
            system.merge_branch(branch)
            system.commit(f"evolve({target.name}): {proposal.summary}")
            return new_metrics
        else:
            # Improvement caused regression elsewhere
            system.delete_branch(branch)
            system.log(f"Rejected: {target.name} improved but caused regressions")
            return baseline_metrics
    else:
        # Target metric did not improve
        system.delete_branch(branch)
        system.log(f"Rejected: {target.name} did not improve")
        return baseline_metrics
```

**The ratchet mechanism:** Once a metric reaches a new high, it never goes back. Every subsequent change must maintain at least the current level:

```python
class MetricRatchet:
    """Ensures metrics never degrade below their historical best."""

    def __init__(self):
        self.high_water_marks = {}

    def update(self, metric_name: str, value: float) -> bool:
        """Returns True if value meets or exceeds the high water mark."""
        current_best = self.high_water_marks.get(metric_name, float("-inf"))
        if value >= current_best:
            self.high_water_marks[metric_name] = value
            return True
        return False

    def check(self, metrics: dict[str, float]) -> list[str]:
        """Returns list of metrics that violate the ratchet."""
        violations = []
        for name, value in metrics.items():
            if name in self.high_water_marks:
                if value < self.high_water_marks[name]:
                    violations.append(
                        f"{name}: {value:.4f} < {self.high_water_marks[name]:.4f}"
                    )
        return violations
```

### 8.5 The aiai Bootstrap Sequence

Based on everything in this document, the recommended bootstrap sequence for aiai:

**Phase 0: Foundation (current state)**

- Human-authored CLAUDE.md, git workflows, CI, model routing config.
- Research documents (this file).
- Project structure and conventions.

**Phase 1: Core Infrastructure**

Build the pieces that make autonomous operation possible:

1. **OpenRouter client** (`src/router/`): Python client for model routing with cascading, cost tracking, and fallback logic.
2. **Test framework**: Comprehensive test suite that agents can run before every commit.
3. **Metrics collection**: Track every API call, every test result, every commit.
4. **Evaluation harness**: Automated benchmarks that measure system capability.

**Phase 2: Agent Runtime**

Build the agent that can modify itself:

1. **Task intake**: Accept tasks as natural language or structured specs.
2. **Planning agent**: Decompose tasks into subtasks with complexity ratings.
3. **Implementation agent**: Write code using the model router for cost-optimized execution.
4. **Testing agent**: Generate and run tests, enforce TDD patterns.
5. **Git integration**: Commit-or-revert pattern, branch-based experimentation, checkpoint system.

**Phase 3: Self-Improvement Engine**

Close the loop:

1. **Metric-driven improvement**: System identifies its weakest metrics and proposes improvements.
2. **Ratchet mechanism**: Improvements are validated against the full metric suite. No regressions allowed.
3. **Evolution engine**: AlphaEvolve-inspired pattern -- maintain a population of approaches, mutate with LLMs, evaluate with automated metrics, select the best.
4. **Cost optimization**: System learns which model tiers work for which task types and optimizes its own routing config.

**Phase 4: Autonomous Operation**

The system runs without human intervention:

1. Receives tasks from issues, metrics, or its own improvement engine.
2. Plans, implements, tests, commits, and pushes.
3. Monitors its own metrics and identifies improvement opportunities.
4. Evolves its own code, prompts, and configuration.
5. Tracks costs and optimizes spend.

**Key constraints at every phase:**

- Tests are the quality gate. No tests = no commit.
- Git history is the audit trail. Small commits, clear messages, easy reverts.
- Cost tracking is mandatory. Every API call is logged and budgeted.
- The ratchet never goes backward. Improvements are cumulative.
- Model collapse avoidance. Anchor against fixed benchmarks derived from real data.

---

## Appendix A: Key Papers and Systems

| Paper / System | Year | Significance |
|----------------|------|-------------|
| Constitutional AI (Anthropic) | 2022 | Foundational self-improvement via AI feedback (RLAIF) |
| NASNet (Google Brain) | 2017 | First major NAS success |
| DARTS (CMU) | 2018 | Differentiable NAS, 1000x cost reduction |
| EfficientNet (Google) | 2019 | Compound scaling from NAS |
| AlphaFold 2 (DeepMind) | 2020 | Solved protein folding, Nobel Prize 2024 |
| AlphaChip (DeepMind) | 2020+ | RL for chip layout, three generations of TPUs |
| FunSearch (DeepMind) | 2023 | LLM-driven mathematical discovery |
| The AI Scientist v1 (Sakana AI) | Aug 2024 | First full-cycle automated science ($15/paper) |
| Devin (Cognition) | Mar 2024 | First autonomous coding agent (13.86% SWE-bench) |
| Frontier AI Self-Replication (Fudan) | Dec 2024 | 50-90% self-replication rates in lab |
| DeepSeek-R1 | Jan 2025 | Pure RL reasoning without SFT |
| METR Time Horizons | Mar 2025 | 7-month doubling of AI task capability |
| RepliBench (UK AISI) | Apr 2025 | Systematic self-replication evaluation |
| The AI Scientist v2 (Sakana AI) | Apr 2025 | First AI-authored peer-reviewed paper |
| AlphaEvolve (DeepMind) | May 2025 | Evolutionary code optimization, Strassen improvement |
| GPT-5 (OpenAI) | Aug 2025 | 74.9% SWE-bench, 94.6% AIME |
| SWE-RL (Meta) | 2025 | Self-play RL for software engineering |
| Claude Opus 4.5 (Anthropic) | Nov 2025 | Self-improving agent, 4-iteration convergence |
| Amodei: 90%+ of Claude code by Claude | Jan 2026 | AI writing AI at production scale |
| GPT-5.3-Codex (OpenAI) | Feb 2026 | First model that helped create itself |
| Claude Opus 4.6 (Anthropic) | Feb 2026 | 1M context, agent teams, adaptive reasoning |
| METR Time Horizon 1.1 | Jan 2026 | Refined doubling: 4.3 months post-2023 |
| Copilot CLI GA (GitHub) | Feb 2026 | Autopilot mode, specialized sub-agents |

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **AutoML** | Automated Machine Learning -- systems that automate model selection, hyperparameter tuning, and feature engineering |
| **CAI** | Constitutional AI -- Anthropic's method for self-improving AI using self-critique against a constitution |
| **Cascading** | Sequentially trying increasingly capable (and expensive) models until one succeeds |
| **DARTS** | Differentiable Architecture Search -- gradient-based NAS |
| **FunSearch** | DeepMind's method for evolving functions using LLMs |
| **Model Collapse** | Degenerative feedback loop from training on synthetic data |
| **NAS** | Neural Architecture Search -- automated design of neural network architectures |
| **PBT** | Population-Based Training -- parallel training with weight sharing |
| **Ratchet** | Mechanism ensuring metrics never degrade below historical best |
| **RLAIF** | Reinforcement Learning from AI Feedback -- using AI instead of humans for preference labeling |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **Routing** | Selecting a single model per query based on task characteristics |
| **RSI** | Recursive Self-Improvement -- AI improving its own capability to improve |
| **Self-Play** | Training paradigm where agent generates its own training data by playing against itself |
| **SWE-bench** | Benchmark for AI coding agents using real GitHub issues |
| **SWE-RL** | Reinforcement learning for software engineering using code evolution data |
| **TDD** | Test-Driven Development -- write tests before implementation |

## Appendix C: Sources Index

### AI Coding Agents
- [GitHub Copilot Coding Agent](https://github.com/newsroom/press-releases/coding-agent-for-github-copilot)
- [Copilot CLI GA](https://github.blog/changelog/2026-02-25-github-copilot-cli-is-now-generally-available/)
- [Cursor Agent](https://cursor.com/product)
- [Claude Code Overview](https://code.claude.com/docs/en/overview)
- [Eight Trends in Software 2026](https://claude.com/blog/eight-trends-defining-how-software-gets-built-in-2026)
- [Anthropic - Effective Harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Introducing Codex -- OpenAI](https://openai.com/index/introducing-codex/)
- [Devin -- Cognition](https://cognition.ai/blog/introducing-devin)
- [Devin 2025 Performance Review](https://cognition.ai/blog/devin-annual-performance-review-2025)

### Benchmarks and Evaluation
- [SWE-bench Verified -- Epoch AI](https://epoch.ai/benchmarks/swe-bench-verified)
- [SWE-bench Leaderboards](https://www.swebench.com/)
- [METR Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/)
- [SWE-Bench Pro -- SEAL / Scale AI](https://scale.com/leaderboard/swe_bench_pro_public)

### AutoML and NAS
- [RZ-NAS -- ICML 2025](https://proceedings.mlr.press/v267/ji25a.html)
- [LLM-NAS -- arXiv](https://arxiv.org/abs/2510.01472)
- [AutoML: A systematic review](https://www.sciencedirect.com/science/article/pii/S2949715923000604)
- [Advances in NAS -- NSR](https://academic.oup.com/nsr/article/11/8/nwae282/7740455)

### Self-Improvement and Evolution
- [AlphaEvolve -- Google DeepMind](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [AlphaEvolve -- arXiv](https://arxiv.org/abs/2506.13131)
- [OpenEvolve -- Hugging Face](https://huggingface.co/blog/codelion/openevolve)
- [FunSearch -- Google DeepMind](https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
- [FunSearch -- Nature](https://www.nature.com/articles/s41586-023-06924-6)
- [DeepSeek-R1 -- arXiv](https://arxiv.org/abs/2501.12948)
- [Constitutional AI -- Anthropic](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- [Bootstrapping Task Spaces -- arXiv](https://arxiv.org/abs/2509.04575)

### Self-Play and Synthetic Data
- [SWE-RL -- arXiv](https://arxiv.org/abs/2502.18449)
- [Self-Play SWE-RL -- arXiv](https://arxiv.org/abs/2512.18552)
- [SWE-RL -- GitHub (Meta)](https://github.com/facebookresearch/swe-rl)
- [Model Collapse -- Wikipedia](https://en.wikipedia.org/wiki/Model_collapse)
- [AI training in 2026: anchoring synthetic data](https://invisibletech.ai/blog/ai-training-in-2026-anchoring-synthetic-data-in-human-truth)

### AI Research Agents
- [AlphaFold -- Google DeepMind](https://deepmind.google/science/alphafold/)
- [The AI Scientist -- Sakana AI](https://sakana.ai/ai-scientist/)
- [AI Scientist-v2 -- GitHub](https://github.com/SakanaAI/AI-Scientist-v2)
- [Evaluating AI Scientist -- arXiv](https://arxiv.org/abs/2502.14297)
- [MLAgentBench -- arXiv](https://arxiv.org/abs/2310.03302)
- [MLE-bench -- arXiv](https://arxiv.org/abs/2410.07095)

### Cost Optimization
- [Routing and Cascading for LLMs -- arXiv](https://arxiv.org/abs/2410.10347)
- [LLM Cost Optimization Guide](https://ai.koombea.com/blog/llm-cost-optimization)
- [CascadeFlow -- GitHub](https://github.com/lemony-ai/cascadeflow)

### Safety and Replication
- [Frontier AI Self-Replication -- arXiv](https://arxiv.org/html/2412.12140v1)
- [RepliBench -- UK AISI](https://www.aisi.gov.uk/blog/replibench-measuring-autonomous-replication-capabilities-in-ai-systems)
- [Recursive Self-Improvement -- Alignment Forum](https://www.alignmentforum.org/w/recursive-self-improvement)
- [METR](https://metr.org/)

### Industry Trends
- [2026 Agentic Coding Trends Report -- Anthropic](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf)
- [TDD and AI: Quality in the DORA report -- Google Cloud](https://cloud.google.com/discover/how-test-driven-development-amplifies-ai-success)
- [7 Agentic AI Trends to Watch in 2026](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/)
