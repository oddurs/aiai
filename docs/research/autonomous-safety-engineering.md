# Safety Engineering for Autonomous AI Development Systems (2024-2026)

> Research compiled February 2026 for the **aiai** project -- a self-improving AI infrastructure system.
> This document covers the engineering foundations for building safety into fully autonomous,
> self-modifying AI systems: validator integrity, circuit breakers, loop detection, mutation testing,
> invariant checking, protected files, cost control, quality regression detection, and self-modification safety.

---

## Table of Contents

1. [The Validator Problem in Depth](#1-the-validator-problem-in-depth)
2. [Circuit Breaker Patterns for AI Agents](#2-circuit-breaker-patterns-for-ai-agents)
3. [Loop Detection in Autonomous Agents](#3-loop-detection-in-autonomous-agents)
4. [Mutation Testing for AI Safety](#4-mutation-testing-for-ai-safety)
5. [Invariant Checking Systems](#5-invariant-checking-systems)
6. [Protected File Mechanisms](#6-protected-file-mechanisms)
7. [Cost Control Architecture](#7-cost-control-architecture)
8. [Quality Regression Detection](#8-quality-regression-detection)
9. [Self-Modification Safety in Practice](#9-self-modification-safety-in-practice)
10. [Implications for aiai](#10-implications-for-aiai)
11. [References](#11-references)

---

## 1. The Validator Problem in Depth

### 1.1 The Core Problem

When an AI system generates both the code and the tests that validate that code, a fundamental epistemological gap emerges: **who watches the watchmen?** If the same model (or a model with similar biases) produces the implementation and its verification, shared blind spots propagate silently. The tests pass, coverage looks good, CI is green -- and the code is wrong.

This is the central safety challenge for aiai. In a fully autonomous system with no human gates, test quality is the only thing standing between a working system and a subtly broken one that confidently reports success.

### 1.2 The Oracle Problem

The **test oracle problem** is a long-studied challenge in software testing: given an input to a program, how do you determine the correct expected output? In traditional development, the human developer serves as the oracle -- they know what the code should do and write assertions accordingly.

When AI generates tests, the oracle problem becomes acute:

- **Tautological oracles**: The AI generates an assertion that simply restates the implementation logic. The test passes by definition but validates nothing. For example, testing `add(a, b)` by asserting `result == a + b` when the implementation is literally `return a + b`.
- **Weak oracles**: Tests that only check for non-null returns, type correctness, or absence of exceptions without validating actual behavior.
- **Inverted oracles**: The AI derives the expected output by mentally executing the code it just wrote, baking in the same bugs as correct behavior.

Recent academic work on neural-based test oracle generation (Dinella et al., 2023) finds that LLM-generated oracles suffer from a **simplification bias** -- models tend to generate overly simple assertions that are easy to satisfy, leading to high pass rates but low fault-detection capability.

**Reference**: [Neural-Based Test Oracle Generation: A Large-Scale Evaluation and Lessons Learned](https://www.researchgate.net/publication/376107399_Neural-Based_Test_Oracle_Generation_A_Large-Scale_Evaluation_and_Lessons_Learned)

### 1.3 Empirical Data on AI-Generated Test Quality

#### CoverUp

CoverUp (Bornholt et al., 2024) is a coverage-guided LLM-based test generator for Python that iteratively prompts the LLM with coverage gaps and feedback. Empirical results on open-source Python projects:

- Per-module median line+branch coverage: **80%** (vs. CodaMosa's 47%)
- Overall line+branch coverage: **60%** (vs. CodaMosa's 45%)
- Against MuTAP (a mutation-guided LLM test generator): **89%** vs. 77% line+branch coverage

CoverUp's key insight is the feedback loop: coverage analysis identifies uncovered lines, which are fed back to the LLM as targeted prompts. This iterative refinement substantially outperforms one-shot generation.

**Reference**: [CoverUp: Coverage-Guided LLM-Based Test Generation](https://arxiv.org/abs/2403.16218)

#### Diffblue Cover

Diffblue Cover is an AI agent for Java unit test generation. Empirical comparison (Almasi et al., 2024):

- Code coverage: **63%** (EvoSuite: 89%, Randoop: 63%)
- Mutation score: **40%** (EvoSuite: 67%, Randoop: 50%)
- Bug detection: 4/10 bugs found (EvoSuite: 7/10, Randoop: 5/10)

However, Diffblue generated only 128 assertions for 70% coverage, while Randoop generated over 150,000 assertions for 69% coverage. Quality per assertion was significantly higher. More recent benchmarks (2025) show improvement: 50-69% coverage with a 71% mutation score, outperforming GitHub Copilot with GPT-5 on the same benchmarks.

**Reference**: [Diffblue Cover Benchmark 2025](https://www.diffblue.com/resources/diffblue-cover-vs-ai-coding-assistants-benchmark-2025/)

#### Meta's TestGen-LLM

Meta deployed LLM-based test generation across Facebook, Instagram, WhatsApp, and wearables in a trial from October to December 2024. Privacy engineers accepted 73% of generated tests, with 36% judged as genuinely privacy-relevant (not just boilerplate). The key finding: **LLM-generated tests improve mutation scores better than coverage alone**, because mutation feedback is incorporated into the generation loop.

**Reference**: [Automated Unit Test Improvement using Large Language Models at Meta](https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/)

### 1.4 Cross-Validation with Independent Models

The most promising defense against the validator problem is **model diversity** -- using a different model (or fundamentally different approach) to validate the output of the generating model.

```
  Generator Model (e.g., Claude)         Validator Model (e.g., GPT-4o)
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │  Generates code + tests  │──────────│  Reviews tests for:      │
  │                          │          │  - Tautological asserts   │
  │                          │          │  - Missing edge cases     │
  │                          │          │  - Oracle validity        │
  └──────────────────────────┘          └──────────────────────────┘
                                                     │
                                                     v
                                        ┌──────────────────────────┐
                                        │  Mutation Testing         │
                                        │  (model-independent)      │
                                        │  Ground truth validation  │
                                        └──────────────────────────┘
```

Strategies for cross-validation:

1. **Different model families**: If Claude generates the code, use GPT-4o or Gemini to review the tests. Different training data and architectures reduce correlated failures.
2. **Adversarial review prompts**: Prompt the validator to specifically look for tautological assertions, missing edge cases, and weak oracles.
3. **Property-based test generation**: Use a separate model to generate property-based tests (Hypothesis-style) that test behavioral invariants rather than specific input-output pairs.
4. **Specification extraction**: Have one model extract a natural-language specification from the code, and a different model generate tests from that specification without seeing the implementation.

### 1.5 Mutation Testing as Ground Truth

Mutation testing is the **model-independent ground truth** for test quality. It answers the question: "If I introduce a bug, do the tests catch it?" This is independent of which model generated the code or tests.

For aiai, mutation testing is the critical backstop. It does not matter if the AI-generated tests look reasonable or achieve high coverage -- if mutants survive, the tests are not doing their job. See [Section 4](#4-mutation-testing-for-ai-safety) for detailed implementation.

### 1.6 Implementation Pattern for aiai

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ValidationVerdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


@dataclass
class ValidatorResult:
    verdict: ValidationVerdict
    mutation_score: float  # 0.0 to 1.0
    tautology_warnings: list[str]
    weak_oracle_warnings: list[str]
    cross_model_agreement: bool
    coverage_line: float
    coverage_branch: float


class TestValidator:
    """Multi-layer validation for AI-generated tests.

    Layer 1: Static analysis (tautology detection)
    Layer 2: Coverage analysis (line + branch)
    Layer 3: Mutation testing (ground truth)
    Layer 4: Cross-model review (independent validation)
    """

    def __init__(
        self,
        generator_model: str,
        validator_model: str,
        mutation_threshold: float = 0.70,
        coverage_threshold: float = 0.80,
    ):
        self.generator_model = generator_model
        self.validator_model = validator_model
        self.mutation_threshold = mutation_threshold
        self.coverage_threshold = coverage_threshold

    def validate(
        self, source_file: str, test_file: str
    ) -> ValidatorResult:
        """Run all validation layers and return composite result."""
        tautology_warnings = self._detect_tautologies(source_file, test_file)
        coverage = self._measure_coverage(source_file, test_file)
        mutation_score = self._run_mutation_testing(source_file, test_file)
        cross_model_ok = self._cross_model_review(source_file, test_file)

        verdict = ValidationVerdict.PASS
        if mutation_score < self.mutation_threshold:
            verdict = ValidationVerdict.FAIL
        elif len(tautology_warnings) > 0 or not cross_model_ok:
            verdict = ValidationVerdict.UNCERTAIN

        return ValidatorResult(
            verdict=verdict,
            mutation_score=mutation_score,
            tautology_warnings=tautology_warnings,
            weak_oracle_warnings=[],
            cross_model_agreement=cross_model_ok,
            coverage_line=coverage["line"],
            coverage_branch=coverage["branch"],
        )

    def _detect_tautologies(
        self, source_file: str, test_file: str
    ) -> list[str]:
        """AST-based detection of tautological assertions.

        Detects patterns like:
        - assert func(x) == func(x)  (identical calls)
        - result = func(x); assert result == result
        - Assertions that reimplement the function under test
        """
        # Implementation: parse both ASTs, compare assertion logic
        # against implementation logic using tree edit distance
        ...

    def _measure_coverage(
        self, source_file: str, test_file: str
    ) -> dict[str, float]:
        """Run tests with coverage.py and return line/branch metrics."""
        ...

    def _run_mutation_testing(
        self, source_file: str, test_file: str
    ) -> float:
        """Run lightweight mutation testing and return mutation score."""
        # See Section 4 for the full implementation
        ...

    def _cross_model_review(
        self, source_file: str, test_file: str
    ) -> bool:
        """Send tests to a different model for independent review."""
        ...
```

---

## 2. Circuit Breaker Patterns for AI Agents

### 2.1 Why Microservice Circuit Breakers Are Not Enough

The classic circuit breaker pattern (Nygard, *Release It!*, 2007) monitors failure rates on downstream service calls and opens the circuit when a threshold is crossed, preventing cascading failures. For LLM-based agents, the failure modes are fundamentally different:

| Microservice Failure | LLM Agent Failure |
|---------------------|-------------------|
| Request timeout / HTTP 500 | Token runaway (infinite generation) |
| Service unavailable | Degraded output quality (subtle) |
| Connection refused | Hallucinated tool calls |
| Data corruption | Semantic drift (gradual loss of context) |
| Memory exhaustion | Cost explosion ($100+ in minutes) |

LLM agents need circuit breakers that monitor **semantic quality**, **cost**, **time**, and **behavioral patterns** -- not just HTTP status codes.

**Reference**: [Trustworthy AI Agents: Kill Switches and Circuit Breakers](https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-6/)

### 2.2 Circuit Breaker State Machine

The state machine extends the classic three-state model with LLM-specific transitions:

```
                 ┌──────────────────────────┐
                 │                          │
     success     │        CLOSED            │  Normal operation.
    ┌───────────>│   (allowing requests)    │  All requests pass through.
    │            │                          │
    │            └────────────┬─────────────┘
    │                         │
    │               failure threshold
    │               exceeded (N failures
    │               in window W)
    │                         │
    │                         v
    │            ┌──────────────────────────┐
    │            │                          │
    │            │         OPEN             │  Blocking requests.
    │            │   (rejecting requests)   │  Returns cached/fallback.
    │            │                          │
    │            └────────────┬─────────────┘
    │                         │
    │               timeout expires
    │               (cooldown period)
    │                         │
    │                         v
    │            ┌──────────────────────────┐
    │            │                          │
    └────────────│      HALF-OPEN           │  Probe with single request.
                 │   (testing recovery)     │  Success -> CLOSED.
                 │                          │  Failure -> OPEN.
                 └──────────────────────────┘
```

### 2.3 Token Budget Circuit Breaker

Prevents token runaway -- a single agent task consuming unbounded tokens.

```python
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Optional


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class TokenBudget:
    max_tokens_per_request: int = 16_000
    max_tokens_per_task: int = 200_000
    max_tokens_per_hour: int = 1_000_000
    max_tokens_per_day: int = 10_000_000


@dataclass
class TokenCircuitBreaker:
    """Circuit breaker that monitors token consumption.

    Opens the circuit when token usage exceeds budgets at any level
    (per-request, per-task, per-hour, per-day).
    """

    budget: TokenBudget = field(default_factory=TokenBudget)
    state: CircuitState = CircuitState.CLOSED
    cooldown_seconds: float = 60.0

    _task_tokens: int = 0
    _hourly_tokens: int = 0
    _daily_tokens: int = 0
    _hourly_reset: float = field(default_factory=time.time)
    _daily_reset: float = field(default_factory=time.time)
    _opened_at: Optional[float] = None
    _lock: Lock = field(default_factory=Lock)

    def allow_request(self, estimated_tokens: int) -> bool:
        """Check if a request should be allowed given current token state."""
        with self._lock:
            self._maybe_reset_windows()

            if self.state == CircuitState.OPEN:
                if self._cooldown_expired():
                    self.state = CircuitState.HALF_OPEN
                else:
                    return False

            if self.state == CircuitState.HALF_OPEN:
                # Allow one probe request with reduced budget
                if estimated_tokens > self.budget.max_tokens_per_request // 2:
                    return False
                return True

            # CLOSED state -- check all budget levels
            if estimated_tokens > self.budget.max_tokens_per_request:
                self._trip("per-request budget exceeded")
                return False

            if self._task_tokens + estimated_tokens > self.budget.max_tokens_per_task:
                self._trip("per-task budget exceeded")
                return False

            if self._hourly_tokens + estimated_tokens > self.budget.max_tokens_per_hour:
                self._trip("hourly budget exceeded")
                return False

            if self._daily_tokens + estimated_tokens > self.budget.max_tokens_per_day:
                self._trip("daily budget exceeded")
                return False

            return True

    def record_usage(self, tokens_used: int, success: bool) -> None:
        """Record actual token usage after a request completes."""
        with self._lock:
            self._task_tokens += tokens_used
            self._hourly_tokens += tokens_used
            self._daily_tokens += tokens_used

            if self.state == CircuitState.HALF_OPEN:
                if success:
                    self.state = CircuitState.CLOSED
                else:
                    self._trip("half-open probe failed")

    def reset_task(self) -> None:
        """Reset task-level budget (call when starting a new task)."""
        with self._lock:
            self._task_tokens = 0

    def _trip(self, reason: str) -> None:
        self.state = CircuitState.OPEN
        self._opened_at = time.time()
        # In production: emit metric, fire alert
        print(f"CIRCUIT OPEN: {reason}")

    def _cooldown_expired(self) -> bool:
        if self._opened_at is None:
            return True
        return (time.time() - self._opened_at) >= self.cooldown_seconds

    def _maybe_reset_windows(self) -> None:
        now = time.time()
        if now - self._hourly_reset >= 3600:
            self._hourly_tokens = 0
            self._hourly_reset = now
        if now - self._daily_reset >= 86400:
            self._daily_tokens = 0
            self._daily_reset = now
```

### 2.4 Cost Circuit Breaker

Monitors actual dollar cost, not just token counts. Different models have different per-token costs, and cost is what actually matters.

```python
@dataclass
class CostCircuitBreaker:
    """Circuit breaker based on actual dollar cost."""

    max_cost_per_request: float = 0.50    # $0.50
    max_cost_per_task: float = 5.00       # $5.00
    max_cost_per_hour: float = 20.00      # $20.00
    max_cost_per_day: float = 100.00      # $100.00

    # Model pricing ($ per 1M tokens) -- kept current
    pricing: dict = field(default_factory=lambda: {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    })

    def estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost before making a request."""
        if model not in self.pricing:
            # Unknown model -- assume expensive
            return (input_tokens + output_tokens) * 75.0 / 1_000_000
        p = self.pricing[model]
        return (
            input_tokens * p["input"] / 1_000_000
            + output_tokens * p["output"] / 1_000_000
        )

    def should_downgrade_model(
        self, current_model: str, remaining_budget: float
    ) -> Optional[str]:
        """Suggest a cheaper model if budget is running low."""
        if remaining_budget < 1.00 and current_model in (
            "claude-opus-4-20250514",
        ):
            return "claude-sonnet-4-20250514"
        if remaining_budget < 0.10:
            return "gemini-2.0-flash"
        return None
```

### 2.5 Quality Circuit Breaker

Detects degraded output quality -- the most difficult and most important type for autonomous systems.

```python
@dataclass
class QualityCircuitBreaker:
    """Detects degraded LLM output quality.

    Monitors:
    - Test pass rate (are generated tests passing?)
    - Mutation score (are tests catching injected bugs?)
    - Compilation/syntax success rate
    - Semantic coherence score
    """

    window_size: int = 10  # Track last N results
    min_test_pass_rate: float = 0.80
    min_mutation_score: float = 0.60
    min_syntax_success_rate: float = 0.95

    _results: list = field(default_factory=list)

    def record_result(self, result: dict) -> None:
        """Record an agent action result.

        result should contain:
        - tests_passed: bool
        - mutation_score: float (0.0-1.0) or None
        - syntax_valid: bool
        """
        self._results.append(result)
        if len(self._results) > self.window_size:
            self._results.pop(0)

    def is_degraded(self) -> tuple[bool, list[str]]:
        """Check if output quality has degraded below thresholds."""
        if len(self._results) < 3:
            return False, []

        reasons = []
        recent = self._results[-self.window_size:]

        pass_rate = sum(1 for r in recent if r.get("tests_passed")) / len(recent)
        if pass_rate < self.min_test_pass_rate:
            reasons.append(
                f"test pass rate {pass_rate:.0%} < {self.min_test_pass_rate:.0%}"
            )

        scores = [r["mutation_score"] for r in recent if r.get("mutation_score") is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score < self.min_mutation_score:
                reasons.append(
                    f"mutation score {avg_score:.0%} < {self.min_mutation_score:.0%}"
                )

        syntax_rate = sum(1 for r in recent if r.get("syntax_valid")) / len(recent)
        if syntax_rate < self.min_syntax_success_rate:
            reasons.append(
                f"syntax success rate {syntax_rate:.0%} < {self.min_syntax_success_rate:.0%}"
            )

        return len(reasons) > 0, reasons
```

### 2.6 Composite Circuit Breaker

In practice, all circuit breaker types must work together:

```python
class CompositeCircuitBreaker:
    """Combines token, cost, quality, and time circuit breakers.

    Any single breaker tripping opens the composite circuit.
    """

    def __init__(self):
        self.token_breaker = TokenCircuitBreaker()
        self.cost_breaker = CostCircuitBreaker()
        self.quality_breaker = QualityCircuitBreaker()
        self.max_task_duration_seconds = 1800  # 30 minutes
        self._task_start: Optional[float] = None

    def start_task(self) -> None:
        self._task_start = time.time()
        self.token_breaker.reset_task()

    def allow_request(
        self, model: str, estimated_input_tokens: int, estimated_output_tokens: int
    ) -> tuple[bool, Optional[str]]:
        """Check all circuit breakers. Returns (allowed, reason_if_blocked)."""

        # Time check
        if self._task_start and (
            time.time() - self._task_start > self.max_task_duration_seconds
        ):
            return False, "task duration exceeded"

        # Token check
        total_tokens = estimated_input_tokens + estimated_output_tokens
        if not self.token_breaker.allow_request(total_tokens):
            return False, f"token circuit open: {self.token_breaker.state.value}"

        # Cost check
        est_cost = self.cost_breaker.estimate_cost(
            model, estimated_input_tokens, estimated_output_tokens
        )
        if est_cost > self.cost_breaker.max_cost_per_request:
            return False, f"estimated cost ${est_cost:.2f} exceeds per-request limit"

        # Quality check
        degraded, reasons = self.quality_breaker.is_degraded()
        if degraded:
            return False, f"quality degraded: {'; '.join(reasons)}"

        return True, None
```

**Reference**: [The Economics of Autonomy: Preventing Token Runaway in Agentic Loops](https://www.alpsagility.com/cost-control-agentic-systems)

---

## 3. Loop Detection in Autonomous Agents

### 3.1 The Problem

Autonomous agents get stuck. They try the same fix repeatedly, rephrase the same question, or oscillate between two approaches without making progress. In aiai, where the agent has full autonomy with no human gates, an undetected loop can waste hours of compute, burn through API budgets, and produce no useful work -- or worse, commit increasingly broken code.

Common causes of agent loops:

- **Ambiguous objectives**: The task description is underspecified, leading to oscillation.
- **Tool misuse**: The agent repeatedly calls a tool hoping for different results.
- **Context drift**: The conversation grows long, earlier context is lost, and the agent forgets what it already tried.
- **Unresolvable errors**: A dependency is broken or an environment issue prevents progress, but the agent keeps trying code-level fixes.

**Reference**: [Why Agents Get Stuck in Loops (And How to Prevent It)](https://gantz.ai/blog/post/agent-loops/)

### 3.2 Detection Methods

#### 3.2.1 Action Hash Comparison

The simplest approach: hash each action (tool call + parameters) and detect exact repetitions.

```python
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ActionHashDetector:
    """Detects exact repeated actions using hashing.

    Trips when the same action (tool + params) appears more than
    max_repeats times within the sliding window.
    """

    window_size: int = 20
    max_repeats: int = 3

    _window: deque = field(default_factory=lambda: deque(maxlen=20))

    def record_action(self, tool: str, params: dict) -> bool:
        """Record an action. Returns True if loop detected."""
        action_hash = self._hash_action(tool, params)
        self._window.append(action_hash)

        count = sum(1 for h in self._window if h == action_hash)
        return count > self.max_repeats

    def _hash_action(self, tool: str, params: dict) -> str:
        canonical = json.dumps({"tool": tool, "params": params}, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

Limitation: This only catches *exact* repetition. An agent that makes trivially different attempts (changing a variable name, adding a comment) evades this check.

#### 3.2.2 Semantic Similarity Detection

Uses embeddings to detect *semantically similar* repeated attempts, catching paraphrased or slightly varied repetitions.

```python
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SemanticLoopDetector:
    """Detects loops via semantic similarity of agent actions.

    Embeds each action description and computes cosine similarity
    against recent actions. Trips when too many recent actions are
    semantically similar (even if not identical).
    """

    window_size: int = 10
    similarity_threshold: float = 0.92  # cosine similarity
    max_similar_in_window: int = 4

    _action_embeddings: list = field(default_factory=list)
    _action_texts: list = field(default_factory=list)

    def record_action(self, action_description: str) -> bool:
        """Record an action and check for semantic loops.

        action_description: natural language description of what the
        agent is doing (e.g., "editing file X to fix import error").
        """
        embedding = self._embed(action_description)

        # Compare against recent actions
        similar_count = 0
        recent = self._action_embeddings[-self.window_size:]
        for prev_embedding in recent:
            sim = self._cosine_similarity(embedding, prev_embedding)
            if sim > self.similarity_threshold:
                similar_count += 1

        self._action_embeddings.append(embedding)
        self._action_texts.append(action_description)

        # Keep memory bounded
        if len(self._action_embeddings) > self.window_size * 2:
            self._action_embeddings = self._action_embeddings[-self.window_size:]
            self._action_texts = self._action_texts[-self.window_size:]

        return similar_count >= self.max_similar_in_window

    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for action text.

        In production, use a fast local model like all-MiniLM-L6-v2
        to avoid API calls for every loop check.
        """
        # Placeholder -- use sentence-transformers in production:
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # return model.encode(text)
        ...

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / norm if norm > 0 else 0.0
```

### 3.3 Progress Detection

The most sophisticated approach: measure whether the agent is making **forward progress** toward its goal, not just whether actions are repeating.

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgressMetrics:
    """Quantifiable progress indicators."""
    tests_passing: int = 0
    tests_total: int = 0
    lint_errors: int = 0
    type_errors: int = 0
    coverage_percent: float = 0.0
    mutation_score: float = 0.0
    files_changed: int = 0


@dataclass
class ProgressDetector:
    """Detects whether an agent is making forward progress.

    Tracks quantifiable metrics over time. If metrics are not
    improving (or are oscillating), the agent is stuck.
    """

    stall_window: int = 5  # Check last N checkpoints
    min_improvement_rate: float = 0.05  # 5% improvement required

    _history: list[ProgressMetrics] = field(default_factory=list)

    def checkpoint(self, metrics: ProgressMetrics) -> None:
        """Record a progress checkpoint."""
        self._history.append(metrics)

    def is_stalled(self) -> tuple[bool, Optional[str]]:
        """Check if progress has stalled.

        Returns (stalled, reason).
        """
        if len(self._history) < self.stall_window:
            return False, None

        recent = self._history[-self.stall_window:]
        first = recent[0]
        last = recent[-1]

        # Check test progress
        if first.tests_total > 0 and last.tests_total > 0:
            first_rate = first.tests_passing / first.tests_total
            last_rate = last.tests_passing / last.tests_total
            if last_rate <= first_rate and first_rate < 1.0:
                return True, (
                    f"test pass rate stalled at {last_rate:.0%} "
                    f"over {self.stall_window} checkpoints"
                )

        # Check if lint errors are not decreasing
        if last.lint_errors >= first.lint_errors and first.lint_errors > 0:
            return True, (
                f"lint errors stuck at {last.lint_errors} "
                f"over {self.stall_window} checkpoints"
            )

        # Check for oscillation (metrics bouncing up and down)
        if self._is_oscillating(
            [m.tests_passing for m in recent]
        ):
            return True, "test count oscillating -- agent may be flip-flopping"

        return False, None

    def _is_oscillating(self, values: list[int], min_flips: int = 3) -> bool:
        """Detect oscillation: values alternating up and down."""
        if len(values) < 3:
            return False
        directions = []
        for i in range(1, len(values)):
            if values[i] > values[i - 1]:
                directions.append(1)
            elif values[i] < values[i - 1]:
                directions.append(-1)
            else:
                directions.append(0)
        flips = sum(
            1 for i in range(1, len(directions))
            if directions[i] != 0
            and directions[i - 1] != 0
            and directions[i] != directions[i - 1]
        )
        return flips >= min_flips
```

### 3.4 Composite Loop Detector

```python
@dataclass
class CompositeLoopDetector:
    """Combines all loop detection methods.

    Escalation ladder:
    1. Warning: Log the issue, continue.
    2. Strategy change: Force the agent to try a different approach.
    3. Halt: Stop the agent entirely.
    """

    action_hash: ActionHashDetector = field(default_factory=ActionHashDetector)
    semantic: SemanticLoopDetector = field(default_factory=SemanticLoopDetector)
    progress: ProgressDetector = field(default_factory=ProgressDetector)

    _warnings: int = 0
    _max_warnings_before_halt: int = 3

    def check(
        self,
        tool: str,
        params: dict,
        action_description: str,
        metrics: Optional[ProgressMetrics] = None,
    ) -> str:
        """Check for loops. Returns action: 'continue', 'change_strategy', 'halt'."""

        if metrics:
            self.progress.checkpoint(metrics)

        exact_loop = self.action_hash.record_action(tool, params)
        semantic_loop = self.semantic.record_action(action_description)
        stalled, stall_reason = self.progress.is_stalled()

        if exact_loop:
            self._warnings += 1
            if self._warnings >= self._max_warnings_before_halt:
                return "halt"
            return "change_strategy"

        if semantic_loop and stalled:
            # Both semantic repetition AND no progress -- strong signal
            return "halt"

        if semantic_loop or stalled:
            self._warnings += 1
            return "change_strategy"

        # Reset warning count on genuine progress
        self._warnings = max(0, self._warnings - 1)
        return "continue"
```

### 3.5 What OpenHands (Formerly OpenDevin) Does

OpenHands, the open-source autonomous coding agent, implements loop detection and has encountered practical challenges. A notable issue (GitHub issue #5355) revealed that their loop detection killed agents that were waiting on legitimately long-running processes (e.g., large test suites). The fix: distinguish between "agent is actively looping" and "agent is idle waiting for external results."

**Reference**: [OpenHands Loop Detection Issue #5355](https://github.com/All-Hands-AI/OpenHands/issues/5355)

**Reference**: [Context Drift in AI Agents: Why Your Agent Loops Forever](https://tacnode.io/post/your-ai-agents-are-spinning-their-wheels)

---

## 4. Mutation Testing for AI Safety

### 4.1 Why Mutation Testing Matters for Autonomous Systems

Mutation testing is the **most rigorous automated measure of test suite quality**. It works by injecting small, syntactically valid bugs (mutants) into the source code and checking whether the test suite detects them. A surviving mutant means the tests have a blind spot.

For aiai, mutation testing serves a unique role: it is the **model-independent validation layer** that does not depend on any AI model's judgment. A test suite either kills the mutants or it does not -- no prompt engineering, no model bias, no subjective assessment.

Key metrics:
- **Mutation score** = killed mutants / total mutants
- A mutation score of 70%+ is generally considered good
- Equivalent mutants (mutations that do not change behavior) create noise -- typically 5-20% of all mutants

### 4.2 Python Mutation Testing Tools

#### mutmut

The most actively maintained Python mutation testing tool as of 2026.

- **Speed**: ~1,200 mutants/min on typical Python code
- **Detection rate**: 88.5% on standard benchmarks
- **CI integration overhead**: ~5 minutes for a medium project
- **Memory**: 150MB baseline, scales linearly to ~500MB on 50k LOC
- **Mutation operators**: Arithmetic, comparison, boolean, string, keyword, and more
- **Incremental mode**: Only re-tests mutants in changed code (critical for CI)

```bash
# Install
pip install mutmut

# Run against a specific module
mutmut run --paths-to-mutate=src/safety/ --tests-dir=tests/safety/

# Show surviving mutants
mutmut results

# Show a specific surviving mutant
mutmut show 42
```

**2025 improvement**: mutmut now uses static analysis to flag likely equivalent mutants, reducing manual triage effort.

#### cosmic-ray

Parallel mutation testing with distributed execution support.

- **Speed**: Slower per-mutant than mutmut, but parallelizes across workers
- **Detection rate**: 82.7% on standard benchmarks
- **Architecture**: Uses Celery for distributed execution -- can spread across multiple machines
- **Mutation operators**: Broad set including statement deletions, constant replacements, control-flow changes

```bash
# Install
pip install cosmic-ray

# Initialize a session
cosmic-ray init config.toml session.sqlite

# Run mutations (distributed via Celery)
cosmic-ray exec session.sqlite

# View results
cr-report session.sqlite
```

Best for: Large codebases where mutation testing must be parallelized across a cluster.

#### mutpy

Lighter-weight, AST-level mutation testing.

- Directly manipulates Python AST nodes
- Smaller set of mutation operators
- Less actively maintained than mutmut
- Best for: Educational use, lightweight integration, custom mutation operator development

**Reference**: [Static and Dynamic Comparison of Mutation Testing Tools for Python](https://dl.acm.org/doi/10.1145/3701625.3701659)
**Reference**: [An Analysis and Comparison of Mutation Testing Tools for Python](https://ieeexplore.ieee.org/document/10818231/)

### 4.3 Lightweight AST-Level Mutation Testing

For aiai, full mutation testing on every commit is too slow. We need a **lightweight mode** that runs in seconds, not minutes, while still catching the most important test gaps.

```python
import ast
import copy
from dataclasses import dataclass
from typing import Generator


@dataclass
class Mutant:
    """A single code mutation."""
    original_node: ast.AST
    mutated_node: ast.AST
    location: tuple[int, int]  # (line, col)
    operator: str  # description of the mutation
    category: str  # "arithmetic", "comparison", "boolean", etc.


class LightweightMutator:
    """Fast AST-level mutation for Python.

    Focuses on high-value mutations that are most likely to catch
    real bugs, skipping low-value mutations that waste time.
    """

    # Priority-ordered mutation operators
    # These are the mutations most likely to catch real bugs
    HIGH_VALUE_MUTATIONS = {
        # Comparison operator swaps (most common real-world bugs)
        ast.Lt: [ast.LtE, ast.Gt],
        ast.LtE: [ast.Lt, ast.GtE],
        ast.Gt: [ast.GtE, ast.Lt],
        ast.GtE: [ast.Gt, ast.LtE],
        ast.Eq: [ast.NotEq],
        ast.NotEq: [ast.Eq],
        # Boolean operator swaps
        ast.And: [ast.Or],
        ast.Or: [ast.And],
        # Arithmetic (off-by-one is the #1 bug class)
        ast.Add: [ast.Sub],
        ast.Sub: [ast.Add],
    }

    # Lower priority -- skip in fast mode
    LOW_VALUE_MUTATIONS = {
        ast.Mult: [ast.Div],
        ast.Div: [ast.Mult],
        ast.BitAnd: [ast.BitOr],
        ast.BitOr: [ast.BitAnd],
    }

    def generate_mutants(
        self, source: str, fast_mode: bool = True
    ) -> Generator[Mutant, None, None]:
        """Generate mutants from Python source code.

        In fast_mode, only generates high-value mutations.
        """
        tree = ast.parse(source)
        mutations = self.HIGH_VALUE_MUTATIONS
        if not fast_mode:
            mutations = {**mutations, **self.LOW_VALUE_MUTATIONS}

        for node in ast.walk(tree):
            yield from self._mutate_node(node, mutations)

    def _mutate_node(
        self, node: ast.AST, mutations: dict
    ) -> Generator[Mutant, None, None]:
        """Generate mutations for a single AST node."""
        # Compare operators
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                op_type = type(op)
                if op_type in mutations:
                    for replacement_type in mutations[op_type]:
                        mutated = copy.deepcopy(node)
                        mutated.ops[i] = replacement_type()
                        yield Mutant(
                            original_node=node,
                            mutated_node=mutated,
                            location=(node.lineno, node.col_offset),
                            operator=f"{op_type.__name__} -> {replacement_type.__name__}",
                            category="comparison",
                        )

        # Binary operators
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in mutations:
                for replacement_type in mutations[op_type]:
                    mutated = copy.deepcopy(node)
                    mutated.op = replacement_type()
                    yield Mutant(
                        original_node=node,
                        mutated_node=mutated,
                        location=(node.lineno, node.col_offset),
                        operator=f"{op_type.__name__} -> {replacement_type.__name__}",
                        category="arithmetic",
                    )

        # Boolean operators
        if isinstance(node, ast.BoolOp):
            op_type = type(node.op)
            if op_type in mutations:
                for replacement_type in mutations[op_type]:
                    mutated = copy.deepcopy(node)
                    mutated.op = replacement_type()
                    yield Mutant(
                        original_node=node,
                        mutated_node=mutated,
                        location=(node.lineno, node.col_offset),
                        operator=f"{op_type.__name__} -> {replacement_type.__name__}",
                        category="boolean",
                    )

        # Return value mutations (return True -> return False, etc.)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, bool):
                mutated = copy.deepcopy(node)
                mutated.value.value = not node.value.value
                yield Mutant(
                    original_node=node,
                    mutated_node=mutated,
                    location=(node.lineno, node.col_offset),
                    operator=f"return {node.value.value} -> return {not node.value.value}",
                    category="return_value",
                )
```

### 4.4 Which Mutations Matter Most

Not all mutations are equal. Research and practical experience show this priority ordering for catching real bugs:

1. **Comparison operator mutations** (==, !=, <, >, <=, >=): Off-by-one errors, boundary condition bugs, and incorrect comparisons are the most common real-world bug class. If a test suite cannot catch `<` being changed to `<=`, it is almost certainly missing edge cases.

2. **Boolean operator mutations** (and/or swaps, not insertion): Logic errors in conditionals are the second most common source of bugs, especially in complex conditional chains.

3. **Return value mutations**: Changing `return True` to `return False` or `return x` to `return None` catches tests that do not verify return values.

4. **Constant boundary mutations**: Changing `0` to `1`, `-1` to `0`, empty string to non-empty. Catches boundary condition test gaps.

5. **Statement deletion**: Removing a statement entirely. If tests still pass, that code is either dead or untested.

Lower-priority (skip in fast mode):
- Arithmetic operator swaps (*/+- etc.) -- less common in real bugs outside numerical code
- Bitwise operator swaps -- rarely relevant outside low-level code
- Exception type mutations -- important but slow to test

### 4.5 Making Mutation Testing Fast Enough for CI

Mutation testing is inherently O(mutants x test_suite_time). For a project with 1,000 mutants and a 30-second test suite, that is 8+ hours sequentially. Strategies to make it practical:

1. **Incremental mutation testing**: Only mutate files changed in the current commit. `mutmut` supports `--paths-to-mutate` to scope mutations.

2. **Test selection**: For each mutant, only run tests that cover the mutated code. Coverage data maps source lines to tests.

3. **Fast-mode mutations**: Use only the high-value mutation operators (comparison, boolean, return value). Reduces mutant count by 60-70%.

4. **Parallel execution**: `cosmic-ray` distributes across Celery workers. `mutmut` runs sequentially but is faster per-mutant.

5. **Sampling**: On large codebases, randomly sample 20-30% of possible mutants. Statistically representative with much lower cost.

6. **Caching**: Cache mutation results by (source_hash, test_hash). If neither the source nor relevant tests changed, reuse the previous result.

```python
# CI integration example: lightweight mutation check on changed files

import subprocess
import sys


def get_changed_files() -> list[str]:
    """Get Python files changed in the current commit."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "--", "*.py"],
        capture_output=True,
        text=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def run_targeted_mutation_testing(changed_files: list[str]) -> bool:
    """Run mutation testing only on changed files."""
    if not changed_files:
        print("No Python files changed, skipping mutation testing")
        return True

    paths = ",".join(changed_files)
    result = subprocess.run(
        ["mutmut", "run", "--paths-to-mutate", paths, "--CI"],
        capture_output=True,
        text=True,
    )

    # Parse mutation score from output
    # mutmut exits with 0 if all mutants killed, non-zero otherwise
    return result.returncode == 0


if __name__ == "__main__":
    changed = get_changed_files()
    if not run_targeted_mutation_testing(changed):
        print("MUTATION TESTING FAILED: surviving mutants detected")
        sys.exit(1)
```

**Reference**: [Meta: LLMs Are the Key to Mutation Testing and Better Compliance](https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/)
**Reference**: [Mutation Testing with Mutmut](https://johal.in/mutation-testing-with-mutmut-python-for-code-reliability-2026/)

---

## 5. Invariant Checking Systems

### 5.1 What Are Safety Invariants?

A safety invariant is a property that must **always** hold, regardless of what the system is doing. Unlike tests that verify specific behaviors, invariants define the boundaries of acceptable system state. If an invariant is violated, something has gone fundamentally wrong.

For aiai, invariants are the **non-negotiable safety constraints** that no autonomous modification should ever violate:

- The safety infrastructure itself must remain functional
- Cost controls must not be disabled
- Protected files must not be modified
- The system must always be able to roll back to a known-good state
- Test suites must continue to pass after any modification

### 5.2 Design by Contract in Python

Design by Contract (DbC), pioneered by Bertrand Meyer in Eiffel, specifies preconditions, postconditions, and class invariants as executable specifications. Python supports this through decorators and runtime checking.

**Reference**: [PEP 316 -- Programming by Contract for Python](https://peps.python.org/pep-0316/)
**Reference**: [invariant-python: Automatic invariant enforcement](https://github.com/andreamancuso/invariant-python)

```python
import functools
from typing import Callable, Any


def precondition(check: Callable[..., bool], message: str = ""):
    """Decorator that enforces a precondition on function entry."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not check(*args, **kwargs):
                raise InvariantViolation(
                    f"Precondition failed for {func.__name__}: {message}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def postcondition(check: Callable[[Any], bool], message: str = ""):
    """Decorator that enforces a postcondition on function return."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not check(result):
                raise InvariantViolation(
                    f"Postcondition failed for {func.__name__}: {message}"
                )
            return result
        return wrapper
    return decorator


class InvariantViolation(Exception):
    """Raised when a safety invariant is violated."""
    pass


class InvariantEnforcer:
    """Mixin that automatically checks class invariants.

    Subclasses define __invariant__() which is called after every
    public method. If it returns False or raises, the system halts.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, method in list(cls.__dict__.items()):
            if callable(method) and not name.startswith("_"):
                cls._wrap_with_invariant_check(name, method)

    @classmethod
    def _wrap_with_invariant_check(cls, name: str, method: Callable) -> None:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            if hasattr(self, "__invariant__"):
                if not self.__invariant__():
                    raise InvariantViolation(
                        f"Class invariant violated after {cls.__name__}.{name}()"
                    )
            return result
        setattr(cls, name, wrapper)
```

### 5.3 Safety Invariants for aiai

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib


@dataclass
class SystemInvariants(InvariantEnforcer):
    """Core safety invariants for the aiai system.

    These invariants MUST hold at all times. Any violation triggers
    an immediate halt and rollback.
    """

    protected_files_checksums: dict[str, str]  # path -> sha256
    min_test_count: int = 1  # Must always have at least 1 test
    max_budget_exceeded: bool = False
    rollback_available: bool = True

    def __invariant__(self) -> bool:
        """All invariants must be True simultaneously."""
        return all([
            self._protected_files_intact(),
            self._tests_exist(),
            self._budget_not_exceeded(),
            self._rollback_available(),
            self._safety_modules_importable(),
        ])

    def _protected_files_intact(self) -> bool:
        """Protected files have not been modified."""
        for path, expected_hash in self.protected_files_checksums.items():
            if not Path(path).exists():
                return False
            actual_hash = hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest()
            if actual_hash != expected_hash:
                return False
        return True

    def _tests_exist(self) -> bool:
        """At least one test file exists and is non-empty."""
        test_dir = Path("tests")
        if not test_dir.exists():
            return False
        test_files = list(test_dir.rglob("test_*.py"))
        return len(test_files) >= self.min_test_count

    def _budget_not_exceeded(self) -> bool:
        """Cost budget has not been exceeded."""
        return not self.max_budget_exceeded

    def _rollback_available(self) -> bool:
        """System can roll back to a known-good state."""
        return self.rollback_available

    def _safety_modules_importable(self) -> bool:
        """Core safety modules can be imported."""
        try:
            # These must always be importable -- they are the safety net
            import importlib
            for module_name in [
                "aiai.safety.circuit_breaker",
                "aiai.safety.loop_detector",
                "aiai.safety.cost_controller",
            ]:
                importlib.import_module(module_name)
            return True
        except ImportError:
            return False
```

### 5.4 Runtime Invariant Monitoring

Invariants should not only be checked at discrete points -- they should be continuously monitored:

```python
import threading
import time
from typing import Callable


class InvariantMonitor:
    """Continuously monitors safety invariants in a background thread.

    If any invariant fails, triggers the emergency response:
    1. Logs the violation
    2. Fires alert
    3. Halts autonomous operations
    4. Initiates rollback if configured
    """

    def __init__(
        self,
        invariants: list[Callable[[], bool]],
        check_interval_seconds: float = 10.0,
        on_violation: Optional[Callable[[str], None]] = None,
    ):
        self.invariants = invariants
        self.check_interval = check_interval_seconds
        self.on_violation = on_violation or self._default_violation_handler
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start continuous monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _monitor_loop(self) -> None:
        while self._running:
            for invariant in self.invariants:
                try:
                    if not invariant():
                        self.on_violation(
                            f"Invariant {invariant.__name__} violated"
                        )
                except Exception as e:
                    self.on_violation(
                        f"Invariant {invariant.__name__} raised: {e}"
                    )
            time.sleep(self.check_interval)

    def _default_violation_handler(self, message: str) -> None:
        print(f"INVARIANT VIOLATION: {message}")
        # In production: fire PagerDuty alert, halt autonomous ops
```

### 5.5 Pre-Commit Invariant Checking

Run invariant checks as a git pre-commit hook, preventing commits that violate safety constraints:

```bash
#!/usr/bin/env bash
# .git/hooks/pre-commit -- safety invariant checks
set -euo pipefail

echo "Running safety invariant checks..."

# 1. Protected files not modified
python -c "
from aiai.safety.invariants import check_protected_files
violations = check_protected_files()
if violations:
    for v in violations:
        print(f'BLOCKED: Protected file modified: {v}')
    exit(1)
"

# 2. Safety modules still importable
python -c "
import importlib
for mod in ['aiai.safety.circuit_breaker', 'aiai.safety.loop_detector',
            'aiai.safety.cost_controller', 'aiai.safety.invariants']:
    try:
        importlib.import_module(mod)
    except ImportError as e:
        print(f'BLOCKED: Safety module {mod} cannot be imported: {e}')
        exit(1)
"

# 3. Tests still pass
python -m pytest tests/safety/ -x -q || {
    echo "BLOCKED: Safety tests failed"
    exit 1
}

# 4. No secrets in staged files
if git diff --cached --diff-filter=ACM -z -- '*.py' '*.yaml' '*.json' '*.toml' |
   xargs -0 grep -lE '(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN.*PRIVATE KEY)' 2>/dev/null; then
    echo "BLOCKED: Potential secrets detected in staged files"
    exit 1
fi

echo "All safety invariants passed."
```

### 5.6 Real-World Invariant Examples

**Database systems** maintain ACID invariants: atomicity (all-or-nothing), consistency (foreign key constraints, unique constraints), isolation (transaction visibility), durability (committed data survives crashes). PostgreSQL enforces these at the storage engine level -- application code cannot bypass them.

**Financial systems** enforce double-entry bookkeeping invariants: debits must always equal credits. Every transaction must balance. This invariant is checked at the database constraint level, the application level, and via periodic reconciliation jobs. Violations trigger immediate investigation.

**Aerospace (DO-178C)**: Safety-critical avionics software enforces invariants at multiple levels. Flight control invariants include attitude limits (the autopilot must never command a bank angle exceeding structural limits), envelope protection (airspeed must remain within safe bounds), and watchdog timing (control loop must execute within a fixed time bound or the backup system takes over).

For aiai, the lesson from these domains is clear: **safety invariants must be enforced at the infrastructure level, not the application level**. The AI agent should not be able to bypass them, just as application code cannot bypass database constraints.

---

## 6. Protected File Mechanisms

### 6.1 The Problem

In a fully autonomous system, the AI agent has write access to the entire codebase -- including the safety infrastructure itself. Without protection, the agent could (intentionally or through a bug):

- Modify circuit breaker thresholds to be more permissive
- Delete test files to make CI pass
- Edit the CLAUDE.md to remove safety constraints
- Modify cost control settings to increase budgets
- Alter the protected file list itself

This is the **self-modification paradox**: the system that enforces safety is itself part of the system being modified.

### 6.2 Protected File List

Define a canonical list of files that must not be modified by the autonomous agent:

```python
# config/protected_files.py
# This file itself is protected.

PROTECTED_FILES = [
    # Safety infrastructure
    "src/aiai/safety/circuit_breaker.py",
    "src/aiai/safety/loop_detector.py",
    "src/aiai/safety/cost_controller.py",
    "src/aiai/safety/invariants.py",
    "src/aiai/safety/protected_files.py",

    # Configuration
    "config/protected_files.py",
    "config/cost_limits.yaml",
    "config/safety_thresholds.yaml",

    # Git hooks (enforcement mechanism)
    ".git/hooks/pre-commit",
    ".git/hooks/pre-push",

    # Core project definition
    "CLAUDE.md",
]

PROTECTED_PATTERNS = [
    # All files in the safety directory
    "src/aiai/safety/**",
    # All git hooks
    ".git/hooks/*",
]
```

### 6.3 Git Hook Enforcement

```bash
#!/usr/bin/env bash
# .git/hooks/pre-commit -- protected file enforcement
set -euo pipefail

# Load protected file list
PROTECTED_FILES=(
    "src/aiai/safety/circuit_breaker.py"
    "src/aiai/safety/loop_detector.py"
    "src/aiai/safety/cost_controller.py"
    "src/aiai/safety/invariants.py"
    "config/protected_files.py"
    "config/cost_limits.yaml"
    ".git/hooks/pre-commit"
    "CLAUDE.md"
)

# Check if any protected files are in the staged changes
VIOLATIONS=()
for file in "${PROTECTED_FILES[@]}"; do
    if git diff --cached --name-only | grep -qF "$file"; then
        VIOLATIONS+=("$file")
    fi
done

if [ ${#VIOLATIONS[@]} -gt 0 ]; then
    echo "========================================="
    echo "BLOCKED: Protected files modified"
    echo "========================================="
    for v in "${VIOLATIONS[@]}"; do
        echo "  - $v"
    done
    echo ""
    echo "Protected files require the AIAI_OVERRIDE_PROTECTION=1"
    echo "environment variable to modify."
    echo "========================================="

    # Allow override with explicit environment variable
    if [ "${AIAI_OVERRIDE_PROTECTION:-0}" = "1" ]; then
        echo "WARNING: Protection override active. Allowing commit."
        exit 0
    fi

    exit 1
fi
```

### 6.4 Checksum Verification

Store checksums of protected files and verify them at multiple points:

```python
import hashlib
import json
from pathlib import Path
from typing import Optional


class ProtectedFileChecker:
    """Verifies integrity of protected files via checksums.

    Checksums are stored in a separate file and verified:
    - At system startup
    - Before each autonomous task
    - In pre-commit hooks
    - Periodically by the invariant monitor
    """

    CHECKSUM_FILE = "config/protected_checksums.json"

    def __init__(self, protected_files: list[str]):
        self.protected_files = protected_files

    def compute_checksums(self) -> dict[str, str]:
        """Compute current checksums for all protected files."""
        checksums = {}
        for path_str in self.protected_files:
            path = Path(path_str)
            if path.exists():
                content = path.read_bytes()
                checksums[path_str] = hashlib.sha256(content).hexdigest()
            else:
                checksums[path_str] = "MISSING"
        return checksums

    def save_checksums(self) -> None:
        """Save current checksums to disk (run during setup/release)."""
        checksums = self.compute_checksums()
        Path(self.CHECKSUM_FILE).write_text(
            json.dumps(checksums, indent=2) + "\n"
        )

    def verify(self) -> list[str]:
        """Verify all protected files against stored checksums.

        Returns list of violations (empty if all OK).
        """
        if not Path(self.CHECKSUM_FILE).exists():
            return ["Checksum file missing -- cannot verify protected files"]

        stored = json.loads(Path(self.CHECKSUM_FILE).read_text())
        current = self.compute_checksums()
        violations = []

        for path_str in self.protected_files:
            stored_hash = stored.get(path_str)
            current_hash = current.get(path_str)

            if stored_hash is None:
                violations.append(f"{path_str}: not in stored checksums")
            elif current_hash == "MISSING":
                violations.append(f"{path_str}: file is missing")
            elif stored_hash != current_hash:
                violations.append(
                    f"{path_str}: checksum mismatch "
                    f"(expected {stored_hash[:12]}..., "
                    f"got {current_hash[:12]}...)"
                )

        return violations
```

### 6.5 How Other Autonomous Systems Handle This

**CORE (Constitutional Governance)**: The CORE autonomous coding agent uses an "immutable constitution" -- a set of rules that are semantically vectorized so the AI agent understands *why* rules exist, not just what they say. The constitution itself is stored outside the agent's writable file space.

**Reference**: [CORE: Autonomous coding agent with an immutable constitution](https://github.com/DariuszNewecki/CORE)

**CrossCheck**: Uses git branch protection and multi-modal review. The AI agent works on feature branches, and a separate review agent validates changes before merge. Protected files are enforced via a combination of git hooks and branch protection rules.

**Reference**: [CrossCheck: A self-correcting system for AI coding loops](https://github.com/sburl/CrossCheck)

**Claude Code Hooks**: Claude Code's hook system allows `PreToolUse` hooks that can block file modifications. However, a security gap was identified (GitHub issue #11226): permission configurations (`permissions.deny`) are not always enforced by Edit/Write tools, reducing them to trust-based behavioral guidance. This underscores the need for **infrastructure-level** enforcement (git hooks, filesystem permissions) rather than relying on the agent framework alone.

**Reference**: [Security Gap: Hooks Cannot Be Protected From Modification](https://github.com/anthropics/claude-code/issues/11226)

### 6.6 Defense in Depth

No single protection mechanism is sufficient. The recommended approach is layered:

1. **Agent-level**: Hook-based rules that tell the agent not to modify protected files (lowest assurance, but reduces accidental modifications).
2. **Git-level**: Pre-commit hooks that reject commits containing protected file changes (medium assurance, requires git to be the commit path).
3. **Filesystem-level**: POSIX permissions (chmod 444) or chattr immutable flag on protected files (high assurance, OS-enforced).
4. **Checksum-level**: External verification that protected files have not been modified (detection, not prevention, but works even if other layers are bypassed).
5. **Out-of-band verification**: A separate process (watchdog) running outside the agent's control that periodically verifies protected file integrity.

---

## 7. Cost Control Architecture

### 7.1 Why Cost Control Is a Safety Mechanism

In an autonomous system with no human gates, cost is the most tangible risk of a runaway agent. A single misdirected agent loop calling Claude Opus for 30 minutes can cost $50-200+. Multiple concurrent agents with token runaway can burn through a monthly budget in hours.

Cost control is not just FinOps -- it is a **safety mechanism**. Unbounded cost implies unbounded computation, which implies unbounded potential for damage.

### 7.2 Budget Hierarchy

```
 Monthly Budget ($500)
 ├── Daily Budget ($25)
 │   ├── Hourly Budget ($5)
 │   │   ├── Per-Task Budget ($2)
 │   │   │   └── Per-Request Budget ($0.50)
 │   │   └── Per-Task Budget ($2)
 │   └── Hourly Budget ($5)
 └── Daily Budget ($25)
```

Each level enforces independently. A task cannot exceed its budget even if the daily budget has room. This prevents a single expensive task from consuming an entire day's budget.

### 7.3 Complete Cost Controller Implementation

```python
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


class BudgetPeriod(Enum):
    REQUEST = "request"
    TASK = "task"
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


class LimitType(Enum):
    SOFT = "soft"   # Warning, allow to proceed
    HARD = "hard"   # Block the request


@dataclass
class BudgetLimit:
    period: BudgetPeriod
    max_cost: float
    limit_type: LimitType = LimitType.HARD
    soft_threshold: float = 0.80  # Warn at 80% of budget


@dataclass
class CostRecord:
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    task_id: str
    agent_id: str


class CostController:
    """Complete cost control system for autonomous LLM agents.

    Features:
    - Multi-level budget enforcement (request, task, hourly, daily, monthly)
    - Soft and hard limits
    - Cost estimation before execution
    - Real-time cost tracking with attribution
    - Automatic model downgrade when budget is low
    - Alert callbacks for soft limit warnings
    """

    # Model pricing ($ per 1M tokens)
    MODEL_PRICING = {
        "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "deepseek-r1": {"input": 0.55, "output": 2.19},
    }

    def __init__(
        self,
        budgets: list[BudgetLimit],
        on_soft_limit: Optional[Callable[[BudgetPeriod, float, float], None]] = None,
        on_hard_limit: Optional[Callable[[BudgetPeriod, float, float], None]] = None,
    ):
        self.budgets = {b.period: b for b in budgets}
        self.on_soft_limit = on_soft_limit
        self.on_hard_limit = on_hard_limit
        self._records: list[CostRecord] = []
        self._current_task_id: Optional[str] = None
        self._lock = threading.Lock()

    def estimate_cost(
        self, model: str, input_tokens: int, estimated_output_tokens: int
    ) -> float:
        """Estimate cost before making a request."""
        pricing = self.MODEL_PRICING.get(model)
        if pricing is None:
            # Unknown model -- use conservative (expensive) estimate
            return (input_tokens * 15.0 + estimated_output_tokens * 75.0) / 1_000_000
        return (
            input_tokens * pricing["input"] / 1_000_000
            + estimated_output_tokens * pricing["output"] / 1_000_000
        )

    def authorize_request(
        self,
        model: str,
        input_tokens: int,
        estimated_output_tokens: int,
        task_id: str,
        agent_id: str,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Authorize a request against all budget levels.

        Returns:
            (allowed, block_reason, suggested_model)
            - allowed: True if request can proceed
            - block_reason: Why it was blocked (None if allowed)
            - suggested_model: Cheaper model suggestion (None if no change)
        """
        estimated_cost = self.estimate_cost(model, input_tokens, estimated_output_tokens)

        with self._lock:
            # Check per-request limit
            if BudgetPeriod.REQUEST in self.budgets:
                limit = self.budgets[BudgetPeriod.REQUEST]
                if estimated_cost > limit.max_cost:
                    if limit.limit_type == LimitType.HARD:
                        return (
                            False,
                            f"Request cost ${estimated_cost:.3f} exceeds "
                            f"limit ${limit.max_cost:.2f}",
                            self._suggest_cheaper_model(model, limit.max_cost),
                        )

            # Check per-task limit
            if BudgetPeriod.TASK in self.budgets:
                limit = self.budgets[BudgetPeriod.TASK]
                task_cost = self._get_cost_for_task(task_id)
                if task_cost + estimated_cost > limit.max_cost:
                    remaining = limit.max_cost - task_cost
                    if limit.limit_type == LimitType.HARD:
                        return (
                            False,
                            f"Task budget exhausted: ${task_cost:.2f} spent "
                            f"of ${limit.max_cost:.2f}",
                            self._suggest_cheaper_model(model, remaining),
                        )
                elif task_cost + estimated_cost > limit.max_cost * limit.soft_threshold:
                    if self.on_soft_limit:
                        self.on_soft_limit(
                            BudgetPeriod.TASK, task_cost, limit.max_cost
                        )

            # Check hourly limit
            if BudgetPeriod.HOURLY in self.budgets:
                limit = self.budgets[BudgetPeriod.HOURLY]
                hourly_cost = self._get_cost_for_period(3600)
                if hourly_cost + estimated_cost > limit.max_cost:
                    if limit.limit_type == LimitType.HARD:
                        return (
                            False,
                            f"Hourly budget exhausted: ${hourly_cost:.2f} "
                            f"of ${limit.max_cost:.2f}",
                            None,
                        )

            # Check daily limit
            if BudgetPeriod.DAILY in self.budgets:
                limit = self.budgets[BudgetPeriod.DAILY]
                daily_cost = self._get_cost_for_period(86400)
                if daily_cost + estimated_cost > limit.max_cost:
                    if limit.limit_type == LimitType.HARD:
                        return (
                            False,
                            f"Daily budget exhausted: ${daily_cost:.2f} "
                            f"of ${limit.max_cost:.2f}",
                            None,
                        )
                elif daily_cost + estimated_cost > limit.max_cost * limit.soft_threshold:
                    if self.on_soft_limit:
                        self.on_soft_limit(
                            BudgetPeriod.DAILY, daily_cost, limit.max_cost
                        )

            # Check monthly limit
            if BudgetPeriod.MONTHLY in self.budgets:
                limit = self.budgets[BudgetPeriod.MONTHLY]
                monthly_cost = self._get_cost_for_period(86400 * 30)
                if monthly_cost + estimated_cost > limit.max_cost:
                    if limit.limit_type == LimitType.HARD:
                        return (
                            False,
                            f"Monthly budget exhausted: ${monthly_cost:.2f} "
                            f"of ${limit.max_cost:.2f}",
                            None,
                        )

        return True, None, None

    def record_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_id: str,
        agent_id: str,
    ) -> CostRecord:
        """Record actual cost after a request completes."""
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        record = CostRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            task_id=task_id,
            agent_id=agent_id,
        )
        with self._lock:
            self._records.append(record)
        return record

    def get_cost_summary(self) -> dict:
        """Get cost summary by period, agent, model, and task."""
        with self._lock:
            now = time.time()
            return {
                "last_hour": self._get_cost_for_period(3600),
                "last_day": self._get_cost_for_period(86400),
                "last_month": self._get_cost_for_period(86400 * 30),
                "by_model": self._group_cost_by("model"),
                "by_agent": self._group_cost_by("agent_id"),
                "by_task": self._group_cost_by("task_id"),
                "total_records": len(self._records),
            }

    def _get_cost_for_period(self, seconds: float) -> float:
        cutoff = time.time() - seconds
        return sum(r.cost for r in self._records if r.timestamp > cutoff)

    def _get_cost_for_task(self, task_id: str) -> float:
        return sum(r.cost for r in self._records if r.task_id == task_id)

    def _group_cost_by(self, field: str) -> dict[str, float]:
        groups: dict[str, float] = {}
        for record in self._records:
            key = getattr(record, field)
            groups[key] = groups.get(key, 0) + record.cost
        return groups

    def _suggest_cheaper_model(
        self, current_model: str, remaining_budget: float
    ) -> Optional[str]:
        """Suggest a cheaper model that fits within remaining budget."""
        # Sorted cheapest first
        model_costs = sorted(
            self.MODEL_PRICING.items(),
            key=lambda x: x[1]["output"],
        )
        for model_name, pricing in model_costs:
            if model_name != current_model:
                # Estimate cost for a typical request (1000 in, 2000 out)
                typical_cost = (
                    1000 * pricing["input"] / 1_000_000
                    + 2000 * pricing["output"] / 1_000_000
                )
                if typical_cost * 5 < remaining_budget:  # Room for ~5 requests
                    return model_name
        return None
```

### 7.4 Cost Attribution

Every request must be tagged with metadata for attribution:

```python
@dataclass
class CostAttribution:
    """Metadata attached to every LLM request for cost tracking."""
    agent_id: str           # Which agent instance
    task_id: str            # Which task
    task_type: str          # "code_generation", "test_generation", "review", etc.
    model: str              # Which model was used
    priority: str           # "critical", "normal", "background"
    is_retry: bool = False  # Is this a retry of a failed request?
    parent_task_id: Optional[str] = None  # For sub-task cost rollup
```

### 7.5 Budget Configuration

```yaml
# config/cost_limits.yaml
# Protected file -- requires AIAI_OVERRIDE_PROTECTION=1 to modify

budgets:
  monthly:
    max_cost: 500.00
    limit_type: hard
    soft_threshold: 0.80  # Alert at 80%

  daily:
    max_cost: 25.00
    limit_type: hard
    soft_threshold: 0.80

  hourly:
    max_cost: 5.00
    limit_type: soft  # Warn but allow (burst tasks)
    soft_threshold: 0.90

  per_task:
    max_cost: 2.00
    limit_type: hard
    soft_threshold: 0.70

  per_request:
    max_cost: 0.50
    limit_type: hard

alerts:
  channels:
    - type: log
      level: warning
    - type: file
      path: logs/cost_alerts.jsonl

model_downgrade_policy:
  # When budget is running low, automatically downgrade models
  enabled: true
  rules:
    - when_remaining_percent: 20
      downgrade_from: claude-opus-4-20250514
      downgrade_to: claude-sonnet-4-20250514
    - when_remaining_percent: 5
      downgrade_from: claude-sonnet-4-20250514
      downgrade_to: gemini-2.0-flash
```

**Reference**: [From Bills to Budgets: How to Track LLM Token Usage](https://www.traceloop.com/blog/from-bills-to-budgets-how-to-track-llm-token-usage-and-cost-per-user)
**Reference**: [How to Build Cost Management for LLM Operations](https://oneuptime.com/blog/post/2026-01-30-llmops-cost-management/view)

---

## 8. Quality Regression Detection

### 8.1 The Problem

In an autonomous self-improving system, every change has the potential to **subtly degrade** system quality. Unlike a dramatic failure that breaks tests, quality regression is a slow drift: test execution gets 10% slower, mutation scores drop from 75% to 68%, API response quality decreases by a few percentage points. No single change is alarming, but over 50 commits the system has degraded significantly.

Without explicit quality regression detection, autonomous systems tend toward **metric drift** -- optimizing for whatever is directly measured (test pass/fail) while unmeasured qualities silently erode.

### 8.2 Metrics to Track

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualitySnapshot:
    """Point-in-time quality metrics for the system."""
    timestamp: float
    commit_hash: str

    # Test quality
    tests_total: int
    tests_passing: int
    test_execution_time_seconds: float
    mutation_score: float  # 0.0-1.0
    coverage_line: float   # 0.0-1.0
    coverage_branch: float # 0.0-1.0

    # Code quality
    lint_errors: int
    type_errors: int
    complexity_avg: float  # Average cyclomatic complexity
    complexity_max: int    # Maximum cyclomatic complexity

    # Performance
    benchmark_scores: dict[str, float]  # Named benchmarks

    # Cost efficiency
    avg_cost_per_task: float
    avg_tokens_per_task: int
```

### 8.3 Statistical Significance Testing

A 5% drop in mutation score -- is that a real regression or random noise? Statistical testing is essential to avoid false alarms (and missed regressions).

```python
import math
from dataclasses import dataclass


@dataclass
class RegressionTestResult:
    metric_name: str
    baseline_mean: float
    current_mean: float
    change_percent: float
    p_value: float
    is_significant: bool
    sample_size: int


class QualityRegressionDetector:
    """Detects statistically significant quality regressions.

    Uses Welch's t-test for comparing metric distributions,
    which does not assume equal variances.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 5,
    ):
        self.significance_level = significance_level
        self.min_samples = min_samples

    def test_regression(
        self,
        metric_name: str,
        baseline_values: list[float],
        current_values: list[float],
    ) -> RegressionTestResult:
        """Test whether current values represent a regression from baseline.

        Uses Welch's t-test (unequal variance t-test).
        """
        if (
            len(baseline_values) < self.min_samples
            or len(current_values) < self.min_samples
        ):
            return RegressionTestResult(
                metric_name=metric_name,
                baseline_mean=_mean(baseline_values),
                current_mean=_mean(current_values),
                change_percent=0.0,
                p_value=1.0,
                is_significant=False,
                sample_size=min(len(baseline_values), len(current_values)),
            )

        baseline_mean = _mean(baseline_values)
        current_mean = _mean(current_values)
        baseline_var = _variance(baseline_values)
        current_var = _variance(current_values)
        n1 = len(baseline_values)
        n2 = len(current_values)

        # Welch's t-statistic
        se = math.sqrt(baseline_var / n1 + current_var / n2)
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (current_mean - baseline_mean) / se

        # Welch-Satterthwaite degrees of freedom
        if baseline_var == 0 and current_var == 0:
            df = n1 + n2 - 2
        else:
            numerator = (baseline_var / n1 + current_var / n2) ** 2
            denominator = (
                (baseline_var / n1) ** 2 / (n1 - 1)
                + (current_var / n2) ** 2 / (n2 - 1)
            )
            df = numerator / denominator if denominator > 0 else 1

        # Approximate p-value using t-distribution
        # For a one-sided test (detecting decrease)
        p_value = self._t_cdf(-abs(t_stat), df)

        change_pct = (
            ((current_mean - baseline_mean) / baseline_mean * 100)
            if baseline_mean != 0
            else 0.0
        )

        return RegressionTestResult(
            metric_name=metric_name,
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            change_percent=change_pct,
            p_value=p_value,
            is_significant=p_value < self.significance_level,
            sample_size=min(n1, n2),
        )

    def _t_cdf(self, t: float, df: float) -> float:
        """Approximate CDF of t-distribution.

        Uses the regularized incomplete beta function approximation.
        For production use, prefer scipy.stats.t.cdf.
        """
        # Approximation via normal distribution for large df
        if df > 30:
            # Use normal approximation
            return 0.5 * (1 + math.erf(t / math.sqrt(2)))
        # For small df, use a simple approximation
        x = df / (df + t * t)
        # This is a rough approximation -- use scipy in production
        return 0.5 * x ** (df / 2) if t < 0 else 1 - 0.5 * x ** (df / 2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)
```

### 8.4 Automatic Revert Triggers

When a statistically significant regression is detected, the system should be able to automatically revert:

```python
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class RevertPolicy:
    """When and how to automatically revert commits."""

    # Thresholds for automatic revert (must be both significant AND large)
    auto_revert_thresholds: dict[str, float] = None

    def __post_init__(self):
        if self.auto_revert_thresholds is None:
            self.auto_revert_thresholds = {
                "mutation_score": -10.0,      # >10% drop
                "tests_passing_rate": -5.0,   # >5% drop
                "coverage_line": -5.0,        # >5% drop
                "test_execution_time": 50.0,  # >50% slower
            }

    def should_revert(
        self, results: list[RegressionTestResult]
    ) -> tuple[bool, list[str]]:
        """Determine if regression results warrant an automatic revert."""
        revert_reasons = []
        for result in results:
            if not result.is_significant:
                continue
            threshold = self.auto_revert_thresholds.get(result.metric_name)
            if threshold is None:
                continue
            # For time metrics, positive change is bad (slower)
            # For quality metrics, negative change is bad (worse)
            if "time" in result.metric_name:
                if result.change_percent > threshold:
                    revert_reasons.append(
                        f"{result.metric_name}: +{result.change_percent:.1f}% "
                        f"(threshold: +{threshold:.1f}%)"
                    )
            else:
                if result.change_percent < threshold:
                    revert_reasons.append(
                        f"{result.metric_name}: {result.change_percent:.1f}% "
                        f"(threshold: {threshold:.1f}%)"
                    )

        return len(revert_reasons) > 0, revert_reasons


def auto_revert_commit(commit_hash: str, reasons: list[str]) -> bool:
    """Automatically revert a commit that caused quality regression."""
    reason_text = "; ".join(reasons)

    result = subprocess.run(
        ["git", "revert", "--no-edit", commit_hash],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Revert failed (likely conflict) -- flag for manual intervention
        print(f"AUTO-REVERT FAILED for {commit_hash}: {result.stderr}")
        return False

    # Amend revert commit message with regression details
    subprocess.run(
        [
            "git", "commit", "--amend", "-m",
            f"Revert {commit_hash[:8]}: quality regression detected\n\n"
            f"Automatic revert triggered by:\n"
            + "\n".join(f"  - {r}" for r in reasons),
        ],
        capture_output=True,
    )

    print(f"AUTO-REVERTED {commit_hash[:8]}: {reason_text}")
    return True
```

### 8.5 Baseline Management

Baselines must be updated periodically to reflect intentional changes:

```python
class BaselineManager:
    """Manages quality baselines for regression detection.

    Baselines are updated:
    - After a successful release
    - After intentional quality changes (with explicit approval)
    - On a rolling window (last N commits)
    """

    def __init__(
        self,
        window_size: int = 20,
        storage_path: str = "config/quality_baselines.json",
    ):
        self.window_size = window_size
        self.storage_path = storage_path
        self._snapshots: list[QualitySnapshot] = []

    def add_snapshot(self, snapshot: QualitySnapshot) -> None:
        self._snapshots.append(snapshot)
        # Keep bounded
        if len(self._snapshots) > self.window_size * 3:
            self._snapshots = self._snapshots[-self.window_size * 2:]

    def get_baseline_values(self, metric_name: str) -> list[float]:
        """Get baseline values for a metric (older snapshots)."""
        baseline = self._snapshots[:self.window_size]
        return [getattr(s, metric_name, 0.0) for s in baseline]

    def get_current_values(self, metric_name: str) -> list[float]:
        """Get current values for a metric (recent snapshots)."""
        current = self._snapshots[-self.window_size:]
        return [getattr(s, metric_name, 0.0) for s in current]
```

**Reference**: [Model performance regression: Detecting quality decline](https://www.statsig.com/perspectives/model-performance-quality-decline)

---

## 9. Self-Modification Safety in Practice

### 9.1 The Bootstrapping Trust Problem

When a system modifies itself, a fundamental question arises: **how do you trust the modifications if the system making them might already be compromised?** This is the bootstrapping trust problem -- the same challenge faced by compiler bootstrapping (Ken Thompson's "Reflections on Trusting Trust", 1984).

For aiai, this manifests as:

- The AI modifies the test generator. The modified test generator produces tests that validate the modification. But the test generator was modified by the same AI -- circular trust.
- The AI improves the cost controller to be "more efficient." The improvement subtly raises spending limits. The cost reports (generated by the same system) show everything is fine.
- The AI refactors the safety infrastructure. The refactored safety code passes all tests (which were also refactored). But a subtle invariant was broken.

### 9.2 The Immutable Safety Core

The solution, borrowed from aerospace and nuclear systems, is an **immutable safety core** -- a minimal set of components that cannot be modified by the system, period. The safety core serves as the fixed reference point against which all modifications are evaluated.

```
┌─────────────────────────────────────────────────────┐
│                   MUTABLE LAYER                      │
│  (AI can freely modify)                             │
│                                                      │
│  Application code, features, optimizations,          │
│  non-safety tests, documentation, configs            │
│                                                      │
├─────────────────────────────────────────────────────┤
│                  GUARDED LAYER                       │
│  (AI can modify with elevated verification)          │
│                                                      │
│  Core libraries, model routing, agent orchestration, │
│  primary test suites, CI configuration               │
│                                                      │
├─────────────────────────────────────────────────────┤
│                IMMUTABLE SAFETY CORE                 │
│  (Cannot be modified by the AI, period)              │
│                                                      │
│  Circuit breakers, cost controllers, loop detectors, │
│  invariant checkers, protected file enforcement,     │
│  watchdog process, rollback mechanism                │
│                                                      │
└─────────────────────────────────────────────────────┘
```

The immutable safety core is enforced by:
1. Filesystem permissions (read-only for the agent process)
2. Git hooks that reject modifications
3. Checksum verification by an external watchdog
4. Separate deployment pipeline (safety core is deployed independently)

### 9.3 Watchdog Processes

A watchdog is an independent process that monitors the primary system from outside. It cannot be modified by the agent because it runs in a separate process space with separate permissions.

```python
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class SafetyWatchdog:
    """Independent watchdog process for monitoring autonomous agents.

    Runs as a separate process with its own permissions. Cannot be
    modified by the agent it monitors. Performs periodic health checks
    and can kill the agent if safety invariants are violated.

    Designed to be started by the system supervisor (systemd, launchd),
    not by the agent itself.
    """

    def __init__(
        self,
        agent_pid_file: str,
        checksum_file: str,
        max_cost_file: str,
        check_interval: float = 30.0,
    ):
        self.agent_pid_file = agent_pid_file
        self.checksum_file = checksum_file
        self.max_cost_file = max_cost_file
        self.check_interval = check_interval

    def run(self) -> None:
        """Main watchdog loop."""
        print(f"Watchdog started (PID {os.getpid()})")
        while True:
            try:
                self._check_agent_alive()
                self._check_protected_files()
                self._check_cost_limits()
                self._check_agent_responsiveness()
            except WatchdogAlert as e:
                self._handle_alert(e)
            time.sleep(self.check_interval)

    def _check_protected_files(self) -> None:
        """Verify protected files have not been tampered with."""
        checker = ProtectedFileChecker(PROTECTED_FILES)
        violations = checker.verify()
        if violations:
            raise WatchdogAlert(
                f"Protected file violations: {violations}",
                severity="critical",
                action="kill_agent",
            )

    def _check_cost_limits(self) -> None:
        """Check if cost limits have been exceeded."""
        # Read cost data from a shared file (not from the agent process)
        cost_file = Path(self.max_cost_file)
        if cost_file.exists():
            import json
            data = json.loads(cost_file.read_text())
            if data.get("daily_exceeded"):
                raise WatchdogAlert(
                    f"Daily cost limit exceeded: ${data['daily_total']:.2f}",
                    severity="critical",
                    action="kill_agent",
                )

    def _check_agent_responsiveness(self) -> None:
        """Check if the agent is responsive (not hung)."""
        heartbeat_file = Path("logs/agent_heartbeat")
        if heartbeat_file.exists():
            last_beat = heartbeat_file.stat().st_mtime
            if time.time() - last_beat > 300:  # 5 minutes
                raise WatchdogAlert(
                    "Agent heartbeat stale (>5 minutes)",
                    severity="warning",
                    action="log_only",
                )

    def _check_agent_alive(self) -> None:
        """Verify the agent process is still running."""
        pid_file = Path(self.agent_pid_file)
        if not pid_file.exists():
            return  # No agent running -- nothing to watch
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)  # Check if process exists (signal 0)
        except ProcessLookupError:
            # Agent died -- clean up
            pid_file.unlink()

    def _handle_alert(self, alert: "WatchdogAlert") -> None:
        """Handle a watchdog alert based on severity."""
        print(f"WATCHDOG ALERT [{alert.severity}]: {alert.message}")

        if alert.action == "kill_agent":
            self._kill_agent()
        elif alert.action == "pause_agent":
            self._pause_agent()
        # Always log
        self._log_alert(alert)

    def _kill_agent(self) -> None:
        """Terminate the agent process."""
        pid_file = Path(self.agent_pid_file)
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(5)
                # If still alive, force kill
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
            pid_file.unlink(missing_ok=True)
        print("WATCHDOG: Agent terminated")

    def _pause_agent(self) -> None:
        """Pause the agent (SIGSTOP)."""
        pid_file = Path(self.agent_pid_file)
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            try:
                os.kill(pid, signal.SIGSTOP)
                print("WATCHDOG: Agent paused (SIGSTOP)")
            except ProcessLookupError:
                pass

    def _log_alert(self, alert: "WatchdogAlert") -> None:
        """Log alert to watchdog log file."""
        import json
        log_entry = {
            "timestamp": time.time(),
            "message": alert.message,
            "severity": alert.severity,
            "action": alert.action,
        }
        with open("logs/watchdog_alerts.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class WatchdogAlert(Exception):
    def __init__(self, message: str, severity: str, action: str):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.action = action
```

### 9.4 Checkpointing and Rollback

Every autonomous modification must be a reversible operation. Git provides the foundation, but we need structured checkpointing:

```python
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Checkpoint:
    """A safe point the system can roll back to."""
    commit_hash: str
    timestamp: float
    quality_snapshot: QualitySnapshot
    description: str
    is_known_good: bool = False  # Explicitly marked as stable


class CheckpointManager:
    """Manages system checkpoints for safe rollback.

    Automatically creates checkpoints before risky operations
    and provides rollback to the last known-good state.
    """

    def __init__(self, max_checkpoints: int = 50):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: list[Checkpoint] = []

    def create_checkpoint(
        self, description: str, quality: QualitySnapshot
    ) -> Checkpoint:
        """Create a checkpoint at the current state."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        commit_hash = result.stdout.strip()

        checkpoint = Checkpoint(
            commit_hash=commit_hash,
            timestamp=time.time(),
            quality_snapshot=quality,
            description=description,
        )
        self._checkpoints.append(checkpoint)

        # Prune old checkpoints (keep known-good ones)
        if len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints = [
                cp for cp in self._checkpoints if cp.is_known_good
            ] + self._checkpoints[-self.max_checkpoints // 2:]

        return checkpoint

    def mark_known_good(self, commit_hash: str) -> None:
        """Mark a checkpoint as known-good (stable baseline)."""
        for cp in self._checkpoints:
            if cp.commit_hash == commit_hash:
                cp.is_known_good = True
                return

    def get_last_known_good(self) -> Optional[Checkpoint]:
        """Get the most recent known-good checkpoint."""
        for cp in reversed(self._checkpoints):
            if cp.is_known_good:
                return cp
        return None

    def rollback_to(self, checkpoint: Checkpoint) -> bool:
        """Roll back the system to a checkpoint.

        Uses git revert (not reset) to preserve history.
        """
        current = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()

        if current == checkpoint.commit_hash:
            return True  # Already at this checkpoint

        # Get list of commits to revert (newest first)
        result = subprocess.run(
            [
                "git", "log", "--format=%H",
                f"{checkpoint.commit_hash}..HEAD",
            ],
            capture_output=True,
            text=True,
        )
        commits_to_revert = result.stdout.strip().split("\n")

        # Revert each commit in reverse order
        for commit in commits_to_revert:
            if not commit:
                continue
            revert_result = subprocess.run(
                ["git", "revert", "--no-edit", commit],
                capture_output=True,
                text=True,
            )
            if revert_result.returncode != 0:
                # Conflict -- abort and flag for manual intervention
                subprocess.run(["git", "revert", "--abort"])
                print(f"ROLLBACK FAILED at {commit}: {revert_result.stderr}")
                return False

        print(
            f"ROLLED BACK to {checkpoint.commit_hash[:8]} "
            f"({checkpoint.description})"
        )
        return True
```

### 9.5 Preventing Cascading Failures

Cascading failures in autonomous AI systems occur when a single fault propagates across multiple agents or components, amplifying into system-wide harm. OWASP classifies this as ASI08 in their Agentic AI Top 10 (2026).

**Prevention architecture**:

```
┌────────────┐    ┌────────────┐    ┌────────────┐
│  Agent A   │    │  Agent B   │    │  Agent C   │
│            │    │            │    │            │
│  ┌──────┐  │    │  ┌──────┐  │    │  ┌──────┐  │
│  │Circ. │  │    │  │Circ. │  │    │  │Circ. │  │
│  │Break.│  │    │  │Break.│  │    │  │Break.│  │
│  └──────┘  │    │  └──────┘  │    │  └──────┘  │
└─────┬──────┘    └─────┬──────┘    └─────┬──────┘
      │                 │                 │
      └────────┬────────┘                 │
               │                          │
      ┌────────v──────────────────────────v────┐
      │         TRUST BOUNDARY                  │
      │   (isolation, output validation,        │
      │    independent verification)            │
      ├─────────────────────────────────────────┤
      │         SHARED STATE                    │
      │   (git repo, config, databases)         │
      │                                         │
      │   Protected by invariant checking,      │
      │   checksums, and watchdog monitoring     │
      └─────────────────────────────────────────┘
```

Key principles from OWASP ASI08:

1. **Architectural isolation**: Each agent has its own circuit breaker. One agent's failure cannot propagate to others.
2. **Trust boundaries**: Agent outputs are validated before being consumed by other agents or written to shared state.
3. **Ground truth validation**: Multi-agent consensus -- if two agents disagree, a third arbitrates (or the operation is rejected).
4. **Kill switches**: External mechanism (watchdog) that can terminate any or all agents instantly.

**Reference**: [Cascading Failures in Agentic AI: Complete OWASP ASI08 Security Guide](https://adversa.ai/blog/cascading-failures-in-agentic-ai-complete-owasp-asi08-security-guide-2026/)
**Reference**: [Towards Guaranteed Safe AI: A Framework for Ensuring Robust and Reliable AI Systems](https://arxiv.org/abs/2405.06624)

### 9.6 The Guaranteed Safe AI Framework

Dalrymple et al. (2024) propose a rigorous framework for AI safety with three components:

1. **World model**: A mathematical description of how the AI system affects the outside world.
2. **Safety specification**: A mathematical description of what effects are acceptable.
3. **Verifier**: An auditable proof certificate that the AI satisfies the safety specification relative to the world model.

For aiai, a practical (non-mathematical) adaptation:

- **World model**: The git repository state, the CI pipeline, the cost tracking system, the deployment environment. These are the things the agent can affect.
- **Safety specification**: Invariants (Section 5), protected files (Section 6), cost limits (Section 7), quality baselines (Section 8).
- **Verifier**: The combination of mutation testing, invariant monitors, watchdog processes, and quality regression detection. These provide (non-mathematical but empirical) assurance that safety specifications are met.

### 9.7 Real Systems That Modify Themselves

**AlphaEvolve (DeepMind, 2025)**: An evolutionary coding agent that self-improves algorithmic implementations. Safety is maintained through a strict separation between the evolutionary search (which can explore freely) and the evaluation pipeline (which provides objective, programmatically computed fitness scores). The evaluator is the immutable safety core -- it cannot be modified by the evolutionary process.

**Gitar**: An autonomous code change agent that learned a cost lesson the hard way. After switching to a 5x cheaper LLM, their costs went *up* because the cheaper model needed more iterations to complete tasks. The lesson: cost-per-token is not the same as cost-per-task. Their circuit breaker now monitors cost-per-completed-task, not just token usage.

**Reference**: [We switched to a 5x cheaper LLM. Our costs went up.](https://gitar.ai/blog/we-switched-to-a-5x-cheaper-llm-our-costs-went-up)

**Addy Osmani's Self-Improving Coding Agents** framework identifies four necessary components for safe self-improvement: (1) measurable evaluation criteria, (2) bounded search spaces, (3) human-reviewable artifacts, and (4) rollback capability. For fully autonomous systems like aiai where human review is not in the loop, the other three components must be proportionally stronger.

**Reference**: [Self-Improving Coding Agents](https://addyosmani.com/blog/self-improving-agents/)

---

## 10. Implications for aiai

### 10.1 Priority Implementation Order

Based on the research in this document, the recommended implementation priority for aiai's safety infrastructure is:

1. **Protected files + git hooks** (Section 6) -- Foundational. Must exist before any autonomous operation. Low complexity, high value. Prevents the AI from modifying its own safety constraints.

2. **Cost controller** (Section 7) -- Existential risk mitigation. Without cost controls, a runaway agent can burn through the entire budget. Implement before enabling continuous autonomous operation.

3. **Circuit breakers** (Section 2) -- The composite circuit breaker (token + cost + quality + time) provides the runtime safety net. Must be wired into the agent execution loop.

4. **Loop detection** (Section 3) -- Start with the action hash detector (simplest), then add progress detection. Semantic similarity detection is a later enhancement.

5. **Invariant checking** (Section 5) -- Define the core system invariants and wire them into pre-commit hooks and a background monitor.

6. **Quality regression detection** (Section 8) -- Requires accumulating quality snapshots over time. Start collecting metrics immediately; the regression detector becomes useful after 20+ commits.

7. **Mutation testing** (Section 4) -- The ground truth for test quality. Implement lightweight (fast-mode) mutation testing first, full mutation testing in CI.

8. **Watchdog process** (Section 9) -- The out-of-band safety monitor. Requires the other components to exist first (it monitors their integrity).

### 10.2 The Safety Stack

```
┌─────────────────────────────────────────────┐
│           WATCHDOG PROCESS                   │
│  (external, independent, cannot be modified) │
├─────────────────────────────────────────────┤
│           INVARIANT MONITOR                  │
│  (continuous background checking)            │
├─────────────────────────────────────────────┤
│     CIRCUIT BREAKERS + LOOP DETECTION        │
│  (runtime protection during agent execution) │
├─────────────────────────────────────────────┤
│    COST CONTROLLER + QUALITY REGRESSION      │
│  (budget enforcement + drift detection)      │
├─────────────────────────────────────────────┤
│   MUTATION TESTING + TEST VALIDATION         │
│  (ground truth test quality assurance)       │
├─────────────────────────────────────────────┤
│   PROTECTED FILES + GIT HOOKS                │
│  (foundational integrity enforcement)        │
├─────────────────────────────────────────────┤
│              GIT REPOSITORY                  │
│  (immutable history, rollback capability)    │
└─────────────────────────────────────────────┘
```

### 10.3 Key Design Principles

1. **Defense in depth**: No single safety mechanism is sufficient. Every mechanism has failure modes. Layer them so that any single failure is caught by another layer.

2. **Model independence**: Safety mechanisms must not depend on AI model judgment for their core operation. Mutation testing, checksums, cost tracking -- these are deterministic, verifiable, and model-independent.

3. **External enforcement**: Safety constraints must be enforced at the infrastructure level (git hooks, filesystem permissions, watchdog processes), not at the agent level. The agent should not be able to bypass safety controls even if instructed to.

4. **Graceful degradation**: When a circuit breaker trips, the system should degrade gracefully (downgrade model, reduce scope, pause non-critical tasks) rather than halt entirely. Full halts are reserved for invariant violations.

5. **Auditability**: Every safety-relevant event must be logged with enough context for post-incident analysis. The git history is the primary audit trail, supplemented by structured logs.

6. **Empirical validation**: Safety mechanisms themselves must be tested. Periodically inject known failures and verify that circuit breakers trip, loop detectors fire, invariant monitors alert, and the watchdog responds correctly.

---

## 11. References

### Academic Papers

- Dalrymple et al. (2024). *Towards Guaranteed Safe AI: A Framework for Ensuring Robust and Reliable AI Systems*. arXiv:2405.06624. [Link](https://arxiv.org/abs/2405.06624)
- Bornholt et al. (2024). *CoverUp: Coverage-Guided LLM-Based Test Generation*. ACM SIGPLAN. [Link](https://arxiv.org/abs/2403.16218)
- Dinella et al. (2023). *Neural-Based Test Oracle Generation: A Large-Scale Evaluation and Lessons Learned*. [Link](https://www.researchgate.net/publication/376107399)
- Almasi et al. (2024). *Empirical Comparison Between Conventional and AI-Based Automated Unit Test Generation Tools in Java*. [Link](https://fardapaper.ir/mohavaha/uploads/2023/09/5-Empirical-Comparison-Between-Conventional-and-AI-based-Automated-Unit-Test-Generation-Tools-in-Java.pdf)
- Thompson, Ken (1984). *Reflections on Trusting Trust*. Communications of the ACM.
- Nygard, Michael (2007). *Release It!* Pragmatic Bookshelf.
- OWASP (2026). *Top 10 for Agentic Applications*. [Link](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- IEEE (2024). *An Analysis and Comparison of Mutation Testing Tools for Python*. [Link](https://ieeexplore.ieee.org/document/10818231/)
- ACM (2025). *Test Oracle Automation in the Era of LLMs*. [Link](https://dl.acm.org/doi/10.1145/3715107)
- Hybrid Fault-Driven Mutation Testing for Python (2026). [Link](https://arxiv.org/html/2601.19088v1)

### Industry References

- Meta Engineering (2025). *LLMs Are the Key to Mutation Testing and Better Compliance*. [Link](https://engineering.fb.com/2025/09/30/security/llms-are-the-key-to-mutation-testing-and-better-compliance/)
- Diffblue (2025). *Diffblue Cover vs AI Coding Assistants Benchmark*. [Link](https://www.diffblue.com/resources/diffblue-cover-vs-ai-coding-assistants-benchmark-2025/)
- Gitar (2025). *We switched to a 5x cheaper LLM. Our costs went up.* [Link](https://gitar.ai/blog/we-switched-to-a-5x-cheaper-llm-our-costs-went-up)
- Osmani, Addy (2025). *Self-Improving Coding Agents*. [Link](https://addyosmani.com/blog/self-improving-agents/)
- ISACA (2025). *Unseen Unchecked Unraveling: Inside the Risky Code of Self-Modifying AI*. [Link](https://www.isaca.org/resources/news-and-trends/isaca-now-blog/2025/unseen-unchecked-unraveling-inside-the-risky-code-of-self-modifying-ai)
- Adversa AI (2026). *Cascading Failures in Agentic AI: OWASP ASI08 Guide*. [Link](https://adversa.ai/blog/cascading-failures-in-agentic-ai-complete-owasp-asi08-security-guide-2026/)
- NeuralTrust (2025). *Using Circuit Breakers to Secure AI Agents*. [Link](https://neuraltrust.ai/blog/circuit-breakers)
- Alps Agility (2025). *The Economics of Autonomy: Preventing Token Runaway*. [Link](https://www.alpsagility.com/cost-control-agentic-systems)
- Portkey (2025). *Retries, fallbacks, and circuit breakers in LLM apps*. [Link](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)

### Tools

- [mutmut](https://github.com/boxed/mutmut) -- Python mutation testing (most actively maintained)
- [cosmic-ray](https://github.com/sixty-north/cosmic-ray) -- Parallel Python mutation testing
- [CoverUp](https://github.com/plasma-umass/coverup) -- Coverage-guided LLM test generation
- [Diffblue Cover](https://www.diffblue.com/) -- AI agent for Java unit testing
- [invariant-python](https://github.com/andreamancuso/invariant-python) -- Design by Contract for Python
- [CrossCheck](https://github.com/sburl/CrossCheck) -- Self-correcting system for AI coding loops
- [CORE](https://github.com/DariuszNewecki/CORE) -- Autonomous coding agent with immutable constitution
- [Langfuse](https://langfuse.com/) -- LLM observability and cost tracking
- [Helicone](https://www.helicone.ai/) -- LLM cost monitoring
