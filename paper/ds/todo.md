# Decision Sciences Migration Roadmap

This document outlines the tasks required to adapt the COLM paper for the Decision Sciences Special Issue on "Decisions in the Agentic Era" (Track A: Agent-Native Decision Science).

Target submission track: Track A (Priority Fast Track deadline: January 15, 2027).

---

## 1. Manuscript Reframing

We need to rewrite the paper around decision theory, formal hypotheses, and managerial implications rather than presenting it as an empirical model benchmark.

- [ ] **Frame the core problem around organizational delegation**
  - Present the act-or-escalate choice as a fundamental problem of delegating authority to autonomous agents under uncertainty.
  - Position the agent's implicit escalation threshold as a latent behavioral characteristic that organizations must measure before deployment.
  - Avoid framing the paper as an LLM leaderboard or evaluation benchmark.

- [ ] **State formal hypotheses**
  - H1 (Threshold heterogeneity): LLM agents exhibit distinct implicit escalation thresholds that are not predicted by model scale or architecture.
  - H2 (Calibration independence): Agent self-estimated accuracy exhibits systematic miscalibration in both directions, which operates independently from the escalation threshold.
  - H3 (Prompt insufficiency): Passive cost framing in prompts fails to align escalation behavior with optimal decision rules unless combined with deliberate reasoning.
  - H4 (Policy learnability and transfer): Explicit supervision on decision rules installs robust cost-sensitive policies that generalize across decision domains, cost ratios, and prompt variations.

- [ ] **Restructure the related literature**
  - Shift technical LLM calibration and alignment literature to secondary background.
  - Lead with literature on human delegation to algorithms, learning to defer, cost-sensitive classification, algorithm aversion, and overreliance in operational workflows.
  - Cite the special issue foundational papers (Zhang et al., 2026 for Type C agent experiments; Liu et al., 2026 for agent-native artifacts).

- [ ] **Expand managerial and operational implications**
  - Explain how uncharacterized escalation thresholds create operational failure modes in automated pipelines (silent error propagation versus unnecessary labor bottlenecks).
  - Provide actionable guidelines for how operations managers should audit and tune agent escalation policies before production rollouts.
  - Discuss the practical implications of training agents on explicit cost arithmetic versus relying on prompt engineering.

- [ ] **Format for Decision Sciences**
  - Target the standard journal format (double-spaced, clear section hierarchy, under 40 pages total).
  - Keep the mathematical proof of the unique cost-optimal threshold in the main text or theory appendix.

---

## 2. Level 1 Research Configuration (Required)

We must create a standalone, machine-readable specification (`paper/ds/artifacts/level1_config.yaml`) that allows a future agent to replicate the experimental design without reading the manuscript text.

- [ ] **1. Study type and design logic**
  - Encode the experimental design as a randomized multi-condition agent experiment across five decision domains.
  - Define the unit of randomization and the unit of analysis.

- [ ] **2. Actors and information flows**
  - Specify the human principal and the LLM agent.
  - Detail the two-turn interaction protocol (Turn 1: prediction; Turn 2: act vs. escalate).
  - Define all information states, signals, and available actions.

- [ ] **3. Key parameters**
  - List all cost ratios ($R \in \{2, 4, 8, 10, 20, 50\}$), theoretical optimal thresholds ($\tau^* = 1 - 1/R$), prompt variants, and sample sizes per condition.
  - Define the outcome variables: predictive accuracy ($p$), escalation accuracy, and self-estimated accuracy ($\hat{a}$).

- [ ] **4. Data provenance (Type A + Type C)**
  - Type A: Document the public source datasets (HotelBookings, LendingClub, Wikipedia Toxicity, MovieLens, MoralMachine), preprocessing steps, and feature splits.
  - Type C: Specify model providers, snapshot versions, API settings, decoding parameters (greedy, temperature 0), token budgets, prompt templates, retry logic, and output parsers.

- [ ] **5. Analytical targets**
  - Map each formal hypothesis directly to its estimator, test statistic, and statistical validation criteria.
  - Define the linear regression procedure used to calculate the implicit threshold $p^*$.

- [ ] **6. Replication requirements**
  - State sample sizes per condition (250 standard, 100 thinking).
  - Define tolerance bands for replication comparability given stochastic model variation.

- [ ] **7. Scope conditions**
  - Explicitly document boundary conditions where the findings hold and conditions that would break the design.

---

## 3. Level 2 Reasoning Front-End (Recommended)

We should author a structured reasoning document (`paper/ds/artifacts/level2_reasoning.yaml` or `.md`) capturing the intellectual development of the research.

- [ ] **Motivations and research objectives**
  - Detail the initial observations regarding brittle automation and why standard accuracy metrics fail to capture delegation risks.

- [ ] **Theoretical positioning**
  - Construct a typed positioning map relating the work to existing literature (extends selective prediction, bounds prompt-based steerability, synthesizes learning to defer with foundation models).

- [ ] **Reasoning journey and decision history**
  - Document key design decisions, including the transition from single-turn prompts to the two-turn prediction/escalation protocol.
  - Record abandoned directions and pilot attempts (such as early adversarial steering explorations and direct preference optimization variants).
  - Explain why supervised fine-tuning on chain-of-thought traces was selected as the primary intervention mechanism.

- [ ] **Extension directions**
  - Provide structured next steps, such as extending the binary escalation action space to continuous delegation or multi-agent escalation chains.

---

## 4. Submission Package and Ancillary Documents

- [ ] **Draft the Artifact Note**
  - Write a 300 to 500 word summary describing the Level 1 and Level 2 artifacts, their structure, how future agents can execute them, and known limitations.

- [ ] **Prepare the Cover Letter**
  - Clearly mark the submission for Track A (Fast Track).
  - Include the required commitment confirming that all artifact files, code, and trained LoRA weights will be released under an open license upon acceptance.

- [ ] **Audit repository for open release**
  - Ensure all training scripts, evaluation harnesses, and dataset preparation code run cleanly with `uv`.
  - Package LoRA adapters and evaluation logs for public archival.
  - Verify that no API keys or confidential credentials exist in the codebase.
