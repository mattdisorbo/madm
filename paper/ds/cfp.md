# Decision Sciences: Call for Papers — Special Issue

**Decisions in the Agentic Era: Toward Agent-Native Decision Science**

Source slides: [`cfp/01.png`](cfp/01.png)–[`cfp/15.png`](cfp/15.png). Pages 1–6 are the CFP. Pages 7–15 are the Track A Author Guide.

---

## Motivation

Artificial intelligence is undergoing a fundamental transition. Systems that once responded to instructions are increasingly acting as agents. They initiate multi-step workflows, use tools, coordinate with other agents, and operate with meaningful autonomy in organizational and market environments. This shift is happening faster than standard academic review cycles can track.

This special issue has two goals. The first is to advance rigorous understanding of how agentic AI reshapes decisions, markets, organizations, and society. The second is to develop research practices that make decision science more accessible to future AI agents. Such practices should help agents reconstruct, replicate, evaluate, and extend published research while preserving the standards expected by human scholars.

Scientific publication converts a branching and iterative research process into a linear narrative. That format often omits motivations, rejected alternatives, failed attempts, and implementation details. These omissions matter when AI agents are expected to work with published research. The two tracks of this special issue address the phenomenon of agentic AI and the research infrastructure needed to study it.

## Two tracks: overview

### Track A: Agent-Native Decision Science (preferred)

Track A is the preferred track. It invites original decision science research on the behavior, design, and dynamics of agentic AI. Each submission must include an agent-native research artifact, defined as a compact and machine-readable research configuration that a future agent can use to reconstruct, replicate, and extend the study. Track A submissions are eligible for priority processing through the fast track. They may also be submitted on a rolling basis.

Detailed requirements, templates, and examples appear in the companion Author Guide.

### Track B: Agentic AI as a Decision Science Phenomenon

Track B invites rigorous empirical or analytical research on the broader consequences of agentic AI for organizations, markets, labor, and society. Standard manuscript submission requirements apply. No agent-native artifact is required. Track B follows the rolling review process.

The editorial team may suggest redirection between tracks during review.

---

## Track A: Agent-Native Decision Science (preferred)

Agentic AI is both an object of study and a research instrument. Agents increasingly participate in decision systems, where they reason, act, learn, and coordinate with humans and other agents. They can also support research by implementing experiments, collecting data, running analyses, and exploring extensions.

Track A invites work that studies agentic systems and is structured for use by agentic systems. The goal is rigorous and cumulative scholarship on agent behavior and human-agent collaboration. The artifact requirement is intended to make the study’s design logic, data provenance, analytical targets, and scope conditions explicit.

### Scope and topics

Track A concentrates on two clusters:

1. Agent behavior and decision-making, with the agent as the unit of analysis.
2. Human-agent collaboration, with the human-agent pair or system as the unit of analysis.

Research on broader societal, market, or labor consequences belongs in Track B unless the central unit of analysis fits one of the two Track A clusters.

Experiments are the preferred method for Track A because they can support causal inference within a defined design. Track A also welcomes simulations, formal models, public-data studies, and observational studies of agent behavior when they meet the artifact and accessibility requirements.

All data used for Track A must be accessible to future agents. Eligible sources include public or archival data, observational agent data, and data generated through lab or agent experiments. Data that require private access, contain proprietary restrictions, or cannot be shared because of institutional approval requirements do not qualify for Track A.

#### Cluster 1: Agent behavior and decision-making

- Agent reasoning, planning, and decision strategies in task environments
- Multi-agent coordination, competition, and emergent collective behavior
- Agent capabilities, limitations, and failure modes in decision-relevant contexts
- Agent identity, role-taking, and consistency under manipulation
- Trust, deception, and norm violation in agent systems
- Agent learning and behavioral change over repeated interactions

#### Cluster 2: Human–agent collaboration

- Human reliance, override, and complementarity in agentic decision systems
- Oversight, accountability, and governance design in human–agent systems
- Human capacity in human–agent systems. As AI agents take on greater decision-making roles, understanding the capabilities humans bring to human–agent systems becomes a foundational question. Individuals differ in their capacity to understand what agents can and cannot do, to know when to rely on or override them, and to adapt as agents evolve. These are dimensions that collectively shape collaboration quality and system outcomes. Track A welcomes framework development, simulation-based investigation, and public-data studies that characterize, measure, or theorize human capacity in agentic decision contexts (e.g., humans’ AI Quotient, or AQ).
- Interface and system design conditions that shape human–agent collaboration outcomes

### What agent-native means

An agent-native research contribution is structured so that a future agent can do something useful with it without relying on implicit interpretation of manuscript prose. The agent should be able to reconstruct the study, assess its assumptions, replicate its core design, or generate a meaningful extension.

Agent-native describes a property, not a required file format. Track A requires a Level 1 research configuration and strongly recommends a Level 2 reasoning front-end.

### Level 1: Research configuration (required)

A Level 1 artifact is a compact and structured specification of the research design. It is not merely a code archive. Executable code may accompany the configuration, but code does not replace an explicit account of the study logic.

A Level 1 research configuration must specify:

- **Study type and design logic:** the experiment, field study, simulation, formal model, or other design, represented through labeled and structured fields.
- **Actors and information flows:** the relevant human and agent actors, their roles, available information, actions, tools, and interactions.
- **Key parameters:** treatment conditions, prompts, model settings, outcome variables, assumptions, and other values that an agent could read and vary.
- **Data provenance:** how data are obtained, accessed, generated, logged, and transformed.
- **Analytical targets:** the mapping from hypotheses or research questions to estimands, models, success criteria, and interpretation rules.
- **Replication requirements:** the number of runs, randomization logic, exclusion rules, uncertainty estimates, and criteria for judging whether results are comparable.
- **Scope conditions:** the settings in which the design should hold, the assumptions on which it depends, and the conditions that could invalidate it.

#### Experimental agent data: experiments for agents

For experiments with agents as subjects (Zhang et al. 2026), the configuration must also identify:

- model provider, model family, version or snapshot, and access date
- system prompts, task prompts, tools, retrieval sources, and permissions
- sampling parameters and supported randomization settings
- task and condition structure
- controls for order effects, prompt sensitivity, model drift, retries, and failed calls
- behavioral logs and rules for converting outputs into data
- expected stochastic variation and comparison tolerances
- software, package, and environment dependencies

The configuration should be specific enough for another agent to reconstruct the study and produce results that are comparable within a stated tolerance. Exact output duplication is not required when the system is stochastic.

### Level 2: Reasoning front-end (optional but strongly recommended)

A Level 2 artifact documents the intellectual architecture of the research in a structured and agent-readable form (Liu et al., 2026). A Level 2 artifact may include:

- **Motivations and research objectives:** what phenomenon or gap motivated this work, explicit enough that an agent can assess whether they apply in a new context.
- **Theoretical positioning:** where this work sits relative to existing knowledge, with typed relationships (extends, contradicts, bounds, synthesizes, replicates) rather than citation lists.
- **Reasoning journey:** important choices, rejected alternatives, failed attempts, and decision points that shaped the final contribution.
- **Extension directions:** what the structure of this contribution implies about where the field goes next, with explicit scope conditions.

No single format is required. Structured Markdown, JSON, YAML, a typed concept graph, or another machine-readable representation is acceptable.

### Submission package

Every Track A submission must include:

- a manuscript in standard Decision Sciences format, clearly marked as Track A
- a Level 1 research configuration as a structured file
- an Artifact Note of 300 to 500 words
- a cover-letter statement confirming that the final artifact files will be made publicly available under an appropriate open license upon acceptance

Track A submissions may also include:

- a Level 2 reasoning front-end
- executable code, notebooks, prompts, logs, schemas, or validation reports that support the research configuration

During review, all files must preserve author anonymity and must be submitted through a journal-approved channel. Accepted artifacts must be deposited in a stable public archive. Authors must not include API keys, credentials, private data, confidential material, or content that they do not have permission to share.

### Evaluation criteria for Track A

- **Research quality:** theoretical contribution, methodological rigor, novelty, empirical or analytical grounding, and relevance to decision science.
- **Artifact quality:** whether the artifact enables replication, reconstruction, validation, positioning, or extension beyond what the manuscript alone provides.
- **Paradigm contribution:** whether the work advances agent experiments as a research method, agent-native research as a knowledge standard, or understanding of human capacity in agentic decision systems.

A paper with strong research quality but an insufficient artifact may be redirected to Track B. Substituting an agent or large language model for a prior system is not enough by itself. The paper must explain how agentic behavior changes the decision architecture, mechanism, or outcome.

---

## Track B: Agentic AI as a Decision Science Phenomenon

### Scope

Track B invites rigorous empirical or analytical research on the consequences of agentic AI for organizations, markets, labor, and society. Agentic AI must be central to the contribution. For this special issue, agentic systems are systems that act, delegate, coordinate, and operate with meaningful autonomy across multi-step workflows. Any established decision science method is appropriate. Standard manuscript submission requirements apply, and no artifact is required.

**Out of scope:** work treating AI purely as a static recommendation tool or generic automation; pure performance benchmarks without decision science insight; CS-oriented systems papers without grounding in decision science theory or empirics; opinion pieces without empirical or analytical grounding.

### Topics of interest

Topics include, but are not limited to:

- societal consequences of agentic AI, including inequality, power, and human autonomy
- organizational transformation involving strategy, structure, and governance
- labor market change and workforce displacement
- market and competitive dynamics when agents act for firms or consumers
- accountability, transparency, and fairness in agentic decision pipelines
- regulation and governance of autonomous AI systems

### Evaluation criteria for Track B

Standard Decision Sciences criteria apply. The editorial team will also assess whether agentic AI is central to the contribution rather than incidental.

---

## Submission timeline

### Fast track — Track A priority processing

The fast track is available only to Track A submissions. It uses the same research and artifact standards as the rolling process, with a compressed review schedule.

| Milestone | Target date |
| --- | --- |
| CFP Launch & Editorial Piece Published | Mid August 2026 |
| Fast Track Submission Deadline (Track A only) | January 15, 2027 |
| First-Round Decisions | ~ 6 weeks after submission |
| Revisions Due | ~ 6 weeks after receiving the revision decision |
| Acceptance Notifications | January 2027 ~ May 2027 |

### Rolling submissions — both tracks

Track A authors may also submit on a rolling basis. Track B submissions follow the rolling process only.

| Element | Details |
| --- | --- |
| Submissions open | August 2026 (both tracks) |
| Submission Deadline (Track B) | March 15, 2027 |
| Review cycle | Standard Decision Sciences review process (~3 months per round) |
| Maximum review rounds | 2 rounds |
| Issue compilation | 2027, upon sufficient volume |

---

## Guest editors

- Yingjie Zhang (Peking University, yingjiezhang@gsm.pku.edu.cn)
- Shunyuan Zhang (Harvard University, szhang@hbs.edu)
- Tianshu Sun (Cheung Kong Graduate School of Business, tianshusun@ckgsb.edu.cn)
- Liangfei Qiu (University of Florida, liangfei.qiu@warrington.ufl.edu)
- Nagesh Murthy (University of Oregon, nmurthy@uoregon.edu)

## Guest associate editors

- Dominik Gutt (RWTH Aachen University, dominik.gutt@rwth-aachen.de)
- Miguel Godinho de Matos (School of Business and Economics, Católica-Lisbon, miguel.godinhomatos@gmail.com)
- Yicheng Song (University of Minnesota, ycsong@umn.edu)
- Yifan Yu (HongKong University, yifan.yu.research@gmail.com)
- Hyeokkoo Eric Kwon (Nanyang Technological University, Singapore, eric.kwon@ntu.edu.sg)
- Yang Gao (University of Illiois at Urbana-Champaign, ygao1@illinois.edu)
- Zhe Yuan (Zhejiang University, yyyuanzhe@gmail.com)
- Brian Han (University of Illiois at Urbana-Champaign, Brianhan@illinois.edu)
- Yan Leng (University of Texas at Austin, yan.leng@mccombs.utexas.edu)
- Wen Wang (University of Maryland, wenw@umd.edu)
- Liu Ming (The Chinese University of Hong Kong, Shenzhen, mingliu@cuhk.edu.cn)
- Zhenyu Zheng (Zhejiang University / Alibaba Group, zhenyuzeng@hotmail.com)
- Xilin Li (Cheung Kong Graduate School of Business, xilinli@ckgsb.edu.cn)

## References

Liu, J. et al. (2026). The Last Human-Written Paper: Agent-Native Research Artifacts. arXiv:2604.24658.

Zhang, Y., Feng, C., Zhu, W., and Sun, T. (2026). Agents for Experiments, Experiments for Agents: A Design Grammar for AI-Enabled Experimental Science. arXiv:2605.17746v1.

---

# Track A Author Guide: Agent-Native Research

Companion document to the Call for Papers.

## Purpose of this guide

This guide explains the requirements for Track A submissions, provides templates, and illustrates what an agent-native research artifact should contain. The Call for Papers defines the formal requirements. Authors targeting Track A should read this before preparing their submission.

Track A is an experiment in research design and communication. The editorial team does not expect a single artifact format to fit every contribution. It does expect authors to make the structure of their research explicit enough for a future agent to reconstruct, assess, replicate, or extend a meaningful part of the work.

## Track A eligibility

A Track A submission should meet ALL of the following conditions:

- The central topic is agent behavior, agent decision-making, or human-agent collaboration.
- Agentic AI is central to the research question, design, mechanism, or outcome.
- The submission includes a Level 1 research configuration.
- The data, prompts, logs, or other inputs needed for reconstruction are publicly accessible, generated within the study, or otherwise shareable.
- The artifact enables a concrete action beyond reading the manuscript, such as reconstruction, replication, validation, comparison, or extension.
- The accepted artifact will be released publicly under an appropriate license.

Data that require private access, contain proprietary restrictions, or cannot be shared because of institutional approval requirements are not eligible for Track A. Such work may still fit Track B.

## Why agent-native research matters

Agent-native research makes the logical structure of a contribution accessible to both human and machine readers. It does not ask authors to replace the manuscript with code or to force every study into one technical format. It asks authors to identify the information that a future agent would need to act on the research.

An agent-native artifact should reduce the need for implicit interpretation. It should identify the study design, actors, parameters, data sources, analytical targets, assumptions, and scope conditions through labeled and structured fields.

The intellectual responsibility remains with the authors. Machine readability does not reduce the standards for theory, methods, evidence, citation, or interpretation.

## Level 1: Research configuration (required)

### What it is

A Level 1 artifact is a structured specification of the research design. It should contain enough information for a future agent to reconstruct and replicate the study without relying on unstated assumptions in the manuscript.

A Level 1 artifact is not merely a code archive. Code may be included and is often useful, but code does not by itself explain why the design was chosen, what parameters can vary, how evidence maps to claims, or when the results should hold. The research configuration is the stable specification. Implementations may vary across languages, platforms, models, and future technical environments.

### Seven components

A Level 1 artifact specifies seven components. Weight them according to the nature of your contribution.

#### 1. Study type and design logic

The overall research design should be encoded as a typed, structured object, rather than described only in prose, so that an agent can instantiate it. For experimental work, specify the design structure, such as a 2x2 between-subjects design, along with the conditions and treatment logic. For field studies, specify the observational unit, data collection approach, and identification strategy. For formal models, specify the model class, key players, and solution concept. The goal is a specification an agent can recognize, categorize, and act on.

#### 2. Actors and information flows

Identify the relevant human and agent actors, their roles, the information available to each actor, the actions and tools they can use, and how they interact. For studies with multiple agents or human-agent teams, make the sequence and direction of information flows explicit. An agent reading the configuration should be able to determine who knows what, when they know it, and what actions are available at each stage.

#### 3. Key parameters

Treatment conditions, actor definitions, prompts, model settings, outcome variables, and model assumptions should be stated as explicit named values rather than embedded in prose. Parameters should be named, typed, and, where relevant, given ranges or valid values. An agent reading a well-parameterized configuration can vary conditions, substitute contexts, and generate study variants without reverse-engineering the design from a methods section.

#### 4. Data provenance

Specify how data are obtained, accessed, generated, logged, and transformed precisely enough that another agent can acquire or generate equivalent data without human intermediation.

Track A recognizes three data provenance types:

**Type A: Public or archival data.** Specify the dataset name or persistent identifier, access method such as a URL, API endpoint, or package name, any filtering or preprocessing logic, and the version or snapshot date where relevant. The standard is that another agent given this specification can retrieve an equivalent dataset and reproduce the analysis.

**Type B: Observational agent data.** For data collected from naturally occurring agent behavior, such as platform logs, API traces, or digital behavioral records, specify the platform or system, access method, behavioral unit of observation, time window, and filtering criteria. The standard is that another agent can collect a comparable observational dataset from the same source.

**Type C: Experimental agent data — agents for experiments, experiments for agents.** For many Track A topics, including agent behavior, multi-agent dynamics, negotiation, trust, deception, and human-agent collaboration studied from the agent side, the appropriate data source is an experiment designed with agents as subjects (Zhang et al., 2026). This is the most distinctive data provenance type for this special issue. It embodies the dual principle directly: the experiment is used to understand agents, and it is structured so that agents can implement and extend it.

A well-specified Type C configuration must be rigorous enough that another agent can design and run the experiment independently and obtain results comparable to the original within a stated tolerance.

An experimental agent data configuration specifies:

- **Agent subject specification:** Model provider, model family, version or snapshot, access date, and relevant parameters, including context settings and system prompt structure. Name these elements explicitly so that results can be attributed to a reproducible setup and compared across model versions.
- **Task and condition structure:** The experimental scenario, treatment conditions, system and task prompts, tools, retrieval sources, permissions, and the way each condition is instantiated. Specify these elements precisely enough that another agent can reconstruct the task and prompt logic without reading the manuscript.
- **Sampling and randomization:** Sampling parameters and supported randomization settings, including seeds when available, along with the randomization or batching logic used to assign conditions and generate observations.
- **Controls for agent-specific confounds:** Controls for order effects, prompt sensitivity, model drift, retries, failed calls, and known behavioral quirks of the model. State the control approach explicitly. For example, explain what a small change in prompt wording might do to the results and how this sensitivity is assessed.
- **Replication requirements:** number of runs per condition needed for stable estimates, criteria for excluding anomalous outputs, randomization or batching logic, and any pilot results that informed these choices.
- **Behavioral logging and transformation:** The outputs recorded, the rules used to convert outputs into data, the operationalization of agent behavior, and the approach used to assess response validity, including checks for instruction-following failures.
- **Stochastic variation and comparison tolerances:** The expected variation across runs, the uncertainty measures used, and the tolerance or decision rule for judging whether a replication is comparable to the original.
- **Software and environment dependencies:** The software, packages, APIs, and execution environment needed to implement the experiment, including relevant version information.

The standard for Type C is that a future agent, reading only the configuration, should be able to design the experiment, implement it rigorously, collect the data, and produce results comparable to the original within a stated tolerance, without reading the manuscript or contacting the authors. Exact output duplication is not required when the system is stochastic.

#### 5. Analytical targets

Specify the mapping from research questions or hypotheses to estimands, models, test statistics, proof strategies, success criteria, and interpretation rules. State which hypothesis maps to which estimator, what the success criterion is, such as directional significance at “p < .05” or specified parameter bounds, and what boundary conditions apply to the results. An agent with this information can generate the analysis code and interpret the output without reading the manuscript.

#### 6. Replication requirements

Specify the number of runs or observations, randomization logic, exclusion rules, uncertainty estimates, and criteria for judging whether results are comparable. For stochastic systems, state the expected variation and an appropriate comparison tolerance. For other research designs, identify the corresponding replication conditions, such as dataset version, preprocessing decisions, model assumptions, solver settings, or validation checks.

#### 7. Scope conditions

Specify the conditions under which the design should hold, what would break it, and which assumptions are load-bearing. Scope conditions are the artifact's self-description of its limits. They allow an agent to reason about whether the design is valid in a new context before attempting to replicate or extend it.

### Format guidance

No format is mandated. Any machine-readable format with explicit structure is acceptable, including structured Markdown with labeled sections, JSON or YAML with typed fields, or any equivalent representation. The test is simple: can an agent read this file and reconstruct the study without reading the manuscript prose?

A minimal Level 1 artifact that specifies these seven components clearly is more valuable than an elaborate artifact that buries them in narrative. Conciseness is a feature.

### Worked example (partial)

Notes: This example is illustrative. Structure and completeness should fit your contribution.

```yaml
study_type_and_design_logic:
  study_type: randomized two-condition between-dyad experiment
  treatment: information_completeness [complete, incomplete]
  unit_of_randomization: negotiation dyad
  unit_of_analysis:
    H1: negotiation dyad
    H2: dyad-round
  sample: 100 dyads per condition
  assignment: randomized, with initial proposer counterbalanced

actors_and_information_flows:
  actors: [buyer_agent, seller_agent]
  roles:
    buyer_agent: seeks the lowest acceptable price
    seller_agent: seeks the highest acceptable price
  information:
    complete: both agents observe both reservation prices
    incomplete: each agent observes only its own reservation price
  protocol: alternating offers, maximum 10 rounds
  actions: [make_offer, accept, reject, terminate]
  tools: none
  retrieval_sources: none
  permissions: agents receive only condition-permitted information

key_parameters:
  buyer_reservation_price: 200
  seller_reservation_price: 50
  valid_offer_range: [0, 250]
  payoffs: buyer = 200 - price; seller = price - 50; no agreement = 0
  outcomes:
    agreement: acceptance of the most recent valid offer
    rounds_to_agreement: agreement round, censored after round 10
    final_price: accepted offer, reported descriptively for agreement dyads

data_provenance:
  provider: OpenAI
  model_family: GPT-4o
  model_version: gpt-4o-2024-11-20
  access_date: YYYY-MM-DD
  prompts: prompts/
  response_schema: schemas/negotiation_response.json
  sampling: {temperature: 0.7, top_p: 1.0, max_tokens: 300}
  randomization_seed: 20260801
  controls:
    order_effects: initial proposer counterbalanced
    prompt_sensitivity: 20 percent rerun with paraphrased prompts
    model_drift: model snapshot fixed and access dates recorded
    retries: technical failures only, maximum two identical retries
    failed_calls: logged separately from behavioral failures
  logging: prompts, responses, offers, actions, model metadata, and failures
  transformation: explicit rules convert logged actions into outcomes
  dependencies: Python, API client, analysis packages, and version records

analytical_targets:
  H1:
    claim: incomplete information lowers agreement probability
    estimand: difference in agreement probability between conditions
    model: logistic regression
    success_criterion: "negative effect with p < .05"
  H2:
    claim: incomplete information delays agreement
    estimand: difference in round-specific agreement hazard
    model: discrete-time hazard model
    success_criterion: "hazard ratio below 1 with p < .05"

replication_requirements:
  completed_sample: 100 dyads per condition
  exclusions: unrecoverable technical failures only
  behavioral_failures: retained and coded as no agreement
  expected_variation: transcripts and offers may differ across runs
  comparison_tolerance:
    H1_effect: absolute difference from original estimate no greater than 0.05
    H2_log_hazard_ratio: absolute difference no greater than 0.20
  exact_output_duplication_required: false

scope_conditions:
  holds_when:
    - agents follow assigned roles
    - hidden information is not leaked
    - each dyad begins with a fresh context
    - tools, retrieval, and cross-dyad memory remain disabled
  breaks_when:
    - model version or prompts change without documentation
    - agents access counterpart information in the incomplete condition
    - outputs cannot be mapped reliably to the response schema
```

## Level 2: Reasoning front-end (optional but strongly recommended)

### What it is

A Level 2 artifact documents how the research came to be: the motivations, objectives, theoretical positioning, and reasoning journey that shaped the final contribution, presented in a structured, agent-readable form. This recovers what narrative publication systematically discards (Liu et al., 2026): the intellectual history that explains why the contribution took the form it did and what the structure of that contribution implies for where the field goes next.

The key property of a Level 2 artifact is extensibility. It is not enough to state what the research found. The artifact should make the reasoning explicit enough that a future agent or researcher can identify where the logic branches, what assumptions could be relaxed, and what the natural next steps are. A well-constructed Level 2 artifact makes the contribution forward-compatible. Future work can build from it instead of reconstructing the reasoning from scratch.

### Four components

A Level 2 artifact addresses four components. Weight them according to the nature of your contribution.

#### 1. Motivations and research objectives

State what phenomenon or gap drove this work and what the research set out to accomplish. This is not a restatement of your abstract. It is the underlying observations and questions that motivated the inquiry, made explicit enough that an agent can assess whether the same motivation applies in a new context and whether your objectives were met.

A useful structure is to state the observation, the gap it reveals, and the research objective. Keep it concrete. An agent reading “we study agent negotiation behavior” learns little. An agent reading “existing work shows agents cooperate in repeated games, but evidence is sparse for one-shot high-stakes settings; we ask whether strategic deception emerges under incomplete information” can reason about scope and extension.

#### 2. Theoretical positioning

State where the work sits relative to existing knowledge, using typed relationships rather than citation lists. Useful relationship types include:

- **extends** — builds on prior work by adding a condition, mechanism, or scope
- **contradicts** — challenges a prior finding or assumption, with the specific point of departure
- **bounds** — identifies the conditions under which a prior finding holds or fails
- **synthesizes** — integrates two or more prior streams into a unified framework
- **replicates** — tests a prior finding in a new context or population

An agent reading a typed positioning map can locate your contribution in the literature and identify what it leaves open. It can also detect mismatches, including prior work your contribution should engage but does not.

#### 3. Reasoning journey

Explain how the research arrived at its final form, including the intellectual choices made along the way, the alternatives considered, and what was tried and did not work. This is the component that narrative publication most systematically erases and the one with the highest value for future agents and researchers.

Document the main decision points: why this theoretical framing rather than an adjacent one, why this operationalization, and why this research context. For failed approaches, including theoretical framings that did not survive early development, experimental designs abandoned after piloting, or operationalizations that produced confounds, record what was tried, why it failed, and what the failure revealed.

The reasoning journey does not need to be comprehensive. Even two or three documented decision points or dead ends have asymmetric value. An agent that knows a particular framing was tried and abandoned for specific reasons does not need to rediscover that independently.

Note: documenting failures and pivots is normal intellectual history, not an exposure of weakness. The most valuable research often involves the most exploration. Documenting what did not work is a contribution to the field, not a liability.

#### 4. Extension directions

Explain what the structure of this contribution implies about where the field goes next, with explicit scope conditions. For each extension direction, state what it would investigate, what assumption or boundary condition of the current work it would relax, and what would need to be true for it to be a valid next step. A typed list is more useful than a prose future-directions paragraph.

### Format guidance

No format is mandated. The key requirement is that structure is explicit: relationships are labeled, reasoning is stated, and logic is navigable. Authors may use structured Markdown with typed sections, JSON or YAML with typed node and edge structures, a concept graph in any notation, or any other machine-readable format. The test is simple: can an agent follow the reasoning and identify extension points without reading your prose?

## Artifact Note template

The Artifact Note is required for all Track A submissions. It should be 300 to 500 words and does not count toward the manuscript word limit. It enables reviewers to evaluate the artifact efficiently and is itself parseable as metadata.

| Field | What to include |
| --- | --- |
| Artifact name and type | What is the artifact, what format, and which level(s) does it address? |
| Research component represented | Which components are covered: study design, actors and information flow, parameters, data provenance (and type), analytical targets, replication requirements, scope conditions (Level 1); motivations, theoretical positioning, reasoning journey, extension directions (Level 2)? |
| What an agent can do with it | Be concrete: replicate the study from the configuration, design and run the agent experiment, vary a design parameter, follow the reasoning to a new extension. |
| Limitations | What does the current artifact not yet support? Honesty here is valued. |
| Access | File format, location or persistent URL, dependencies. |

## How Track A artifacts are evaluated

The Call for Papers evaluates Track A submissions on three dimensions: research quality, artifact quality, and paradigm contribution. This section explains the artifact quality dimension.

Artifact evaluation uses a single criterion: does this artifact enable something, such as replicating the study, designing and running the agent experiment, reasoning through an extension, or identifying the scope of a finding, that the manuscript alone does not?

Reviewers will ask:

- Can an agent replicate the core study from the Level 1 configuration alone, without reading the manuscript? For Type C data: can an agent design and implement the experiment rigorously from the configuration alone?
- For submissions including Level 2: can an agent follow the reasoning and identify extension points without reading the manuscript prose?
- Does the artifact add genuine value beyond what a well-written paper already provides?

A decorative artifact, meaning one that is formatted as machine-readable but contains only information already in prose form, does not meet the bar. A Level 1 artifact must address all seven required components, but it may do so concisely. Its value depends on whether it enables a concrete action beyond reading the manuscript.

Artifact quality is distinct from research quality and is evaluated blind to process, including the use of AI assistance in artifact construction. What matters is realized value. A paper with strong research quality but an insufficient artifact may be redirected to Track B.

Note to reviewers: Track A is a first attempt to establish community standards for agent-native social science research. Evaluate artifacts with intellectual generosity. The question is whether the artifact represents a genuine attempt to make the research replicable and extensible by future agents, not whether it meets an ideal standard that does not yet exist.

## Frequently asked questions

**Do I need both Level 1 and Level 2?**
Level 1, the research configuration, is required for all Track A submissions. Level 2, the reasoning front-end, is optional but strongly recommended. The strongest Track A submissions will include both. Theoretically oriented contributions in which reasoning and positioning are the primary value should prioritize Level 2 alongside the required Level 1.

**Which data provenance type applies to my study?**
Type A, public or archival data, applies to studies using existing datasets. Type B, observational agent data, applies to studies collecting naturally occurring agent behavior from platforms or APIs. Type C, experimental agent data, applies to studies in which authors design and run experiments with agents as subjects. Type C is likely to be common for Track A topics involving agent behavior, negotiation, multi-agent dynamics, and human-agent collaboration studied from the agent side. Some studies may combine types; specify each in the configuration. All three types qualify for both topic clusters, provided the data are agent-accessible and the study meets the artifact requirements.

**How precise does the experimental agent data specification need to be?**
It should be precise enough that another agent or researcher could design and implement the experiment and obtain results comparable to yours within a stated tolerance, without reading the manuscript or contacting you. Specify the model provider, model family, version or snapshot, access date, prompts, tools, retrieval sources, permissions, sampling parameters, confound controls, retry and failure rules, logging and transformation rules, expected stochastic variation, comparison tolerances, and software and environment dependencies. Exact output duplication is not required for a stochastic system. If you are unsure whether your specification meets this standard, ask: could a capable agent run this study tomorrow from my configuration alone?

**Is there a required format?**
No. Any machine-readable format with explicit structure is acceptable for either level. What matters is that components are labeled, logic is stated, and the artifact is navigable, not that a particular file format is used.

**Can I use AI to construct the artifacts?**
Yes, and we encourage it. Using an AI assistant to help structure your research configuration, specify your experimental design, map your theoretical positioning, or document your reasoning journey is exactly the kind of human-AI collaboration this special issue is designed to support. Evaluation focuses on whether the artifact is useful, not how it was produced.

**How detailed does the reasoning journey need to be?**
Partial is fine. Document the decision points and dead ends you genuinely remember and can articulate. Two or three well-described pivots or failed framings are more useful than a long sanitized account. The goal is honest intellectual history, not comprehensive documentation.

**What if my contribution does not fit these components neatly?**
Describe what you have in the Artifact Note and explain what an agent can do with it. We are genuinely interested in learning what forms agent-native research takes across contribution types. An honest attempt that does not fit the template may be more instructive than a tidy artifact that fits but adds nothing.

**My study is about human capacity in human–agent systems, not agent behavior directly. Does it qualify for Track A?**
Yes, provided the data are agent-accessible and the study is designed so agents can implement it. Track A's second cluster, human-agent collaboration, explicitly includes research on individual capacity to understand, work with, and adapt to AI agents, including constructs such as AI Quotient, or AQ. The key constraint is method. Track A welcomes framework development, simulation-based investigation, and public-data studies on human capacity. Studies that require proprietary or institutionally restricted human-subject data belong in Track B. If your research on human capacity uses agent-simulated behavior, public behavioral traces, or theoretical frameworks grounded in agent-accessible evidence, it is in scope for Track A.

**Will my artifact go through peer review?**
Yes. Artifact quality is evaluated alongside research quality and paradigm contribution. Reviewers will not expect perfection. They will assess whether the artifact represents a genuine attempt to make the research replicable and extensible by future agents and researchers.

## Contact

Authors with questions about track eligibility, artifact format, or any aspect of the Track A requirements are encouraged to contact the editorial team before submission.

## References

Liu, J. et al. (2026). The Last Human-Written Paper: Agent-Native Research Artifacts. arXiv:2604.24658.

Zhang, Y., Feng, C., Zhu, W., and Sun, T. (2026). Agents for Experiments, Experiments for Agents: A Design Grammar for AI-Enabled Experimental Science. Working Paper.
