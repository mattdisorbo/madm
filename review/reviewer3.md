# Reviewer 3

## Review

**Summary:**
The paper investigates when models across different domains choose to escalate. They test different interventions and see how prompting vs SFT impacts different models. The results suggest that escalation is a model-specific property that should be considered prior to deployment.

**Reasons To Accept:**
- Authors introduce the problem as two-pronged: on one hand, models are miscalibrated, on the other hand, models have different thresholds for escalation
- The insight that models have different thresholds for escalation is interesting

**Reasons To Reject:**
- Missing definition on what you mean by escalation — this can look like lots of different forms: getting a human in the loop, asking permission to run something else, retrieving information by calling a tool.
- Unquantifiable costs: Line 76: I'm not convinced by this cost structure that you can directly compare the cost of human labor with the cost of incorrect answers. These are going to be widely different costs based on what task you are operating — and they are not necessarily quantifiable. Imagine an automatic car that could make a wrong mistake to cause an accident. How do you quantify the value of an accident?
- 2.2 Cost Structure: This is assuming that humans are perfect that taking the over in the automation, which has been shown in the STS literature as not true. As automation gets more advanced, it's harder for humans "in-the-loop" to jump in and fix a mistake (see Pseudo-Automation: How Labor-Offsetting Technologies Reconfigure Roles and Relationships in Frontline Retail Work)
- Results from Section 5 are known in the miscalibration and overconfidence literature, and a lot more work needs to be cited.
- Authors could do a better job of positioning their work and explaining why it's distinct from pre-existing literature
- Authors could also discuss at least potential mitigation solutions

**Rating:** 3: Clear rejection
**Confidence:** 4: The reviewer is confident but not absolutely certain that the evaluation is correct
**Ethics Flag:** Yes

## Response

Thank you for the careful review and for raising scope, ethics, and positioning concerns.

We revised the paper to make the scope precise, strengthen the related-work positioning, and foreground the mitigation result. We also expanded the empirical results by adding Claude and Gemma models to the main analysis, two more models to the fine-tuning experiments, and larger samples for the thinking runs.

**Definition of escalation.** We define escalation in the first paragraph of Section 2.1 and depict it in Figure 1. The agent makes a prediction, then chooses to implement that prediction or defer to a human. We have clarified the wording so it is clear we study this act-or-defer decision, not the broader space of tool calls or permission requests.

**Cost commensurability.** The paper does not require a universal exchange rate between human labor and prediction error. The quantitative analysis is restricted to settings where the human decision is itself the ground truth, so the cost of a wrong prediction is well defined. In MovieLens, for example, the ground truth is which movie a person rated higher, so the human is correct by construction. Section 2.2 now states this assumption explicitly and notes that accident-style cases fall outside the scope of the experiments.

**Human-in-the-loop degradation.** Treating the human as a perfect fallback can overstate the value of escalation when overseers lose context. We added this limitation to the discussion and revised claims that could be read as saying escalation recovers full human accuracy. The empirical results do not require perfect human fallback. They require only that the human is the labeling authority in our datasets, which holds for the preference and subjective-judgment tasks we use.

**Ethics flag.** Our experiments use public, de-identified data of recorded human decisions, with no human subjects and no new data collection. We added a brief ethics statement to the paper that says this and notes we use the sensitive domains, loan approval and content moderation, only to study deferral rather than to endorse automating them.

**Mitigation.** Section 6 demonstrates mitigation rather than only discussing it. Prompting alone did not reliably move the escalation threshold, but supervised fine-tuning controlled it across several models. We now emphasize this result. The fine-tuning results also evaluate six cost ratios, from an error being twice as costly as escalation to fifty times as costly.

**Miscalibration literature.** Section 5 now cites the relevant calibration and uncertainty-estimation literature, including Kadavath et al. (2022), Xiong et al. (2024), Tian et al. (2023), Lin et al. (2022), Mielke et al. (2022), and Guo et al. (2017). Our calibration results also changed with the larger model set. Gemma 3 12B is underconfident by 7 points (69% self-estimate against 76% actual), so we no longer claim models are uniformly overconfident. Confidence varies by model, and it does not predict how much a model escalates.

**Positioning and novelty.** We clarified that our contribution is not a new theory of uncertainty estimation or abstention. Instead, we adapt those ideas to LLM agents and empirically characterize a behavior that standard LLM evaluations usually leave unmeasured. Calibration and uncertainty estimation ask whether a model knows how likely it is to be correct. Selective prediction and learning-to-defer study when a predictive system should abstain or route an instance to another decision-maker. Our setting combines these concerns in an LLM agent that first predicts, then decides whether to act on its own prediction under an explicit cost ratio, without access to a calibrated class probability. The LLM-agent context is important because this policy is elicited through language, varies across prompts and models, and is part of the deployed agent behavior rather than a separate threshold that can be inspected or tuned directly. This matters empirically. Gemma 3 4B and Gemma 3 12B both predict at 75 to 76% accuracy and are both underconfident by about 7 points, yet they escalate at 76% vs. 98%. Two models can therefore match on predictive accuracy and calibration while following very different act-or-defer policies. The related work now states:

> In LLM agents, deferral can be analyzed as a threshold decision, but the deployed policy is usually not exposed as a separate classifier threshold that a practitioner can inspect or tune directly. It is elicited through language and can vary across prompts and models.

We also revised the introduction to read:

> Existing LLM evaluations measure what models can do; we characterize what they choose to do under uncertainty. This act-or-defer policy is not captured by accuracy alone, and it can leave a high-accuracy model either propagating errors at scale or eliminating the value of automation.

And the conclusion:

> While most LLM evaluation focuses on capability (accuracy, throughput, cost-savings), our results characterize a related deployment behavior that accuracy alone does not measure: how an agent decides to act or defer under uncertainty. This behavior is latent, model-specific, and not predicted by architecture or scale, but as we show it can be characterized empirically and corrected through targeted training. Capability-focused benchmarks miss this failure mode: a high-accuracy model can still systematically act when it should defer or defer when it should act, undermining the automation it was deployed for.

These revisions are included in the revised manuscript.
