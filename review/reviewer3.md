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

Thank you for the close reading. We address each concern below and have revised the manuscript accordingly. We also expanded the empirical results at the review team's request, adding Claude and Gemma models to the main analysis, two more models to the fine-tuning experiments, and larger samples for the thinking runs.

**Definition of escalation.** We define escalation in the first paragraph of Section 2.1 and depict it in Figure 1. The agent makes a prediction, then chooses to implement that prediction or defer to a human. We have clarified the wording so it is clear we study this act-or-defer decision, not the broader space of tool calls or permission requests.

**Cost commensurability.** You are right that human labor and prediction error are not commensurable in general. A self-driving collision and a wrong movie recommendation differ by orders of magnitude, and some harms resist any dollar value. We do not claim a universal exchange rate. We restrict the quantitative analysis to settings where the human decision is itself the ground truth, so the cost of a wrong prediction is well defined. In MovieLens the ground truth is which movie a person rated higher, so the human is correct by construction. Section 2.2 now states this assumption explicitly and notes that the accident-style cases you raise fall outside it.

**Human-in-the-loop degradation.** Thank you for the Pseudo-Automation reference. We agree that as automation improves, human overseers lose context and intervene less effectively, so treating the human as a perfect fallback overstates the value of escalation. We have added this point to the discussion and softened the claim that escalation recovers full human accuracy. Our results do not require the human to be perfect. They require only that the human is the labeling authority in our datasets, which holds for the preference and subjective-judgment tasks we use.

**Mitigation.** Section 6 demonstrates mitigation rather than only discussing it. Prompting alone did not reliably move the escalation threshold, but supervised fine-tuning controlled it across several models. We now emphasize this result. Section 6.2 also shows the fine-tuning works across cost structures, from an error being twice as costly as escalation to fifty times as costly.

**Miscalibration literature.** You are right that Section 5 needs more citations. We have added Kadavath et al. (2022), Xiong et al. (2024), Tian et al. (2023), Lin et al. (2022), Mielke et al. (2022), and Guo et al. (2017). Our calibration results also changed with the larger model set. Gemma 3 12B is strongly underconfident [FILL: by how much], so we no longer claim models are uniformly overconfident. Confidence varies by model, and it does not predict how much a model escalates.

**Positioning and novelty.** We clarified how our contribution differs from uncertainty estimation and calibration. The distinction is empirical. Two models can share predictive accuracy and calibration yet escalate at very different rates [FILL: name the pair and the two escalation rates], which uncertainty estimation alone cannot explain. We revised the introduction to read:

> Existing LLM evaluations measure what models can do; we instead characterize what they choose to do under uncertainty. This is a model-specific property, orthogonal to capability, that can leave a high-accuracy model either propagating errors at scale or eliminating the value of automation.

And the conclusion:

> While most LLM evaluation focuses on capability (accuracy, throughput, cost-savings), our results characterize an orthogonal dimension that has received far less attention: how an agent decides to act or defer under uncertainty. This dimension is latent, model-specific, and not predicted by architecture or scale, but as we show it can be characterized empirically and corrected through targeted training. Capability-focused benchmarks miss this failure mode entirely: a high-accuracy model can still systematically act when it should defer or defer when it should act, undermining the automation it was deployed for.

Please let us know if you have any more questions before the end of the discussion period.
