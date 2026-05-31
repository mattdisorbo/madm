# Reviewer 4

## Review

**Summary:**
The authors assess the ability of LLMs to appropriately "escalate" a task to a human when the LLM is uncertain and the cost of escalation to a human is sufficiently low. They also test some basic approaches to improving this escalation ability, such as SFT.

**Reasons To Accept:**
- The authors test escalation across several domains, which strengthens the contribution relative to a single domain.

**Reasons To Reject:**
- My central concern is about the novelty and significance of this contribution. What makes escalation importantly different from uncertainty estimation? Isn't escalation just a matter of identifying uncertainty and then setting some threshold for when there is enough uncertainty to escalate? What exactly do we gain from this line of thinking? I am unsure what new knowledge the authors have surfaced of sufficient novelty for publication.
- Some of the language seems rushed or imprecise, such as "the human workload it was meant to replace" when replacement is not necessarily the goal.
- I am unsure why the authors describe as "Theorems" relatively straightforward statements. Why is any mathematical discussion or proof needed here? For example, why do we need a proof that bias increases the cost? These seem less like mathematical discussions and more like the authors put in theorems and proofs just to make the manuscript seem more rigorous or something. The "proofs" in the appendix seem trivial and like a poor use of space.
- Figures 3 and 4 seem like they contain the core contributions, but what do we really learn from these? I don't think it is sufficient for a COLM contribution to just show how some current models score on these, but we need some broader insight about the nature of language modeling. It is unclear to me what is really meant to be novel and interesting in this submission.

**Rating:** 4: Ok but not good enough - rejection
**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct
**Ethics Flag:** No

## Response

Thank you for the direct review. Your central question was what escalation adds beyond uncertainty estimation. We think the answer is our strongest result, so we address it first.

**Novelty and significance.** Escalation is not reducible to uncertainty estimation with a threshold. Two models can share predictive accuracy and calibration yet escalate at very different rates [FILL: name the pair and the two escalation rates]. Uncertainty estimation cannot account for that gap, since the uncertainty is matched. The difference comes from each model's implicit cost structure, which is set during training and does not appear in any capability or calibration metric. This is what the paper contributes. The axis matters for deployment, current benchmarks do not measure it, and we show it can be both characterized and corrected. We reframed the introduction:

> Existing LLM evaluations measure what models can do; we instead characterize what they choose to do under uncertainty. This is a model-specific property, orthogonal to capability, that can leave a high-accuracy model either propagating errors at scale or eliminating the value of automation.

And the conclusion:

> While most LLM evaluation focuses on capability (accuracy, throughput, cost-savings), our results characterize an orthogonal dimension that has received far less attention: how an agent decides to act or defer under uncertainty. This dimension is latent, model-specific, and not predicted by architecture or scale, but as we show it can be characterized empirically and corrected through targeted training. Capability-focused benchmarks miss this failure mode entirely: a high-accuracy model can still systematically act when it should defer or defer when it should act, undermining the automation it was deployed for.

We also expanded the empirical base behind this claim, adding Claude and Gemma models to the main results, two more models to the fine-tuning experiments, and larger samples for the thinking runs.

**Theorems.** We agree that Theorem 2, that bias increases cost, is not necessary, and we have replaced it with a short reference. We kept Theorem 1 because the threshold $\tau^*$ anchors the cost-sensitive decision throughout the paper, and the later analysis refers back to it.

**Language precision.** We tightened the imprecise phrasings. "The human workload it was meant to replace" now reads "an agent that always escalates provides no labor savings over manual review." This states the actual point. Autonomous agents in these settings are not meant to replace human judgment but to reduce how often it is needed, and an agent that escalates everything saves nothing.

Please do not hesitate to reach out with any other questions or comments during the 'Follow-up Discussion' period.
