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

Thank you for taking the time to read our paper and to write this thoughtful review! We have substantially expanded our results based on the review team's feedback: Claude and Gemma models for the main results, two additional models for the supervised fine-tuning interventions and an enhanced sample size for the thinking runs. We also worked hard to incorporate your suggestions specifically:

- We tried to better frame the novelty and significance of our paper. We agree that how models "score" on these tasks is not sufficient for a conference like COLM; instead, we tried to establish the broader insight that models have hidden escalation patterns that vary significantly, cannot be predicted and are non-trivial to control. For example, we added this to the introduction:

> Existing LLM evaluations measure what models can do; we instead characterize what they choose to do under uncertainty. This is a model-specific property, orthogonal to capability, that can leave a high-accuracy model either propagating errors at scale or eliminating the value of automation.

And this to the conclusion:

> While most LLM evaluation focuses on capability (accuracy, throughput, cost-savings), our results characterize an orthogonal dimension that has received far less attention: how an agent decides to act or defer under uncertainty. This dimension is latent, model-specific, and not predicted by architecture or scale, but as we show it can be characterized empirically and corrected through targeted training. Capability-focused benchmarks miss this failure mode entirely: a high-accuracy model can still systematically act when it should defer or defer when it should act, undermining the automation it was deployed for.

- We agree that Theorem 2 ("bias increases cost") is relatively unnecessary, and we have replaced it with a short reference. We decided to keep Theorem 1 because the threshold $\tau^*$ is at the centerpiece of the ensuing discussions.
- We have added precision to multiple concepts and definitions (e.g., escalation accuracy vs. predictive accuracy). More specifically, we have replaced "the human workload it was meant to replace" as "an agent that always escalates provides no labor savings over manual review". This gets more to the crux of our argument: we agree that autonomous agents in these settings are not meant to replace humans, but to supplement them.

Please do not hesitate to reach out with any other questions or comments during the 'Follow-up Discussion' period.
