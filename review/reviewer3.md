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

Thank you for taking the time to review our work! We have substantially expanded our results based on the review team's feedback: Claude and Gemma models for the main results, two additional models for the supervised fine-tuning interventions and an enhanced sample size for the thinking runs. We also wanted to clarify components of our original submission that we believe may address your reasons to reject:

- Escalation was defined in the first paragraph of Section 2.1, and depicted in Figure 1. In our setting, we consider escalation to be the process of the agent deciding to implement its prediction, or to escalate to a human.
- We absolutely agree that different settings require different cost structures! A self-driving car making a mistake is much more severe than a recommendation system highlighting the wrong movie. These differences were reflected in our manuscript. In Sections 3-4, we tested escalation behavior on different settings that spanned moral dilemmas in autonomous driving to movie recommendations, and found that LMs reasonably tend to escalate more in higher stakes settings. However, this is distinct from our main results, which emphasize that each model has a unique escalation profile, or implicit cost structure, that is often substantially different from other models in the same family. Finally, in Section 6.2, we highlighted how our supervised fine-tuning approach was built using a variety of different cost structures, ranging from an implementation mistake being twice as costly as escalation to fifty times as costly as escalation.
- We discussed, and demonstrated, potential mitigative solutions in Section 6. We found that prompting was not fully effective, and that supervised fine-tuning was better at controlling the escalation threshold across a number of models.
- We agree that there are complicated decision settings where humans may not be perfectly accurate, or even less accurate than the AI agent. However, we used data from settings where the human's preference, or decision, is the ground truth. For instance, in the MovieLens dataset, the 'ground truth' is which movie the human rated higher. In this matter, humans are perfectly accurate, as the ground truth is their subjective opinion.

We appreciate your thoughtful comments, and worked hard to incorporate your suggestions into our work. For example:

- Thank you for highlighting our lack of citations regarding the miscalibration literature. We agree that this was underdiscussed, and have added discussions pertaining to a number of these works (e.g., Kadavath et al., 2022; Xiong et al., 2024; Tian et al., 2023; Lin et al., 2022; Mielke et al., 2022; Guo et al., 2017). However, we have extended our analysis to new families of models (e.g., Claude and Gemma) thanks to other reviewer comments, and our confidence results have changed. Some models, like Gemma 3 12B, are extremely underconfident! We have shifted our discussion to reflect that, like escalation behavior, confidence varies substantially by model; interestingly, there does not seem a connection between how confident a model is and how much it escalates.
- We agree that our paper's positioning could be improved. We adjusted the framing to clarify how our paper can be distinguished from the broader literature. For example, in the introduction:

> Existing LLM evaluations measure what models can do; we instead characterize what they choose to do under uncertainty. This is a model-specific property, orthogonal to capability, that can leave a high-accuracy model either propagating errors at scale or eliminating the value of automation.

And the conclusion:

> While most LLM evaluation focuses on capability (accuracy, throughput, cost-savings), our results characterize an orthogonal dimension that has received far less attention: how an agent decides to act or defer under uncertainty. This dimension is latent, model-specific, and not predicted by architecture or scale, but as we show it can be characterized empirically and corrected through targeted training. Capability-focused benchmarks miss this failure mode entirely: a high-accuracy model can still systematically act when it should defer or defer when it should act, undermining the automation it was deployed for.

Please do not hesitate to reach out with any other questions or comments during the 'Follow-up Discussion' period.
