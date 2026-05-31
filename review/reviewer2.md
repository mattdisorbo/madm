# Reviewer 2

## Review

**Summary:**
This paper studies the problem of "effective automation" concerned with when an LLM should decide to act (solve the problem "on its own") or escalate (delegate the problem to a human). This is achieved by studying classification problems in two stages: first, the model is given a problem and a signal that includes some external predictive accuracy and makes a prediction; and second, the model then decides based on its own prediction to either act or escalate. The authors find that the escalation behavior is model dependent and that LLMs are miscalibrated in their self-assessment. Finally, the authors show that prompt-based and finetuning approaches help the models make better escalation decisions.

The paper addresses a relevant and well-motivated problem, and the cost sensitive decision framework is interesting. The clarity of the work is sometimes missing, for example due to the terminology used, making the narrative difficult to follow at times. The significance of the findings could be stronger with a more complete experimental validation of the SFT experiment, helping support the broader claims made in the abstract / introduction.

**Reasons To Accept:**
- The studied problem is very relevant and the cost-sensitive decision setup is very interesting.
- The authors test different LLMs and scenarios, and have made their code available.

**Reasons To Reject:**
- Although the problem setup is interesting and the goal of each experimental section is sensible, the writing was confusing and difficult to follow at times (see Questions and Suggestions for some examples).
- The proposed solutions in Section 6 are applied only to one model. After showing in Sections 4 and 5 that the different models behave very differently (for example in how overconfident/underconfident they are, or how they respond to the predictive accuracy part of the signal), it would be important to test these mitigations in more models to corroborate the claim that "these dynamics, however, can be corrected". Of course, keeping in mind the respective limitations especially for the SFT experiment.

**Questions:**
- How was the training data obtained? In particular, the signal and the corresponding accuracy part of the string "A decision tree …".
- It was unclear to me the role of the prediction after turn 1 in Section 4. My current understanding is that it does not play a role, and that the analysis in this section is solely based on the "predictive accuracy" part of the input signal and the "escalation rate" after turn 2.
- In Section 5, what does the "actual accuracy" correspond to? Is it the accuracy from the signal that in Section 4 was defined as "predictive accuracy" or the accuracy of the turn 1 prediction vs the gold prediction?
- Wouldn't it make sense for Section 6 to include results where the model is not given the signal? I suppose that would be the most realistic scenario, in which there is no auxiliary model. The model is tested without signal, but I think training without signal would also make sense. Right now it seems like the SFT model is learning a rule like "if R * (1 - accuracy in signal) > 1: escalate". In general, I do not understand the need of turn 1 when you are given the prediction of an external model as part of the input. Could you also clarify this?
- What are the "Dollar" and "Wording" columns in Appendix Table 4?

**Suggestions / Typos:**
I encourage improving the clarity of the paper. For example:
- Introducing "Cost ratio 4" and "Thinking" in Section 3.3 but only using them in Section 6 did not help with clarity. Also, it seems like "cost ratio 4" is then used as "cost framing" which was confusing.
- There are many accuracies being used: "predictive accuracy", "actual accuracy", "self-estimate accuracy", "decision accuracy", just "accuracy". At some point it's hard to keep track of what we are actually looking at.
- I think the claims related to scaling are not substantiated. Some models might be of the same "family" but in practice they correspond to very different models. E.g. Llama 3 vs Llama 4, Mistral vs Mixtral.
- The work is missing more qualitative examples showing the model behavior.
- Even though code is shared, it would be good to have hyperparameters for generation, etc. detailed in the Appendix.

**Rating:** 4: Ok but not good enough - rejection
**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct
**Ethics Flag:** No

## Response

Thank you for the detailed and constructive review. Your central concern was that the Section 6 mitigation ran on a single model, and we have fixed that directly.

**Fine-tuning across multiple models.** We now test supervised fine-tuning on three models, Qwen2.5-7B-Instruct, Qwen3.5-4B, and Qwen3.5-9B, and reach high escalation accuracy in each [FILL: per-model escalation accuracy before and after]. The claim that escalation dynamics can be corrected no longer rests on one model.

**Training data.** We built each signal from held-out instances using a decision tree. In the held-out loan data, 91% of applicants with FICO above 700 were approved, which becomes the stated predictive accuracy for that signal.

**Model family pairing.** We replaced the Llama 3 vs. Llama 4 and Mistral vs. Mixtral comparisons, which you correctly noted span very different models, with Qwen3.5-4B vs. Qwen3.5-9B and Gemma-3-4B vs. Gemma-3-12B. These hold architecture and training fixed and vary scale.

**Role of turn 1 and the no-signal setting.** Your reading is right. The Section 4 signal makes the turn 1 prediction easy, for instance "there is a 91% chance of a positive outcome," so turn 1 mainly equalizes predictive quality across models and lets us isolate the turn 2 escalation decision. We do test the no-signal setting in Section 5, where the model must judge its own predictions, and we recover each model's self-estimated accuracy that way. We have clarified the role of each turn so this is explicit.

**Accuracy terminology.** Three reviewers flagged this, so we now define the metrics formally. Prediction accuracy is how often the turn 1 prediction is correct. Self-estimated predictive accuracy is how correct the model implicitly believes it is, read off its escalation rate. Escalation accuracy, the metric we care about most, is how often the act-or-defer decision is correct given the predictive accuracy and the cost threshold. We use these terms consistently throughout now.

**Dollar and Wording columns.** These test the fine-tuned models on prompts that carry the same meaning in different words. Dollar phrases the cost as "Escalation costs \$1. A wrong implementation costs \$4." When behavior matches across phrasings, the model has learned the underlying trade-off rather than the surface wording.

**Cost framing.** We renamed "cost ratio 4" to "cost framing" throughout, and Section 3.3 now flags that these variants first appear in Section 6.

**Qualitative examples.** We added examples of divergent behavior. On the same moderate signal, Opus implements while Sonnet escalates.

**Hyperparameters.** The appendix now reports generation and training details, including the full LoRA and fine-tuning setup (r = 64).

Please do not hesitate to reach out with any other questions or comments during the 'Follow-up Discussion' period.
