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

Thank you for taking the time to read our paper, and for such useful suggestions. We believe your insight has really helped to strengthen our work. We made many updates directly based on your comments. First, we expanded our results:

- At your suggestion, we successfully tested the supervised fine-tuning (SFT) approach on two new Qwen models, and achieved high levels of escalation accuracy. The paper now includes results from Qwen2.5-7B-Instruct, Qwen3.5-4B and Qwen3.5-9B.
- Training data was obtained from held-out instances in the datasets we used. Decision trees helped to construct each signal: for instance, in the held-out instances, 91% of FICO > 700 loans were approved.
- Our main manuscript now includes model pairs that are more strongly related: specifically, Qwen3.5-4B / Qwen3.5-9B and Gemma-3-4B / Gemma-3-12B. We agree that our initial submission that compared Llama 3 to Llama 4, and Mistral to Mixtral, did not achieve the clean comparison we would hope for.

We also made substantial changes to the manuscript for clarity and improved framing:

- In Section 4, we provide the LM with a signal that greatly simplifies the prediction (e.g., "there is a 91% chance of a positive outcome"), essentially rendering the LM's turn 1 prediction uninteresting, since prediction is much simpler with an explicit signal! The purpose of this section is to analyze LM escalation behavior in turn 2, and our main result is that different models exhibit vastly different escalation behavior. We merely used the signal in turn 1 to control LM predictive quality across models and isolate escalation behavior. Finally, we do test models without providing the signal in Section 5, which allow us to elicit how good each model believes its predictions are. We find that, like escalation behavior, confidence calibration (overconfident vs. underconfident vs. just right) varies across models.
- Your point on accuracy is very well taken, and was reflected by other reviewers! We consider three primary accuracies. Prediction accuracy measures how often the model's turn 1 prediction is correct, and self-estimated predictive accuracy measures how often the model implicitly believes its turn 1 prediction is correct, based on how often it escalates. The most important metric is escalation accuracy: how often the model makes the correct escalation decision based on its predictive accuracy and the pertinent cost threshold. We have updated the paper to formally define the different accuracy metrics, and to be more specific throughout about the wording used.
- The 'Dollar' and 'Wording' columns test our fine-tuned models with prompts that have the same meaning but are worded differently (e.g., Dollar is "Escalation costs $1. A wrong implementation costs $4"). This allows us to conclude that our fine-tuned models are not just learning patterns from the original wording, but understanding the underlying reason for escalating or implementing.
- We have renamed "cost ratio 4" as "cost framing" throughout for clarity, and made clear in Section 3.3 that these variants will not appear until Section 6.
- We have added qualitative examples of how different models exhibit very different behavior. For example, given a moderate signal, Opus implements while Sonnet escalates.
- We have added an extensive section in the Appendix covering implementation details. For example, the full LoRA and SFT hyperparameters (e.g., r = 64, etc.).

Please do not hesitate to reach out with any other questions or comments during the 'Follow-up Discussion' period.
