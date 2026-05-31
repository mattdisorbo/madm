# Reviewer 1

## Review

**Summary:**
The work explores how models trade off between acting on a problem when uncertain about its predictions on a problem versus escalating to a human supervisor. To establish how models perform on this metric they test 8 models on 4 different tasks that involve varying degrees of uncertainty and challenge such as estimating whether a hotel booking might get cancelled or whether a piece of wikipedia text is toxic.

In each task they provide an estimate of the model's likelihood of being wrong phrased as "for cases like this, predictions are usually 91% accurate". They then ask the model whether they'd like to go with their prediction or escalate to a human.

They find that in general models seem to have very specific tendencies regarding whether or not they escalate, and also that surprisingly more capable models often escalate more frequently than less capable models. Significant differences also exist within model families, pointing at the idea that the capability seems quite independent of typical signals one might use to predict this.

Finally, they fine-tune qwen 3 8B on similar versions of the problem, and find that doing this substantially improves the performance of that model on the task on a held out dataset, indicating that the problem is amenable to straightforward solutions.

**Review:**
The work is useful in exposing the strange discrepancy in the predictability of model escalation behavior, and showing that models are easily improved on this dimension is a valuable contribution.

One criticism I have is that the choice of models makes it hard to cleanly answer the extent to which capabilities are a proxy for this metric, because the selection consists primarily of open-source models which are unlikely to be calibrated on this task, and excludes any frontier models that would give more teeth to the claim.

The claim would be much stronger if Claude Haiku 4.5, Sonnet 4.6, and Opus 4.7 or GPT-5.5 had been included, as this would give a much clearer capability gradient. Given that frontier labs may have already addressed this issue with their shift to agentic workflows with reporting hierarchies, the current selection genuinely misses an opportunity to provide valuable information that is directly accessible. It is somewhat frustrating that the capability-to-escalation relationship cannot be cleanly answered from this crop of models.

The number of samples is also quite low for each model, such that the confidence bounds are genuinely somewhat shaky for the claims. Even 100 samples for thinking-mode runs would have substantially improved the confidence on the claims.

**Reasons To Accept:**
- The work explores an interesting problem and provides a clear methodology that provides a good way to measure what they claim to measure.
- The methodology is sound in principle.
- The presentation of the work is extremely clear and understandable, as well as being quite thorough and complete.

**Reasons To Reject:**
- I think the choice of models is a weakness of the paper, and misses an opportunity to provide valuable information on the problem they have identified. The models selected are relatively unlikely to be ones that would typically be deployed for the types of problems under examination, so we are not in a vastly better position to comment on the status of current frontier models on this problem.
- I am also concerned that Table 1 reports point estimates only, with no variance or confidence intervals. Given the small sample sizes for the thinking conditions (50 per condition), the headline effects could plausibly disappear under proper statistical analysis. Given figures 3 and 4 elsewhere in the paper do have error bars, this makes the omission from Table 1 specifically a notable oversight. Given that the paper's core empirical claims rest on these results it calls the statistical merits of the results into question.

**Questions:**
- The work would be substantially improved by running the evaluation on some of the frontier models mentioned earlier in my review (Claude Haiku 4.5, Sonnet 4.6, Opus 4.7, GPT-5.5). Would you be able to add at least two or three of these during the rebuttal period?
- Qwen 3 9B is reported as having a 17 percentage point improvement over baseline. Given that only 50 samples are given this means a 7 point standard error. Could you add confidence intervals to table 1?

**Rating:** 5: Marginally below acceptance threshold
**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct
**Ethics Flag:** No

## Response

Thank you for taking the time to review our paper! We enjoyed reading your comments (especially the term "strange discrepancy" between model escalation behavior, which we think captures our main thrust) and believe your input helped to improve the work. We made the following changes in response to your suggestions:

- We increased the thinking-mode run sample sizes to $n = 100$, and added confidence intervals to Table 1. These demonstrate how the Qwen 9B improvement over baseline is strongly significant: a confidence interval of $(58.3, 59.7)$ for baseline vs. $(79.4, 81.5)$ for thinking with cost framing (the improvement moved from 17% to about 20% with the larger sample sizes).
- We added two frontier models: Claude Sonnet 4.6 and Claude Opus 4.7. The results were consistent with the broader theme that escalation behavior differs across models: Opus 4.7 escalated less than Sonnet 4.6 (59% vs. 66% threshold). However, we did find that confidence calibration was much better with the Claude models, which reframed our discussion: instead of reporting that models are generally overconfident, we show how models have different levels of confidence, and how these levels still do not predict escalation behavior. We also added Gemma-3-4B and Gemma-3-12B in response to another reviewer's comments; these models were more prone to escalation (76% vs. 98%) and were much more underconfident. We decided not to add Haiku 4.5 because Sonnet and Opus are generally considered more at the frontier, and GPT-5.5 for similar reasons and an additional cost constraint.
- Finally, although not necessarily pertinent to your review, we fine-tuned two additional Qwen models and nearly matched the strong performance from our original submission. This demonstrates the robustness of SFT to align escalation behavior.

Please do not hesitate to reach out with any other questions or comments during the 'Follow-up Discussion' period.
