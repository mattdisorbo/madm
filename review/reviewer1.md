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

Thank you for the careful review, and for the phrase "strange discrepancy," which captures our central finding. We address both requests directly.

**Confidence intervals (Table 1).** We raised the thinking-mode sample size to $n = 100$ and added Wilson confidence intervals to Table 1. The Qwen 9B effect remains significant. Baseline sits at $(58.3, 59.7)$ and thinking with cost framing at $(79.4, 81.5)$, and the improvement rose from 17% to about 20% with the larger samples. These intervals confirm that the effect is not an artifact of the small samples.

**Frontier models.** We added Claude Sonnet 4.6 and Claude Opus 4.7. Sonnet and Opus give a controlled capability comparison, since they share a developer and the same Claude 4 generation, so the contrast reflects capability rather than differences in training pipeline. The more capable Opus escalates less than Sonnet, 59% vs. 66% threshold, and capability still does not predict escalation across the eight models, where the implicit threshold ranges from 53% (GPT-5-mini) to 98% (Gemma 3 12B). We show this in a new figure of per-model thresholds. We also added Gemma-3-4B and Gemma-3-12B at another reviewer's suggestion. They escalate at 76% vs. 98% and are much more underconfident. The frontier additions also revised one secondary claim. Confidence is not uniformly high, so we now report that calibration varies by model and still does not predict escalation, rather than that models are broadly overconfident.

**Why Sonnet and Opus rather than Haiku or GPT-5.5.** We focused on the Sonnet-Opus pair because they share a developer and generation, which gives the clean capability contrast. Haiku 4.5 sits below both, and GPT-5.5 adds a cross-lab confound, so neither sharpens the gradient the way this pair does.

**Additional fine-tuned models.** We fine-tuned two more Qwen models and nearly matched the original result, reaching 95.7% escalation accuracy for Qwen3.5-4B and 87.3% for Qwen3.5-9B against near 100% for the original Qwen2.5-7B-Instruct, which shows the fine-tuning aligns escalation behavior across models, not only the model from our original submission.

Please let us know if you have any more questions before the end of the discussion period.
