<div id="content" role="main">

<div class="Forum_forum__wS8Fw">

<div class="forum-container">

<div class="forum-note">

<div style="height: auto; overflow: visible;">

<div class="btn-group">

2 Versions <span class="caret"></span>

- [NeurIPS 2025 poster (September 18,
  2025)](https://openreview.net/forum?id=xp7B8rkh7L)
- [Submitted to ICLR 2025 (September 27,
  2024)](https://openreview.net/forum?id=PGNdDfsI6C)

</div>

</div>

<div class="forum-title mt-2 mb-2">

## LoRA vs Full Fine-tuning: An Illusion of Equivalence

<div class="forum-content-link">

<a href="https://openreview.net/pdf?id=xp7B8rkh7L"
class="citation_pdf_url" target="_blank" rel="noreferrer"
title="Download PDF"><img
src="./LoRA%20vs%20Full%20Fine-tuning_%20An%20Illusion%20of%20Equivalence%20_%20OpenReview_files/pdf_icon_blue.svg"
alt="Download PDF" /></a>

</div>

</div>

<div class="forum-authors mb-2">

### <a href="https://openreview.net/profile?id=~Reece_S_Shuttleworth1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Reece_S_Shuttleworth1">Reece S Shuttleworth</a>, <a href="https://openreview.net/profile?id=~Jacob_Andreas1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Jacob_Andreas1">Jacob Andreas</a>, <a href="https://openreview.net/profile?id=~Antonio_Torralba1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Antonio_Torralba1">Antonio Torralba</a>, <a href="https://openreview.net/profile?id=~Pratyusha_Sharma1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Pratyusha_Sharma1">Pratyusha Sharma</a> 

</div>

<div class="clearfix mb-1">

<div class="forum-meta">

<span class="date item"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>Published: 18 Sept 2025, Last Modified: 21 Apr
2026</span><span class="item"><span class="glyphicon glyphicon-folder-open"
aria-hidden="true"></span>NeurIPS 2025
poster</span><span class="readers item" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone&lt;br/&gt;since 29 Oct 2025"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="item"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=xp7B8rkh7L)</span><span class="item"><span class="glyphicon glyphicon-bookmark"
aria-hidden="true"></span><a href="https://openreview.net/forum?id=xp7B8rkh7L#"
data-target="#bibtex-modal" data-toggle="modal"
data-bibtex="%40inproceedings%7B%0Ashuttleworth2026lora%2C%0Atitle%3D%7BLo%7BRA%7D%20vs%20Full%20Fine-tuning%3A%20An%20Illusion%20of%20Equivalence%7D%2C%0Aauthor%3D%7BReece%20S%20Shuttleworth%20and%20Jacob%20Andreas%20and%20Antonio%20Torralba%20and%20Pratyusha%20Sharma%7D%2C%0Abooktitle%3D%7BThe%20Thirty-ninth%20Annual%20Conference%20on%20Neural%20Information%20Processing%20Systems%7D%2C%0Ayear%3D%7B2026%7D%2C%0Aurl%3D%7Bhttps%3A%2F%2Fopenreview.net%2Fforum%3Fid%3Dxp7B8rkh7L%7D%0A%7D">BibTeX</a></span><span class="item"><span class="glyphicon glyphicon-copyright-mark"
aria-hidden="true"></span><a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"
target="_blank" rel="noopener noreferrer" data-toggle="tooltip"
data-placement="top"
data-original-title="Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International">CC
BY-NC-SA 4.0</a></span>

</div>

<div class="invitation-buttons">

</div>

</div>

<div class="note-content">

<div>

**Keywords:** <span class="note-content-value">Parameter Efficient
Fine-tuning (PEFT), Low Rank Adaptation (LoRA), LLMs,
Transformers</span>

</div>

<div>

**Abstract:**

<div class="note-content-value markdown-rendered">

Fine-tuning is a crucial paradigm for adapting pre-trained large
language models to downstream tasks. Recently, methods like Low-Rank
Adaptation (LoRA) have been shown to effectively fine-tune LLMs with an
extreme reduction in trainable parameters. But, \emph{are their learned
solutions really equivalent?} We study how LoRA and full-finetuning
change pre-trained models by analyzing the model's weight matrices
through the lens of their spectral properties. We find that LoRA and
full fine-tuning yield weight matrices whose singular value
decompositions exhibit very different structure: weight matrices trained
with LoRA have new, high-ranking singular vectors, which we call
\emph{intruder dimensions}, while those trained with full fine-tuning do
not. Further, we extend the finding that LoRA forgets less than full
fine-tuning and find its forgetting is vastly localized to the intruder
dimension -- by causally intervening on the intruder dimensions by
changing their associated singular values post-fine-tuning, we show that
they cause forgetting. Moreover, scaling them down significantly
improves modeling of the pre-training distribution with a minimal drop
in downstream task performance. Given this, we should expect
accumulating intruder dimensions to be harmful and lead to more
forgetting. This will be amplified during continual learning because of
sequentially fine-tuning, and we show that LoRA models do accumulate
intruder dimensions here tend to perform worse in this setting,
emphasizing the practicality of our findings.

</div>

</div>

<div>

**Primary Area:** <span class="note-content-value">Deep learning (e.g.,
architectures, generative models, optimization for deep networks,
foundation models, LLMs)</span>

</div>

<div>

**Submission Number:** <span class="note-content-value">21986</span>

</div>

</div>

</div>

<div class="filters-container mt-4">

<div class="wrap">

<div class="form-group expand">

<div class="replies-filter invitations-filter css-b62m3t-container">

<span id="react-select-invitations-filter-live-region"
class="css-7pg0cj-a11yText"></span><span class="css-7pg0cj-a11yText"
aria-live="polite" aria-atomic="false" aria-relevant="additions text"
role="log"></span>

<div class="dropdown-select__control css-1tqhi6y-control">

<div class="dropdown-select__value-container dropdown-select__value-container--is-multi css-1uzcsaf">

<div id="react-select-invitations-filter-placeholder"
class="dropdown-select__placeholder css-1ynal7p-placeholder">

Filter by reply type...

</div>

<div class="dropdown-select__input-container css-1ab7ooq" data-value="">

</div>

</div>

<div class="dropdown-select__indicators css-1wy0on6">

<span class="dropdown-select__indicator-separator css-qgckm3-indicatorSeparator"></span>

<div class="dropdown-select__indicator dropdown-select__dropdown-indicator css-1qajzci-indicatorContainer"
aria-hidden="true">

<img
src="data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjIwIiB3aWR0aD0iMjAiIHZpZXdib3g9IjAgMCAyMCAyMCIgYXJpYS1oaWRkZW49InRydWUiIGZvY3VzYWJsZT0iZmFsc2UiIGNsYXNzPSJjc3MtOG1ta2NnIj48cGF0aCBkPSJNNC41MTYgNy41NDhjMC40MzYtMC40NDYgMS4wNDMtMC40ODEgMS41NzYgMGwzLjkwOCAzLjc0NyAzLjkwOC0zLjc0N2MwLjUzMy0wLjQ4MSAxLjE0MS0wLjQ0NiAxLjU3NCAwIDAuNDM2IDAuNDQ1IDAuNDA4IDEuMTk3IDAgMS42MTUtMC40MDYgMC40MTgtNC42OTUgNC41MDItNC42OTUgNC41MDItMC4yMTcgMC4yMjMtMC41MDIgMC4zMzUtMC43ODcgMC4zMzVzLTAuNTctMC4xMTItMC43ODktMC4zMzVjMCAwLTQuMjg3LTQuMDg0LTQuNjk1LTQuNTAycy0wLjQzNi0xLjE3IDAtMS42MTV6IiAvPjwvc3ZnPg=="
class="css-8mmkcg" />

</div>

</div>

</div>

<div>

</div>

</div>

</div>

<div class="form-group expand">

<div class="replies-filter css-b62m3t-container">

<span id="react-select-signatures-filter-live-region"
class="css-7pg0cj-a11yText"></span><span class="css-7pg0cj-a11yText"
aria-live="polite" aria-atomic="false" aria-relevant="additions text"
role="log"></span>

<div class="dropdown-select__control css-1tqhi6y-control">

<div class="dropdown-select__value-container dropdown-select__value-container--is-multi css-1uzcsaf">

<div id="react-select-signatures-filter-placeholder"
class="dropdown-select__placeholder css-1ynal7p-placeholder">

Filter by author...

</div>

<div class="dropdown-select__input-container css-1ab7ooq" data-value="">

</div>

</div>

<div class="dropdown-select__indicators css-1wy0on6">

<span class="dropdown-select__indicator-separator css-qgckm3-indicatorSeparator"></span>

<div class="dropdown-select__indicator dropdown-select__dropdown-indicator css-1qajzci-indicatorContainer"
aria-hidden="true">

<img
src="data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjIwIiB3aWR0aD0iMjAiIHZpZXdib3g9IjAgMCAyMCAyMCIgYXJpYS1oaWRkZW49InRydWUiIGZvY3VzYWJsZT0iZmFsc2UiIGNsYXNzPSJjc3MtOG1ta2NnIj48cGF0aCBkPSJNNC41MTYgNy41NDhjMC40MzYtMC40NDYgMS4wNDMtMC40ODEgMS41NzYgMGwzLjkwOCAzLjc0NyAzLjkwOC0zLjc0N2MwLjUzMy0wLjQ4MSAxLjE0MS0wLjQ0NiAxLjU3NCAwIDAuNDM2IDAuNDQ1IDAuNDA4IDEuMTk3IDAgMS42MTUtMC40MDYgMC40MTgtNC42OTUgNC41MDItNC42OTUgNC41MDItMC4yMTcgMC4yMjMtMC41MDIgMC4zMzUtMC43ODcgMC4zMzVzLTAuNTctMC4xMTItMC43ODktMC4zMzVjMCAwLTQuMjg3LTQuMDg0LTQuNjk1LTQuNTAycy0wLjQzNi0xLjE3IDAtMS42MTV6IiAvPjwvc3ZnPg=="
class="css-8mmkcg" />

</div>

</div>

</div>

<div>

</div>

</div>

</div>

<div class="form-group expand">

</div>

<div class="form-group no-expand">

Sort: Newest FirstSort: Oldest First

</div>

<div class="form-group no-expand layout-buttons">

<div class="btn-group btn-group-sm" role="group"
aria-label="nesting level">

<img
src="./LoRA%20vs%20Full%20Fine-tuning_%20An%20Illusion%20of%20Equivalence%20_%20OpenReview_files/linear_icon.svg"
title="Linear discussion layout" class="icon" data-toggle="tooltip"
alt="back arrow" /><span class="sr-only">Linear</span>

<img
src="./LoRA%20vs%20Full%20Fine-tuning_%20An%20Illusion%20of%20Equivalence%20_%20OpenReview_files/threaded_icon.svg"
title="Threaded discussion layout" class="icon" data-toggle="tooltip"
alt="back arrow" /><span class="sr-only">Threaded</span>

<img
src="./LoRA%20vs%20Full%20Fine-tuning_%20An%20Illusion%20of%20Equivalence%20_%20OpenReview_files/nested_icon.svg"
title="Nested discussion layout" class="icon" data-toggle="tooltip"
alt="back arrow" /><span class="sr-only">Nested</span>

</div>

<div class="btn-group btn-group-sm" role="group"
aria-label="collapse level">

<span toggle="tooltip"
title="Collapse content">−</span><span class="sr-only">Collapsed</span>

<span toggle="tooltip"
title="Partially expand content">＝</span><span class="sr-only">Default</span>

<span toggle="tooltip"
title="Fully expand content">≡</span><span class="sr-only">Expanded</span>

</div>

<div class="btn-group btn-group-sm" role="group" aria-label="copy url">

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="Copy filter URL"
aria-hidden="true"></span><span class="sr-only">Copy link</span>

</div>

</div>

</div>

<div>

<span class="control-label icon-label"><span class="glyphicon glyphicon-eye-open"
toggle="tooltip" placement="top" title="Visible to"
aria-hidden="true"></span></span>

<div class="form-group readers-filter-container">

<div class="btn-group btn-group-sm toggle-group readers-filter"
role="group">

Everyone

<span class="glyphicon glyphicon-remove" toggle="tooltip"
placement="top" title="Reset"
aria-hidden="true"></span><span class="sr-only">Reset</span>

</div>

</div>

<div class="form-group filtered-reply-count">

*25 / 25 replies shown*

</div>

</div>

</div>

<div class="invitations-container">

<div class="invitation-buttons top-level-invitations">

<span class="hint">Add:</span>

Public Comment

</div>

</div>

<div class="row forum-replies-container layout-default">

<div class="col-xs-12">

<div id="forum-replies">

<div class="note depth-odd" data-id="NoNuQbh2sB">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### **Paper Decision**

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note NoNuQbh2sB"></span><span class="sr-only">Copy
URL of note NoNuQbh2sB</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 255, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Decision</span><span class="signatures">by
Program Chairs</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>17 Sept 2025, 08:51 (modified: 29 Oct 2025,
02:15)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=NoNuQbh2sB)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Decision:** <span class="note-content-value">Accept (poster)</span>

</div>

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

This paper introduces the concept of intruder dimensions in LoRA
fine-tuning, showing their strong correlation with forgetting and
demonstrating, via causal interventions, that mitigating their effect
can recover pre-trained knowledge without sacrificing downstream
performance. Reviewers agreed the paper is novel, clearly written, and
supported by extensive experiments across models, datasets, and
hyperparameters.

While minor issues were raised (e.g., phrasing of claims, reliance on
cosine similarity as a distance measure), these do not undermine the
core contributions. The rebuttal effectively addressed concerns about
hyperparameter bias, causal importance of intruders, and whether LoRA
differences are simply basis rotations. Overall, the paper provides
valuable insights into the spectral properties of LoRA and practical
strategies for mitigating forgetting.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-odd" data-id="t2bUOZttS0">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Author Final Remarks by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note t2bUOZttS0"></span><span class="sr-only">Copy
URL of note t2bUOZttS0</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(204, 255, 136); color: rgb(44, 58, 74);"
original-title="Reply type">Author Final
Remarks</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>15 Aug 2025, 10:59 (modified: 29 Oct 2025,
02:08)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=t2bUOZttS0)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Author Final Remarks:**

<div class="note-content-value markdown-rendered">

We thank the reviewers for their thoughtful reviews. They emphasized
that our work is well written (GjPE, JRuL, 9SVt), novel (BfKg, GjPE),
and experimental setup is “strong...includes a wide range of additional
experiments”(GjPE) and “well designed...based on multiple rigorous
experiments”(9SVt). We recap reviews below.

9SVt had a positive reaction and left a score of 5.

BfKg had concerns that we addressed and raised their score to 4.

JRuL did not raise further concerns after our rebuttal. They state our
paper has “extensive empirical studies” and “logical and straightforward
presentation.” They state they will talk with others before deciding
their updated score.

GjPE state our work “addresses an important problem” and the “paper is
thorough”. Our rebuttal partially answered their concerns after several
rounds of responses. We summarize GjPE’s existing concerns:

- They are concerned that at higher learning rates, full FT may also
  have intruder dimensions. However, we show that this is not the case:
  we use default lr used in previous works on LoRA. We further conduct
  lr sweeps(Fig. 7) for both LoRA and full FT and find that full FT has
  no intruders for all lr's. Increasing lr further leads to divergence
  (even nearly random performance). In contrast, LoRA always has
  intruders. This show our results are not the result of biased
  hyperparams.

- They are concerned that intruders are “not particularly special in
  terms of forgetting compared to other dimensions.” However, we find
  the opposite. We show in causal experiments that intruders account for
  large amounts of the forgetting of the model(Fig. 8). In one example,
  we show that scaling a particular models intruders with λ = 0.3 leads
  to a 0.1% drop in accuracy and a 33.3% drop in forgetting. This is
  unique to intruders(Fig. 12) and is a highly novel post-training
  intervention.

- They are concerned that LoRA might simply “rotate” the basis. We point
  to Fig. 1 as evidence that this is not the case: if these matrices
  were being rotated, we would not expect to see a 1-1 mapping between
  singular vectors in the original and fine-tuned weights (as shown by
  the clear diagonal). Instead, for both LoRA and full FT, almost every
  singular vector of the original matrix has a matching vector
  (determined by high cosine similarity) in the tuned weights. LoRA's
  diagonal is offset because they are displaced by intruders. This shows
  that the differences between full FT and LoRA are primarily reflected
  by the intruders.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-odd" data-id="xn8X6GtKqa">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### **Please carefully read the rebuttal and start discussion**

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note xn8X6GtKqa"></span><span class="sr-only">Copy
URL of note xn8X6GtKqa</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Area Chair
Xh3c</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>03 Aug 2025, 21:10 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=xn8X6GtKqa)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Dear Reviewers and Authors,

Thank you all for your efforts so far. As the author–reviewer discussion
period will conclude on **August 6**, please start the discussion as
soon as possible.

**For Reviewers:** If you have not done so, please read the authors’
responses and, if necessary, continue the discussion with them.

- If your concerns have been addressed, consider updating your review
  and score accordingly.

- If some concerns remain, or if you share concerns raised by other
  reviewers, clearly state these in your review and consider adjusting
  your review (positively or negatively).

- If you feel that your concerns have not been addressed, you may also
  choose to keep your review as is.

- I will follow up with you again during the reviewer–AC discussion
  period (August 7–13) to finalize the reviews and scores.

**For Authors:** If you have not already done so, please respond to all
questions raised by the reviewers. Keep your responses factual, concise,
and ensure that every point raised is addressed.

Best regards,

The AC

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-odd" data-id="BoSimKJd9F">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission21986 by Reviewer BfKg

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note BoSimKJd9F"></span><span class="sr-only">Copy
URL of note BoSimKJd9F</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
BfKg</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>03 Jul 2025, 05:14 (modified: 28 Oct 2025,
23:20)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=BoSimKJd9F)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

This paper studies the difference between full fine-tuning and LoRA
through spectral analysis of weight matrices. The authors identify
intruder dimensions—high-ranking singular vectors in tuned weights that
are dissimilar to any pre-trained directions—and show they are a key
cause of forgetting. Scaling down these components reduces forgetting
without harming downstream too much performance. The accumulation of
intruder dimensions is also shown to hurt LoRA in continual learning.

</div>

</div>

<div>

**Strengths And Weaknesses:**

<div class="note-content-value markdown-rendered">

Strengths:

- The authors provide a novel perspective on the difference between LoRA
  and full fine-tuning. The concept of intruder dimensions reveals a
  concrete structural difference between full-finetuning and LoRA.

- The link between intruder dimensions and forgetting is insightful. It
  offers both theoretical understanding and potential avenues for
  mitigation.

Weaknesses

- Logical mismatch in Section 5. Section 4 convincingly correlates the
  *number* of intruder dimensions with forgetting, yet Section 5 only
  experiments with scaling down the *magnitude* of the top intruder
  dimension. To solidify the argument, the authors should also
  manipulate the *count* of intruder dimensions and measure its effect
  on forgetting.

- Limited practical guidance. The paper does not provide guidance on how
  to prevent the emergence of intruder dimensions during training, which
  is crucial for improving real-world applicability. As it stands,
  intruder dimensions appear to be primarily useful as a post-hoc model
  selection criterion. However, in practice, comparing two models based
  on their performance on pre-training data may be more straightforward.

</div>

</div>

<div>

**Quality:** <span class="note-content-value">3: good</span>

</div>

<div>

**Clarity:** <span class="note-content-value">2: fair</span>

</div>

<div>

**Significance:** <span class="note-content-value">2: fair</span>

</div>

<div>

**Originality:** <span class="note-content-value">3: good</span>

</div>

<div>

**Questions:**

<div class="note-content-value markdown-rendered">

- The paper concludes that intruder dimensions cause forgetting.
  However, full-tuning with few intruder dimensions, still exhibits more
  forgetting than LoRA with an appropriately chose $`\alpha`$. How do we
  understand this based on findings in the paper?

- In the continual learning setup, are the similarity matrices in Figure
  9 all computed with the original model? I am wondering can we observe
  instruder dimensions over a model already tuned on another task(say do
  weights after task 3 have instruder dimension over weights after task
  2)?

</div>

</div>

<div>

**Limitations:**

<div class="note-content-value markdown-rendered">

Yes.

</div>

</div>

<div>

**Rating:** <span class="note-content-value">4: Borderline accept:
Technically solid paper where reasons to accept outweigh reasons to
reject, e.g., limited evaluation. Please use sparingly.</span>

</div>

<div>

**Confidence:** <span class="note-content-value">4: You are confident in
your assessment, but not absolutely certain. It is unlikely, but not
impossible, that you did not understand some parts of the submission or
that you are unfamiliar with some pieces of related work.</span>

</div>

<div>

**Ethical Concerns:** <span class="note-content-value">NO or VERY MINOR
ethics concerns only</span>

</div>

<div>

**Paper Formatting Concerns:**

<div class="note-content-value markdown-rendered">

No.

</div>

</div>

<div>

**Code Of Conduct Acknowledgement:**
<span class="note-content-value">Yes</span>

</div>

<div>

**Responsible Reviewing Acknowledgement:**
<span class="note-content-value">Yes</span>

</div>

<div>

**Final Justification:**

<div class="note-content-value markdown-rendered">

\[object Object\]

Resolved Issues:

- Clarified the relationship between intruder dimensions and forgetting.
- Provided a clear explanation of the method to prevent intruder
  dimensions and addressed potential logic mismatches.

Unresolved Issues:

- Additional evidence is needed to connect Sections 3 and 5 effectively.
- Algorithmic suggestions for mitigating intruder dimensions remain
  limited.

In conclusion, this is a technically solid and engaging paper with some
limitations. I assign it a rating of 4.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="bn6z5rLWAs">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Rebuttal by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note bn6z5rLWAs"></span><span class="sr-only">Copy
URL of note bn6z5rLWAs</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>30 Jul 2025, 20:30 (modified: 29 Oct 2025,
01:12)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=bn6z5rLWAs)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

We thank the reviewer for their thoughtful review. We provide responses
to their review below.

> Logical mismatch in Section 5. Section 4 convincingly correlates the
> number of intruder dimensions with forgetting, yet Section 5 only
> experiments with scaling down the magnitude of the top intruder
> dimension. To solidify the argument, the authors should also
> manipulate the count of intruder dimensions and measure its effect on
> forgetting.

We actually do study altering the count of intruder dimensions in
section 5. When scaling with $`\lambda = 0`$, we remove that intruder
dimension. In Section 5, we remove the top intruder dimension *in each
weight matrix*. This has the effect of removing up to 72 intruder
dimensions from a RoBERTa model and has the effect of significantly
reducing forgetting while causing less extreme drop in test performance.
For examples, see Fig. 8 when $`\lambda = 0`$. We hope this clarifies
the reviewers confusion.

> Limited practical guidance. The paper does not provide guidance on how
> to prevent the emergence of intruder dimensions during training, which
> is crucial for improving real-world applicability. As it stands,
> intruder dimensions appear to be primarily useful as a post-hoc model
> selection criterion.

This paper provides several concrete suggestions on how to **reduce the
emergence of intruder dimensions at training time**. We provide evidence
that certain initializations (Fig. 10), lower learning rates (Fig. 7),
and setting $`\alpha = 2r`$ (Section N) all lead to fewer intruder
dimensions. These are concrete prescriptions that can be used to prevent
the emergence of intruder dimensions during training.

Moreover, we show that scaling down the top intruder dimensions after
LoRA fine-tuning recovers the drop in pretraining performance without
affecting downstream adaptation. In one such case, we see forgetting
reduced by 33.2% with no change in test accuracy (lines 292-293). This
effectively **resurfaces forgotten** information without compromising
task performance—a novel and actionable intervention that leads to
improved OOD generalization which was not previously known.

These are a few practical guidelines offered by the paper.

> The paper concludes that intruder dimensions cause forgetting.
> However, full-tuning with few intruder dimensions, still exhibits more
> forgetting than LoRA with an appropriately chose alpha. How do we
> understand this based on findings in the paper?

It is important to note that intruder dimensions are **a cause of
forgetting** in LoRA'd models, but not the **only cause** of forgetting
in fine-tuning in general. This is obvious, since we should expect any
sort of deviation from the pre-trained weights, which were trained to
minimize language modelling loss, to lead to an increase in language
modelling loss (aka forgetting). This finding demonstrates that these
two methods update weight matrices in fundamentally different ways. This
explains the structural basis of forgetting with LoRA'd models (and
variants of LoRA). However, a functional explanation of forgetting in
full-finetuned models can be different and an interesting avenue for
future study.

> In the continual learning setup, are the similarity matrices in Figure
> 9 all computed with the original model? I am wondering can we observe
> intruder dimensions over a model already tuned on another task(say do
> weights after task 3 have intruder dimension over weights after task
> 2)?

Yes, all similarity matrices in Figure 9 are computed using the original
model. That said, we do observe what you're describing: as we train on
more tasks, new intruder dimensions emerge in **distinct locations**
with every round of adaptation to a new task (highlighted in pink),
indicating that intruder dimensions can indeed arise relative to earlier
task weights (e.g., after Task 3 vs. Task 2).

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="YNvqOD6xRz">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Rebuttal by Authors

</div>

<div class="heading">

#### Official Comment by Reviewer BfKg

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note YNvqOD6xRz"></span><span class="sr-only">Copy
URL of note YNvqOD6xRz</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Reviewer
BfKg</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>04 Aug 2025, 09:30 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=YNvqOD6xRz)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thank you for the rebuttal; it addresses most of my concerns, and I will
raise my rating to 4.

Regarding my first question, I agree that when $`\lambda = 0`$, one
intruder dimension is removed for each weight matrix. However, the paper
does not include a figure showing the trend of pre-train loss versus the
number of intruder dimensions. Moreover, this decrease follows a
specific pattern, where exactly one intruder dimension is removed per
matrix. I believe additional discussion or results linking the number of
intruder dimensions to their magnitudes would strengthen the logical
flow of the paper. At present, Section 3 focuses entirely on the count
of intruder dimensions, while Section 5 discusses their magnitudes—two
related but not equivalent concepts.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="KFbOJH6yqK">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Official Comment by Reviewer BfKg

</div>

<div class="heading">

#### Official Comment by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note KFbOJH6yqK"></span><span class="sr-only">Copy
URL of note KFbOJH6yqK</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>04 Aug 2025, 20:57 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=KFbOJH6yqK)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

We thank the reviewer for their thoughtful responses and their decision
to update their score.

> Regarding my first question, I agree that when $`\lambda = 0`$, one
> intruder dimension is removed for each weight matrix. However, the
> paper does not include a figure showing the trend of pre-train loss
> versus the number of intruder dimensions. Moreover, this decrease
> follows a specific pattern, where exactly one intruder dimension is
> removed per matrix. I believe additional discussion or results linking
> the number of intruder dimensions to their magnitudes would strengthen
> the logical flow of the paper. At present, Section 3 focuses entirely
> on the count of intruder dimensions, while Section 5 discusses their
> magnitudes—two related but not equivalent concepts.

Thank you for this valuable suggestion. We will look into measuring and
linking the number of intruder dimensions to their magnitudes in order
to strengthen the logical flow of this work as you suggest. Due to the
limited time remaining in the rebuttal cycle, we will be unable to
complete this measure before the response deadline but will include this
measure in the updated version of the manuscript.

We thank you again for your thoughtful responses and your positive
reaction to our work.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="J9ke6u7W9I">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Rebuttal by Authors

</div>

<div class="heading">

#### Mandatory Acknowledgement by Reviewer BfKg

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note J9ke6u7W9I"></span><span class="sr-only">Copy
URL of note J9ke6u7W9I</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Mandatory
Acknowledgement</span><span class="signatures">by Reviewer
BfKg</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>07 Aug 2025, 01:35 (modified: 12 Nov 2025,
10:34)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=J9ke6u7W9I)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Mandatory Acknowledgement:** <span class="note-content-value">I have
read the author rebuttal and considered all raised points., I have
engaged in discussions and responded to authors., I have filled in the
"Final Justification" text box and updated "Rating" accordingly (before
Aug 13) that will become visible to authors once decisions are
released., I understand that Area Chairs will be able to flag up
Insufficient Reviews during the Reviewer-AC Discussions and shortly
after to catch any irresponsible, insufficient or problematic behavior.
Area Chairs will be also able to flag up during Metareview grossly
irresponsible reviewers (including but not limited to possibly
LLM-generated reviews)., I understand my Review and my conduct are
subject to Responsible Reviewing initiative, including the desk
rejection of my co-authored papers for grossly irresponsible behaviors.
<a
href="https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/"
rel="noopener noreferrer"
target="_blank">https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/</a></span>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="UOFAXrz1mI">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission21986 by Reviewer GjPE

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note UOFAXrz1mI"></span><span class="sr-only">Copy
URL of note UOFAXrz1mI</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
GjPE</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>02 Jul 2025, 14:46 (modified: 28 Oct 2025,
23:20)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=UOFAXrz1mI)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

The paper analyzes the differences between LoRA and full fine-tuning of
LLMs by examining the spectral properties of the resulting weight
matrices. Specifically, the authors compare the singular vectors of the
pre-trained and fine-tuned weights and find that, while full fine-tuning
largely preserves the original spectrum, LoRA introduces intruder
dimensions—singular vectors with large singular values that have low
similarity to any singular vector from the pre-trained model. The
authors conduct thorough experiments to study how various
hyperparameters, such as LoRA rank and alpha, learning rate, the
parameters used to compute the number of intruder dimensions, and even
variations of the LoRA method, affect these intruder dimensions. They
then analyze the forgetting behavior of LoRA in both fine-tuning and
continual learning settings, and argue that the intruder dimensions
contribute to it.

</div>

</div>

<div>

**Strengths And Weaknesses:**

<div class="note-content-value markdown-rendered">

Strengths:

1.  The paper addresses an important problem: understanding the
    properties of solutions obtained with LoRA is highly relevant, given
    its widespread use.
2.  To the best of my knowledge, the analysis of spectral differences
    between LoRA and full fine-tuning is novel, and the results are
    nontrivial and interesting.
3.  The experimental setup is strong. The use of different architectures
    and datasets supports the generality of the intruder dimension
    phenomenon.
4.  The paper is thorough and includes a wide range of additional
    experiments and ablations.
5.  The paper is clearly written and easy to follow.

Main weaknesses:

1.  The focus of the paper. The paper focuses on intruder dimensions as
    the main factor distinguishing the spectral properties of LoRA and
    full fine-tuning, but it is not entirely clear whether this choice
    captures the core difference. According to Figure 2b, the difference
    appears to be more uniform: all singular vectors in the LoRA
    solution tend to have lower cosine similarity, and there is no clear
    small subset of outlier directions with significantly lower
    similarity. This suggests that LoRA broadly rotates the basis rather
    than introducing a few distinct intruder dimensions. The paper does
    not provide a clear justification for why the top intruder
    dimensions should be considered more important than the overall
    rotation of the spectrum.
2.  Reasons behind intruder dimensions. The paper does not provide a
    clear explanation for why LoRA introduces intruder dimensions.
    Moreover, while the results convincingly show that LoRA solutions
    have intruder dimensions, the causal relationship is not fully
    established. In particular, it is possible that these dimensions are
    not specific to LoRA, but instead arise from larger overall changes
    to the weights. The experiments varying the learning rate for LoRA
    show that higher learning rates lead to more intruder dimensions,
    and it is plausible that full fine-tuning would show a similar
    pattern if higher learning rates were used. This raises the question
    of whether the observed differences between LoRA and full
    fine-tuning are due to specific properties of the LoRA method, or
    simply a consequence of differences in optimal hyperparameter
    ranges. Additional experiments that vary the learning rate in full
    fine-tuning and analysis of the relationship between the number of
    intruder dimensions and the magnitude of weight updates would help
    clarify this point.
3.  Practical importance of intruder dimensions. The analyses in
    Sections 4 and 5 do not clearly establish a causal link between
    intruder dimensions and forgetting or generalization. Moreover, they
    do not provide evidence that the top intruder dimensions contribute
    more significantly than the rest of the changes in the weights.
    - The correlation between the number of intruder dimensions and
      forgetting observed in Section 4 may result from a shared
      underlying cause rather than a direct causal relationship.
      Specifically, larger changes in weights during fine-tuning (such
      as those resulting from higher learning rates in Figure 7) can
      lead to both a greater number of intruder dimensions and stronger
      adaptation to the fine-tuning data, which in turn increases
      forgetting. In this case, even though there is no strong
      correlation between the number of intruder dimensions and test
      accuracy on the fine-tuning task, there is likely a correlation
      with training accuracy/loss.
    - While Section 4 claims that the number of intruder dimensions
      correlates with forgetting, Figure 7 shows that full fine-tuning
      has the fewest intruder dimensions, yet exhibits more forgetting
      than LoRA with a low learning rate.
    - In Section 5, the experiments do not convincingly show that
      intruder dimensions contribute to forgetting more than other
      changes in weights. The comparison between downscaling intruder
      dimensions and regular singular vectors effectively compares
      removing part of the fine-tuning versus removing part of the
      pre-training. To show that intruder dimensions are specifically
      responsible for forgetting, a more appropriate comparison would be
      between downscaling intruder dimensions and downscaling the
      overall fine-tuning update. This distinction is important, since
      downscaling the overall fine-tuning update is also known to
      improve generalization
      (<a href="http://arxiv.org/abs/2109.01903" target="_blank"
      rel="noopener noreferrer">http://arxiv.org/abs/2109.01903</a>).
      From the current experiments, it remains unclear whether intruder
      dimensions contribute more to forgetting than other changes
      introduced during fine-tuning.

Additional concerns, comments and questions:

1.  The analysis of the effect of LoRA rank on intruder dimensions could
    be improved. Figure 4 clearly shows a non-monotonic trend: as the
    LoRA rank increases, the number of intruder dimensions initially
    grows and then decreases. While the text notes that intruders
    decrease and converge toward full-rank behavior at high ranks, it
    does not discuss the initial increase at low ranks.
2.  The claim that LoRA forgets less than full fine-tuning, even at
    comparable performance levels, is not particularly strong. The
    results for LoRA and full fine-tuning in Table 2 differ
    significantly, and the observed differences in forgetting appear to
    correlate, at least to some extent, with differences in accuracy.
    For example, on the QQP task, both accuracy and forgetting differ
    between high-rank LoRA and full fine-tuning, while on MNLI, the
    results are almost identical in both respects.
3.  The results in Figure 17 do not seem to fully support the
    conclusions drawn from Figure 9a. Across different tasks, LoRA
    exhibits both more and less forgetting, suggesting that the effect
    may not generalize consistently.
4.  A more detailed analysis of intruder dimensions of different LoRA
    variants would be beneficial. There appears to be a non-monotonic
    dependence of the number of intruder dimensions on the epsilon
    parameter, which likely reflects properties of specific methods
    used.
5.  Are Figures 1, 2, and 3 based on the same experimental setting? In
    Figure 2, only one low-epsilon intruder dimension appears among the
    top 10 singular vectors, whereas Figures 1 and 3 seem to show a
    higher number of such dimensions.
6.  In Figure 2c, it seems odd to refer to a "normal" singular vector
    with cosine similarity equal to 1, given that Figure 2b clearly
    shows that the similarity is significantly lower for most vectors
    under both fine-tuning methods.
7.  It would be helpful to briefly define what alpha represents in the
    LoRA setup in the Background section, especially since its effect is
    analyzed later in the paper.
8.  Why do the full fine-tuning results differ Tables 1 and 2?
9.  In Figure 23, the line type for both VeRA experiments is the same.

</div>

</div>

<div>

**Quality:** <span class="note-content-value">2: fair</span>

</div>

<div>

**Clarity:** <span class="note-content-value">3: good</span>

</div>

<div>

**Significance:** <span class="note-content-value">2: fair</span>

</div>

<div>

**Originality:** <span class="note-content-value">3: good</span>

</div>

<div>

**Questions:**

<div class="note-content-value markdown-rendered">

All questions and concerns are detailed in the Strengths and Weaknesses
section. The main weaknesses 1–3 are the most critical and will have the
greatest impact on my final evaluation after the rebuttal.

</div>

</div>

<div>

**Limitations:**

<div class="note-content-value markdown-rendered">

I believe the limitations section should provide a more thorough
discussion noting that the paper focuses only on one aspect of the
spectral differences between LoRA and full fine-tuning, the intruder
dimensions, while leaving aside the effect of the overall rotation of
the spectrum and the changes in singular values.

</div>

</div>

<div>

**Rating:** <span class="note-content-value">3: Borderline reject:
Technically solid paper where reasons to reject, e.g., limited
evaluation, outweigh reasons to accept, e.g., good evaluation. Please
use sparingly.</span>

</div>

<div>

**Confidence:** <span class="note-content-value">5: You are absolutely
certain about your assessment. You are very familiar with the related
work and checked the math/other details carefully.</span>

</div>

<div>

**Ethical Concerns:** <span class="note-content-value">NO or VERY MINOR
ethics concerns only</span>

</div>

<div>

**Paper Formatting Concerns:**

<div class="note-content-value markdown-rendered">

--

</div>

</div>

<div>

**Code Of Conduct Acknowledgement:**
<span class="note-content-value">Yes</span>

</div>

<div>

**Responsible Reviewing Acknowledgement:**
<span class="note-content-value">Yes</span>

</div>

<div>

**Final Justification:**

<div class="note-content-value markdown-rendered">

After careful consideration, I still hold the opinion that **the paper
is not ready for publication and requires significant revision**. The
rebuttal and discussion mostly addressed my concern about the focus of
the paper (Weakness 1). However, the additional results largely
confirmed that my concerns regarding the reasons behind the intruder
dimensions and their connection to forgetting (Weaknesses 2 and 3) were
reasonable. **The main reason for my negative score is the section on
forgetting (Weakness 3):** the current claim that intruder dimensions
are “special” with respect to forgetting is not convincing and may be
incorrect.

**Final comments on the main weaknesses:**

- **Focus on intruder dimensions (Weakness 1).** After the discussion, I
  am convinced that intruder dimensions are indeed present in many LoRA
  experiments and that the paper’s focus is reasonable. However, some
  experiments still show very different behaviour (Figure 2b). Looking
  at Figure 4, both types of behaviours appear in practical experiments:
  clear outlier intruder dimensions result in an up–constant–up pattern
  (as in low-rank LoRA in the MNLI experiment), while a uniform change
  as in Figure 2b results in a constant–up pattern (as in high-rank LoRA
  in the MNLI experiment). I believe the paper should explicitly discuss
  this distinction.
- **Reasons behind intruder dimensions (Weakness 2).** Given that the
  LoRA weight update has a much higher L2 norm, it is not clear whether
  intruder dimensions are specifically related to LoRA or simply to a
  larger weight update in general. Based on the current results, in my
  opinion, the only justified conclusion is that LoRA and full
  fine-tuning *under optimal hyperparameters* have different update
  structures. This conclusion is interesting in itself, but the paper
  would then need to adjust its claims accordingly. Alternatively, the
  paper could include additional experiments, e.g., using a higher
  learning rate or longer training for full fine-tuning (or a lower
  learning rate for LoRA), to confirm whether LoRA and full fine-tuning
  still produce different weight structures when the weight update norms
  are similar.
- **Practical importance of intruder dimensions (Weakness 3).** After
  the discussion, I remain unconvinced that intruder dimensions are in
  any way special with respect to forgetting. Based on the previous
  point, the high correlation between the number of intruder dimensions
  and forgetting can most likely be explained by the strong correlation
  between the weight update norm and forgetting. Moreover, the
  similarity in the effects of downscaling intruder dimensions and
  downscaling the entire weight update in LoRA (Q4 in the discussion)
  contradicts the claim that intruder dimensions are particularly
  special.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="E7HdI942Id">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Rebuttal by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note E7HdI942Id"></span><span class="sr-only">Copy
URL of note E7HdI942Id</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>30 Jul 2025, 20:32 (modified: 29 Oct 2025,
01:12)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=E7HdI942Id)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

Thank you for your thoughtful review. We have incorporated your
suggestions, including the mistakes you caught like VeRA having the same
line twice in Fig. 23, into our updated manuscript.

Where required, we provide responses to your review below:

> Main weaknesses: The focus of the paper. The paper focuses on intruder
> dimensions as the main factor distinguishing the spectral properties
> of LoRA and full fine-tuning, but it is not entirely clear whether
> this choice captures the core difference.

Our primary objective was to assess **whether different fine-tuning
methods converged to functionally equivalent models despite differing
parameterizations**. To investigate this, we perform a spectral analysis
to characterize how the sequence of transformations each method applied
differ.

Since the SVD breaks up the principal components of a matrix, studying
how these principal components change (via cosine similarity) is an
intuitive way to examine how a weight matrix is changed during
fine-tuning. Empirically, we observe that while full fine-tuning, which
directly manipulates the weights, makes small adjustments to the
magnitude and direction of the singular vectors, LoRA, which uses a low
rank product, introduces new singular vectors with low cosine similarity
to the existing singular vectors. We provide some intuition for why
intruder dimensions are the right framework in Section B.2, where we
showed that adding the outer product of a random vector to a weight
matrix introduces an intruder dimension. This is analogous to the matrix
product of BA in LoRA when rank is 1, and implies that intruder
dimensions occur because the B and A vectors are uncorrelated to the
columns/rows of the weight matrix W_0.

We have conducted an extremely detailed examination and would have every
reason to report a better measure had it been found. Does the reviewer
have a suggestion of measure?

> The paper does not provide a clear explanation for why LoRA introduces
> intruder dimensions. Moreover, while the results convincingly show
> that LoRA solutions have intruder dimensions, the causal relationship
> is not fully established.

It is important to note that intruder dimensions are an empirical
observation of LoRA. However, we provide a few mathematical
justifications for their occurrence. **We showed that adding the outer
product of a random vector to a weight matrix introduces an intruder
dimension. This is analogous to the matrix product of BA in LoRA when
rank is 1, and implies that intruder dimensions occur because the B and
A vectors are uncorrelated to the columns/rows of the weight matrix W_0
(Section B.2).** Further, we showed empirically that just training the B
matrix, while freezing the A matrix with all singular values of 1,
eliminates the amplification of the singular values because of the
matrix product and therefore reduces the number of high ranking intruder
dimensions. This suggests that this multiplicative property may cause
new singular vectors to have large singular value. These experiments
provide evidence towards why LoRA causes intruder dimensions to be
introduced.

> it is possible that these dimensions are not specific to LoRA, but
> instead arise from larger overall changes to the weights ... The
> experiments varying the learning rate for LoRA show that higher
> learning rates lead to more intruder dimensions, and it is plausible
> that full fine-tuning would show a similar pattern if higher learning
> rates were used. This raises the question of whether the observed
> differences between LoRA and full fine-tuning are due to specific
> properties of the LoRA method, or simply a consequence of differences
> in optimal hyperparameter ranges.

The reviewer is absolutely correct to be curious about this and we
indeed investigated this. In our investigation, we used standard
learning rates for both full fine-tuning and LoRA. When we conduct a
large learning rate sweep for both methods, we observe that increasing
learning rate significantly over the default setting leads to training
instabilities and divergence. Decreasing learning rate has difficulty
converging to similar performance, even with more training steps.
Because of this, the resulting models perform significantly worse and
therefore cannot be used in our analysis. **For models that do converge
to similar, near optimal performance (Fig. 7), we observe that full
fine-tuning contains no intruder dimensions while LoRA contains many.**

> Practical importance of intruder dimensions. The analyses in Sections
> 4 and 5 do not clearly establish a causal link between intruder
> dimensions and forgetting or generalization.

In Section 4, we find a strong correlation between intruder dimensions
and forgetting. The reviewer is absolutely right to point out that this
could be the result of a third variable that is causing both to
increase. **To show that this is not the case, in section 5 we intervene
on the intruder dimensions.** In it, we scale down the singular values
of the intruder dimensions, which has the effect of reducing their
contribution on the fine-tuned weights. By performing this causal
experiment, we find that there is little change in test accuracy but a
large impact on forgetting. One example of this (reported in the main
text in lines 292-293) shows that using $`\lambda = 0.7`$ on our model
trained on QQP leads to *no change* in test accuracy but a 33.2%
reduction in forgetting.

These findings lead to the following conclusion: intervening on intruder
dimensions and scaling them down results in a big drop in forgetting but
little change in test accuracy, showing that these intruder dimensions,
and in particular their magnitude (singular vector), causes a large
amount of the forgetting in LoRA models.

We do not claim that intruder dimensions are not the *only* possible
cause of forgetting. Any sort of change to the pre-trained weights
should be expected to impact a models base language modelling ability.
Indeed, we observe that full fine-tuning makes small adjustments to the
direction and magnitude of the pre-trained singular vectors. This still
changes the fine-tuned weights, and we should expect forgetting to
occur. Our causal experiment scaling intruder dimensions is specific to
LoRA.

Intruder dimensions and section 5 explain the structural basis of
forgetting with LoRA'd models (and variants of LoRA). However, a
functional explanation of forgetting in full-finetuned models can be
different and an interesting avenue for future study.

> Additional concerns, comments and questions: The claim that LoRA
> forgets less than full fine-tuning, even at comparable performance
> levels, is not particularly strong. The results for LoRA and full
> fine-tuning in Table 2 differ significantly, and the observed
> differences in forgetting appear to correlate, at least to some
> extent, with differences in accuracy. For example, on the QQP task,
> both accuracy and forgetting differ between high-rank LoRA and full
> fine-tuning, while on MNLI, the results are almost identical in both
> respects.

Our conclusion that "LoRA forgets less than full fine-tuning, even at
comparable performance levels" stems from our results in Table 2 and
Fig. 6. Table 2 contains the test accuracies of all of our models. In
it, we see models trained to approximately the same accuracy. We are not
sure why the reviewer claims that the results for LoRA and full
fine-tuning in Table 2 differ significantly. For example, in Table 2,
all our models fine-tuned on MNLI perform within 0.5% of each other.
Looking horizontally, comparing full fine-tuning and LoRA r=16, without
loss of generality, we see similar performance, with LoRA sometimes
outperforming full fine-tuning and vice versa. Further, when correlating
test accuracy and forgetting within datasets for Table 2 (as indicated
by the reviewer), we get no statistically significant result (test:
Spearman's rank-order correlation, p-value\>0.33 for each.).
**Therefore, this means that observed differences in forgetting do not
correlate with differences in accuracy.**

When looking at Fig. 6b, we see that full fine-tuning always forgets
more than LoRA. This leads us to conclude that LoRA forgets less than
full fine-tuning, even at comparable performance levels, which exends
the findings of \[1\]. We hope this clarifies the reviewers concern.

\[1\] - <a href="https://arxiv.org/pdf/2405.09673" target="_blank"
rel="noopener noreferrer">https://arxiv.org/pdf/2405.09673</a>

> A more detailed analysis of intruder dimensions of different LoRA
> variants would be beneficial. There appears to be a non-monotonic
> dependence of the number of intruder dimensions on the epsilon
> parameter, which likely reflects properties of specific methods used.

We provide a preliminary study to show that our findings scale to other
variants in Section P. We find that LoRA variants (LoRA+, VeRA, AdaLoRA,
PiSSA) all have intruder dimensions, showing that these methods are not
adequate for preventing intruder dimensions. We emphasize that our study
of LoRA variants is orthogonal to our main study, and that it is
difficult to conduct every detailed experiment we have on all LoRA
variants. While we save an in depth analysis for future work, we have no
reason to suspect these will impact the claims we make in this paper.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="NWM4lFoYvG">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Rebuttal by Authors

</div>

<div class="heading">

#### Official Comment by Reviewer GjPE

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note NWM4lFoYvG"></span><span class="sr-only">Copy
URL of note NWM4lFoYvG</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Reviewer
GjPE</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>01 Aug 2025, 17:25 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=NWM4lFoYvG)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thanks for the detailed response!

The main weaknesses I pointed out are only partially addressed, so I
have some clarification questions below. Questions 1/2/4 are more
critical.

**Weakness 1. Focus of the paper**

Q1. I fully agree that analyzing how the principal components of weight
matrices change is a natural choice here. My concern is not about the
idea of analyzing SVD components, but rather about the focus on only the
top subset of these components, rather than considering the average
change across all of them. As I mentioned in the initial review,
according to Figure 2b, the difference between LoRA and full fine-tuning
appears more uniform: all singular vectors in the LoRA solution tend to
have lower cosine similarity, with no clear small subset of outlier
directions showing significantly lower similarity. Is it not the case
usually? If it is, could you please elaborate on why the paper focuses
only on changes in the top directions and does not analyze all
components?

**Weakness 2. Reasons behind intruder dimensions**

Q2. Could you please provide the results for the learning rate values
that led to reasonable training outcomes for full fine-tuning in your
learning rate sweep (i.e., accuracy and number of intruder dimensions at
different threshold levels)? Results even for 2-3 of them would already
be valuable.

Q3. If possible, a comparison of the distances between pre-training and
fine-tuning checkpoints for LoRA and full fine-tuning would also be very
helpful to make sure that intruder dimensions are mostly related to LoRA
and not higher changes in weights in general.

**Weakness 3. Practical importance of intruder dimensions**

Q4. While I agree that this intervention experiment is a good starting
point for making a causal claim, I believe it currently lacks a proper
baseline. To support the idea that there is something specific about the
forgetting properties of intruder dimensions, the result should be
compared to a baseline where the full fine-tuning weight update is
downscaled. Without this comparison, it’s unclear whether the observed
effect is related specifically to the intruder dimensions or is simply a
general consequence of downscaling the fine-tuning update. Prior work
(e.g., <a href="http://arxiv.org/abs/2109.01903" target="_blank"
rel="noopener noreferrer">http://arxiv.org/abs/2109.01903</a>) shows
that the latter can also lead to similar effects. A strong argument
would be to show that downscaling only the intruder dimensions results
in less loss of accuracy at the same level of forgetting than
downscaling the entire fine-tuning update.

Q5. If possible, could you please provide the results on the correlation
between the number of intruder dimensions and the training loss on the
fine-tuning task (instead of test accuracy provided in the paper)?

**Additional concerns**

Thanks for the correlation results, I found them very useful. Regarding
the additional analysis of LoRA variants, I meant it only as a
suggestion, and I fully agree with you that this is an interesting
direction for future work!

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="O3sw923TiD">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Official Comment by Reviewer GjPE

</div>

<div class="heading">

#### Official Comment by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note O3sw923TiD"></span><span class="sr-only">Copy
URL of note O3sw923TiD</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>02 Aug 2025, 22:35 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=O3sw923TiD)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thank you for your detailed response and for your quick answer to our
rebuttal. We provide responses to your follow up questions below. Please
let us know if any of our below responses are unsatisfactory, and we
would be happy to follow up.

> Q1.

You are absolutely correct to bring up this point. We included Fig. 2b
to help provide intuition on measuring the cosine similarity between
pre-trained and fine-tuned singular vectors and is **not** what is
typically observed. We instead observe that all pre-trained singular
vectors are preserved and that new, high ranking singular vectors are
introduced, as shown in Fig 2a. We apologize that this selection of
graphic was confusing to the claims we make. Thank you for helping us
clarify this.

> Q2.

As requested, we provide accuracy and number of intruder dimensions for
several full fine-tuning runs below. While we cannot update the PDF, we
print out accuracyand number of intruder dimensions for several
different runs. Each value in a list is for each epoch (the first entry
is for after epoch 1, etc.. note that these values would apply to the
settings used in Fig. 7).

Full fine-tuning with lr=2.5e-6:

Number of Intruders: \[0, 0, 0, 0, 0\]

Test Accuracy: \[0.8554, 0.8668, 0.8704, 0.8680, 0.8655\]

Full fine-tuning with lr=5e-6:

Number of Intruders: \[0, 0, 0, 0, 0\]

Test Accuracy: \[0.8601, 0.8742, 0.8699, 0.8648, 0.8656\]

Full fine-tuning with lr=1e-5:

Number of Intruders: \[0, 0, 0, 0, 0\]

Test Accuracy: \[0.8607, 0.8745, 0.8703, 0.8694, 0.8671\]

We hope these values are helpful and encourage the reviewer to reach out
with futher requests or clarifying questions should they come up.

> Q3.

To measure this, we select a model and measure the weight norm of the
update to the weights ($`\Delta W`$). We measure full fine-tuning's
update has an average weight norm of 0.000172, while LoRA's update has
an average weight norm of 0.002839. This is a ~16x difference. However,
this is consistent with our observation of intruder dimensions. Since
LoRA has intruder dimensions (new singular vectors with large singular
value), we should expect LoRA to have larger weight norm difference
because $`\Delta W`$ will contain the intruder dimension. In contrast,
full fine-tuning, which makes subtle changes to singular values/vectors,
will have smaller weight norm difference because the update matrix will
contain very small values (since $`W_{f} = W_{0} + \Delta W`$).

> Q4.

This is indeed a good baseline to measure. When we compare LoRA'd models
that either have their entire updated scaled or just their intruder
dimensions scaled, we get similar pareto curves on the forgetting vs
performance graph. However, we argue that this makes sense. Since we
argue that most of the LoRA update is in the intruder dimensions,
scaling these intruder dimensions down should have a similar effect to
scaling the entire update down. We thank the reviewer for suggesting
this measure and pointing us to this piece of literature.

> Q5.

Unfortunately, we only log training loss at batch level only, which
means we cannot calculate a global train loss measure. However, we
attempt to calculate this value by averaging the batch training loss
across the previous 100 batches in order to estimate the value you
request. When calculating the spearman correlation, we measure a
statistic of -0.2174 and pvalue of 0.3573. This means that there is no
statistically significant relationship between training loss and
intruder dimensions. This further shows that our results are not simply
based on the fact that certain models are overfit to the training set.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="730aiyb0b0">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Official Comment by Reviewer GjPE

</div>

<div class="heading">

#### Official Comment by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note 730aiyb0b0"></span><span class="sr-only">Copy
URL of note 730aiyb0b0</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>07 Aug 2025, 00:14 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=730aiyb0b0)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

We would thank you again for your thoughtful review and response. Given
that the rebuttal period is closing soon, we are reaching out to ensure
we have addressed all the concerns raised. We would also be happy to
answer any further questions if any have come up.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="qK0UiuQbej">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Official Comment by Authors

</div>

<div class="heading">

#### Official Comment by Reviewer GjPE

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note qK0UiuQbej"></span><span class="sr-only">Copy
URL of note qK0UiuQbej</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Reviewer
GjPE</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>08 Aug 2025, 18:38 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=qK0UiuQbej)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

**Weakness 1.** I don’t find this argument convincing. Figure 2a is a
motivational image, while Figure 2b is an actual plot of the results.
So, if Figure 2b is not what is usually observed in the experiments,
could you please provide some data on what is observed? For example, a
histogram of cosine values from other experiments? And please explain
why the results in Figure 2b are different.

Based on all the results from the paper, it seems that while the cosines
for non-intruder vectors are close to one for the first several vectors
(Figure 3), this is not the case for the vectors in general (Figures 1
and 2). The analysis in the paper is clearly not focused only on the
first several vectors.

**Weakness 2, Q1.** Thanks for the additional results. However, my
question was more about learning rates higher than the one used in the
original experiment from the paper. It is expected that for lower
learning rates, the number of intruder dimensions would also be 0.

**Weakness 2, Q2.** Thanks for the additional results. I agree with your
discussion here; however, this result also supports the idea that the
effect of intruder dimensions could be connected to the learning rate
used.

**Weakness 3, Q4.** Again, I agree with your discussion here, but this
result demonstrates that intruder dimensions are not particularly
special in terms of forgetting compared to other dimensions.

**Weakness 3, Q5.** Thanks for the result, it is indeed an interesting
observation.

I understand that there is not much time left in the discussion period,
so I will keep that in mind. I would appreciate any response possible
within the remaining time.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="1il6bhHfSX">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Official Comment by Reviewer GjPE

</div>

<div class="heading">

#### Official Comment by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note 1il6bhHfSX"></span><span class="sr-only">Copy
URL of note 1il6bhHfSX</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>08 Aug 2025, 23:58 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=1il6bhHfSX)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thank you again for your thoughtful responses and involvement with this
rebuttal.

> Weakness 1. I don’t find this argument convincing. Figure 2a is a
> motivational image, while Figure 2b is an actual plot of the results.
> So, if Figure 2b is not what is usually observed in the experiments,
> could you please provide some data on what is observed? For example, a
> histogram of cosine values from other experiments? And please explain
> why the results in Figure 2b are different.

We are apologetic about any confusion that has been caused by this plot.
Fig 2b is a graph specifically selected to motivate this paper and to
suggest that LoRA and Full fine-tuning are updating weight matrices
differently. The proper plot to examine for the standard difference
between full fine-tuning and LoRA is Fig. 1. In it, we see an 'offset'
in LoRA due to intruder dimensions, whereas we see no offset for full
fine-tuning. In particular, we call attention to the fact that both LoRA
and full fine-tuning have both a diagonal with the same slope and a
similarly wide band- except for intruder dimensions causing the offset,
there is little different between the two models.

While we cannot update the PDF or add any links to include any new
figures due to rebuttal rules, we hope that Figure 1 can clarify this
issue.

> Based on all the results from the paper, it seems that while the
> cosines for non-intruder vectors are close to one for the first
> several vectors (Figure 3), this is not the case for the vectors in
> general (Figures 1 and 2). The analysis in the paper is clearly not
> focused only on the first several vectors.

We are unsure of what the reviewer means by the last sentence, but while
we agree that low-ranked singular vectors in the fine-tuned models do
not map as full as high-ranking singular vectors(this is trivial: even
with small changes to high ranked singular vectors, low ranked singular
vectors will be forced to change in order to continue to span the vector
space), we do not observe a particularly pronounced effect on this.
Figure 1, as you mention above, actually supports this: We see that both
full fine-tuning and LoRA have a clear diagonal, indicating a ordered
mapping to the pre-trained singular vectors. Further, there is no clear
difference in the width of the band between full fine-tuning and LoRA.
We emphasize that the purpose of this paper is to study the differences
between these two models.

> Weakness 2, Q1. Thanks for the additional results. However, my
> question was more about learning rates higher than the one used in the
> original experiment from the paper. It is expected that for lower
> learning rates, the number of intruder dimensions would also be 0.

We are sorry if we misinterpreted your request from your earlier
response. Unfortunately, due to the lack of time remaining in the
rebuttal, we are unable to conduct new experiments. We hope this is
acceptable to you. However, it is a common observation that increasing
the learning rate leads to training instabilities and even divergence.

> Weakness 2, Q2. Thanks for the additional results. I agree with your
> discussion here; however, this result also supports the idea that the
> effect of intruder dimensions could be connected to the learning rate
> used.

As we detail in Fig 7. of the paper, we agree that learning rate plays a
role on intruder dimensions. However, we find it important to mention
that **we adapt no special hyperparameter settings for fine-tuning, but
rather we replicate the training settings of existing works**
(<a href="https://arxiv.org/abs/2405.09673" target="_blank"
rel="noopener noreferrer">https://arxiv.org/abs/2405.09673</a>,
<a href="https://arxiv.org/abs/2106.09685" target="_blank"
rel="noopener noreferrer">https://arxiv.org/abs/2106.09685</a>). To
reiterate, we use standard hyperparameter settings that have been
selected by others for their success in well-optimizing these models.
This shows that our results are not due to our failure to well-optimize
our LoRA models, but rather are for LoRA's standard setting and
use-case.

> Weakness 3, Q4. Again, I agree with your discussion here, but this
> result demonstrates that intruder dimensions are not particularly
> special in terms of forgetting compared to other dimensions.

We would like to re-emphasize the claim that the large portion of the
update of LoRA is in its intruder dimensions. If this were to the case,
scaling both down the intruder dimensions only or the full update should
lead to similar results. In fact, *this is exactly what we observe,* and
show that the magnitude of these intruder dimensions is responsible for
forgetting and can be reduced in magnitude while reducing forgetting and
maintaining performance.

> Weakness 3, Q5. Thanks for the result, it is indeed an interesting
> observation.

Thank you again for your suggestions. We are glad that this particular
question was helpful to you.

> I understand that there is not much time left in the discussion
> period, so I will keep that in mind.

We thank you for your understanding and are grateful for your thoughtful
feedback which has improved this paper.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="SvvXwWsVdj">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Rebuttal by Authors

</div>

<div class="heading">

#### Mandatory Acknowledgement by Reviewer GjPE

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note SvvXwWsVdj"></span><span class="sr-only">Copy
URL of note SvvXwWsVdj</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Mandatory
Acknowledgement</span><span class="signatures">by Reviewer
GjPE</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>13 Aug 2025, 13:47 (modified: 12 Nov 2025,
10:34)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=SvvXwWsVdj)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Mandatory Acknowledgement:** <span class="note-content-value">I have
read the author rebuttal and considered all raised points., I have
engaged in discussions and responded to authors., I have filled in the
"Final Justification" text box and updated "Rating" accordingly (before
Aug 13) that will become visible to authors once decisions are
released., I understand that Area Chairs will be able to flag up
Insufficient Reviews during the Reviewer-AC Discussions and shortly
after to catch any irresponsible, insufficient or problematic behavior.
Area Chairs will be also able to flag up during Metareview grossly
irresponsible reviewers (including but not limited to possibly
LLM-generated reviews)., I understand my Review and my conduct are
subject to Responsible Reviewing initiative, including the desk
rejection of my co-authored papers for grossly irresponsible behaviors.
<a
href="https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/"
rel="noopener noreferrer"
target="_blank">https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/</a></span>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="zAzkhQXKZz">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission21986 by Reviewer JRuL

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note zAzkhQXKZz"></span><span class="sr-only">Copy
URL of note zAzkhQXKZz</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
JRuL</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>30 Jun 2025, 03:16 (modified: 28 Oct 2025,
23:20)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=zAzkhQXKZz)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

This paper thoroughly investigates the structural differences between
weight updates from LoRA and full-finetuning through the lens of SVD
decomposition, and introduces a new concept *intruder dimension* to
quantify these differences. Extensive empirical stuides and discussions
are provided to better exemplify the properties and behaviors of this
new concept.

</div>

</div>

<div>

**Strengths And Weaknesses:**

<div class="note-content-value markdown-rendered">

**Strengths**

1.  The paper is well-written and clearly structured, with a logical and
    straightforward presentation.
2.  The concept of *intruder dimension* is well-defined and intuitively
    aligns with the mathematical underpinnings of representation
    learning, particularly in the context of SVD, where directions with
    higher variance tend to carry more informative content.

**Weaknesses**

1.  The core concern is that the paper primarily presents extensive
    empirical comparisons between LoRA and full fine-tuning to
    illustrate the emergence and impact of *intruder dimensions*,
    without proposing new methods or actionable improvements based on
    these insights. As per the NeurIPS regulation, it may be better
    suited for the **Datasets & Benchmarks track** rather than the main
    conference.

2.  Despite the extensive empirical evidence for the impact of *intruder
    dimension*, the theoretical explanation for why they appear in
    fine-tuning LoRA is still lacking. Mathematically, it is expected
    that constrained updates (e.g., via LoRA) tend to concentrate on
    directions associated with the largest singular values, while full
    fine-tuning distributes updates more evenly with sufficiently large
    update space. Without rigorous theoretical support, the introduction
    of the *intruder dimension* appears to be a restatement of this
    known behavior within the specific framing of LoRA.

3.  I've several concerns regarding the empirical analyses:

    1.  Regarding the correlation between the number of *intruder
        dimension* and catastrophic forgetting (Lines 256-263), the
        experiments using a larger learning rate show that LoRA is not
        well optimized, as evidenced by a significant performance gap.
        In such cases, the increase in intruder dimensions could be a
        byproduct of suboptimal training rather than an inherent issue.
        For a well-optimized LoRA model (e.g., with lr=1e-4), the
        observation in \[1\] (LoRA learns less but forgets less) still
        holds. Thus, conclusions based on poorly trained models are less
        convincing.
    2.  In Appendix H.2, the observation that performance drops when
        learned knowledge is perturbed and LoRA's preserved knowledge is
        exaggerated is unsurprising. This behavior is consistent with
        general deep learning dynamics and does not convincingly support
        the claim that *intruder dimensions* degrade OOD performance.
        The argument feels more like an expected consequence of
        disrupting well-internalized features.
    3.  In Appendix I, was the hyperparameter search of learning rate
        conducted to ensure that LoRA with different $`\alpha`$
        converged to similar test accuracy? If not, it's trivial that
        improper/insufficient learning naturally leads to low rank
        solutions.

\[1\] Dan, Biderman, et al., "LoRA Learns Less and Forgets Less," TMLR
2024.

</div>

</div>

<div>

**Quality:** <span class="note-content-value">3: good</span>

</div>

<div>

**Clarity:** <span class="note-content-value">3: good</span>

</div>

<div>

**Significance:** <span class="note-content-value">2: fair</span>

</div>

<div>

**Originality:** <span class="note-content-value">3: good</span>

</div>

<div>

**Questions:**

<div class="note-content-value markdown-rendered">

See Weaknesses.

</div>

</div>

<div>

**Limitations:**

<div class="note-content-value markdown-rendered">

yes.

</div>

</div>

<div>

**Rating:** <span class="note-content-value">4: Borderline accept:
Technically solid paper where reasons to accept outweigh reasons to
reject, e.g., limited evaluation. Please use sparingly.</span>

</div>

<div>

**Confidence:** <span class="note-content-value">4: You are confident in
your assessment, but not absolutely certain. It is unlikely, but not
impossible, that you did not understand some parts of the submission or
that you are unfamiliar with some pieces of related work.</span>

</div>

<div>

**Ethical Concerns:** <span class="note-content-value">NO or VERY MINOR
ethics concerns only</span>

</div>

<div>

**Paper Formatting Concerns:**

<div class="note-content-value markdown-rendered">

NA.

</div>

</div>

<div>

**Code Of Conduct Acknowledgement:**
<span class="note-content-value">Yes</span>

</div>

<div>

**Responsible Reviewing Acknowledgement:**
<span class="note-content-value">Yes</span>

</div>

<div>

**Final Justification:**

<div class="note-content-value markdown-rendered">

Overall Assessment:

1.  I thank the authors for their clarifications, and I find this
    submission to be in alignment with the guidelines of the main
    conference.
2.  As noted by reviewers BfKg and GjPE, the exploration of an “intruder
    dimension” for LoRA seems to have limited practicality in real-world
    deployments, considering that the primary value of PEFT lies in
    enabling the rapid application of LLMs in industrial scenarios.
3.  I appreciate the authors’ substantial effort and the detailed
    clarifications provided.

After considering the discussions throughout the rebuttal phase, I view
this paper as being around the borderline. I have accordingly adjusted
my rating to 4, while noting that this does not imply a definitive
decision regarding acceptance.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="SUvpVsoKsq">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Rebuttal by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note SUvpVsoKsq"></span><span class="sr-only">Copy
URL of note SUvpVsoKsq</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>30 Jul 2025, 20:32 (modified: 29 Oct 2025,
01:12)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=SUvpVsoKsq)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

We thank the reviewer for their thoughtful review. We provide responses
to their review below.

> The core concern is that the paper primarily presents extensive
> empirical comparisons between LoRA and full fine-tuning to illustrate
> the emergence and impact of intruder dimensions, without proposing new
> methods or actionable improvements based on these insights. As per the
> NeurIPS regulation, it may be better suited for the Datasets &
> Benchmarks track rather than the main conference.

We are confused by your suggestion to change tracks. First, under the
main track call for papers, the website states that NeurIPS "encourage
in-depth analysis of existing methods that provide new insights in terms
of their limitations or behaviour beyond the scope of the original
work". Moreover, this paper provides neither a dataset nor a benchmark.
Further, we do indeed provide actionable insights: we show several ways
that the number of intruder dimensions can be reduced, including with
certain initializations (Fig. 10), lower learning rates (Fig. 7), and
setting $`\alpha = 2r`$ (Section N). We even identify a simple
post-training intervention motivated by section 5: scaling down the
magnitude of intruder dimensions can reduce the forgetting a model has
while retaining nearly identical adaptation performance. These are
concrete prescriptions that can be used to train better LoRA models that
exhibit reduced forgetting and better out-of-distribution
generalization.

> Despite the extensive empirical evidence for the impact of intruder
> dimension, the theoretical explanation for why they appear in
> fine-tuning LoRA is still lacking.

We do provide a few mathematical justifications for the occurrence of
intruder dimensions in Section B.1. We showed that adding the outer
product of a random vector to a weight matrix introduces an intruder
dimension (Section B.1). This is analogous to the matrix product of BA
in LoRA when rank is 1, and implies that intruder dimensions occur
because the B and A vectors are uncorrelated to the columns/rows of the
weight matrix W_0. Further, we showed empirically that just training the
B matrix, while freezing the A matrix with all singular values of 1,
reduces the amplification of the singular values because of the matrix
product and therefore reduces the number of high ranking intruder
dimensions, aligning with our mathematical intuition.

> Mathematically, it is expected that constrained updates (e.g., via
> LoRA) tend to concentrate on directions associated with the largest
> singular values, while full fine-tuning distributes updates more
> evenly with sufficiently large update space. Without rigorous
> theoretical support, the introduction of the intruder dimension
> appears to be a restatement of this known behavior within the specific
> framing of LoRA.

What you describe as expected behavior actually doesn’t align with what
our observations. Rather than changes occurring primarily along the
existing high-ranking singular vectors, what happens is that a few new
high-ranking singular vectors emerge, while the original ones remain
largely unchanged or only slightly shifted. This suggests that the
dynamics introduced by constrained updates like LoRA are more nuanced
than simply concentrating on pre-existing dominant directions like you
describe. Whereas full fine-tuning distributes updates across all
singular vectors.

It is important to note that, before our work, it was not expected that
LoRA would update a weight matrix in a **structurally different** manner
than full fine-tuning. Rather, It was expected that LoRA approximated
full fine-tuning
(<a href="https://arxiv.org/pdf/2106.09685" target="_blank"
rel="noopener noreferrer">https://arxiv.org/pdf/2106.09685</a>), which
would result in the LoRA looking like an approximation of the fully
fine-tuned weight updates that is concentrated on the top existing
singular values, like you mention.

> Regarding the correlation between the number of intruder dimension and
> catastrophic forgetting (Lines 256-263), the experiments using a
> larger learning rate show that LoRA is not well optimized, as
> evidenced by a significant performance gap. In such cases, the
> increase in intruder dimensions could be a byproduct of suboptimal
> training rather than an inherent issue. For a well-optimized LoRA
> model (e.g., with lr=1e-4), the observation in \[1\] (LoRA learns less
> but forgets less) still holds. Thus, conclusions based on poorly
> trained models are less convincing.

We are concerned the reviewer is misinterpretting this section. All the
models are equally well trained (as measured by equal validation and
train loss) on the adaptation distribution--which are typical measures
used when finetuning models. Could you please explain which measure
would make one of the models sub-optimal over the other?

In the section you refer to, we sweep learning rates to see its effect
on optimization. We find that we get *similar* test loss but
significantly different forgetting profiles. For example, in Fig. 7, let
us examine LoRA with lr=1e-4 and 2e-4. We see that the models have
trained to within 0.5% test accuracy performance across epochs (middle
plot). However, we see that 2e-4 consistently forgets a much more than
1e-4 when examining their corresponding pseudo loss. These are the
datapoints that are used to indicate that intruder dimensions correlate
with forgetting. We infact support the conclusion of ("LoRA learns less
but forgets less") and extend it in this paper (lines 235-247).

Our main conclusions of our paper are all based on well trained models.
We welcome the reviewer to follow up and provide clarification if we
have misunderstood their concern. If there is another criterion by which
one model is considered suboptimal and is typically monitored at
finetuning, we would appreciate further clarification.

> In Appendix H.2, the observation that performance drops when learned
> knowledge is perturbed and LoRA's preserved knowledge is exaggerated
> is unsurprising. This behavior is consistent with general deep
> learning dynamics and does not convincingly support the claim that
> intruder dimensions degrade OOD performance. The argument feels more
> like an expected consequence of disrupting well-internalized features.

Your suggested intuition is actually opposite to our findings, making
this a surprising result. Our finding is not that "performance drops
when learned knowledge is perturbed", but rather that it is *barely*
changed when intruder dimensions are scaled down while forgetting has
decreased *significantly*. We provide a key passage in our main text
(lines 290-295) here:

"In one example for LLaMA2-7B fine-tuned on MetaMath with LoRA r = 256,
we observe that scaling the top intruder dimension in each matrix with
$`\lambda = 0.3`$ leads to a 0.1% drop in test accuracy and a 33.3% drop
in the forgetting induced by fine-tuning. In another for RoBERTa-base
fine-tuned on QQP, using $`\lambda = 0.7`$ leads to equivalent in test
accuracy and a 33.2% reduction in the forgetting induced by fine-tuning.
In certain scenarios, we even see test accuracy improve along with a
drop in forgetting. If we instead increase their contribution
($`\lambda > 1`$), we observe more forgetting."

This is an unexpected result. We welcome the reviewer to follow up and
provide clarification if we have misunderstood their concern.

> In Appendix I, was the hyperparameter search of learning rate
> conducted to ensure that LoRA with different alpha converged to
> similar test accuracy? If not, it's trivial that improper/insufficient
> learning naturally leads to low rank solutions.

Yes. We follow standard machine learning practice while conduct learning
rate sweeps and ensure that the different models compared converged to
similar test accuracy as shown in Table 2. In Appendix I, all models
train to similar test loss and accuracy, making our investigation fair
across ranks.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="CC5qId3DvK">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Rebuttal by Authors

</div>

<div class="heading">

#### Official Comment by Reviewer JRuL

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note CC5qId3DvK"></span><span class="sr-only">Copy
URL of note CC5qId3DvK</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Reviewer
JRuL</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>05 Aug 2025, 00:13 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=CC5qId3DvK)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thank you for the detailed response and I’ve read it carefully. I will
discuss with the other reviewers to update my score as the discussion
converges.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="1gfHV1SHDq">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Official Comment by Reviewer JRuL

</div>

<div class="heading">

#### Official Comment by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note 1gfHV1SHDq"></span><span class="sr-only">Copy
URL of note 1gfHV1SHDq</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>07 Aug 2025, 00:12 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=1gfHV1SHDq)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thank you again for your thoughtful review. We are happy to answer any
further questions you may have in case any come up.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="ArwW4LHC4H">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Rebuttal by Authors

</div>

<div class="heading">

#### Mandatory Acknowledgement by Reviewer JRuL

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note ArwW4LHC4H"></span><span class="sr-only">Copy
URL of note ArwW4LHC4H</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Mandatory
Acknowledgement</span><span class="signatures">by Reviewer
JRuL</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>15 Aug 2025, 10:58 (modified: 12 Nov 2025,
10:34)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=ArwW4LHC4H)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Mandatory Acknowledgement:** <span class="note-content-value">I have
read the author rebuttal and considered all raised points., I have
engaged in discussions and responded to authors., I have filled in the
"Final Justification" text box and updated "Rating" accordingly (before
Aug 13) that will become visible to authors once decisions are
released., I understand that Area Chairs will be able to flag up
Insufficient Reviews during the Reviewer-AC Discussions and shortly
after to catch any irresponsible, insufficient or problematic behavior.
Area Chairs will be also able to flag up during Metareview grossly
irresponsible reviewers (including but not limited to possibly
LLM-generated reviews)., I understand my Review and my conduct are
subject to Responsible Reviewing initiative, including the desk
rejection of my co-authored papers for grossly irresponsible behaviors.
<a
href="https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/"
rel="noopener noreferrer"
target="_blank">https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/</a></span>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="fVy0M3YRod">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission21986 by Reviewer 9SVt

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note fVy0M3YRod"></span><span class="sr-only">Copy
URL of note fVy0M3YRod</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
9SVt</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>23 Jun 2025, 15:49 (modified: 28 Oct 2025,
23:20)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=fVy0M3YRod)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

This work introduces the concept of intruder dimensions that arise
during fine-tuning of language models via low rank adaptation (LoRA).
Intruder dimensions are directions in the parameter space that disrupt
the original structure of pre-trained weights and strongly correlates
with forgetting of pre-trained knowledge. Furthermore, the authors point
out that fine-tuning via LoRA not always results in less forgetting and
strongly depends on the number of introduced intruder dimensions. Via
causal intervention, the authors verify that by scaling down singular
values associated with intruder dimensions leads to recovering
pre-trained knowledge while maintaining downstream performance.

</div>

</div>

<div>

**Strengths And Weaknesses:**

<div class="note-content-value markdown-rendered">

**Strengths**

The paper is well written and most of the claims are well supported.

Definition of a new concept that results from fine-tuning with low-rank
adaptation, which is very relevant.

The experiments are well designed with interesting results and
takeaways.

The key takeawas are based on multiple rigorous experiments, not just
single seeds/experiments.

**Weaknesses**

I am generally very positive about this work, the following weaknesses
are minor issues.

**Unsupported claims**

There are a few claims that are not entirely supported:

- line 166: "LoRA introduces new singular vectors that have a large
  contribution to the norm of the updated parameter matrix." - As far as
  I can see there is no support for a change in norm, only in directions
  because of using cosine similarity.
- line 210: "LoRA consistently has more intruder dimensions than full
  fine-tuning" - not entirely correct, see Appendix J Figure 15, LoRA
  r=64 has less intruder dimensions than full fine-tuning
- line 244: "This extends the finding that LoRA learns less " - I assume
  this is a typo and should be "forgets less" rather than "learns less",
  as there is no support for the learning less argument as far as I can
  see

**Different distance measures**

Currently, the authors only consider cosine similarity as distance
measures between singular vectors. It would be interesting if intruder
dimensions could be identified via different distance measures, or
whether it is really only the difference in directions that enables
identifying them.

**Potential additional experiments**

It is very interesting that initialization has a huge effect on intruder
dimensions (Figure 23). It would be interesting how the number of
intruder dimensions vary for different initialization schemes,
specifically for data-driven initialization, e.g. \[1,2,3\].

\[1\] Wang et al., LoRA-GA: Low-Rank Adaptation with Gradient
Approximation, NeurIPS 2024

\[2\] Paischer et al., One Initialization to Rule them All: Fine-tuning
via Explained Variance Adaptation, NeurIPS 2024 ENLSP Workshop

\[3\] Yang et al., CorDA: Context-Oriented Decomposition Adaptation of
Large Language Models for Task-Aware Parameter-Efficient Fine-tuning,
NeurIPS 2024

Another very interesting point would be showing where exactly the
intruder dimensions appear in. Are they only apparent in certain
projection matrices (Q/K/V, etc.) and layers and can they be related
with implicit biases as introduced in \[4\]?

\[4\] Sun et al., Massive Activations in Large Language Models, COLM
2024

Finally, it would be interesting how the number of intruder dimensions
changes with larger models.

**Presentation**

Generally, I recommend increasing the fontsize of the plots, they are
partly a bit hard too read without zooming in a lot. It would be cool to
have a zoomed in version of the top left part of Figure 1, right, to
better identify intruder dimensions (same for Figure 2b). There are
typos in line 188 and 202.

</div>

</div>

<div>

**Quality:** <span class="note-content-value">4: excellent</span>

</div>

<div>

**Clarity:** <span class="note-content-value">3: good</span>

</div>

<div>

**Significance:** <span class="note-content-value">3: good</span>

</div>

<div>

**Originality:** <span class="note-content-value">3: good</span>

</div>

<div>

**Questions:**

<div class="note-content-value markdown-rendered">

See weaknesses.

</div>

</div>

<div>

**Limitations:**

<div class="note-content-value markdown-rendered">

Some limitations are in the appendix, but they could be a bit more
elaborate. To accomodate those in the main body of the paper, I suggest
moving Algorithm 1 in the appendix, it does not add too much value.

</div>

</div>

<div>

**Rating:** <span class="note-content-value">5: Accept: Technically
solid paper, with high impact on at least one sub-area of AI or
moderate-to-high impact on more than one area of AI, with
good-to-excellent evaluation, resources, reproducibility, and no
unaddressed ethical considerations.</span>

</div>

<div>

**Confidence:** <span class="note-content-value">5: You are absolutely
certain about your assessment. You are very familiar with the related
work and checked the math/other details carefully.</span>

</div>

<div>

**Ethical Concerns:** <span class="note-content-value">NO or VERY MINOR
ethics concerns only</span>

</div>

<div>

**Paper Formatting Concerns:**

<div class="note-content-value markdown-rendered">

No major issues.

</div>

</div>

<div>

**Code Of Conduct Acknowledgement:**
<span class="note-content-value">Yes</span>

</div>

<div>

**Responsible Reviewing Acknowledgement:**
<span class="note-content-value">Yes</span>

</div>

<div>

**Final Justification:**

<div class="note-content-value markdown-rendered">

My initial rating of the paper was already positive and all my raised
points have been addressed during the rebuttal. I see this work having
an impact in understanding and improving LoRA-style fine-tuning,
therefore I reside with my original rating.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="jTJEsPyxgT">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Rebuttal by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note jTJEsPyxgT"></span><span class="sr-only">Copy
URL of note jTJEsPyxgT</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>30 Jul 2025, 20:31 (modified: 29 Oct 2025,
01:12)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=jTJEsPyxgT)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

We thank you for your thoughtful review and are grateful for your
positivity about our work. We have incorporated the suggestions you have
requested, including typos, presentation changes, and updating claims,
into the updated manuscript. We provide responses and more context where
required below:

> There are a few claims that are not entirely supported: line 166:
> "LoRA introduces new singular vectors that have a large contribution
> to the norm of the updated parameter matrix." - As far as I can see
> there is no support for a change in norm, only in directions because
> of using cosine similarity.

Thank you for pointing this out. While Figure 5 shows an increase in the
singular value corresponding to the intruder dimension—suggesting a
potential increase in the matrix norm—we currently don't report the norm
itself. We will include this statistic in future revisions to more
concretely support the claim.

> Different distance measures Currently, the authors only consider
> cosine similarity as distance measures between singular vectors. It
> would be interesting if intruder dimensions could be identified via
> different distance measures, or whether it is really only the
> difference in directions that enables identifying them.

Thank you for the suggestion; it's a valuable point. While we focus on
cosine similarity, we don’t claim it's the only viable measure. We chose
it because, in the context of SVD, comparing singular vectors via cosine
similarity offers a natural and interpretable way to track changes in
the principal directions of the weight matrix. Since these directions
define how the matrix transforms input, shifts in them are meaningful.
That said, exploring alternative distance measures is an interesting
direction for future work and could offer complementary insights.

> Another very interesting point would be showing where exactly the
> intruder dimensions appear in. Are they only apparent in certain
> projection matrices (Q/K/V, etc.) and layers and can they be related
> with implicit biases as introduced in \[4\]?

Across our experiments, intruders appear in all weight matrices, with no
systematic difference in the number of intruder dimensions between MLP
and attention matrices. We will add in a section outlining these details
to the appendix.

> It would be interesting how the number of intruder dimensions changes
> with larger models.

We thank the reviewer for there extensive suggestions of additional
experiments. Due to the limited length of the rebuttal, we will leave
these experiments to future work to extend this work.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="Zs18ZtqRTi">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Rebuttal by Authors

</div>

<div class="heading">

#### Official Comment by Reviewer 9SVt

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note Zs18ZtqRTi"></span><span class="sr-only">Copy
URL of note Zs18ZtqRTi</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Reviewer
9SVt</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>01 Aug 2025, 02:39 (modified: 29 Oct 2025,
03:09)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=Zs18ZtqRTi)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thank you for answering all of my raised points. Since I was already
positive about the paper at the time of submission, I will retain my
positive rating.

</div>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

<div class="note depth-even" data-id="w6KVgInFUL">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="parent-title">

##### <span class="glyphicon glyphicon-share-alt" aria-hidden="true"></span> Replying to Rebuttal by Authors

</div>

<div class="heading">

#### Mandatory Acknowledgement by Reviewer 9SVt

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note w6KVgInFUL"></span><span class="sr-only">Copy
URL of note w6KVgInFUL</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Mandatory
Acknowledgement</span><span class="signatures">by Reviewer
9SVt</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>01 Aug 2025, 02:39 (modified: 12 Nov 2025,
10:34)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=w6KVgInFUL)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Mandatory Acknowledgement:** <span class="note-content-value">I have
read the author rebuttal and considered all raised points., I have
engaged in discussions and responded to authors., I have filled in the
"Final Justification" text box and updated "Rating" accordingly (before
Aug 13) that will become visible to authors once decisions are
released., I understand that Area Chairs will be able to flag up
Insufficient Reviews during the Reviewer-AC Discussions and shortly
after to catch any irresponsible, insufficient or problematic behavior.
Area Chairs will be also able to flag up during Metareview grossly
irresponsible reviewers (including but not limited to possibly
LLM-generated reviews)., I understand my Review and my conduct are
subject to Responsible Reviewing initiative, including the desk
rejection of my co-authored papers for grossly irresponsible behaviors.
<a
href="https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/"
rel="noopener noreferrer"
target="_blank">https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/</a></span>

</div>

</div>

</div>

<div class="ForumReplyForm_container__w2VwR">

<div class="ForumReplyForm_buttons__eGXo_">

<span class="ForumReplyForm_hint__azwm6">Add:</span>

Public Comment

</div>

</div>

</div>

</div>

</div>

</div>

</div>

</div>

</div>

</div>

</div>
