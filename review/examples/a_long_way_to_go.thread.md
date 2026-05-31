<div id="content" role="main">

<div class="Forum_forum__wS8Fw">

<div class="forum-container">

<div class="forum-note">

<div style="height: auto; overflow: visible;">

<div class="btn-group">

2 Versions <span class="caret"></span>

- [COLM (July 10, 2024)](https://openreview.net/forum?id=G8LaO1P0xv)
- [Submitted to ICLR 2024 (September 23,
  2023)](https://openreview.net/forum?id=sNtDKdcI1f)

</div>

</div>

<div class="forum-title mt-2 mb-2">

## A Long Way to Go: Investigating Length Correlations in RLHF

<div class="forum-content-link">

<a href="https://openreview.net/pdf?id=G8LaO1P0xv"
class="citation_pdf_url" target="_blank" rel="noreferrer"
title="Download PDF"><img
src="./A%20Long%20Way%20to%20Go_%20Investigating%20Length%20Correlations%20in%20RLHF%20_%20OpenReview_files/pdf_icon_blue.svg"
alt="Download PDF" /></a>

</div>

</div>

<div class="forum-authors mb-2">

### <a href="https://openreview.net/profile?id=~Prasann_Singhal1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Prasann_Singhal1">Prasann Singhal</a>, <a href="https://openreview.net/profile?id=~Tanya_Goyal1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Tanya_Goyal1">Tanya Goyal</a>, <a href="https://openreview.net/profile?id=~Jiacheng_Xu2"
data-toggle="tooltip" data-placement="top"
data-original-title="~Jiacheng_Xu2">Jiacheng Xu</a>, <a href="https://openreview.net/profile?id=~Greg_Durrett1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Greg_Durrett1">Greg Durrett</a> 

</div>

<div class="clearfix mb-1">

<div class="forum-meta">

<span class="date item"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>Published: 10 Jul 2024, Last Modified: 25 Aug
2024</span><span class="item"><span class="glyphicon glyphicon-folder-open"
aria-hidden="true"></span>COLM</span><span class="readers item"
toggle="tooltip" placement="top" title=""
original-title="Visible to &lt;br/&gt;everyone&lt;br/&gt;since 25 Aug 2024"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="item"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=G8LaO1P0xv)</span><span class="item"><span class="glyphicon glyphicon-bookmark"
aria-hidden="true"></span><a href="https://openreview.net/forum?id=G8LaO1P0xv#"
data-target="#bibtex-modal" data-toggle="modal"
data-bibtex="%40inproceedings%7B%0Asinghal2024a%2C%0Atitle%3D%7BA%20Long%20Way%20to%20Go%3A%20Investigating%20Length%20Correlations%20in%20%7BRLHF%7D%7D%2C%0Aauthor%3D%7BPrasann%20Singhal%20and%20Tanya%20Goyal%20and%20Jiacheng%20Xu%20and%20Greg%20Durrett%7D%2C%0Abooktitle%3D%7BFirst%20Conference%20on%20Language%20Modeling%7D%2C%0Ayear%3D%7B2024%7D%2C%0Aurl%3D%7Bhttps%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DG8LaO1P0xv%7D%0A%7D">BibTeX</a></span><span class="item"><span class="glyphicon glyphicon-copyright-mark"
aria-hidden="true"></span><a href="https://creativecommons.org/licenses/by/4.0/" target="_blank"
rel="noopener noreferrer" data-toggle="tooltip" data-placement="top"
data-original-title="Licensed under Creative Commons Attribution 4.0 International">CC
BY 4.0</a></span>

</div>

<div class="invitation-buttons">

</div>

</div>

<div class="note-content">

<div>

**Research Area:** <span class="note-content-value">Alignment, Data,
Evaluation, Learning algorithms for LMs</span>

</div>

<div>

**Keywords:** <span class="note-content-value">Natural Language
Processing, Large Language Models, RLHF, Reward Hacking</span>

</div>

<div>

**TL;DR:** <span class="note-content-value">Many of the gains in open
RLHF work, particularly due to flaws in reward modeling, are
attributable to length</span>

</div>

<div>

**Abstract:**

<div class="note-content-value markdown-rendered">

Great success has been reported using Reinforcement Learning from Human
Feedback (RLHF) to align large language models, with open preference
datasets enabling wider experimentation, particularly for "helpfulness"
in tasks like dialogue and web question answering. Alongside these
improvements, however, RLHF also often drives models to produce longer
outputs. This paper demonstrates, on three diverse settings, that
optimizing for response length is, much more than previously thought, a
significant factor behind RLHF. Studying the strategies RL optimization
uses to maximize reward, we find improvements in reward to largely be
driven by increasing response length, instead of other features. Indeed,
we find that even a *purely* length-based reward reproduces most
downstream RLHF improvements over supervised fine-tuned models. Testing
a comprehensive set of length-countering interventions, we identify the
dominant source of these biases to be reward models, which, by studying
training dynamics, we find are non-robust and easily influenced by
length biases in preference data.

</div>

</div>

<div>

**Code Of Ethics:** <span class="note-content-value">I acknowledge that
I and all co-authors of this work have read and commit to adhering to
the COLM Code of Ethics on
<a href="https://colmweb.org/CoE.html" rel="noopener noreferrer"
target="_blank">https://colmweb.org/CoE.html</a></span>

</div>

<div>

**Author Guide:** <span class="note-content-value">I certify that this
submission complies with the submission instructions as described on
<a href="https://colmweb.org/AuthorGuide.html" rel="noopener noreferrer"
target="_blank">https://colmweb.org/AuthorGuide.html</a></span>

</div>

<div>

**Submission Number:** <span class="note-content-value">865</span>

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
src="./A%20Long%20Way%20to%20Go_%20Investigating%20Length%20Correlations%20in%20RLHF%20_%20OpenReview_files/linear_icon.svg"
title="Linear discussion layout" class="icon" data-toggle="tooltip"
alt="back arrow" /><span class="sr-only">Linear</span>

<img
src="./A%20Long%20Way%20to%20Go_%20Investigating%20Length%20Correlations%20in%20RLHF%20_%20OpenReview_files/threaded_icon.svg"
title="Threaded discussion layout" class="icon" data-toggle="tooltip"
alt="back arrow" /><span class="sr-only">Threaded</span>

<img
src="./A%20Long%20Way%20to%20Go_%20Investigating%20Length%20Correlations%20in%20RLHF%20_%20OpenReview_files/nested_icon.svg"
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

*16 / 16 replies shown*

</div>

</div>

</div>

<div class="row forum-replies-container layout-default">

<div class="col-xs-12">

<div id="forum-replies">

<div class="note depth-odd" data-id="Af00TYoQpj">

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
original-title="Copy URL of note Af00TYoQpj"></span><span class="sr-only">Copy
URL of note Af00TYoQpj</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 255, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Decision</span><span class="signatures">by
Program Chairs</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>10 Jul 2024, 06:13 (modified: 25 Aug 2024,
20:55)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=Af00TYoQpj)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Decision:** <span class="note-content-value">Accept</span>

</div>

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

This paper investigates the influence of RLHF on features of
LM-generated text, particularly focusing on how it influences length of
generated responses (in particular, causing an increase in length). The
experiments suggest that reward improvements are primarily due to
increases in length, e.g., by using response length as a reward and
finding most improvements from RLHF are retained after RL-based
fine-tuning, and find that this is mostly due to trained reward models
learning a strong length bias (i.e., assigning higher reward to longer
responses).

The experiments are well-designed, and the insights this paper provides
are really important given that RL-based finetuning (especially RLHF) is
currently the dominant paradigm in training LMs.

I suggest the authors incorporate feedback and suggestions from the
reviewers.

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="YVSCQXvczy">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### **Discussion period is now open**

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note YVSCQXvczy"></span><span class="sr-only">Copy
URL of note YVSCQXvczy</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Area Chair
4ZkL</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>02 Jun 2024, 15:09 (modified: 29 Aug 2024,
09:36)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=YVSCQXvczy)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Hi reviewers, please take a look at the author's rebuttals and the other
reviews for this paper!

If the rebuttals addressed your concerns, please let the authors know
about this and update your review. If not, please continue to engage
with the authors and the other reviewers in the discussion forum.

Three of our five reviewers are very positive about this paper, and Bn2p
and WB4w are less positive.

Bn2p:

- One of your concerns was that experiments were not performed on larger
  models; however, the authors conducted these experiments during the
  rebuttal period. Does this address that concern?
- You also were concerned about the choices of datasets. Do you find the
  authors' response addresses this concern?

WB4w:

- Your question about different models is also addressed by the authors.
- Your concerns about length bias correlation with tasks, as well as
  concerns about human preferences, also received a response from the
  authors. Do you find this addresses your concerns?

For the other reviewers, would you like to argue for this paper's
acceptance?

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="Zd5f8810Q1">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission865 by Reviewer rdjK

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note Zd5f8810Q1"></span><span class="sr-only">Copy
URL of note Zd5f8810Q1</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
rdjK</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>24 May 2024, 04:38 (modified: 25 Aug 2024,
20:55)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=Zd5f8810Q1)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

In this paper, the authors present a thorough investigation into how
RLHF tends to optimize for longer output lengths, often at the expense
of true quality improvements. They study this phenomenon across three
diverse task settings - WebGPT, Stack, and RLCD. On WebGPT and RLCD, the
bulk of the reward improvement from RLHF can be attributed simply to
increases in output length, with little gain coming from optimizing
other features. Attempts to mitigate the length bias through various
interventions in the RLHF pipeline, such as adjusting the reward
function, policy rollouts, KL loss, etc. do reduce length increases to
some degree, but fail to eliminate the bias and often hurt overall
performance. Overall, this is a well-executed and insightful study that
reveals some concerning flaws in current RLHF approaches. The authors
convincingly show that a significant portion of recently reported
"progress" may be illusory and simply due to producing longer outputs.
The finding that even preference datasets balanced by length still
produce biased reward models is particularly notable. I believe this
paper is great with its originality and rigorous experiment setting.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

- The experiments are quite neat. I truly appreiciate the rigorous
  experiment pipeline to understand the underlying mechanism for rlhf.
  The overall 'causal' experiments are quite convincing, make it a
  strong candidate for the venue.
- Insightful Analysis. Beyond just showing the existence of length
  biases, the paper provides a detailed look into their causes and the
  training dynamics involved. The finding that biases stem from the
  preference data itself and remain even after balancing attempts is an
  important insight with implications for data collection practices.

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

I enjoy reading this paper but the only confusing section for me is Sec
3.1. I can not get why non-length reward gain (NRG) is calculated as
average ∆R within each bucket weighted by the number of examples in each
bucket. What does this stand for? For me the most intuitive way to
investigate the correlation between reward gained by ppo and its length
is just to calculate $`\mathrm{\Delta}R/\mathrm{\Delta}L`$, where
$`\mathrm{\Delta}L`$ is the length difference between sft model and ppo
model.

</div>

</div>

<div>

**Questions To Authors:**

<div class="note-content-value markdown-rendered">

- For experiment setup in Section 3.1, I wonder is sft output or ppo
  output you would stratify based on the length, or combined?
- For Table 2, Could you please win/tie/lose rate in addition to win
  rate only? I believe this could strengthen the performance comparision
  between models.
- For Table 3, why std ppo has failed reward optimization? As it has
  length and reward, I think it should be accessed by sim perf also?

</div>

</div>

<div>

**Rating:** <span class="note-content-value">8: Top 50% of accepted
papers, clear accept</span>

</div>

<div>

**Confidence:** <span class="note-content-value">4: The reviewer is
confident but not absolutely certain that the evaluation is
correct</span>

</div>

<div>

**Ethics Flag:** <span class="note-content-value">No</span>

</div>

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="Zjiy6o9KIV">

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
original-title="Copy URL of note Zjiy6o9KIV"></span><span class="sr-only">Copy
URL of note Zjiy6o9KIV</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>28 May 2024, 17:42 (modified: 28 Aug 2024,
22:32)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=Zjiy6o9KIV)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

Thanks for the thoughtful comments! We’ll try to address the questions
below:

> I enjoy reading this paper but the only confusing section for me is
> Sec 3.1… most intuitive way is to just calculate ∆𝑅/∆𝐿

This is a great question! We’ll try to clarify here.

Intuitively, we wanted to compute reward increase $`\Delta R`$ when
$`\Delta L`$ = 0. Because these cases are quite rare, our approximation
of this is to compute the $`\Delta R`$ within each length-stratified
bucket, as $`\Delta L`$ within buckets is quite small. We take an
average of this across buckets, weighted by the number of instances
(SFT + PPO combined) in each bucket.

We like the suggestion of reporting $`\Delta R`$/$`\Delta L`$ or
correlation between $`\Delta R`$ and $`\Delta L`$! One slight limitation
of $`\Delta R`$/$`\Delta L`$ is that it changes the units and is
therefore not comparable to $`\Delta R`$ in the table. Furthermore, if
we computed $`\Delta R`$/$`\Delta L`$ (assuming you meant
$`\sum\limits_{i}(\Delta R)/\sum\limits_{i}(\Delta L)`$), it could be
the case that length increases and reward increases, but these are
unconnected. Controlling for length with our approach helps disentangle
this. Note that we already report corr(𝑅, 𝐿) within a batch in Table 4.

> For experiment setup in Section 3.1, I wonder is sft output or ppo
> output you would stratify based on the length, or combined?

We stratify based on length here, where we compare all the outputs from
SFT that are in some length range (e.g. 20-40 tokens) with all the PPO
outputs in the same length range (20-40 tokens). Note that the overall
prompt set is the same, but each bin may not have comparable prompts
between SFT and PPO.

> For Table 2, Could you please win/tie/lose rate in addition to win
> rate only? I believe this could strengthen the performance comparison
> between models.

Our prompts (same as AlpacaFarm) ask GPT-4 to return one model name as
output, corresponding to the better output. Because of this we rarely
observe ties, thus the lose rate is just the reverse of win-rate.

> For Table 3, why std ppo has failed reward optimization? As it has
> length and reward, I think it should be accessed by sim perf also?

Thanks for pointing this out! Standard PPO doesn’t have failed reward
optimization; we used a - in the sim pref column for it because it is
the reference point. We can replace those numbers with 50% for better
clarity.

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="qLU2tvpQRk">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission865 by Reviewer kVck

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note qLU2tvpQRk"></span><span class="sr-only">Copy
URL of note qLU2tvpQRk</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
kVck</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>23 May 2024, 13:58 (modified: 25 Aug 2024,
20:55)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=qLU2tvpQRk)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

This paper presents a well-scoped investigation of Reinforcement
Learning with Human Feedback (RLHF), examining an observed tendency to
prefer longer outputs and how it may be mitigated. The paper is concise
and convincing, highlighting an important and overlooked feature of
RLHF, by questioning what our preferences actually encourage in the
generated outputs. This is an important step toward better evaluation of
and methods for alignment techniques such as RLHF.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

To study the effect of length in RLHF, the authors investigate a
satisfactory span of data: a long form and a short form question
answering dataset (WebGPT, Stack), and a multi-turn dialogue dataset
(RLCD). They provide sufficient, detailed information about the models
and hyperparameters used. Their argument is well-structured, each step
culminating with empirical results. First, the paper presents an
empirical justification that there exists a difference in preferred
length with Figure 3. Then, an investigation of how much the length
explains the reward with Table 1. Finally, a study of what interventions
can mitigate rewarding long responses supported by figures and tables
throughout section 4.

I believe this is an important paper, because it questions and
systematically observes a flaw in a method that otherwise appears
ubiquitously applied. The paper investigates whether and to what degree
a popular method, Reinforcement Learning using Human Feedback (RLHF), is
susceptible to producing text generations that are longer but not truly
better. It is important to question contemporary, widely used
techniques, and this paper does so in a principled way. This will be
useful as a step towards further critiquing RLHF, and refining alignment
methods.

The interventions proposed by the paper are reasonable and
comprehensive. They bring clarity to the different parts of the RLHF
components and how each can be influenced, and how effective this is.

The paper is clear and concise, explaining their decisions succinctly
and motivating them well. Their background section had just enough
detail to be relevant without extraneous information. Their claim and
findings are clear in the abstract, and I found it easy to keep them in
mind while reading the rest of the paper. I did not notice inconsistent
reasoning.

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

I did not understand the last sentence of 2.1, "To control for length
bias, we focus on results where shorter outputs show greater win-rates".
Specifically, how this was done throughout the paper (is it distinct
from showing that shorter outputs are less frequently associated with
greater win-rates?) and why this is a representative way to support the
argument.

</div>

</div>

<div>

**Rating:** <span class="note-content-value">9: Top 15% of accepted
papers, strong accept</span>

</div>

<div>

**Confidence:** <span class="note-content-value">3: The reviewer is
fairly confident that the evaluation is correct</span>

</div>

<div>

**Ethics Flag:** <span class="note-content-value">No</span>

</div>

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="oIUksoiMGz">

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
original-title="Copy URL of note oIUksoiMGz"></span><span class="sr-only">Copy
URL of note oIUksoiMGz</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>28 May 2024, 17:43 (modified: 28 Aug 2024,
22:32)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=oIUksoiMGz)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

Thanks for the thoughtful comments!

> the last sentence of 2.1…

Thanks for pointing this out! We mainly mentioned this to qualify the
win-rate preference results, since these may themselves have an
undesired length-bias (e.g. STACK on SFT-LONG in Table 2), which has
been supported in other recent work as well \[1\]. We will clarify this
more in future versions.

\[1\] Yann Dubois, Balazs Galambosi, Percy Liang, and Tatsunori B.
Hashimoto. LengthCorrected AlpacaEval: A Simple Debiasing of Automatic
Evaluators. <a href="https://github/" target="_blank"
rel="noopener noreferrer">https://github</a>. com/tatsu-lab/alpaca_eval,
2024.

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="3aQclhRQG5">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission865 by Reviewer Bn2p

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note 3aQclhRQG5"></span><span class="sr-only">Copy
URL of note 3aQclhRQG5</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
Bn2p</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>14 May 2024, 09:41 (modified: 25 Aug 2024,
20:55)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=3aQclhRQG5)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

The paper investigates the relationship between output length and
performance in Reinforcement Learning from Human Feedback (RLHF) for
large language models. The authors demonstrate across three diverse
settings (WebGPT, Stack, and RLCD) that RLHF tends to significantly
increase output length, and a substantial portion of the reported
improvements in reward can be attributed to this length increase rather
than other meaningful quality improvements. Through controlled
experiments, the authors show that optimizing for length alone (using a
length-based reward) can reproduce most of the gains seen in standard
RLHF. They explore various interventions to mitigate the length bias,
including adjustments to the reward model training, preference data, and
PPO optimization, but find that length biases persist across these
interventions. Further analysis reveals that reward models exhibit
strong correlations with length, potentially due to overfitting to a
small set of "easy" length-biased examples during training.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

- The paper provides a comprehensive analysis of the relationship
  between output length and RLHF performance, which has been noted but
  not thoroughly studied in prior works. The experimental setup covers
  diverse settings and tasks, lending robustness to the findings.
- The analysis is multi-faceted, involving controlled experiments,
  interventions, and in-depth investigation of training dynamics, and
  the paper proposes practical interventions and evaluation techniques
  (e.g., NRG) to mitigate and assess length biases in RLHF.
- I think the findings are meaningful and call for more careful
  consideration of preference data quality, reward model robustness, and
  evaluation metrics in RLHF research.

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

- The chosen evaluation datastes are limited and more results on general
  benchmarks like MT-bench or AlpacaEval 2.0 can make the results more
  convincing. Also, it would be interesting to see the performance
  change on evaluations that does not rely on the lengths of the outputs
  (like MMLU).
- Due to the computational resources, the authors did not conduct
  experiments on larger models, and the the PPO training is based on
  LoRA. Even without using a larger model, the authors can try the
  experimental results on more powerful small models (such as mistral,
  llama-2 and llama-3).

</div>

</div>

<div>

**Questions To Authors:**

<div class="note-content-value markdown-rendered">

- In page 7, Reesults -\> Results.

</div>

</div>

<div>

**Rating:** <span class="note-content-value">6: Marginally above
acceptance threshold</span>

</div>

<div>

**Confidence:** <span class="note-content-value">4: The reviewer is
confident but not absolutely certain that the evaluation is
correct</span>

</div>

<div>

**Ethics Flag:** <span class="note-content-value">No</span>

</div>

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="JljKoCJAo6">

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
original-title="Copy URL of note JljKoCJAo6"></span><span class="sr-only">Copy
URL of note JljKoCJAo6</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>28 May 2024, 17:45 (modified: 28 Aug 2024,
22:32)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=JljKoCJAo6)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

Thanks for the thoughtful comments! We address the main concerns here:

> larger models

We conducted additional experiments with scaling up our RM experiments
up to LLaMA-2 13B. We report intrinsic reward modeling accuracy results
below (compare with Table 4 in the paper):

| Model       | WebGPT | Stack | RLCD  |
|-------------|--------|-------|-------|
| LLaMA 7B    | 61.5%  | 70%   | 80%   |
| LLaMA-2 13B | 64.5%  | 71.3% | 81.2% |

On these datasets, reward modeling accuracy is only marginally better
than it was before, suggesting that increasing model scale isn’t
necessarily the main bottleneck on these tasks.

We will add these experiments in any future version.

> The chosen evaluation datasets are limited and more results on general
> benchmarks like MT-bench or AlpacaEval 2.0 can make the results more
> convincing

This is a fair point. We aim to cover a fairly diverse set of realistic
tasks (multi-turn dialogue QA, single-turn long-form QA, and technical
question answering) to demonstrate our findings on publicly available
preference datasets at the time the work was conducted. We do note that
the pairwise preference metric we report (setting specific) is based on
GPT-4 evaluation which is used on MT-Bench and alpaca-eval and thus
should give a reasonable approximation to what one may find in those
settings, the main difference being the prompt set.

As for MMLU, we think this is a nice suggestion, but the fine-tuning
recipes (datasets used, SFT, etc.) necessary to induce good
chain-of-thought-based question answering are a bit different than the
recipes for the kind of LLM alignment we study here.

</div>

</div>

</div>

</div>

</div>

<div class="note depth-even" data-id="3ZXtsi2ejE">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Comment by Area Chair 4ZkL

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note 3ZXtsi2ejE"></span><span class="sr-only">Copy
URL of note 3ZXtsi2ejE</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Area Chair
4ZkL</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>04 Jun 2024, 18:59 (modified: 29 Aug 2024,
09:36)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=3ZXtsi2ejE)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Hi Bn2p, can you check the authors' response and update your review if
it addressed your concern (or participate in discussion with the authors
if it did not)?

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="X4MKlneuwd">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission865 by Reviewer WB4w

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note X4MKlneuwd"></span><span class="sr-only">Copy
URL of note X4MKlneuwd</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
WB4w</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>10 May 2024, 23:17 (modified: 25 Aug 2024,
20:55)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=X4MKlneuwd)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

This paper explores the tendency of Reinforcement Learning from Human
Feedback (RLHF) to produce longer outputs when aligning LLMs with
desired properties, such as helpfulness in dialogue and web question
answering tasks. The authors delve into the strategies RL optimization
employs to maximize reward and find that improvements are often driven
by increasing response length rather than other features.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

- The length bias issue is worth a deep investigation. The final
  conclusion of reward modeling is well-supported by experiments.
- The controlled experiments on length-only reward clearly show length
  shortcuts.

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

- Would the length bias issue correlate with tasks? The selected
  datasets are question answering and chatting, which may naturally
  bring the length bias issue. I suggest selecting some length-unrelated
  tasks such as math, coding, or length-constrained instructions for a
  more comprehensive evaluation.
- In my opinion, before intervening in the RLHF process to mitigate the
  lengthy outputs, it is more important to recognize whether human
  annotators and LLM judges commonly prefer lengthy outputs, and when it
  is a bias.

</div>

</div>

<div>

**Questions To Authors:**

<div class="note-content-value markdown-rendered">

- Would developing more capable reward models benefit this issue? With
  limited GPU resources, the authors could at least evaluate whether
  top-ranked models on RewardBench (e.g. Eurus-RM-7B and
  Starling-RM-34B) still prefer longer responses.

</div>

</div>

<div>

**Rating:** <span class="note-content-value">6: Marginally above
acceptance threshold</span>

</div>

<div>

**Confidence:** <span class="note-content-value">4: The reviewer is
confident but not absolutely certain that the evaluation is
correct</span>

</div>

<div>

**Ethics Flag:** <span class="note-content-value">No</span>

</div>

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="4sxAHpskD1">

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
original-title="Copy URL of note 4sxAHpskD1"></span><span class="sr-only">Copy
URL of note 4sxAHpskD1</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>28 May 2024, 17:47 (modified: 28 Aug 2024,
22:32)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=4sxAHpskD1)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

Thanks for the thoughtful comments! We’ll try to address concerns here.

> more capable models

We conducted additional experiments with scaling up our RM experiments
up to LLaMA-2 13B. We report intrinsic reward modeling accuracy results
below (compare with Table 4 in the paper):

| Model       | WebGPT | Stack | RLCD  |
|-------------|--------|-------|-------|
| LLaMA 7B    | 61.5%  | 70%   | 80%   |
| LLaMA-2 13B | 64.5%  | 71.3% | 81.2% |

On these datasets, reward modeling accuracy is only marginally better
than it was before, suggesting that increasing model scale isn’t
necessarily the main bottleneck on these tasks. While the settings are a
bit different, we will definitely consider extending our study to
evaluate other models in future work.

> length-unrelated tasks such as math, coding, length-constrained
> instructions…

This is a great point. It’s worth noting that we actually already have
such a setting (the Stack setting) which we chose because it represents
a realistic setting where math, coding and more complex reasoning may be
used. Indeed, we do note this setting to demonstrate most of the
discussed length-correlated behavior throughout the RLHF pipeline,
though to a lesser degree.

> Would the length bias issue correlate with tasks?” “In my opinion,
> before intervening in the RLHF process to mitigate the lengthy
> outputs, it is more important to recognize whether human annotators
> and LLM judges commonly prefer lengthy outputs

We agree with this point and discuss it in the paper (introduction,
3.1): length may be a valid feature corresponding to quality for humans.
Two of the datasets we run experiments on, Stack and WebGPT, use human
preference labels and these labels do show length biases (Table 5).
However, these are fairly small, suggesting that length may not be a
dominant feature for human annotation. These small length biases are not
proportional to the dominance of length we find experimentally after
RLHF on these same settings, which is particularly concerning given how
commonplace these techniques / evaluations are. Regardless of whether
length is a good or bad feature, we find it concerning if it comes at
the cost of other potential good features.

</div>

</div>

</div>

</div>

</div>

<div class="note depth-even" data-id="CqkTM7q4pE">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Comment by Area Chair 4ZkL

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note CqkTM7q4pE"></span><span class="sr-only">Copy
URL of note CqkTM7q4pE</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Area Chair
4ZkL</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>04 Jun 2024, 18:59 (modified: 29 Aug 2024,
09:36)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=CqkTM7q4pE)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Hi WB4w, can you check the authors' response and update your review if
it addressed your concern (or participate in discussion with the authors
if it did not)?

</div>

</div>

</div>

</div>

</div>

<div class="note depth-even" data-id="Ix1mu6xs7L">

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

#### **Reply**

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note Ix1mu6xs7L"></span><span class="sr-only">Copy
URL of note Ix1mu6xs7L</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Reviewer
WB4w</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>05 Jun 2024, 06:16 (modified: 29 Aug 2024,
09:36)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=Ix1mu6xs7L)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thanks for the reply. The additional results are great. I will increase
my score.

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="Md4kFWM9TH">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission865 by Reviewer zdy6

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note Md4kFWM9TH"></span><span class="sr-only">Copy
URL of note Md4kFWM9TH</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
zdy6</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>04 May 2024, 07:19 (modified: 25 Aug 2024,
20:55)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=Md4kFWM9TH)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

This paper reveals an issue regarding the verbosity of large language
models (LLMs). Specifically, the authors demonstrate that the response
length is a significant factor behind RLHF. The authors further find
that even a purely length-based reward reproduces most downstream RLHF
improvements over supervised fine-tuned models. Testing a comprehensive
set of length-countering interventions, this paper identifies the
dominant source of the biases for the reward models. The main
contribution of this paper is to reveal the vulnerability of the RLHF
process.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

I think this paper does an excellent job of identifying the bias
regarding the output length of LLMs.

**Clarity.** This paper is well-written and organized. I think this
paper is easy to follow.

**Importance.** Evaluation of LLMs is a very timely and important topic
in the field of AI. Verboisity bias has been highlighted in several
concurrent works (please see the specific paper titles in the Reason to
Reject section). This paper investigates such bias in LLMs in detail. I
think the empirical analysis of this paper is quite convincing, and
there is little room for objection.

**Empirical experiments.** I think the empirical experiments are
well-designed. The results are consistent with other papers, and I think
they are reasonable.

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

I don't come up with a strong reason to reject this paper, but I think
there are a few missing citations regarding length or verbosity bias in
RLHF/DPO. These papers can be regarded as concurrent work, so it is
optional for authors to refer to them.

- Park, Ryan, et al. "Disentangling length from quality in direct
  preference optimization." arXiv preprint arXiv:2403.19159 (2024).
- Saito, Keita, et al. "Verbosity bias in preference labeling by large
  language models." arXiv preprint arXiv:2310.10076 (2023).
- Wang, Haoxiang, et al. "Arithmetic Control of LLMs for Diverse User
  Preferences: Directional Preference Alignment with Multi-Objective
  Rewards." arXiv preprint arXiv:2402.18571 (2024).

</div>

</div>

<div>

**Questions To Authors:**

<div class="note-content-value markdown-rendered">

N/A

</div>

</div>

<div>

**Rating:** <span class="note-content-value">8: Top 50% of accepted
papers, clear accept</span>

</div>

<div>

**Confidence:** <span class="note-content-value">4: The reviewer is
confident but not absolutely certain that the evaluation is
correct</span>

</div>

<div>

**Ethics Flag:** <span class="note-content-value">No</span>

</div>

</div>

</div>

<div class="note-replies">

<div class="note depth-even" data-id="srpy0f5niL">

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
original-title="Copy URL of note srpy0f5niL"></span><span class="sr-only">Copy
URL of note srpy0f5niL</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>28 May 2024, 17:48 (modified: 28 Aug 2024,
22:32)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=srpy0f5niL)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

Thanks for the thoughtful review! We will add discussion of those
citations you mention in any future version.

</div>

</div>

</div>

</div>

</div>

<div class="note depth-even" data-id="f16jDg4ToY">

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

#### **Acknowledgement**

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note f16jDg4ToY"></span><span class="sr-only">Copy
URL of note f16jDg4ToY</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Reviewer
zdy6</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>04 Jun 2024, 18:46 (modified: 29 Aug 2024,
09:36)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=f16jDg4ToY)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

I've read other reviews and authors' rebuttals. I did not change my mind
and will keep the original score.

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

</div>

</div>
