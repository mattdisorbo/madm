<div id="content" role="main">

<div class="Forum_forum__wS8Fw">

<div class="forum-container">

<div class="forum-note">

<div style="height: auto; overflow: visible;">

<div class="btn-group">

2 Versions <span class="caret"></span>

- [COLM (July 10, 2024)](https://openreview.net/forum?id=wS7PxDjy6m)
- [CoRR 2024 (December 31,
  2023)](https://openreview.net/forum?id=fOeugZHuV0)

</div>

</div>

<div class="forum-title mt-2 mb-2">

## Dated Data: Tracing Knowledge Cutoffs in Large Language Models

<div class="forum-content-link">

<a href="https://openreview.net/pdf?id=wS7PxDjy6m"
class="citation_pdf_url" target="_blank" rel="noreferrer"
title="Download PDF"><img
src="./Dated%20Data_%20Tracing%20Knowledge%20Cutoffs%20in%20Large%20Language%20Models%20_%20OpenReview_files/pdf_icon_blue.svg"
alt="Download PDF" /></a>

</div>

</div>

<div class="forum-authors mb-2">

### <a href="https://openreview.net/profile?id=~Jeffrey_Cheng2"
data-toggle="tooltip" data-placement="top"
data-original-title="~Jeffrey_Cheng2">Jeffrey Cheng</a>, <a href="https://openreview.net/profile?id=~Marc_Marone1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Marc_Marone1">Marc Marone</a>, <a href="https://openreview.net/profile?id=~Orion_Weller1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Orion_Weller1">Orion Weller</a>, <a href="https://openreview.net/profile?id=~Dawn_Lawrie1"
data-toggle="tooltip" data-placement="top"
data-original-title="~Dawn_Lawrie1">Dawn Lawrie</a>, <a href="https://openreview.net/profile?id=~Daniel_Khashabi2"
data-toggle="tooltip" data-placement="top"
data-original-title="~Daniel_Khashabi2">Daniel Khashabi</a>, <a href="https://openreview.net/profile?id=~Benjamin_Van_Durme2"
data-toggle="tooltip" data-placement="top"
data-original-title="~Benjamin_Van_Durme2">Benjamin Van Durme</a> 

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
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=wS7PxDjy6m)</span><span class="item"><span class="glyphicon glyphicon-bookmark"
aria-hidden="true"></span><a href="https://openreview.net/forum?id=wS7PxDjy6m#"
data-target="#bibtex-modal" data-toggle="modal"
data-bibtex="%40inproceedings%7B%0Acheng2024dated%2C%0Atitle%3D%7BDated%20Data%3A%20Tracing%20Knowledge%20Cutoffs%20in%20Large%20Language%20Models%7D%2C%0Aauthor%3D%7BJeffrey%20Cheng%20and%20Marc%20Marone%20and%20Orion%20Weller%20and%20Dawn%20Lawrie%20and%20Daniel%20Khashabi%20and%20Benjamin%20Van%20Durme%7D%2C%0Abooktitle%3D%7BFirst%20Conference%20on%20Language%20Modeling%7D%2C%0Ayear%3D%7B2024%7D%2C%0Aurl%3D%7Bhttps%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DwS7PxDjy6m%7D%0A%7D">BibTeX</a></span><span class="item"><span class="glyphicon glyphicon-copyright-mark"
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

**Research Area:** <span class="note-content-value">Data,
Evaluation</span>

</div>

<div>

**Keywords:** <span class="note-content-value">knowledge cutoffs,
training data, temporal alignment</span>

</div>

<div>

**TL;DR:** <span class="note-content-value">Singular knowledge cutoff
dates do not capture the entirety of LLM training corpora, so we design
a simple probing method using time spanning datasets and analyze a large
set of open access pretraining corpora.</span>

</div>

<div>

**Abstract:**

<div class="note-content-value markdown-rendered">

Large Language Models (LLMs) are often paired with a reported cutoff
date, the time at which training data was gathered. Such information is
crucial for applications where the LLM must provide up-to-date
information. However, a reported cutoff only scratches the surface. Do
all sub-resources in the training data share the same cutoff? Does the
model's demonstrated knowledge for these sub-resources closely align to
their cutoff? We define the notion of an effective cutoff, which is
distinct from the LLM's reported cutoff and differs between
sub-resources. We propose a simple approach to estimate effective
cutoffs of an LLM on the resource-level by probing across versions of
the data. Crucially, our method does not require access to a model's
pre-training data. Through our analysis, we find that effective cutoffs
often drastically differ from reported cutoffs. To understand the root
cause of this observation, we conduct a large-scale analysis on open
pre-training datasets. Our analysis reveals two reasons for these
inconsistencies: (1) temporal misalignments of CommonCrawl data due to
non-trivial amounts of old data in new dumps; and (2) complications in
LLM deduplication schemes involving semantic duplicates and lexical
near-duplicates. Overall, our results show that cutoffs are not as
simple as they have seemed and that care must be taken both by LLM
dataset curators as well as practitioners who seek to use these models.

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

**Submission Number:** <span class="note-content-value">289</span>

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
src="./Dated%20Data_%20Tracing%20Knowledge%20Cutoffs%20in%20Large%20Language%20Models%20_%20OpenReview_files/linear_icon.svg"
title="Linear discussion layout" class="icon" data-toggle="tooltip"
alt="back arrow" /><span class="sr-only">Linear</span>

<img
src="./Dated%20Data_%20Tracing%20Knowledge%20Cutoffs%20in%20Large%20Language%20Models%20_%20OpenReview_files/threaded_icon.svg"
title="Threaded discussion layout" class="icon" data-toggle="tooltip"
alt="back arrow" /><span class="sr-only">Threaded</span>

<img
src="./Dated%20Data_%20Tracing%20Knowledge%20Cutoffs%20in%20Large%20Language%20Models%20_%20OpenReview_files/nested_icon.svg"
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

*11 / 11 replies shown*

</div>

</div>

</div>

<div class="row forum-replies-container layout-default">

<div class="col-xs-12">

<div id="forum-replies">

<div class="note depth-odd" data-id="4jE5UiKqAq">

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
original-title="Copy URL of note 4jE5UiKqAq"></span><span class="sr-only">Copy
URL of note 4jE5UiKqAq</span>

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
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=4jE5UiKqAq)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Decision:** <span class="note-content-value">Accept</span>

</div>

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

This paper introduces the interesting concept of an "effective knowledge
cutoff" in LLM training: the date associated with specific resources
using during training. This work exposes the conceptual ambiguity of
"reported knowledge cutoffs" as reported by LLM developers by
demonstrating how different pre-training sub-collections have different
knowledge cutoffs, introduces a simple perplexity-based measure for
assessing the effective cutoff of a resource, and carries out several
analyses investigating the misalignment between effective cutoffs and
reported ones in several LLMs. Reviewers generally found this work to be
creative, conceptually innovative and experimentally robust, with
important consequences for data documentation in LLMs.

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="SDAjfVXKCf">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission289 by Reviewer Z85G

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note SDAjfVXKCf"></span><span class="sr-only">Copy
URL of note SDAjfVXKCf</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
Z85G</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>14 May 2024, 09:24 (modified: 25 Aug 2024,
20:54)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=SDAjfVXKCf)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

The paper presents a technique and application for discovering the
effective training data cutoff date, via perplexity.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

The approach is clever and revealing. The technique itself is extremely
simple and therefore seemingly quite robust. The experiments
demonstrating the success are well-thought out and also very convincing.
The problem is important in LLM accountability, transparency for the
user to hold the correct expectations of the model.

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

None.

</div>

</div>

<div>

**Questions To Authors:**

<div class="note-content-value markdown-rendered">

no questions

</div>

</div>

<div>

**Rating:** <span class="note-content-value">9: Top 15% of accepted
papers, strong accept</span>

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

<div class="note depth-even" data-id="nnTdKHDvP1">

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
original-title="Copy URL of note nnTdKHDvP1"></span><span class="sr-only">Copy
URL of note nnTdKHDvP1</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>31 May 2024, 00:02 (modified: 28 Aug 2024,
22:31)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=nnTdKHDvP1)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

Thank you for your review and thoughtful comments! We appreciate that
you found our work "revealing" and "simple" which were the goals of our
analysis to provide transparency for users and creators.

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="CzzJZRhNNW">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission289 by Reviewer eFeV

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note CzzJZRhNNW"></span><span class="sr-only">Copy
URL of note CzzJZRhNNW</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
eFeV</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>10 May 2024, 23:23 (modified: 25 Aug 2024,
20:54)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=CzzJZRhNNW)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

LLMs report a cutoff date which indicates the date till which the LLMs
can be considered knowledgeable. This paper introduces the concept of a
resource-specific "effective cutoff date," for an LLM. The effective
cutoff date, distinct from the "reported cutoff date" of training data
for an LLM, is an approximate date of earlier versions of the resource
in the training data from which LLMs draw their knowledge. The paper
presents an automated technique designed to identify the effective
cutoff date for a resource in LLM training data without needing explicit
access to training data. Experiments on different LLMs reveal that there
are discrepancies between the reported and effective cut-offs that
largely stem from two reasons: incomplete deduplication resulting in
multiple versions of the same resource in training data and inadvertent
leakage of old data in new data dumps.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

The primary strength of the paper is shifting the notion of cutoff dates
from a reported cutoff to an empirically calculated effective cutoff.
The paper's proposed method for estimating the approximate effective
cutoff for a resource is straightforward and easy to replicate. The
empirical analysis backs up the thought that the effective date is
different from the reported cutoff. A side-effect of the analysis in
this paper is that the issues found around data deduplication can serve
as valuable insights for further pre-training data curation. The
experiments were detailed and provided a clear understanding. Overall,
the reporting of a resource-level effective cutoff can increase the
transparency of an LLM.

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

My biggest exception to this paper is the nebulous description of some
concepts. Though the paper advances the notion of effective cutoff date,
it never formally defines this concept. We're only given a procedure of
finding an effective cutoff date and a few examples to illustrate the
concept. Similarly, it may not always be clear how one would define a
resource within training data and if there is a need to have an
effective cutoff date at the resource level or at some other level. In
the paper, for example, news from a single media outlet (e.g., nytimes)
is considered a resource. But the tax example given in the paper
suggests the need for an effective date on a topic rather than a
specific news outlet. Moreover, sometimes news articles are sourced and
reproduced from common reporting agencies such as Reuters or Associated
Press, making it difficult to truly isolate the boundaries of a
resource.

</div>

</div>

<div>

**Questions To Authors:**

<div class="note-content-value markdown-rendered">

- In section 3.1, under Wikispan, I didn't follow how the documents were
  slotted into T topics and which topics were filtered out. Why is the
  topic distinction even necessary?
- In section 3.2, under normalization, it was hard for me to follow how
  the perplexities were aggregated. What does "average of the median 95%
  mean"? It would have been helpful to rigorously layout the
  calculations as math equations to remove any ambiguity.
- In figure 6, the minima in the left panel for both the plots looks to
  be coinciding. I understand the minima for the dedup curve is not as
  sharp as for non-dedup, but if they coincide wouldn't the effective
  cutoff dates be the same with and without deduplication?
- Possible additional citations
  - "Whose Language Counts as High Quality? Measuring Language
    Ideologies in Text Data Selection" by Gururangan et. al. also
    discuss the tole of quality filters in training data selection
  - "Speak, Memory: An Archaeology of Books Known to ChatGPT/GPT-4" by
    Chang et. al. as an example of membership inference testing without
    the need to compute the LLMs perplexity
- In section 2, under continual learning, the objective of continual
  learning is a modeling concern which is different from the analytic
  concern of this paper. What is the more specific connection between
  the two besides both "examine temporal knowledge"?

</div>

</div>

<div>

**Rating:** <span class="note-content-value">7: Good paper,
accept</span>

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

<div class="note depth-even" data-id="xgd2KM4XmJ">

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
original-title="Copy URL of note xgd2KM4XmJ"></span><span class="sr-only">Copy
URL of note xgd2KM4XmJ</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>31 May 2024, 00:07 (modified: 28 Aug 2024,
22:31)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=xgd2KM4XmJ)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

> My biggest exception … is the nebulous description of some concepts

We will clarify the definition in Sec. 3. We define the effective cutoff
date with respect to a model and resource as the date of the version of
that resource that most closely aligns with a model. Alignment can be
measured in several ways, but we use minimum perplexity.

In the tax example, resource refers to a version of the tax code, not a
specific section. In general, **our approach is for resources (e.g. a
set of docs) rather than for attribution to a single doc.**

We agree - as in your news example - that a resource can be more
nebulous, and in those cases it would depend on if there are enough
unique articles in that resource. If not, then perhaps it’s not a good
resource definition and should be re-defined (as AP news or news
broadly).

> I didn't follow how the docs were slotted into T topics

A wikipedia topic is effectively the document title. We use this term to
disambiguate between versions of an article and the subject of the
article. Wikispan has 94 doc versions (1/month) for each of the 5k
topics.

Our pipeline filters more recently created topics as these are blank for
certain timespans (e.g. the “Covid-19” page is blank in the year 2017).

> What does "average of the median 95% mean"

To remove any outlier ppl values (resulting from unfiltered stub
articles/redirects) we first take the mean and 95% confidence interval,
also known as a
<a href="https://en.wikipedia.org/wiki/Truncated_mean" target="_blank"
rel="noopener noreferrer">truncated mean</a>. We will update the wording
to make it more clear and include math equations to remove ambiguity!

> if they coincide wouldn't the effective cutoff dates be the same

Interpreting relatively flat basins in the curves indeed requires
nuanced considerations. If the two minima coincided, that would be
correct. However in Figure 6, while the min of the non-dedup curve
coincides with a local minima of the dedup curve, the global min of the
dedup curve is around a year earlier.

> connection besides both "examine temporal knowledge"?

That is the connection, as our work explores **if** there is temporal
alignment and continual learning explores **how to** aligning them. We
will revise this section to make this more clear with our analysis - for
example, our analysis could be used “in-the-loop” to determine that a
model should undergo additional learning on more recent sources.

We will add the relevant citations!

</div>

</div>

</div>

</div>

</div>

<div class="note depth-even" data-id="3dUkx6iIxS">

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

#### **Response**

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note 3dUkx6iIxS"></span><span class="sr-only">Copy
URL of note 3dUkx6iIxS</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(187, 187, 255); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Comment</span><span class="signatures">by Reviewer
eFeV</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>05 Jun 2024, 11:04 (modified: 29 Aug 2024,
09:33)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=3dUkx6iIxS)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

Thanks for the response to the original review. I appreciate the authors
taking the effort to engage with all parts of the review.

> We will clarify the definition in Sec. 3. We define the effective
> cutoff date with respect to a model and resource as the date of the
> version of that resource that most closely aligns with a model.
> Alignment can be measured in several ways, but we use minimum
> perplexity.

> In the tax example, resource refers to a version of the tax code, not
> a specific section. In general, our approach is for resources (e.g. a
> set of docs) rather than for attribution to a single doc.

> We agree - as in your news example - that a resource can be more
> nebulous, and in those cases it would depend on if there are enough
> unique articles in that resource. If not, then perhaps it’s not a good
> resource definition and should be re-defined (as AP news or news
> broadly).

The part that the resource is somewhat subjectively decided will remain
a point of contention for me although at least giving clear definitions
would be useful for future researchers.

> A wikipedia topic is effectively the document title. We use this term
> to disambiguate between versions of an article and the subject of the
> article. Wikispan has 94 doc versions (1/month) for each of the 5k
> topics.

> Our pipeline filters more recently created topics as these are blank
> for certain timespans (e.g. the “Covid-19” page is blank in the year
> 2017).

Thanks for this explanation. It makes more sense now.

> To remove any outlier ppl values (resulting from unfiltered stub
> articles/redirects) we first take the mean and 95% confidence
> interval, also known as a truncated mean. We will update the wording
> to make it more clear and include math equations to remove ambiguity!

Thanks

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="Dhjk9HziLJ">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission289 by Reviewer VAWs

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note Dhjk9HziLJ"></span><span class="sr-only">Copy
URL of note Dhjk9HziLJ</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
VAWs</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>09 May 2024, 20:31 (modified: 25 Aug 2024,
20:54)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=Dhjk9HziLJ)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

This work proposes to reexamine the reported knowledge cutoff dates of
LLMs by demonstrating that different "sub-resources" in LLM training
might come with different cutoff dates. This work calculates perplexity
over a temporally shifting domain corpora to illustrate the perplexity
changes corresponding to document time stamps. Experiments found various
artifacts of existing models for NYT and Wikipedia domains.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

- this work has the potential to facilitate greater transparency in
  training data
- the proposed approach is straightforward

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

- To verify the actual cutoff date for a certain "sub-resource" or
  domain, the proposed methodology would need a collection of documents
  in the domain for a long time period. I wonder how dependent is the
  approach on those documents: Do we need a lot of documents for each
  time stamp, or are a few/under 10 often enough? It might be
  challenging to collect a lot of documents for a domain-specific
  application and use the proposed method.

- In addition to visualizing the perplexity changes through time, it
  might be nice to have a methodology to quantitatively decide on a
  specific cutoff date. In this way, users could decide on whether to
  trust the LLM when their query concerns information after that
  specific time stamp. I wonder if the authors might have ideas about
  how to obtain a specific cutoff date estimation.

- The approach focuses on perplexity: however, it might be the case that
  certain long-tail documents from a newer date were there in the
  training data, but were just not properly "memorized" thus leading to
  high perplexities. I wonder if the authors believe that this might be
  a confounding factor, and how the proposed methodology mitigates this.

- It would be nice to include a qualitative analysis showing example
  documents that are before the reported cutoff date but after the
  "actual" cutoff date, along with perplexity scores/metadata etc.

- After identifying the "actual" cutoff dates of different
  sub-resources, what could people do with those dates? There could be
  social aspects such as better issuing a disclaimer about
  domain-specific knowledge cutoffs, there could also be technical
  implications such as adapting the model to more recent data on certain
  domains, instead of training on new data indiscrimnitaly. It would be
  nice to propose some applications/recommendations that build on this
  methodology.

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

<div class="note depth-even" data-id="IhtUHMxMA6">

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
original-title="Copy URL of note IhtUHMxMA6"></span><span class="sr-only">Copy
URL of note IhtUHMxMA6</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>31 May 2024, 00:05 (modified: 28 Aug 2024,
22:31)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=IhtUHMxMA6)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

> Do we need a lot of documents

You bring up an important point about the number of documents required
for our method. We empirically found the perplexity of documents in a
sub-resource can vary greatly (due to the inherent differences in
perplexity in language). As such, we use the median 95% of the
perplexity values and take larger samples to obtain more unbiased
samples of the true distribution.

Upon your suggestion we are currently running this experiment as we feel
it would add to our paper. We are varying the number of docs per
timestamp and will provide this in the appendix. It should be done in
the next few days and we will post the results here. Thank you for the
suggestion!

> methodology to quantitatively decide on a specific cutoff date

We do provide a baseline approach to quantitatively define the effective
cutoff — we use the minima of the perplexity curve, after truncating the
distribution to the median 95%. This is what is shown in Figures 3-7.
The quantitative way to do this automatically is simply to select the
global minima.

> certain long-tail documents from a newer date were there in the
> training data, but were just not properly "memorized"

This could happen, and we have already attempted to prevent this by
using the median 95% of the distribution when calculating values. Thus
if the values are on the tails, they will not be included. If there are
a large amount of these, then that would be a problem, however, we are
unaware of any one collection that contains a large subset of documents
that are hard to memorize.

> qualitative analysis … before the reported cutoff date but after the
> "actual" cutoff date

We included a qualitative analysis of duplicated documents, but agree
that some examples of perplexity/docs from before/after the cutoff would
be helpful also. We will update the paper to include examples in a new
appendix section!

> what could people do with those dates?

Great question! As discussed in the introduction and conclusion, as an
initial step we believe it serves as a warning to both LLM creators and
users to be mindful of the temporal biases of LLMs.

For what users could do with that date, it will have to be
resource-specific – perhaps if you need the LM up to date, you either
pick a different LM or try some techniques to improve its temporal
alignment. However, this paper focuses on finding and notifying the
community of this problem, we leave the rest to future work.

</div>

</div>

</div>

</div>

</div>

<div class="note depth-even" data-id="52nPW6aq6P">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Comment by Authors

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note 52nPW6aq6P"></span><span class="sr-only">Copy
URL of note 52nPW6aq6P</span>

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
aria-hidden="true"></span>04 Jun 2024, 11:41 (modified: 29 Aug 2024,
09:33)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=52nPW6aq6P)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Comment:**

<div class="note-content-value markdown-rendered">

These files in the linked repository show the ablation results where the
number of documents in each month bucket are reduced. To be specific,
rather than considering the 5000 most edited topics, we repeat the setup
described in the paper instead with the x most edited topics. We plot
those averages for LLaMA, RedPajamas, and OLMo. Note that when x=5000,
this is the same result as in the paper (Figure 4).

We find that for x\>50, the effective cutoffs of the three models is
consistent with the full results. x=50 is the threshold where the trends
and effective cutoffs are less consistent with the original results due
to the more apparent variability when taking few samples. Ultimately, we
find that these extra results are a useful addition to the paper and
confirm our main hypotheses for reasonable number of document bucket
sizes.

Thank you for your suggestions and comments!

<a href="https://anonymous.4open.science/r/dated-data/README.md"
target="_blank"
rel="noopener noreferrer">https://anonymous.4open.science/r/dated-data/README.md</a>

</div>

</div>

</div>

</div>

</div>

</div>

</div>

<div class="note depth-odd" data-id="3RHqiEoU4Q">

<div class="btn-group-vertical btn-group-xs collapse-controls-v"
role="group" aria-label="Collapse controls">

−

＝

≡

</div>

<div class="heading">

#### Official Review of Submission289 by Reviewer 6tHA

<span class="glyphicon glyphicon-link" toggle="tooltip" placement="top"
title="" aria-hidden="true"
original-title="Copy URL of note 3RHqiEoU4Q"></span><span class="sr-only">Copy
URL of note 3RHqiEoU4Q</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 187, 187); color: rgb(44, 58, 74);"
original-title="Reply type">Official
Review</span><span class="signatures">by Reviewer
6tHA</span><span class="created-date" toggle="tooltip" placement="top"
title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>09 May 2024, 04:03 (modified: 25 Aug 2024,
20:54)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=3RHqiEoU4Q)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Summary:**

<div class="note-content-value markdown-rendered">

This paper investigates the concept of knowledge cutoffs in Large
Language Models, which is critical for applications requiring up-to-date
information. The authors challenge the assumption that a reported cutoff
date for an LLM implies uniformity across all training data and propose
a method to estimate the effective cutoff at the resource level. Their
approach does not require access to the model's pre-training data and
reveals significant discrepancies between reported and effective
cutoffs. The paper also conducts a large-scale analysis to understand
the causes of these inconsistencies, identifying issues with data
deduplication and temporal misalignments in CommonCrawl data.

</div>

</div>

<div>

**Reasons To Accept:**

<div class="note-content-value markdown-rendered">

- The paper introduces a novel method for evaluating the temporal
  dynamics of knowledge within LLMs. By measuring the perplexity across
  different versions of datasets, the authors provide a granular
  understanding of knowledge cutoffs, which is a significant advancement
  in LLM analysis.

- The large-scale analysis of pre-training datasets is particularly
  impressive. The identification of issues such as temporal
  misalignments in CommonCrawl data and deduplication challenges is a
  substantial finding that contributes to the broader understanding of
  LLM training data nuances.

- The detailed examination of the reasons behind the misalignment of
  cutoff dates provides diagnostic value to both LLM creators and users.
  The insights can guide the development of better training protocols
  and inform users about the potential limitations of LLM knowledge
  bases.

</div>

</div>

<div>

**Reasons To Reject:**

<div class="note-content-value markdown-rendered">

- The paper does not discuss the scalability of the proposed method to
  even larger models or datasets.
- It is unclear how these findings generalize to LLMs trained with
  different data or methods.

</div>

</div>

<div>

**Questions To Authors:**

<div class="note-content-value markdown-rendered">

- Have you considered or tested the applicability of your approach to
  other types of resources, such as news articles, books, or legal
  documents, which may have different update frequencies and versioning
  challenges?
- How does the model architecture or size influence the identification
  of effective cutoffs? Would smaller models or those with different
  training methodologies exhibit similar discrepancies between reported
  and effective cutoffs?

</div>

</div>

<div>

**Rating:** <span class="note-content-value">9: Top 15% of accepted
papers, strong accept</span>

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

<div class="note depth-even" data-id="O6MT8oMLDg">

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
original-title="Copy URL of note O6MT8oMLDg"></span><span class="sr-only">Copy
URL of note O6MT8oMLDg</span>

</div>

<div class="subheading">

<span class="invitation highlight" toggle="tooltip" placement="top"
title=""
style="background-color: rgb(255, 136, 204); color: rgb(44, 58, 74);"
original-title="Reply type">Rebuttal</span><span class="signatures">by
Authors</span><span class="created-date" toggle="tooltip"
placement="top" title=""
original-title="Date created"><span class="glyphicon glyphicon-calendar"
aria-hidden="true"></span>31 May 2024, 00:04 (modified: 28 Aug 2024,
22:31)</span><span class="readers" toggle="tooltip" placement="top"
title=""
original-title="Visible to &lt;br/&gt;everyone"><span class="glyphicon glyphicon-eye-open"
aria-hidden="true"></span>Everyone</span><span class="revisions"><span class="glyphicon glyphicon-duplicate"
aria-hidden="true"></span>[Revisions](https://openreview.net/revisions?id=O6MT8oMLDg)</span>

</div>

<div class="note-content-container">

<div class="note-content">

<div>

**Rebuttal:**

<div class="note-content-value markdown-rendered">

Thank you for your review and for finding our work "impressive" and a
"substantial finding that contributes to the broader understanding of
LLM training data nuances"!

> The paper does not discuss the scalability of the proposed method

We show results for models from 7B to 65B in Figure 5, but agree that we
did not test models beyond 65B parameters. We will add more discussion
to Section 5.2 discussing this – we believe that the trends from 7B to
65B will continue, as they show consistent results.

As the resource size increases our method will work better as the sample
size will be larger and less biased, but we agree that computationally,
our method will increase linearly with the size of the dataset.

We will add these comments to Section 4.1 and 4.2.

> How does the model architecture or size influence the identification
> of effective cutoffs?

Great questions! We evaluate a large variety of language models of
varying transformer architectures and training schemes (see Table 1).
Moreover, we also evaluate different sized models in Pythia and LLaMA
suites (see Figure 5 in section 5.2). Overall, we find that that
effective cutoffs are consistent between different sized models that
were pre-trained on the same data.

> Have you considered or tested the applicability of your approach to
> other types of resources

In our paper we showed results for two types of resources (Wikipedia and
news articles, as you suggest – that is our NYT collection), as they
represent the streaming and updating types of datasets (Section 3).
Unfortunately due to paper space we could not add more datasets
comprehensively, but our initial tests showed that it holds on other
resources as well.

We leave it to future work (and to LLM creators) to document these
results for all other resources that users are interested in – there are
a lot of resources out there and we agree this is an important area of
future work!

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
