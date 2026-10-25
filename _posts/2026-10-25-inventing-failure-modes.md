---
title: "Inventing Failure Modes"
date: 2026-10-25
image:
  path: /assets/images/og/inventing-failure-modes.png
  width: 1200
  height: 630
  alt: "Inventing Failure Modes — jitrc.com"
description: "k-means invents structure that isn't there, and HDBSCAN's noise label is not the novelty detector people think it is. A small controlled experiment on agent failures."
kind: essay
tags: [agents, evals, clustering]
---


An agent fails in production. Someone reads the trace, writes one sentence about what broke, and the sentence joins a pile. A month later the pile is the raw material for everything the improvement loop does — the eval cases, the proposed fixes, the evidence shown to whoever owns the workflow.

Turning that pile into a small set of named failure modes is the first real job, and it is unforgiving. Everything downstream inherits whatever this step gets wrong. Bad eval cases are measured against, fixes chase phantom problems, and a domain expert is shown clusters that don't correspond to anything real. It is one of the few places where a wrong default doesn't just slow the loop down; it quietly makes the output meaningless while looking like evidence.

Production failures have an awkward shape. They are short, noisy and long-tailed: a few modes recur constantly, a long tail happens once or twice, and a good number of individual failures are genuinely idiosyncratic. Any method that assumes a cleaner structure will invent one, and invented structure is worse than none.

The two defaults people reach for both assume otherwise. I put them side by side on a corpus with known answers to see how badly.

## k-means answers the question you're asking

k-means needs `k` before it starts. But "how many distinct failure modes are there, and is this new one real?" is the question you were trying to answer. Setting `k` up front answers it before you have looked.

The second problem is worse. k-means assigns every point to a cluster — that is the algorithm. It has no representation of "this belongs to nothing." A one-off failure gets pulled into the nearest centroid, the centroid is usually a mode it superficially resembles, and now that mode's label is wrong for both of them.

I built a corpus of 58 failures: four recurring modes, twelve paraphrases each, plus ten deliberately idiosyncratic one-offs. Then I told k-means the true number of modes, which is more help than it gets in practice.

| Designed mode | Cluster 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| validation | 2 | 0 | **10** | 0 |
| timeout | 0 | 0 | 1 | **11** |
| auth | 0 | **11** | 1 | 0 |
| parse | 1 | 0 | **11** | 0 |
| one-off | **9** | 0 | 0 | 1 |

Four modes went in. Three clean clusters came out, plus a junk drawer — cluster 0 — that absorbed nine of the ten one-offs along with three real failures from the modes it was supposed to separate. Validation and parse were merged into cluster 2 despite being different failures with different fixes. Agreement with the ground truth was an adjusted Rand index of 0.564.

This is not a matter of picking a better `k`. Silhouette prefers `k=4` at 0.066, and `k=5` and `k=6` are worse on both silhouette and ARI. The junk cluster is not a parameter error; it is what the algorithm does with points that belong nowhere. You asked for four failure modes and got three modes and a trash can with a mode's name on it, which is exactly the input that produces a confident, useless eval case.

## Topic models need more words than a failure message has

LDA and its relatives model documents as mixtures of topics, and estimating that mixture requires seeing terms co-occur across a document. A failure message is a sentence or two. The document-term matrix is nearly all zeros, so the posterior is estimated from very little.

That is measurable, and it was measured on short text well before agents existed. Hong and Davison
([2010](https://www.cse.lehigh.edu/~brian/pubs/2010/SOMA/twitter-topic-modeling.pdf)) trained LDA on individual short messages and found the content "very limited" such that "the model may not have enough information to learn the topic patterns" — they got a usable model only by aggregating messages into longer documents. The finding isn't that topic models are bad. It's that their input assumption is document-shaped and a failure log isn't, so the natural failure mode is a broad topic that swallows the rare-but-real case.

## What each piece of the usual stack is actually for

Three components, each earning its place for one property. None is novel; the argument is about the combination and what it does not do.

**Sentence embeddings, for grouping by meaning.** "Connection pool exhausted" and "ran out of database connections" are the same failure and share almost no tokens. Reimers and Gurevych state the problem their work addresses precisely — "the construction of BERT makes it unsuitable for semantic similarity search as well as for unsupervised tasks like clustering"
([Sentence-BERT, 2019](https://arxiv.org/abs/1908.10084)) — and the reason is computational: vanilla BERT needs both sentences fed through the network together, so finding the closest pair among 10,000 sentences costs roughly 65 hours. SBERT derives embeddings you can compare with cosine distance and does the same search in about five seconds.

**UMAP, because density is hard to estimate in 384 dimensions.** The UMAP docs make this case with a number: on MNIST, HDBSCAN over 50-dimensional PCA data clustered
[17% of the points](https://umap-learn.readthedocs.io/en/latest/clustering.html) and called the rest noise; after UMAP reduction it clustered over 99%. That is the effect worth having. The same page also opens by calling this pipeline "somewhat controversial, and should be attempted with care," because UMAP "does not completely preserve density" and "can also create false tears in clusters, resulting in a finer clustering than is necessarily present in the data."

Keep that sentence. It means the cluster count is an upper bound. If the pipeline reports nine modes, the honest reading is *at most nine, and some of those are tears in the embedding*, not "we have nine failure modes."

**HDBSCAN, for the count and the outliers.** It builds a hierarchy over density and extracts clusters by stability, so you don't supply `k`. And it has the property that makes the whole design sound in theory: "any point not in a selected cluster is simply a noise point (and assigned the label −1)"
([how HDBSCAN works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html);
[McInnes et al., JOSS 2017](https://joss.theoj.org/papers/10.21105/joss.00205)).

**An LLM to name each cluster.** A cluster is a set of points; a name is what a person can react to. The name becomes the eval-case name.

So far this is the standard recommendation, and the first two claims hold up. The third one is where I stopped believing the blog posts.

## The noise label does not find the one-offs

On the same corpus, with UMAP to 5 dimensions and `min_cluster_size=5`:

| `min_samples` | Clusters | Noise points | One-offs in noise | Real failures in noise |
|---|---|---|---|---|
| 1 | 6 | 10 | 3/10 | 7 |
| 2 | 5 | 13 | 3/10 | 10 |
| 3 | 4 | 8 | 1/10 | 7 |
| 5 | 3 | 11 | 4/10 | 7 |
| 8 | 2 | 12 | 3/10 | 9 |
| 12 | 2 | 33 | 7/10 | 26 |
| 20 | 0 | 58 | 10/10 | 48 |

There is no setting where the label is useful. At the strict end it catches one of ten one-offs while flagging seven ordinary failures. Push it until it catches them all — `min_samples=20` — and it labels the entire corpus noise, including all 48 real failures. Precision and recall fail together, which is the signature of a signal that isn't measuring what you want rather than one that's badly tuned.

Looking at *which* points it flagged explains why. Of the eight noise points at `min_samples=3`, seven were perfectly ordinary members of known modes — "argument 'limit' must be an integer, got string", "required parameter 'account_id' was not provided" — and exactly one was a designed one-off. The score agrees: GLOSH outlier scores rank the one-offs at an AUC of 0.564, and HDBSCAN's own membership uncertainty does no better at 0.565. Both are coin flips.

The reason is that `-1` does not mean "novel". It means "this point contributes nothing useful to the density structure I found," which is a statement about local geometry given a `min_cluster_size` you chose. A failure mode can be real, recurring and important while sitting at slightly lower local density than its neighbours, and an idiosyncratic failure can sit right next to a mode because it shares all its vocabulary. On this corpus the one-offs were geometrically distant — 0.11 to 0.42 cosine similarity to the nearest mode centroid, against a within-mode average of 0.61 — and HDBSCAN still couldn't see them, because distance from a global centroid is not the quantity density clustering measures.

Treating `-1` as "new failure mode, write a fix" does both bad things at once: it opens eval cases for ordinary failures, and it misses genuinely novel ones. That is the loop's trust budget being spent on noise.

## What does work is the thing you already have

A loop that has run for a while already owns labelled failures — the eval set from previous rounds. That turns novelty from a clustering question into a distance question. With six labelled examples per mode as "history", scoring each new failure by its distance from the nearest known mode gave:

| Method | ROC AUC for "is a one-off" |
|---|---|
| GLOSH outlier score | 0.564 |
| HDBSCAN membership uncertainty | 0.565 |
| Distance from known modes | **0.987** |

Nine of the top ten by that distance were designed one-offs. The single miss was the failure that genuinely belongs to a known mode.

That inverts the roles of the two steps, which is the part I had wrong. Clustering is for seeing the shape of the corpus and for naming modes you don't have names for yet. Deciding whether a given failure is new is a comparison against what you already know. Expecting the clustering to hand you a novelty flag conflates them, and the corpus above shows how completely the second job fails when you ask the first tool to do it.

## Rules I'd now write down before building this

- **A cluster is not an eval case until a human recognises the name.** The LLM label is a proposal. The recognition test is the only one that matters, because the name is what the rest of the loop is measured against.
- **`-1` routes to a human; it never triggers a fix on its own.** It's a triage signal with poor precision either way, and the cost of a wrong eval case is paid every round after.
- **Report the cluster count as an upper bound.** UMAP can tear a mode in two; if two clusters get near-identical names, that is the tear, not two modes.
- **`min_cluster_size` is `k` wearing a hat.** Five means "a mode needs five members" — a prior about frequency that you should state out loud rather than accept from a default.
- **Never tune the parameters against the thing you're trying to detect.** I could reach 10/10 one-off detection with `min_samples=20`. I'd also have labelled every real failure noise. Any tuning loop scored on the answer key will find that setting and report success.
- **The embedding model decides everything downstream.** Swap it and the modes rename themselves, so pin it alongside the eval set the way you'd pin a dependency.

## What this doesn't prove

The corpus is synthetic and I wrote both the failures and the answer key, which makes every number here a demonstration rather than a measurement of production behaviour. Fifty-eight messages is small, one embedding model is one data point, and a distance-to-centroid baseline is crude next to whatever a real eval set would support. What I'd defend is the direction: the noise label is not a novelty detector, and the comparison that actually works is against failures you have already named.

The reproducible script and corpus are
[in the repo](https://github.com/jitrc/jitrc.github.io/tree/master/scripts/cluster-experiment) if you want to change the corpus and break the result — which is the useful thing to do with it.

The last thing worth saying is that all of this is instrumentation for a judgment. A clustering pipeline produces candidate names. A domain expert either recognises *"silently overwrote a column because it shared a name with the index"* as a real thing their operation does, or doesn't. Every number above only decides which candidates get shown to them.
