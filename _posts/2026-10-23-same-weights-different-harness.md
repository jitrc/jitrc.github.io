---
title: "Same Weights, Different Harness"
date: 2026-10-23
image:
  path: /assets/images/og/same-weights-different-harness.png
  width: 1200
  height: 630
  alt: "Same Weights, Different Harness — jitrc.com"
description: "ARC-AGI-3 shows the same model scoring 63% or 99% depending on what surrounds it. My own benchmark shows why that isn't the whole story."
kind: essay
tags: [agents, evals, benchmarks]
---


This started in May 2026 as a note after instrumenting tau2-bench. The ARC-AGI-3 results below landed in July and August and forced me to rewrite the conclusion.

The cleanest result in agent evaluation right now is a table with two columns. On [ARC Prize's own verified results](https://arcprize.org/results/openai-gpt-6-astra) for GPT-6 Astra, the same model, at the same reasoning effort, scores:

| Effort | Standard harness | Provider adapter harness |
|---|---|---|
| Max | 62.71% | 98.55% |
| High | 54.82% | 99.95% |
| Medium | 38.59% | 98.44% |
| Low | 17.45% | 98.03% |

Nothing about the weights changed between those columns. What changed is the harness: how the model receives context, keeps notes, carries reasoning state forward, and recovers from a dead end. At low effort the gap is over 80 points.

That is not a benchmark curiosity. It is the whole deployment problem in one table.

## The public demonstrations

Three independent systems made the same point inside two months.

**NVIDIA's AVO** [reported 100.00 RHAE](https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents) across all 25 environments and 183 levels of the ARC-AGI-3 public set, using 6,624 environment actions — roughly 12% fewer than VISTA, an earlier system that solved the same set with the same model. Claude Opus 5 alone scores [about 30%](https://arcprize.org/results/anthropic-claude-opus-5) on that benchmark. The model didn't get smarter; the shell around it did. AVO's two load-bearing pieces are persistent memory and a supervisor that keeps a long-horizon run on task.

**OpenAI** [described two harness settings](https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores) — retaining reasoning state and enabling compaction — that tripled its scores, and noted a game where no frontier model cleared the first level but their harness solved all six. Compaction is the interesting one: it is a memory-management decision, not a capability.

**Schema**, an academic harness, [reports ~99%](https://schema-harness.github.io/) on the public set by making the model write each game's mechanism as an executable program, test it against observations, and plan inside it. It does not touch weights, and the authors are careful to note ARC Prize has not verified the result.

The pattern across all three: the harness supplies what the model lacks at the edges — durable state, a world model, and a feedback loop that turns an action's result into the next decision.

## Where that story breaks

I spent a chunk of May building the opposite experiment. [tau2-enhanced](https://github.com/jitrc/tau2-enhanced) wraps Sierra's tau2-bench — multi-turn airline and retail support tasks — with structured logging, then with two agents that fix the two most obvious execution problems: a retry agent that recovers from parameter-validation errors, and a context agent that keeps the working set under the point where performance collapses.

The instrumentation said execution was the problem. Action-execution failures were 89% of all failures in my aggregate report, ranging from 48% to 87% across models. So the fix should work.

It didn't, at the level that matters. Ten tasks, two trials, 54 expected actions:

| Agent | Tool success | Task success |
|---|---|---|
| Baseline | 57.4% | 60.0% |
| Retry | 68.5% | 60.0% |
| Context | 59.3% | 60.0% |
| Combined | 70.4% | 55.0% |

Tool success improved 13 points. Task success didn't move, and the combined agent — the one that fixes both problems — was five points *worse* than baseline. Precision stayed near 3.9%, recall between 26% and 37%. Self-loops made up 37% of calls, with one lookup function repeated hundreds of times in a single evaluation.

Retry and context management are real improvements to the execution layer. They are not a harness in the sense that ARC-AGI-3 is measuring. My agents fixed *how* a call was made; they never touched *what the agent believed about the world*. A malformed argument is a symptom of a shallow model of the task, and patching the argument leaves the shallowness in place.

## What the difference actually is

Put the two results side by side and the line is clear:

- **Execution harness** — retries, validation, context trimming, tool-call formatting. Buys you single-digit-to-low-double-digit tool metrics. Cheap, general, and easy to mistake for progress.
- **Cognitive harness** — persistent memory, an explicit world model the agent can test, a supervisor that decides when to change strategy, and a loop that carries what was learned into the next attempt. This is what moved a 30% model to 100% on ARC-AGI-3.

The second one is not a wrapper you bolt on. It is the part of the system that encodes the environment. ARC-AGI-3 rewards it because the benchmark is explicitly about learning a novel environment's rules. Enterprise workflows reward it for the same reason, minus the clean scoring function.

## The uncomfortable implication

Two of the four cells in my table got better and the row that matters got worse. If I had shipped on tool success — the metric that moved first, and the one easiest to instrument — I would have reported a 13-point win for a change that made the workflow worse.

That is the evaluation trap the ARC-AGI-3 results expose from the other direction. A harness is necessary and not sufficient, and the difference between +13 points on tools and +68 points on tasks is exactly the knowledge the harness holds about the world. Measuring execution tells you whether the agent can act. It does not tell you whether it knows what it is doing.

The model gap closed. The deployment gap is a harness design gap — and most of the harness work being sold today is still the cheap half.
