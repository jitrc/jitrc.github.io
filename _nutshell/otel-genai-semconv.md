---
layout: nutshell
title: "OTel GenAI Semconv in a Nutshell"
date: 2026-10-24
image:
  path: /assets/images/og/otel-genai-semconv.png
  width: 1200
  height: 630
  alt: "OTel GenAI Semconv in a Nutshell — jitrc.com"
summary: "Cheat sheet for OpenTelemetry's GenAI conventions: the span names, the attributes that matter, the metrics that got renamed, and how MCP trace context travels."
tags: [agents, observability, opentelemetry]
---


<div class="callout">
<strong>Verified 2026-10-24</strong> against <code>open-telemetry/semantic-conventions-genai</code> (the conventions moved out of the main semconv repo — the opentelemetry.io pages now redirect there) and <code>opentelemetry-sdk</code> 1.45.1. Every signal is status <strong>Development</strong>, i.e. experimental: attribute names are still moving. The span snippet below was executed and produces the trace shown.
</div>

## What it does

The GenAI semantic conventions are a naming scheme for telemetry from model calls, agents and
tools: what a span is called, which attributes it carries, which metrics exist. They do not
change what you instrument — they make the result comparable across providers, SDKs and
dashboards.

## Why it exists

- **One set of field names for sixteen providers.** `gen_ai.provider.name: "aws.bedrock"` and
  `"openai"` sit in the same column, so a cost dashboard does not fork per vendor.
- **A trace that matches the way agents actually run.** `invoke_agent` → `chat` → `execute_tool`
  nests naturally, so "why was this turn slow" has one obvious query.
- **Token accounting that survives caching.** Input, output, cache-read, cache-write and
  reasoning tokens are separate attributes, because they have separate prices.

<figure class="diagram">
<svg viewBox="0 0 640 250" role="img" aria-labelledby="otel-diagram-title">
  <title id="otel-diagram-title">A trace tree for one agent turn: invoke_agent at the root, chat spans for each model call, and an execute_tool span containing an MCP tools/call span</title>
  <defs>
    <marker id="spine" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#4b0082"/>
    </marker>
  </defs>

  <line x1="60" y1="18" x2="60" y2="232" stroke="#4b0082" stroke-width="2"/>

  <rect x="80" y="26" width="330" height="34" rx="4" fill="#fff" stroke="#4b0082"/>
  <text x="92" y="47" font-size="11" fill="#222">invoke_agent support-bot</text>
  <text x="300" y="47" font-size="9" fill="#666">CLIENT</text>

  <rect x="110" y="76" width="300" height="30" rx="4" fill="#ededed" stroke="#666"/>
  <text x="122" y="95" font-size="10" fill="#222">chat gpt-4.1-mini</text>
  <text x="330" y="95" font-size="9" fill="#666">tokens</text>

  <rect x="110" y="118" width="300" height="30" rx="4" fill="#ededed" stroke="#666"/>
  <text x="122" y="137" font-size="10" fill="#222">execute_tool read_file</text>
  <text x="330" y="137" font-size="9" fill="#666">CLIENT</text>

  <rect x="160" y="160" width="270" height="28" rx="4" fill="#fff" stroke="#666" stroke-dasharray="3 2"/>
  <text x="172" y="178" font-size="9.5" fill="#222">tools/call read_file  (MCP span)</text>
  <text x="390" y="178" font-size="8" fill="#666">SERVER</text>

  <rect x="110" y="200" width="300" height="30" rx="4" fill="#ededed" stroke="#666"/>
  <text x="122" y="219" font-size="10" fill="#222">chat gpt-4.1-mini</text>
  <text x="330" y="219" font-size="9" fill="#666">tokens</text>

  <line x1="60" y1="43" x2="76" y2="43" stroke="#4b0082" stroke-width="1.5" marker-end="url(#spine)"/>
  <line x1="60" y1="91" x2="106" y2="91" stroke="#666" stroke-width="1"/>
  <line x1="60" y1="133" x2="106" y2="133" stroke="#666" stroke-width="1"/>
  <line x1="110" y1="133" x2="156" y2="174" stroke="#666" stroke-width="1" stroke-dasharray="3 2"/>
  <line x1="60" y1="215" x2="106" y2="215" stroke="#666" stroke-width="1"/>

  <text x="470" y="95" font-size="9" fill="#666">one per model call</text>
  <text x="470" y="137" font-size="9" fill="#666">one per tool</text>
  <text x="470" y="178" font-size="9" fill="#666">merges into the</text>
  <text x="470" y="190" font-size="9" fill="#666">tool span when</text>
  <text x="470" y="202" font-size="9" fill="#666">already traced</text>
</svg>
<figcaption>Span names carry the identity (`invoke_agent support-bot`); attributes carry the numbers.</figcaption>
</figure>

## Cheat sheet: the spans

| `gen_ai.operation.name` | Span name format | Kind | Where it comes from |
|---|---|---|---|
| `invoke_agent` | `invoke_agent {gen_ai.agent.name}` | CLIENT | Calling a remote agent service |
| `invoke_agent` | `invoke_agent {gen_ai.agent.name}` | INTERNAL | In-process agent loop |
| `create_agent` | `create_agent {gen_ai.agent.name}` | CLIENT | Creating a hosted agent resource |
| `invoke_workflow` | framework-specific | INTERNAL | Multi-agent or graph orchestration |
| `plan` | `plan` | INTERNAL | Planning / task decomposition |
| `execute_tool` | `execute_tool {gen_ai.tool.name}` | CLIENT or INTERNAL | One tool invocation |
| `chat` | `chat {gen_ai.request.model}` | CLIENT | A model call |
| `generate_content` | `generate_content {gen_ai.request.model}` | CLIENT | Gemini-style multimodal call |
| `embeddings` | `embeddings {gen_ai.request.model}` | CLIENT | Vector generation |
| `retrieval` | `retrieval {gen_ai.data_source.id}` | CLIENT | RAG fetch |
| `fetch_response` | `fetch_response` | CLIENT | Re-read a stored response (no tokens) |

**Required on every one of them:** `gen_ai.operation.name` and `gen_ai.provider.name`. If a
provider is proxied, record what the instrumentation *believes* — the spec says the attribute may
differ from the actual upstream when a gateway relays calls.

Two naming rules that matter operationally: the model name goes in the span name, the session
never does. `chat gpt-4.1-mini` is low cardinality; `chat 2026-10-24T18:31:07Z` destroys every
aggregation you wanted.

## Cheat sheet: the attributes

| Attribute | Level | Notes |
|---|---|---|
| `gen_ai.operation.name` | Required | Drives the span name |
| `gen_ai.provider.name` | Required | `openai`, `anthropic`, `aws.bedrock`, `gcp.vertex_ai`, `deepseek`, … |
| `gen_ai.request.model` | Cond. required | Exact vendor model string |
| `gen_ai.response.finish_reasons` | Recommended | Array, one per choice; use `error` for a truncated position |
| `gen_ai.usage.input_tokens` | Recommended | **Includes cached tokens** |
| `gen_ai.usage.output_tokens` | Recommended | Billed count, not "consumed", where providers differ |
| `gen_ai.usage.cache_read.input_tokens` | Recommended | Subset of input; the discount line |
| `gen_ai.usage.cache_write.input_tokens` | Recommended | Subset of input; the premium line |
| `gen_ai.usage.reasoning.output_tokens` | Recommended | Thinking tokens |
| `gen_ai.conversation.id` | Cond. required | Only if the framework has a real one |
| `gen_ai.agent.name` / `.id` / `.version` | Cond. required | `id` is the hosted resource ID — not an in-memory object id |
| `gen_ai.input.messages` | Opt-in | Prompt content. Sensitive |
| `gen_ai.output.messages` | Opt-in | Completion content. Sensitive |
| `gen_ai.system_instructions` | Opt-in | System prompt. Sensitive |
| `gen_ai.tool.definitions` | Opt-in | Can be large; not on by default |

The token rule that catches everyone: **`input_tokens` is the total, and the cache counters are
subsets of it**, not additions. A request with 1,200 input tokens of which 900 were cached is
`input_tokens=1200`, `cache_read.input_tokens=900` — not 2,100. Summing them double-bills you.

Also note `gen_ai.conversation.id` must not be invented. The spec is explicit: a new UUID, the
trace id, or a hash of the request are **not** acceptable fallbacks. If the framework has no
conversation identifier, leave the attribute off.

## The metrics were renamed — your dashboards are probably wrong

This is the single most disruptive change in the current draft. The old
`gen_ai.client.token.usage` histogram is **gone** from the docs, replaced by one metric per token
kind:

| Metric | Type | Unit |
|---|---|---|
| `gen_ai.client.inference.usage.input_tokens` | Histogram | `{token}` |
| `gen_ai.client.inference.usage.output_tokens` | Histogram | `{token}` |
| `gen_ai.client.inference.usage.cache_read.input_tokens` | Histogram | `{token}` |
| `gen_ai.client.inference.usage.cache_write.input_tokens` | Histogram | `{token}` |
| `gen_ai.client.inference.usage.reasoning.output_tokens` | Histogram | `{token}` |
| `gen_ai.client.inference.operation.input_tokens` | Histogram | `{token}` |

If you have a Grafana panel on `gen_ai.client.token.usage`, it has been flat-lining. The
`operation.*` variants exist for calls whose usage you only learn after the fact — like
`fetch_response` — where per-request attribution is impossible.

MCP gets its own duration metrics with explicit bucket boundaries:
`mcp.client.operation.duration` and `mcp.server.operation.duration` on
`[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]` seconds. If you override
buckets, override them to that list or cross-server comparisons stop working.

## MCP tracing, and the `_meta` trick

MCP spans are named `{mcp.method.name} {target}` — `tools/call read_file`, `tools/list` — with
`mcp.method.name` **required**. When the client is already tracing the tool execution as a GenAI
`execute_tool` span, the MCP instrumentation should **not** create a second span; it should add
MCP attributes to the existing one. Duplicate spans per tool call are the most common tracing
mistake in agent stacks.

Context propagation is the interesting part. MCP is transport-independent, and one HTTP request
can carry many JSON-RPC messages, so HTTP trace headers are not enough. The convention is to
inject W3C context into the JSON-RPC `params._meta` bag:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "_meta": {
      "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
      "tracestate": "rojo=00f067aa0ba902b7"
    }
  },
  "id": 1
}
```

Two subtleties: the keys are written **unprefixed** even though MCP normally DNS-prefixes `_meta`
keys (that is SEP-414), and the receiver should treat `_meta` as the remote parent while *linking*
the ambient transport context rather than parenting on it. Retries and multiplexing mean one HTTP
request may serve several MCP requests, so the two contexts are genuinely independent.

## Content is a decision, not a default

Instrumentations **must not** capture prompts or completions by default. The spec lays out three
postures, and picking one deliberately is the difference between an observability bill you can
explain and one you cannot:

| Posture | Records | Use for |
|---|---|---|
| Default | Nothing — only counts, latency, model, finish reason | Production, unless you have a reason |
| Attributes | `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions` on the span | Pre-prod, or where telemetry storage meets your privacy bar |
| External reference | A pointer; content uploaded via a hook to storage with its own access control | Production with content-level debugging needs |

The hook runs **regardless of the sampling decision**, so sampled-out traces can still have their
content archived — which is how you debug a 0.1% sample without paying to store everything.

## Snippets

### A correctly-shaped trace (run, not transcribed)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("demo")

with tracer.start_as_current_span("invoke_agent support-bot") as agent:
    agent.set_attribute("gen_ai.operation.name", "invoke_agent")
    agent.set_attribute("gen_ai.provider.name", "openai")
    agent.set_attribute("gen_ai.agent.name", "support-bot")
    agent.set_attribute("gen_ai.conversation.id", "conv_123")

    with tracer.start_as_current_span("chat gpt-4.1-mini") as call:
        call.set_attribute("gen_ai.operation.name", "chat")
        call.set_attribute("gen_ai.provider.name", "openai")
        call.set_attribute("gen_ai.request.model", "gpt-4.1-mini")
        call.set_attribute("gen_ai.usage.input_tokens", 1200)
        call.set_attribute("gen_ai.usage.cache_read.input_tokens", 900)
        call.set_attribute("gen_ai.usage.output_tokens", 180)
        call.set_attribute("gen_ai.response.finish_reasons", ["stop"])
```

Run with `opentelemetry-sdk` 1.45.1 and the console exporter prints exactly that nesting —
`parent_id` of the `chat` span is the `span_id` of `invoke_agent`, and both names match the
required formats. No instrumentation library, no collector: the conventions are a naming
discipline first.

### Injecting context into an MCP request

```python
from opentelemetry.propagate import inject

params = {"name": "read_file", "arguments": {"path": "/repo/README.md"}, "_meta": {}}
inject(params["_meta"])          # adds traceparent / tracestate / baggage
```

The default propagator is W3C trace context plus baggage, which is exactly the key set the
convention expects.

### What a compliant span looks like on the wire

```json
{
  "name": "execute_tool read_file",
  "kind": "SPAN_KIND_CLIENT",
  "attributes": [
    { "key": "gen_ai.operation.name", "value": { "stringValue": "execute_tool" } },
    { "key": "gen_ai.provider.name", "value": { "stringValue": "openai" } },
    { "key": "gen_ai.tool.name", "value": { "stringValue": "read_file" } },
    { "key": "gen_ai.tool.call.id", "value": { "stringValue": "call_001" } }
  ]
}
```

## Typical use cases

| Want | Instrument |
|---|---|
| "Which model is expensive?" | `chat` spans grouped by `gen_ai.request.model`, sum `output_tokens` |
| "Why is this turn slow?" | `invoke_agent` span duration split across child `chat` / `execute_tool` |
| "Does caching help?" | `cache_read.input_tokens` ÷ `input_tokens` per model |
| "Which tool fails?" | `execute_tool {tool}` spans with `error.type` |
| "Is the MCP server the bottleneck?" | `mcp.server.operation.duration` vs the client span |
| "Was this answer grounded?" | `retrieval` spans — but only with the opt-in document attributes |

## Gotchas

- **It is all `Development`.** Pin your semconv version next to your SDK version, or an upgrade
  will silently rename your metrics again.
- **`gen_ai.system` no longer exists.** Current conventions require `gen_ai.provider.name`;
  anything still emitting the old key will not join.
- **Span names must stay low cardinality.** Model or tool name, never session, user or timestamp.
- **Durations are seconds** (`s` in UCUM), not milliseconds. The MCP buckets above are seconds.
- **`input_tokens` includes cached tokens.** Cache counters are subsets. Do not sum the family.
- **`fetch_response` has no token usage.** The spec says not to report it, because no inference
  happened.
- **MCP `mcp.session.id` will be empty against new servers.** The `2026-07-28` MCP revision
  removed protocol sessions entirely, so the attribute has nothing to carry — see
  [the MCP note](/nutshell/model-context-protocol/).
- **Content attributes are PII.** One `gen_ai.input.messages` attribute can hold an entire
  document, and it lands in your tracing backend, not your database.

## Where it sits next to the other notes

[The MCP note](/nutshell/model-context-protocol/) describes the wire this instrument measures, and
[the MCP auth note](/nutshell/mcp-authorization/) explains why a `401` shows up as a span error
with `error.type` set to the JSON-RPC code. [Sandboxing](/nutshell/agent-sandboxing/) covers the
other half of the same question: telemetry tells you what the agent *did*, the sandbox decides
what it *could* have done.
