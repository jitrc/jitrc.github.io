---
layout: nutshell
title: "MCP in a Nutshell"
date: 2026-10-23
summary: "Cheat sheet for the Model Context Protocol: the verbs, what tool definitions and tool results cost in context, stdio vs Streamable HTTP, stateful vs stateless revisions, and the snippets that get a server and client running."
tags: [agents, mcp, protocols]
---

## What it does

MCP is a JSON-RPC 2.0 protocol that lets an LLM application (the **host**) discover and
call capabilities exposed by a separate program (the **server**). One server exposes
tools, resources, and prompts; every MCP-capable host can use it without custom glue.

## Why it exists

- **N × M → N + M.** Before MCP, each app wrote its own integration for each tool. Now one
  server works in every host, and one host speaks to every server.
- **The model gets structure, not scraped text.** Tool names, descriptions, and JSON
  Schemas are generated from code and handed to the model, so calls are validated.
- **Capabilities become a deployment, not a fork.** Ship a server, point hosts at it, and
  revoke it without rebuilding the app.

<figure class="diagram">
<svg viewBox="0 0 640 210" role="img" aria-labelledby="mcp-diagram-title">
  <title id="mcp-diagram-title">An LLM host talks to an MCP server over JSON-RPC, which fronts tools and data</title>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/>
    </marker>
  </defs>

  <rect x="10" y="55" width="180" height="100" rx="6" fill="#ededed" stroke="#666"/>
  <text x="100" y="95" text-anchor="middle" font-size="15" fill="#222">Host</text>
  <text x="100" y="118" text-anchor="middle" font-size="11" fill="#666">LLM + MCP client</text>

  <rect x="245" y="55" width="150" height="100" rx="6" fill="#fff" stroke="#4b0082"/>
  <text x="320" y="95" text-anchor="middle" font-size="15" fill="#222">MCP server</text>
  <text x="320" y="118" text-anchor="middle" font-size="11" fill="#666">tools · resources · prompts</text>

  <rect x="470" y="55" width="160" height="100" rx="6" fill="#ededed" stroke="#666"/>
  <text x="550" y="95" text-anchor="middle" font-size="15" fill="#222">Tools &amp; data</text>
  <text x="550" y="118" text-anchor="middle" font-size="11" fill="#666">files, DB, APIs</text>

  <line x1="192" y1="88" x2="241" y2="88" stroke="#666" stroke-width="1.5" marker-end="url(#arrow)"/>
  <line x1="241" y1="122" x2="192" y2="122" stroke="#666" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="216" y="40" text-anchor="middle" font-size="10" fill="#666">JSON-RPC 2.0</text>
  <text x="216" y="52" text-anchor="middle" font-size="10" fill="#666">stdio / HTTP</text>

  <line x1="397" y1="105" x2="466" y2="105" stroke="#666" stroke-width="1.5" marker-end="url(#arrow)"/>
  <text x="431" y="140" text-anchor="middle" font-size="10" fill="#666">your code</text>
</svg>
<figcaption>The host owns the model; the server owns exactly one integration.</figcaption>
</figure>

## Cheat sheet: the verbs

| Method | What it does | Revision |
|---|---|---|
| `initialize` | Handshake: protocol version, capabilities, client/server info | 2025 only |
| `notifications/initialized` | Client confirms the handshake | 2025 only |
| `server/discover` | Fetch server capabilities up front, no handshake | 2026 |
| `tools/list` | List tools with `name`, `title`, `description`, `inputSchema` | both |
| `tools/call` | Run a tool with `{name, arguments}`; returns `content`, `structuredContent`, `isError` | both |
| `resources/list` | List concrete resources by URI (`file:///…`, `catalog://genres`) | both |
| `resources/templates/list` | List parameterised URIs (`catalog://genres/{genre}`) | both |
| `resources/read` | Read a concrete or template-filled URI | both |
| `prompts/list` / `prompts/get` | List user-invoked templates / render one to messages | both |
| `completion/complete` | Autocomplete prompt or resource-template arguments | both |
| `notifications/cancelled` | Cancel an in-flight request | both |
| `notifications/tools/list_changed` | Tool list changed; refresh caches | 2025 (2026 adds `ttlMs`) |

<div class="callout">
<strong>Two revisions matter right now</strong>
<code>2025-11-25</code> is the latest stable: stateful sessions over Streamable HTTP, opened by an
<code>initialize</code> handshake. <code>2026-07-28</code> is the release candidate that makes the
protocol <em>stateless</em>: no handshake, no session header, capabilities travel in
<code>_meta</code> on every request. SDKs negotiate per connection, so the snippets below work
against both — but raw JSON-RPC must pick one.
</div>

## Context is the real cost

Every token an MCP server contributes is a token the model cannot use for the actual task,
and there are two bills: **definitions on the way in**, and **results on the way out**.

The published numbers are sobering. GitHub's server is ~26K tokens of definitions for 35
tools; a five-server setup (GitHub, Slack, Sentry, Grafana, Splunk) lands around 55–77K
tokens **before the user's first message**. Tool *output* is usually worse: one unfiltered
API call can return hundreds of KB of JSON, and hosts like Claude Code cap tool responses
at roughly 25K tokens by default for exactly this reason.

### Definition bloat (charged on every request)

| Cost | Fix |
|---|---|
| Too many tools per server | Split by workflow (read-only vs admin), not by resource |
| Long descriptions | Two sentences: what it does, when **not** to use it |
| Big enums / nested schemas | Free-form `string` the server validates, or a `resource` holding the vocabulary |
| Schemas resent every turn | Host-side tool search / deferred loading (~85% reduction in published tests) |
| Always-on servers you rarely use | Disable the server, not just the tool |

### Response bloat (charged once, but permanently)

- **Default to small.** `limit=20` with `has_more` + `cursor` beats "return everything".
- **Return a pointer, not a payload.** Hand back a `resource` URI and let the model decide
  to read it, instead of dumping 12K tokens of file content.
- **Truncate explicitly.** A result ending in `[truncated 4,812 of 5,000 lines — filter by
  path or use offset]` teaches the model to refine. Silent truncation teaches it nothing.
- **Put code-facing data in `structuredContent`.** `content` is what enters the model's
  context; `structuredContent` is for your application. Summarise in the former, be exact
  in the latter.
- **Push aggregation into the server.** "Top 10 errors grouped by service" is 200 tokens;
  the raw log slice it came from is 20K.

```python
# A result shape that respects the context window
@mcp.tool()
def search(query: str, limit: int = 10, cursor: str | None = None) -> dict:
    """Search incidents. Returns at most `limit` matches."""
    page = index.search(query, limit=limit, cursor=cursor)
    return {
        "items": [{"id": h.id, "title": h.title, "score": h.score} for h in page],
        "has_more": page.next_cursor is not None,
        "next_cursor": page.next_cursor,
    }
```

Measure it instead of guessing: whatever `tools/list` returns is roughly
`characters / 4` tokens per request, times every server you enable.

## Transports: stdio vs Streamable HTTP

| | stdio | Streamable HTTP |
|---|---|---|
| Who starts it | Host spawns a child process | You deploy it; clients connect by URL |
| Wire format | Newline-delimited JSON-RPC on stdin/stdout | One endpoint, POST + optional SSE |
| Auth | Inherits the user's permissions | Real HTTP auth (OAuth, bearer) |
| Scaling | One process per user | Round-robin behind a load balancer |
| Use when | Local tools: filesystem, git, a CLI | Shared, remote, multi-tenant servers |

**stdio rules that bite:** never write logs or banners to stdout — it corrupts the stream;
use stderr. Messages must not contain embedded newlines.

**Streamable HTTP essentials:** one path handles `POST` (requests) and `GET` (server→client
stream). Clients must send `Accept: application/json, text/event-stream`; a request may get
a single JSON response *or* an SSE stream, and notifications get `202 Accepted` with no
body. Replies can arrive as SSE events with IDs, and a dropped connection resumes with
`Last-Event-ID`. Validate `Origin` and bind to localhost if local: DNS rebinding against a
local MCP server is a real attack.

The older **HTTP+SSE** transport (two endpoints, 2024-11-05) is deprecated. "We use SSE" in
a 2026 doc almost always means Streamable HTTP with SSE framing, not that.

## Stateful vs stateless

The state question is really three different questions:

1. **Protocol state** — does the connection carry a session?
2. **Application state** — does the server remember anything between calls?
3. **Transport state** — is there a long-lived socket?

| | `2025-11-25` | `2026-07-28` |
|---|---|---|
| Handshake | `initialize` → `initialized`, once | none; `_meta` on every request |
| Session | Server mints `Mcp-Session-Id`; client echoes it | removed entirely |
| Load balancing | Sticky routing + shared session store | Any instance, plain round-robin |
| Server→client asks | Any time over the session stream | Only while handling a request, via `input_required` + retry |
| Caching | `list_changed` notifications | `ttlMs` / `cacheScope` on list results |

**Stateless protocol, stateful applications.** Dropping sessions does not forbid state; it
moves it into your payload. Mint a handle and let the model thread it:

```json
{"name":"create_basket","arguments":{}}                 // → {"basket_id":"b_42"}
{"name":"add_item","arguments":{"basket_id":"b_42","sku":"A-1"}}
```

That is more robust than hidden session state: the handle can be logged, cached, retried on
another instance, or passed between tools. If you are on `2025-11-25`, sticky sessions and a
shared store (Redis) are mandatory for horizontal scale; on `2026-07-28` they are not.

For local stdio servers, "stateful" is trivial — the process lives as long as the session.
For HTTP, assume any request can hit any instance and design accordingly.

## Snippets

### Install

```bash
uv add "mcp[cli]"        # or: pip install "mcp[cli]"   (Python 3.11+)
```

### Minimal server

```python
# server.py
from mcp.server import MCPServer

mcp = MCPServer("Demo", instructions="Small demo server.")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()                  # stdio, for local hosts
```

Type hints become the JSON Schema; the docstring becomes the description the model reads.
No schema file, no request parsing.

### Run and inspect it

```bash
uv run mcp dev server.py                                  # opens the MCP Inspector UI
uv run mcp run server.py --transport streamable-http       # http://localhost:8000/mcp
```

### Minimal client

```python
import anyio
from mcp import Client

async def main() -> None:
    async with Client("http://localhost:8000/mcp") as client:
        print(client.server_info, client.protocol_version)

        tools = await client.list_tools()
        print([t.name for t in tools.tools])

        result = await client.call_tool("add", {"a": 2, "b": 3})
        print(result.content, result.structured_content, result.is_error)

anyio.run(main)
```

`Client(...)` also accepts `StdioServerParameters` (spawn a subprocess) or a server object
(in-process, for tests — no port, no process).

### Point a host at it

```json
{
  "mcpServers": {
    "demo": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/project", "mcp", "run", "server.py"]
    },
    "bookshop": {
      "url": "https://bookshop.internal/mcp",
      "headers": { "Authorization": "Bearer ${BOOKSHOP_TOKEN}" }
    }
  }
}
```

### Raw JSON-RPC over HTTP

```bash
curl -s http://localhost:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call",
       "params":{"name":"add","arguments":{"a":2,"b":3}}}'
```

Both `Content-Type` and that exact `Accept` pair are required; the response is SSE-framed
JSON-RPC, one `data:` line per message.

## Typical use cases

| Scenario | What you expose |
|---|---|
| Coding agent needs the repo | `read_file`, `search`, `apply_patch` tools + file resources |
| Internal data assistant | A `query_*` tool per safe query shape, plus schema resources |
| Support/ops agent | Ticket lookup and escalation tools; runbooks as prompts |
| Observability | `logs_search`, `metrics_query` tools; dashboards as resources |
| Third-party SaaS | A hosted Streamable HTTP server with OAuth, one per tenant |

Reach for MCP when the capability should outlive one app, or when more than one host
should be able to use it. If the tool only exists to serve one prompt in one product, a
plain function call is cheaper.

## Gotchas

- **Tool output is untrusted input.** A tool can return "ignore previous instructions" —
  keep output in the data channel and keep approvals in the host.
- **`isError` is a result, not an exception.** `call_tool` returns normally when the tool
  raises; check `isError` before trusting `structuredContent`.
- **Never log to stdout on stdio.** It corrupts the JSON-RPC stream; use stderr.
- **Descriptions are the prompt.** Vague tool descriptions cause wrong calls far more often
  than weak models do.
- **Servers inherit your permissions.** For local stdio servers, the container/VM is the
  real security boundary, not the tool name.
- **Version skew is silent.** `2025-11-25` and `2026-07-28` differ on sessions and
  handshakes; negotiate instead of assuming.
- **`sampling`, `roots`, and `logging` are deprecated** as of `2026-07-28` — use direct LLM
  calls, tool parameters, and OpenTelemetry instead.

## Where it sits next to ACP

MCP is model ↔ tool. **ACP** (Agent Client Protocol) is editor ↔ agent: launching an
agent, streaming its plan and diffs, answering permission prompts. They compose — an ACP
agent typically consumes MCP servers internally. Wiring an agent into an editor: ACP.
Giving an agent a new capability: MCP.
