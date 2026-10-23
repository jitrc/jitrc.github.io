---
layout: nutshell
title: "MCP in a Nutshell"
date: 2026-10-23
summary: "Cheat sheet for the Model Context Protocol: the verbs, the snippets that get a server and client running, and when to reach for it instead of a bespoke tool API."
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

| Method | What it does |
|---|---|
| `initialize` | Handshake: exchange protocol version, capabilities, client/server info |
| `notifications/initialized` | Client confirms the handshake; nothing else may be sent before this |
| `tools/list` | List tools with `name`, `title`, `description`, `inputSchema` |
| `tools/call` | Run a tool with `{name, arguments}`; returns `content`, `structuredContent`, `isError` |
| `resources/list` | List concrete resources by URI (`file:///…`, `catalog://genres`) |
| `resources/templates/list` | List parameterised URIs (`catalog://genres/{genre}`) |
| `resources/read` | Read a concrete or template-filled URI |
| `prompts/list` / `prompts/get` | List user-invoked templates / render one to messages |
| `completion/complete` | Autocomplete prompt or resource-template arguments |
| `notifications/cancelled` | Cancel an in-flight request |
| `notifications/tools/list_changed` | Server signal that the tool list changed; refresh caches |

<div class="callout">
<strong>Protocol versions move fast</strong>
Latest stable revision is <code>2025-11-25</code>; the <code>2026-07-28</code> revision is a
release candidate that removes sessions and the <code>initialize</code> handshake. SDK v2
negotiates per connection, so the snippets below work against both — but pin versions when
you write raw JSON-RPC.
</div>

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
- **Never log to stdout on stdio transport.** It corrupts the JSON-RPC stream; use stderr.
- **Descriptions are the prompt.** Vague tool descriptions cause wrong calls far more often
  than weak models do.
- **Servers inherit your permissions.** For local stdio servers, the container/VM is the
  real security boundary, not the tool name.
- **Version skew is silent.** `2025-11-25` and `2026-07-28` differ on sessions and
  handshakes; negotiate instead of assuming.

## Where it sits next to ACP

MCP is model ↔ tool. **ACP** (Agent Client Protocol) is editor ↔ agent: launching an
agent, streaming its plan and diffs, answering permission prompts. They compose — an ACP
agent typically consumes MCP servers internally. Wiring an agent into an editor: ACP.
Giving an agent a new capability: MCP.
