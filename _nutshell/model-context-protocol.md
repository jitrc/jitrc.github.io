---
layout: nutshell
title: "MCP in a Nutshell"
date: 2026-10-23
summary: "The Model Context Protocol is JSON-RPC between an LLM app and the servers that expose its tools and data. Here is the whole shape of it in one page."
tags: [agents, mcp, protocols]
---

MCP (Model Context Protocol) standardises one thing: how an LLM application asks for
capabilities it did not ship with. Instead of every IDE and chat app inventing its own
plugin API, both sides speak the same JSON-RPC 2.0 dialect.

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
  <text x="320" y="118" text-anchor="middle" font-size="11" fill="#666">capability provider</text>

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
<figcaption>The host owns the model and the client connection; the server owns one integration.</figcaption>
</figure>

## The three roles

- **Host** — the app the user runs: an IDE, a chat client, an agent runtime. It owns the
  model, the user consent UI, and one **client** per connected server.
- **Client** — the connection object inside the host. One client ↔ one server.
- **Server** — a small process or HTTP service that exposes capabilities. It does not
  call the model; it only answers requests and emits notifications.

A single host typically runs several servers side by side (filesystem, git, a database,
an internal API). That is the whole point: integrations compose without the host
knowing anything about them.

## What a server can expose

| Primitive | Direction | Use it for |
|---|---|---|
| **Tools** | model-invoked | Actions with side effects: run a query, write a file, call an API |
| **Resources** | app-controlled | Read-only context addressed by URI: a file, a schema, a log |
| **Prompts** | user-invoked | Slash-command style templates the user picks explicitly |

Tools are what most people mean by "MCP". Resources are the quiet workhorse for
giving the model grounded context without pretending it is an action.

Two newer capabilities cut the other way: **sampling** lets a server ask the host's model
to generate text, and **elicitation** lets it ask the user a structured question. Both
require explicit host support, so negotiate them rather than assuming.

## The handshake

Every session starts with `initialize`, and nothing else may be sent first:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
  "protocolVersion":"2025-06-18",
  "capabilities":{"roots":{"listChanged":true}},
  "clientInfo":{"name":"my-ide","version":"1.2.0"}}}
```

The server replies with its own `protocolVersion`, `capabilities`, and `serverInfo`, and
the client sends an `initialized` notification to close the loop. After that, calls look
like ordinary JSON-RPC:

```json
{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{
  "name":"query_orders","arguments":{"since":"2026-10-01"}}}
```

<div class="callout">
<strong>Version skew is the usual failure</strong>
The protocol ships revisions, not semantic-version patches. If the versions differ, the
client must fall back to one it supports or abort — parsing a newer server's response with
older assumptions is how you get silent truncation of tool results.
</div>

## Transport

- **stdio** — host spawns the server as a child process and speaks newline-delimited
  JSON-RPC on stdin/stdout. Simplest and the default for local tools. Never write logs to
  stdout; use stderr or you corrupt the stream.
- **Streamable HTTP** — a single endpoint handling POST and optional SSE, with session
  IDs and resumability. Use it for shared or remote servers, and put real auth in front
  of it. The older HTTP+SSE transport is deprecated.

## Gotchas worth knowing before you ship

- **Tool results are untrusted input.** A tool can return text that says "ignore previous
  instructions". Treat server output as data, never as instructions, and keep the approval
  step in the host.
- **Describe tools for a model, not for a human.** Names, descriptions, and JSON schemas
  are the prompt. Vague descriptions produce wrong calls far more often than weak models do.
- **Listings change at runtime.** `tools/list_changed` (and the resources/prompts
  equivalents) means the host must refresh; cached tool lists go stale when a server adds
  or drops a tool mid-session.
- **Timeouts belong to the host.** Servers are ordinary programs that can hang. Bound every
  call, and make cancellation (`notifications/cancelled`) actually tear down the work.
- **Sandbox the server, not just the tool.** A local stdio server runs with your user's
  permissions. If it shells out, the container or VM boundary is the real security control.

## Where it sits next to ACP

MCP is model ↔ tool. **ACP** (Agent Client Protocol) is editor ↔ agent: it standardises
how an IDE launches an agent, streams its plan and diffs, and answers permission prompts.
They are complementary — an ACP agent will usually consume MCP servers internally, and a
given product can be both. If you are wiring an agent into an editor, reach for ACP; if you
are giving an agent a new capability, reach for MCP.
