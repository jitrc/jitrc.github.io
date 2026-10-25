---
layout: nutshell
title: "ACP in a Nutshell"
date: 2026-10-24
image:
  path: /assets/images/og/agent-client-protocol.png
  width: 1200
  height: 630
  alt: "ACP in a Nutshell — jitrc.com"
summary: "Cheat sheet for the Agent Client Protocol: who calls whom, the prompt-turn lifecycle, and why the agent calls back into your editor."
tags: [agents, acp, protocols]
---


<div class="callout">
<strong>Verified against</strong> ACP schema <code>v1.26.0</code> and Rust crate <code>v1.12.0</code>, both
released 2026-10-16 · <code>agent-client-protocol</code> 0.12.1 (PyPI) · <code>@agentclientprotocol/sdk</code>
2.0.2 (npm). Checked 2026-10-24. A v2 schema exists as <code>v2.0.0-alpha.9</code>; the differences are
in <a href="#revisions">Revisions</a> below.
</div>

## What it does

ACP is a JSON-RPC 2.0 protocol between a **client** (an editor or IDE) and an **agent** (a
coding agent). The client launches the agent as a subprocess and they talk over stdio. The
agent streams messages, plans, tool calls and diffs back; when it needs to read a file, run a
command, or get approval, it calls the *client* and the editor does the work.

## Why it exists

- **N × M → N + M.** Every editor used to write a bespoke integration per agent, and every
  agent a bespoke adapter per editor. Now one editor speaks to every ACP agent, and one agent
  works in every ACP editor. The docs make the LSP comparison explicitly.
- **The agent borrows the editor instead of reimplementing it.** This is the part people miss:
  the callbacks flow *backwards*. The agent gets the user's unsaved buffers, the user's
  terminal, and the user's approval UI — instead of shadowing a copy of the repo and inventing
  its own permission prompt.
- **It composes with MCP rather than competing with it.** ACP carries an `mcpServers` list into
  `session/new`; the agent connects to those servers itself. ACP is the editor↔agent wire, MCP
  is the agent↔tool wire.

<figure class="diagram">
<svg viewBox="0 0 640 215" role="img" aria-labelledby="acp-diagram-title">
  <title id="acp-diagram-title">An editor spawns an agent over JSON-RPC on stdio; the agent calls back for permissions, files and terminals, and reaches the model and MCP servers itself</title>
  <defs>
    <marker id="acp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/>
    </marker>
  </defs>

  <rect x="10" y="35" width="160" height="110" rx="6" fill="#ededed" stroke="#666"/>
  <text x="90" y="75" text-anchor="middle" font-size="15" fill="#222">Client</text>
  <text x="90" y="96" text-anchor="middle" font-size="11" fill="#666">editor / IDE</text>
  <text x="90" y="114" text-anchor="middle" font-size="11" fill="#666">files · terminal · UI</text>

  <rect x="250" y="35" width="160" height="110" rx="6" fill="#fff" stroke="#4b0082"/>
  <text x="330" y="75" text-anchor="middle" font-size="15" fill="#222">Agent</text>
  <text x="330" y="96" text-anchor="middle" font-size="11" fill="#666">subprocess, stdio</text>
  <text x="330" y="114" text-anchor="middle" font-size="11" fill="#666">owns the loop</text>

  <rect x="490" y="14" width="140" height="60" rx="6" fill="#ededed" stroke="#666"/>
  <text x="560" y="42" text-anchor="middle" font-size="13" fill="#222">Model API</text>
  <text x="560" y="60" text-anchor="middle" font-size="10" fill="#666">your key, your choice</text>

  <rect x="490" y="106" width="140" height="60" rx="6" fill="#ededed" stroke="#666"/>
  <text x="560" y="134" text-anchor="middle" font-size="13" fill="#222">MCP servers</text>
  <text x="560" y="152" text-anchor="middle" font-size="10" fill="#666">passed at session/new</text>

  <line x1="172" y1="90" x2="246" y2="90" stroke="#666" stroke-width="1.5"
        marker-start="url(#acp-arrow)" marker-end="url(#acp-arrow)"/>
  <text x="209" y="80" text-anchor="middle" font-size="9" fill="#666">JSON-RPC 2.0</text>
  <text x="209" y="106" text-anchor="middle" font-size="9" fill="#666">stdio</text>

  <line x1="412" y1="62" x2="486" y2="46" stroke="#666" stroke-width="1.5" marker-end="url(#acp-arrow)"/>
  <line x1="412" y1="120" x2="486" y2="134" stroke="#666" stroke-width="1.5" marker-end="url(#acp-arrow)"/>

  <text x="320" y="180" text-anchor="middle" font-size="10" fill="#666">Client → Agent: session/new · session/prompt · session/cancel</text>
  <text x="320" y="196" text-anchor="middle" font-size="10" fill="#666">Agent → Client: session/update · request_permission · fs/* · terminal/*</text>
</svg>
<figcaption>Both ends are servers. The editor answers file, terminal and permission requests; the agent never gets ambient access to your machine.</figcaption>
</figure>

## Cheat sheet: the methods

| Method | Direction | What it does |
|---|---|---|
| `initialize` | C→A | Negotiate `protocolVersion` (a single integer), capabilities, `authMethods` |
| `authenticate` | C→A | Run an advertised `type: "agent"` login method |
| `logout` | C→A | End the authenticated state (needs `agentCapabilities.auth.logout`) |
| `session/new` | C→A | Create a session from `cwd` + `mcpServers`; returns `sessionId` |
| `session/load` | C→A | Restore a session **and replay history** (needs `loadSession`) |
| `session/resume` | C→A | Reattach without replay (`sessionCapabilities.resume`) |
| `session/close` | C→A | Cancel work and free the session (`sessionCapabilities.close`) |
| `session/prompt` | C→A | Send a turn; resolves only when the turn ends, with a `stopReason` |
| `session/cancel` | C→A, notif | Interrupt the turn; no response |
| `session/set_mode` | C→A | Switch operating mode (legacy — see config options) |
| `session/set_config_option` | C→A | Set a model / mode / reasoning selector; returns the full config |
| `session/update` | A→C, notif | Chunks, tool calls, plans, usage, notices, compaction |
| `session/request_permission` | A→C | Ask the user; returns a selected `optionId` or `cancelled` |
| `fs/read_text_file` | A→C | Read a file — **including unsaved editor changes** |
| `fs/write_text_file` | A→C | Write through the editor so it can track/diff the change |
| `terminal/create` … | A→C | `output`, `wait_for_exit`, `kill`, `release` against a real terminal |
| `elicitation/create` | A→C | Ask for structured input (form or out-of-band URL) |

Direction columns aside, the split that matters is **capabilities**: `clientCapabilities` gates
what the agent may ask the client to do, `agentCapabilities` gates what the client may expect.
Anything omitted is **unsupported** — not "probably fine".

## The prompt turn, in order

1. `initialize` — once per connection.
2. `session/new` — once per conversation; `cwd` must be absolute and is the root boundary.
3. `session/prompt` — the client sends content blocks and waits.
4. Meanwhile, the agent emits `session/update` notifications: `agent_message_chunk`,
   `agent_thought_chunk`, `plan`, `tool_call`, `tool_call_update`, `available_commands_update`,
   `current_mode_update`, `config_option_update`, `usage_update`, `notice`,
   `compaction_update`.
5. If it needs approval, it calls `session/request_permission` and blocks on the answer.
6. The turn ends with `session/prompt` resolving to one `stopReason`:

| `stopReason` | Meaning |
|---|---|
| `end_turn` | The model finished without asking for more tools |
| `max_tokens` | Token limit hit |
| `max_turn_requests` | Too many model round trips in one turn |
| `refusal` | The agent declined |
| `cancelled` | The client cancelled |

<figure class="diagram">
<svg viewBox="0 0 640 168" role="img" aria-labelledby="acp-turn-title">
  <title id="acp-turn-title">A prompt turn: session/prompt opens it, session/update notifications stream in between, the response closes it with a stop reason</title>
  <defs>
    <marker id="turn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#4b0082"/>
    </marker>
  </defs>

  <line x1="20" y1="40" x2="620" y2="40" stroke="#ccc" stroke-width="2"/>
  <line x1="20" y1="40" x2="620" y2="40" stroke="#4b0082" stroke-width="2" marker-end="url(#turn-arrow)"/>

  <circle cx="40" cy="40" r="6" fill="#4b0082"/>
  <circle cx="600" cy="40" r="6" fill="#4b0082"/>

  <text x="40" y="22" text-anchor="start" font-size="11" fill="#222">session/prompt</text>
  <text x="600" y="22" text-anchor="end" font-size="11" fill="#222">stopReason</text>

  <circle cx="200" cy="40" r="4" fill="#fff" stroke="#4b0082" stroke-width="2"/>
  <circle cx="320" cy="40" r="4" fill="#fff" stroke="#4b0082" stroke-width="2"/>
  <circle cx="440" cy="40" r="4" fill="#fff" stroke="#4b0082" stroke-width="2"/>

  <text x="200" y="66" text-anchor="middle" font-size="10" fill="#666">update</text>
  <text x="320" y="66" text-anchor="middle" font-size="10" fill="#666">tool_call</text>
  <text x="440" y="66" text-anchor="middle" font-size="10" fill="#666">request_permission</text>

  <text x="200" y="82" text-anchor="middle" font-size="10" fill="#666">agent_message_chunk</text>
  <text x="320" y="82" text-anchor="middle" font-size="10" fill="#666">or terminal/*</text>
  <text x="440" y="82" text-anchor="middle" font-size="10" fill="#666">→ approve / reject</text>

  <text x="320" y="128" text-anchor="middle" font-size="10" fill="#666">The client can end the turn early with session/cancel; the agent must still answer</text>
  <text x="320" y="145" text-anchor="middle" font-size="10" fill="#666">session/prompt with stopReason: cancelled — not an error.</text>
</svg>
<figcaption>One method opens and closes the turn; everything interesting happens as notifications in between.</figcaption>
</figure>

### Cancellation is semantic, not exceptional

The spec is unusually blunt here, because implementations keep getting it wrong: abort APIs
usually **throw**, and an agent that lets that exception escape turns a user pressing Escape
into a red error toast. Agents must catch the abort and return `stopReason: "cancelled"`. On
the client side, every pending `session/request_permission` must be answered with
`outcome: {"outcome": "cancelled"}` when the turn is cancelled.

## Revisions {#revisions}

<div class="callout">
<strong>v1 is stable, v2 is a draft.</strong> Schema <code>v1.26.0</code> is the latest stable release
(2026-10-16). <code>v2.0.0-alpha.9</code> shipped the same day and the v2 docs are published as a draft.
Build against v1; read the v2 notes so nothing surprises you.
</div>

What v2 changes, per the draft overview:

| v1 | v2 draft |
|---|---|
| `authenticate` | `auth/login` — and `auth/logout` becomes mandatory whenever `authMethods` is non-empty |
| `logout` (optional) | `auth/logout` (baseline when auth is advertised) |
| `session/load` replays history | `session/resume` is baseline; resume handles optional replay |
| `session/list`, `session/close` optional | Both baseline |
| `session/prompt` resolves at the end of the turn | Resolves when the prompt is **accepted**; completion arrives as an idle `state_update` with a stop reason |

Within v1, two deprecations are already live and will bite you:

- **`session/set_mode` → config options.** Agents that expose modes should send both during
  the transition; clients that understand `configOptions` should ignore `modes`.
- **`session/load` capability lives at the top level** (`agentCapabilities.loadSession`) while
  `resume`/`close`/`delete` live under `sessionCapabilities`. The spec says this will be
  unified; today you have to look in two places.

## Transports, auth, and who owns what

| | stdio | Streamable HTTP / WebSocket |
|---|---|---|
| Status | The only specified transport | RFD in progress |
| Who starts | Client spawns the agent | Would be remote |
| Use | Local agents — every real integration today | Future cloud agents |

**stdio rules that bite:** messages are newline-delimited and must not contain embedded
newlines. The agent **must not** write anything to stdout that is not an ACP message — a stray
`print()` or a progress bar corrupts the stream. Logs go to stderr. Same rule in reverse for
the client's stdin.

**Auth** happens at `initialize`: the agent returns `authMethods`, and those come in two types.

| `type` | What the client does |
|---|---|
| `agent` (default) | Calls `authenticate {methodId}` and the agent runs its own flow |
| `terminal` | Launches the agent binary interactively with the method's `args`/`env`, waits for exit 0, then reconnects — and **never** calls `authenticate` |

A `terminal` method may only be advertised when the client sent `clientCapabilities.auth.terminal`.
That capability means "I can reproduce your launch command in a real terminal" — not "I support
auth".

## Modes are becoming config options

Agents can expose arbitrary session selectors: model, mode, reasoning level, or a boolean flag.
The client renders them, the agent owns the semantics.

```json
{
  "configOptions": [
    { "id": "model", "category": "model", "type": "select",
      "currentValue": "fast",
      "options": [ { "value": "fast", "name": "Fast" },
                   { "value": "deep", "name": "Deep" } ] }
  ]
}
```

- `session/set_config_option` returns the **complete** option list, so agents can express
  dependencies (changing the model may change which reasoning levels exist).
- Categories — `mode`, `model`, `model_config`, `thought_level` — are UX hints only. Clients
  must work when they are missing or unknown.
- Boolean options require the client to advertise
  `session.configOptions.boolean: {}` first.
- Every option must have a default, because the agent has to run when the client shows nothing.

## Permissions are the actual security boundary

Tool calls carry a `kind` (`read`, `edit`, `delete`, `move`, `search`, `execute`, `think`,
`fetch`, `switch_mode`, `other`) and a `status` (`pending`, `in_progress`, `completed`,
`failed`). Both are **display hints** — they do not grant or restrict anything.

Authorization is the four option kinds the agent offers, and the client's answer:

| Kind | Meaning |
|---|---|
| `allow_once` | This operation, this time |
| `allow_always` | Remember the choice |
| `reject_once` | No, this time |
| `reject_always` | No, and remember it |

Clients may auto-approve according to their own settings — which is why the option list, not
the tool name, is the contract. An agent that invents an option kind out of band (say, a
"trust this repo" checkbox) is outside the protocol and every other client will drop it.

## Snippets

### Install

```bash
uv add agent-client-protocol          # Python; needs 3.10–3.14
npm install @agentclientprotocol/sdk  # TypeScript
```

### A minimal agent (verified over real stdio)

```python
# agent.py
import asyncio
import uuid

from acp import (
    NewSessionResponse,
    PromptResponse,
    run_agent,
    update_agent_message_text,
)


class Echo:
    def __init__(self):
        self._conn = None

    def on_connect(self, conn):
        self._conn = conn

    async def initialize(self, protocol_version, **kwargs):
        return {"protocolVersion": 1}

    async def new_session(self, cwd, **kwargs):
        return NewSessionResponse(sessionId=str(uuid.uuid4()))

    async def prompt(self, session_id, prompt, **kwargs):
        text = "".join(b.text for b in prompt if getattr(b, "type", "") == "text")
        for word in text.split():
            await self._conn.session_update(
                session_id, update_agent_message_text(word + " ")
            )
        return PromptResponse(stopReason="end_turn")


if __name__ == "__main__":
    asyncio.run(run_agent(Echo()))
```

Run against it with raw JSON-RPC and you get exactly this:

```console
$ (printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{}}}' \
    '{"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}}'
   sleep 2) | python agent.py
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1}}
{"jsonrpc":"2.0","id":2,"result":{"sessionId":"4ec75b25-…"}}
```

Two things the docs gloss over and this snippet encodes: `run_agent()` is a **coroutine**
(`asyncio.run(...)`, not a bare call), and stdio must actually be a pipe — redirecting a file
into stdin fails with `ValueError: Pipe transport is for pipes/sockets only`. Use a pipeline,
not `< file`.

### A client that drives it

```python
import asyncio
import sys
from pathlib import Path

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block


class DemoClient:
    async def request_permission(self, session_id, tool_call, options, **kwargs):
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(self, session_id, update, **kwargs):
        if update.session_update == "agent_message_chunk":
            print(update.content.text, end="", flush=True)


async def main():
    script = Path(__file__).with_name("agent.py")
    async with spawn_agent_process(DemoClient(), sys.executable, str(script)) as (conn, _proc):
        await conn.initialize(protocol_version=PROTOCOL_VERSION)
        session = await conn.new_session(cwd="/tmp", mcp_servers=[])
        result = await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("hello from a test client")],
        )
        print("\nstopReason:", result.stop_reason)


asyncio.run(main())
```

### Point an editor at it

```json
{
  "agent_servers": {
    "Echo Agent (Python)": {
      "type": "custom",
      "command": "uv",
      "args": ["run", "/abs/path/to/agent.py"]
    }
  }
}
```

That is Zed's `settings.json`. JetBrains, Neovim (CodeCompanion, avante.nvim, agentic.nvim),
Emacs (`agent-shell.el`), Sublime, Qt Creator, marimo, Jupyter and Obsidian all have clients;
the launch config shape differs, the wire does not.

### Check whether you are actually conformant

```bash
git clone https://github.com/agentclientprotocol/acp-tck && cd acp-tck
# drives your agent through init, sessions, prompts, cancel and error handling
```

The TCK is explicitly **experimental** — it doesn't cover the whole protocol and a green run
is not a conformance claim. It is still the fastest way to find the mistakes that make one
editor work and another hang.

## Typical use cases

| Scenario | What you build |
|---|---|
| Ship one agent into every editor | An ACP agent; skip nine bespoke integrations |
| Add a second agent to your editor | An ACP client; skip a parser per agent |
| Drive a coding agent from CI or a script | `spawn_agent_process` + `session/prompt`, no TUI |
| Put an agent behind a chat platform | A client that maps `session/update` to messages and permission requests to buttons |
| Run agents on your own hardware | stdio locally; the remote transports are still RFD |

Reach for ACP when the agent needs a *workspace* — files, a shell, a human approving things.
If the agent only needs tools and data, MCP is the shorter path; if it only produces text, you
need neither.

## Gotchas

- **stdout is the wire.** Any non-ACP byte on stdout breaks the session; `print()` debugging is
  the single most common way to hang a client.
- **`session/prompt` blocks for the whole turn.** Awaiting it in the UI loop freezes the editor
  until the agent is done. Render `session/update` as it arrives.
- **Omitted capability means no.** Don't call `terminal/*` because the platform has a terminal;
  check `clientCapabilities.terminal`.
- **Never `authenticate` a `terminal` method.** That path is exit-code-based, not JSON-RPC.
- **Terminal handles leak.** `terminal/kill` does not free; `terminal/release` does, and the
  agent must call it.
- **Paths absolute, lines 1-based.** The spec requires it, and relative paths silently resolve
  against `cwd` in some agents and the launch directory in others.
- **`cwd` + `additionalDirectories` are the intended sandbox root.** A tool that writes outside
  them is a bug in the agent, not a feature.
- **Group chunks by `messageId`.** Agents may include it; consecutive chunks with the same ID
  are one message. Ignoring it turns done/retry into duplicated text.
- **Content blocks are MCP's `ContentBlock`.** That is deliberate: MCP tool output can be
  forwarded to the client without translation.
- **Keep both mode APIs in sync during the transition.** Send `configOptions` *and* `modes`, or
  half your clients will show a stale mode.

## Where it sits next to MCP

MCP is model ↔ tool. **ACP is editor ↔ agent** — launching an agent, streaming its plan and
tool calls, answering permission prompts, handing it the user's files and terminal. They
compose: ACP carries the `mcpServers` list into `session/new` and the agent connects to them
itself, so a single ACP integration inherits every MCP server your users already run.

The mental test: **does this need a workspace?** Files, a shell, approvals, a diff — ACP. Just
capabilities for the model — MCP.
