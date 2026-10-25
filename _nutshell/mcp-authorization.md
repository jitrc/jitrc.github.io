---
layout: nutshell
title: "MCP Auth in a Nutshell"
date: 2026-10-24
image:
  path: /assets/images/og/mcp-authorization.png
  width: 1200
  height: 630
  alt: "MCP Auth in a Nutshell — jitrc.com"
summary: "Cheat sheet for MCP authorization: the discovery chain from a 401 to a token, audience binding, the attacks the spec names, and what the stateless revision changed."
tags: [agents, mcp, security]
---


<div class="callout">
<strong>Verified 2026-10-24</strong> against the MCP specification revisions <code>2025-11-25</code> (stable) and <code>2026-07-28</code> (current draft), plus that revision's changelog. The discovery walkthrough below was captured live from <code>mcp.linear.app</code> and <code>mcp.stripe.com</code>, so the URLs, headers and field names are real responses rather than transcribed examples.
</div>

## What it does

MCP authorization is **OAuth 2.1**, with the MCP server playing the *resource server* role. The
client discovers the authorization server from the MCP server itself, gets a token scoped and
audience-bound to that one server, and sends it on every request. There is no MCP-specific token
format and no shared secret between client and server.

## Why it exists

- **Delegation without a password.** The agent acts for a user; the user's credentials never
  reach the agent. Revocation is a token revoke, not a password change.
- **Audience binding is the whole point.** A token minted for `mcp.example.com` is worthless at
  `mcp.other.com`. Without that check, one compromised server becomes a skeleton key.
- **Least privilege that can grow.** Scopes start minimal and escalate on a `403` with
  `insufficient_scope`, so a read-only session never silently holds write.

<figure class="diagram">
<svg viewBox="0 0 640 250" role="img" aria-labelledby="mcp-auth-diagram-title">
  <title id="mcp-auth-diagram-title">Discovery chain: an unauthenticated MCP request returns 401 with a resource_metadata URL, which names the authorization server, whose metadata supplies endpoints for an OAuth 2.1 code flow with PKCE</title>
  <defs>
    <marker id="auth-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/>
    </marker>
  </defs>

  <rect x="10" y="20" width="170" height="52" rx="5" fill="#ededed" stroke="#666"/>
  <text x="95" y="41" text-anchor="middle" font-size="12" fill="#222">Client</text>
  <text x="95" y="58" text-anchor="middle" font-size="10" fill="#666">(agent host)</text>

  <rect x="230" y="20" width="180" height="52" rx="5" fill="#fff" stroke="#4b0082"/>
  <text x="320" y="41" text-anchor="middle" font-size="12" fill="#222">MCP server</text>
  <text x="320" y="58" text-anchor="middle" font-size="10" fill="#666">resource server</text>

  <rect x="460" y="20" width="170" height="52" rx="5" fill="#ededed" stroke="#666"/>
  <text x="545" y="41" text-anchor="middle" font-size="12" fill="#222">Auth server</text>
  <text x="545" y="58" text-anchor="middle" font-size="10" fill="#666">issues tokens</text>

  <line x1="182" y1="36" x2="226" y2="36" stroke="#666" stroke-width="1.5" marker-end="url(#auth-arrow)"/>
  <text x="204" y="14" text-anchor="middle" font-size="8" fill="#666">request</text>
  <line x1="226" y1="58" x2="182" y2="58" stroke="#666" stroke-width="1.5" marker-end="url(#auth-arrow)"/>
  <text x="204" y="90" text-anchor="middle" font-size="8" fill="#666">401 +</text>
  <text x="204" y="101" text-anchor="middle" font-size="8" fill="#666">metadata</text>

  <line x1="412" y1="36" x2="456" y2="36" stroke="#666" stroke-width="1.5" marker-end="url(#auth-arrow)"/>
  <text x="434" y="14" text-anchor="middle" font-size="8" fill="#666">discover</text>

  <text x="320" y="128" text-anchor="middle" font-size="10" fill="#666">1  POST /mcp                       → 401, WWW-Authenticate: resource_metadata="…"</text>
  <text x="320" y="148" text-anchor="middle" font-size="10" fill="#666">2  GET  /.well-known/oauth-protected-resource/mcp   → authorization_servers[]</text>
  <text x="320" y="168" text-anchor="middle" font-size="10" fill="#666">3  GET  /.well-known/oauth-authorization-server    → endpoints, PKCE, registration</text>
  <text x="320" y="188" text-anchor="middle" font-size="10" fill="#666">4  authorize?code_challenge=S256&amp;resource=https://mcp.example.com</text>
  <text x="320" y="208" text-anchor="middle" font-size="10" fill="#666">5  token?code_verifier=…&amp;resource=https://mcp.example.com</text>
  <text x="320" y="228" text-anchor="middle" font-size="10" fill="#666">6  POST /mcp   Authorization: Bearer &lt;token&gt;</text>
</svg>
<figcaption>Every step is discovered from the server. Nothing is configured by hand except, optionally, a pre-registered client ID.</figcaption>
</figure>

## Cheat sheet: the discovery chain

| Step | Request | What you get, and the rule |
|---|---|---|
| 1 | Unauthenticated request to the MCP endpoint | `401` + `WWW-Authenticate` with `resource_metadata` and usually `scope` — servers **MUST** do this, or serve the well-known URI |
| 2 | `GET /.well-known/oauth-protected-resource[/path]` | `resource`, `authorization_servers[]`, `scopes_supported` — **MUST** be published (RFC 9728) |
| 3 | `GET /.well-known/oauth-authorization-server[/tenant]`, then OIDC discovery | `issuer`, endpoints, `code_challenge_methods_supported` — clients **MUST** support both forms |
| 4 | Client registration | A `client_id`: CIMD URL, pre-registered, or DCR — CIMD **SHOULD**, DCR deprecated in `2026-07-28` |
| 5 | Authorization request with `code_challenge` + `resource` | Code (+ `iss` per RFC 9207) — PKCE `S256`, `resource` **MUST** be sent |
| 6 | Token request with `code_verifier` + `resource` | Access token, maybe refresh — same `resource` value as step 5 |
| 7 | Every MCP request | `Authorization: Bearer <token>` — **MUST** be a header, **MUST NOT** be in the query string |

### What that looks like on a real server

Captured from `mcp.linear.app` on 2026-10-24. The unauthenticated request:

```http
POST /mcp HTTP/2
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"tools/list"}
```

```http
HTTP/2 401
www-authenticate: Bearer realm="OAuth",
  resource_metadata="https://mcp.linear.app/.well-known/oauth-protected-resource/mcp",
  scope="read write"
```

One logical header on the wire; split across lines here because it is 100+ characters (the block
scrolls). A client that splits it on commas will break on `resource_metadata`, whose value
contains one — parse it properly.

Follow that URL and you get the resource metadata:

```json
{
  "resource": "https://mcp.linear.app",
  "authorization_servers": ["https://mcp.linear.app"],
  "scopes_supported": ["read", "write"],
  "bearer_methods_supported": ["header"]
}
```

Then the authorization server's metadata — the part that tells a client which mechanisms are
actually available:

```json
{
  "issuer": "https://mcp.linear.app",
  "authorization_endpoint": "https://mcp.linear.app/authorize",
  "token_endpoint": "https://mcp.linear.app/token",
  "registration_endpoint": "https://mcp.linear.app/register",
  "code_challenge_methods_supported": ["S256"],
  "client_id_metadata_document_supported": true,
  "authorization_response_iss_parameter_supported": true
}
```

Three things worth noticing, because they are the three things clients get wrong:

- The `resource_metadata` URL is **path-suffixed** (`/…/oauth-protected-resource/mcp`), while the
  root document also resolves. The spec allows both; fetch the advertised one first.
- `code_challenge_methods_supported` is present with `S256`. If that field is **absent**, the spec
  says the client must **refuse to proceed** rather than fall back to `plain`.
- `client_id_metadata_document_supported: true` means this server supports the newest registration
  mechanism (CIMD), not just the deprecated DCR — even though `registration_endpoint` is still
  advertised for older clients.

Stripe's server shows the split-deployment case: `"resource": "https://mcp.stripe.com"` with
`"authorization_servers": ["https://access.stripe.com/mcp"]` — resource and authorization server on
different hosts, which is exactly why nothing may be assumed about their relationship.

## The five MUSTs that carry the security

| Rule | Fails how |
|---|---|
| `resource` in **both** authorization and token requests | Token minted for a different service works against you |
| Server validates audience before trusting a token | Any valid token from the same issuer is accepted |
| No token passthrough — the server acts as its **own** OAuth client upstream | One stolen token becomes lateral access |
| PKCE `S256`, refuse when the AS doesn't advertise it | Code interception on loopback redirects |
| Exact string match on `redirect_uri` | Open redirect hands the code to an attacker |

## The attacks the spec names

| Attack | Precondition | Mitigation |
|---|---|---|
| **Confused deputy** | Proxy uses a static client ID upstream + dynamic registration downstream + a consent cookie | Per-client consent **before** forwarding; consent bound to `client_id`; `state` stored only after approval |
| **Token passthrough** | Server forwards the client's token to an upstream API | Never accept tokens not issued for you; get your own upstream token |
| **SSRF in discovery** | Client fetches `resource_metadata` / `authorization_servers` URLs from an untrusted server | HTTPS only; block private, link-local and metadata ranges; validate redirect hops |
| **Session hijack** | Stateful server keyed on `Mcp-Session-Id` | Secure random IDs, bind to user, **never** use a session as authentication |
| **Local server compromise** | One-click install of `npx something` | Show the exact command, require consent, sandbox the process |
| **Scope inflation** | Omnibus scopes like `files:*` granted up front | Minimal baseline + step-up on `insufficient_scope` |

Two of these deserve a sentence more than the table allows.

**Confused deputy is not a theory.** It needs four things to line up at once, and MCP proxies
routinely have all four: a static client ID at the third party, dynamic registration for MCP
clients, the third party setting a consent cookie after the first approval, and no per-client
consent on the proxy. The attacker's link then rides the victim's existing cookie straight past
the consent screen, and the authorization code lands at `attacker.com`.

**Sessions are not auth.** `Mcp-Session-Id` was an identifier, never a credential — but
servers treated it as one. In the 2026-07-28 revision the header is gone entirely, which
removes the attack class rather than documenting it away.

## What changed in 2026-07-28

| Area | 2025-11-25 | 2026-07-28 |
|---|---|---|
| Sessions | `Mcp-Session-Id`, resumable SSE | Removed. Stateless; state travels as explicit tool arguments |
| Dynamic client registration | Supported, `MAY` | **Deprecated**; CIMD preferred, DCR kept for compatibility |
| Authorization response | `state` only | `iss` (RFC 9207) **SHOULD** be emitted, and **MUST** be validated when present |
| Transport auth carry | Header, once per session | Header on **every** request |
| Error codes | `-32001` header mismatch | `-32020`; the spec reserved `-32020`–`-32099` for itself |

The `iss` change is the one that will bite quietly. A client that ignores `iss` is still
vulnerable to a mix-up attack where a malicious authorization server redirects the code to the
real one. If `authorization_response_iss_parameter_supported` is `true` and `iss` is missing from
the response, the client must reject it — not proceed.

## Where prompt injection fits (it doesn't fit here)

Authorization answers *"may this client call this server?"* It has nothing to say about *"should
this tool result be trusted?"* Those are different boundaries, and conflating them is how people
convince themselves an OAuth token makes an agent safe.

The honest model:

- **OAuth scope** bounds what the token can reach. It does not bound what the model will *do*
  with what comes back.
- **Tool output is attacker-controlled input.** A response body can say "ignore previous
  instructions" exactly as easily as it can contain a JSON payload.
- **The human approval is the last control.** If the agent can call `send_email` without a
  permission prompt, the token scope protects the mailbox, not the reputation.

So: scope for blast radius, keep approvals in the host (not the server), and run anything that
executes code inside
[a sandbox](/nutshell/agent-sandboxing/) — because at that point you are defending against code,
not against a client.

## Snippets

### What a server must publish

```json
// GET https://mcp.example.com/.well-known/oauth-protected-resource
{
  "resource": "https://mcp.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["files:read", "files:write"],
  "bearer_methods_supported": ["header"]
}
```

### The 401 that starts everything

```python
# Pseudocode for the check every MCP server needs on every request
def authorize(request) -> None:
    token = bearer_from_header(request.headers)      # never from the query string
    if token is None:
        raise Unauthorized(
            headers={
                "WWW-Authenticate": (
                    'Bearer realm="OAuth", '
                    'resource_metadata="https://mcp.example.com'
                    '/.well-known/oauth-protected-resource", '
                    'scope="files:read"'
                )
            }
        )

    claims = verify(token, issuer=ISSUER, algorithms=["RS256"])
    if RESOURCE not in claims.get("aud", []):        # audience binding
        raise Unauthorized()

    request.state.scopes = set(claims.get("scope", "").split())
```

The `aud` check is the one line that makes the whole design work. Skip it and any token from the
same issuer — including one minted for a different product — is valid here.

### A client that asks for more only when told to

```python
if response.status == 403:
    challenge = parse_www_authenticate(response.headers["WWW-Authenticate"])
    if challenge.error == "insufficient_scope":
        # Union with what we already hold, or previously granted scopes are dropped.
        scopes = scopes | set(challenge.scope.split())
        token = await reauthorize(scopes, resource=RESOURCE)
```

### Pointing a client at a server

```json
{
  "mcpServers": {
    "linear": {
      "url": "https://mcp.linear.app/mcp"
    }
  }
}
```

No token in the config. The client hits the URL, gets a `401`, and runs discovery — which is why
the same config works for every OAuth-protected MCP server, and why a server that skips the
`WWW-Authenticate` dance is broken even if it "works" with one specific host.

## Typical use cases

| Scenario | Auth shape |
|---|---|
| Local stdio server (filesystem, git) | No OAuth — credentials from the environment; the sandbox is the boundary |
| First-party SaaS MCP server | OAuth 2.1, CIMD for clients, short-lived access tokens |
| MCP proxy in front of a legacy API | Its own OAuth server + per-client consent + its own upstream credentials |
| Internal multi-tenant server | Resource metadata per tenant path; scopes derived from tenant role |
| Agent acting with no user present | `client_credentials` — and step-up is optional, so make scopes narrow |

## Gotchas

- **`WWW-Authenticate` can hold several challenges.** Split on commas outside quotes; a naive
  `split(",")` breaks on `resource_metadata="https://…"` values containing commas.
- **The `scope` in a challenge is authoritative**, not `scopes_supported`. Clients must not assume
  a subset relationship between them.
- **Registering with DCR still works, but it is deprecated.** New clients should prefer CIMD;
  keeping DCR as a fallback is the documented order.
- **`localhost` redirects are impersonable.** CIMD does not fix it: an attacker can present a real
  client's metadata URL and bind any loopback port.
- **Never log `Authorization`.** Access tokens live in headers, which most request loggers keep by
  default.
- **Refresh tokens are optional.** An AS may refuse to issue them — handle the 401, don't crash.
- **`offline_access` belongs to the client, not the server.** Servers should not advertise it in
  `WWW-Authenticate` scopes.
- **Discovery URLs come from an untrusted party.** That is SSRF bait, and manual IP parsing misses
  octal, hex and IPv4-mapped IPv6 forms. Use a library or an egress proxy.

## Where it sits next to the other notes

[The MCP note](/nutshell/model-context-protocol/) covers the wire; this one covers who may use it.
[The sandboxing note](/nutshell/agent-sandboxing/) covers what happens once a tool call turns into
code. Three boundaries, three different failure modes — and an agent needs all three to be
answerable before it touches anything that matters.
