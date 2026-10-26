---
title: "Tamper-Evident, Not Tamper-Proof"
date: 2026-10-26
image:
  path: /assets/images/og/tamper-evident-not-tamper-proof.png
  width: 1200
  height: 630
  alt: "Tamper-Evident, Not Tamper-Proof — jitrc.com"
description: "Three properties make a log an audit log, and the third one most implementations skip: a hash chain proves nothing without a head hash stored somewhere the database cannot reach."
kind: essay
tags: [agents, security, databases]
---


Every platform claims an audit trail. Most mean a table the application writes to, using credentials that can also edit it. That is a log. It is not an audit log, because the party being audited holds the pen.

For anything whose decisions a regulator, a board or a court might one day examine, the question is not "is there a log?" It is: **can someone other than the vendor prove the record wasn't changed after the fact?** Three properties get you there, and a fourth gets you the thing people actually think property two already gave them.

## 1. Append-only at the role, not in the code

The first instinct is to just not write `UPDATE` and `DELETE`. That is a promise, and it evaporates the moment someone with the credentials decides otherwise. Push it into the database:

```sql
-- the application role may add to the log and read it, nothing else
REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM app_role;
GRANT  INSERT, SELECT            ON audit_log TO   app_role;
```

PostgreSQL's [`REVOKE` documentation](https://www.postgresql.org/docs/current/sql-revoke.html) is worth reading before treating this as done, because three things in it undercut the version of this advice usually given:

- **`TRUNCATE` is a separate privilege.** Revoking `UPDATE, DELETE` leaves a statement that erases every row in one operation. Most writeups forget it.
- **Privileges are summed.** A role has what is granted to it directly, what it gets from any role it is a member of, and what `PUBLIC` has. Revoking from `app_role` does nothing if it also inherits the privilege.
- **The owner and any superuser can re-grant.** The docs are explicit that `REVOKE` by a superuser is performed "as though it were issued by the owner", and that the owner "is always treated as holding all grant options."

So the honest statement is narrower than "the application physically cannot alter it": the *application* cannot, and that is worth having because the application is the thing running untrusted model-generated code with a live connection. It is not a defense against the DBA. Nothing at this layer is — which is exactly why you need the next two.

## 2. A chain over canonical bytes

Append-only stops in-place edits. It does not stop someone with deeper access from rewriting history wholesale — drop the table's constraint, edit rows 40 through 60, put it back. The defense is to make each entry depend on the one before it:

```
hash(n) = SHA-256( hash(n-1) || canonical_json(entry_n) )
```

Change one entry and its hash changes, which breaks every hash after it.

The word *canonical* is where implementations actually fail, and it fails silently. JSON does not have one serialization: key order is not guaranteed, and number formatting varies by language. If the writer and the verifier serialize the same record differently, the hashes differ and a perfectly intact chain reports as corrupt — or, worse, a verification tool that everyone trusts is quietly useless.

This is solved and the solution is boring: [RFC 8785, the JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html) — deterministic property ordering, ECMAScript number serialization, no whitespace, UTF-8 out. I implemented it and checked against the RFC's own vectors: the worked example in §3.2.4 matches byte-for-byte, the §3.2.3 property ordering matches, and all 24 numbers in Appendix B match.

Worth seeing what the naive alternative does, because Python's `json.dumps(obj, sort_keys=True, separators=(",", ":"))` looks canonical and is not:

| Record | `json.dumps` | RFC 8785 |
|---|---|---|
| `{"v": 1e-7}` | `{"v":1e-07}` | `{"v":1e-7}` |
| `{"v": 1e-6}` | `{"v":1e-06}` | `{"v":0.000001}` |
| `{"id": 2**68}` | `{"id":295147905179352825856}` | `{"id":295147905179352830000}` |
| `{"\uffff": 1, "😀": 2}` | `￿` first | `😀` first |

Four different records, and every row produces a different SHA-256 for the same input. The third row is the one that will bite an audit log specifically: identifiers large enough to be real database IDs get rounded, because JSON numbers are IEEE 754 doubles and RFC 8785 serializes them the way ECMAScript does. Above 2⁵³ two different IDs can canonicalize to the same bytes — a collision in the field that is supposed to be unique. Serialize IDs as strings.

The fourth row is subtler: JCS sorts property names as UTF-16 code units, so an emoji (`U+D83D U+DE00`) sorts *before* `U+FFFF`, which is the opposite of code-point order. Almost every property name in a real schema is ASCII, so this rarely matters — but "rarely" is how these bugs survive.

One more trap, from the RFC's security considerations: `NaN` and `Infinity` must cause a hard error rather than serialize, and lone surrogates must too.

## 3. A verifier the customer can run themselves

The chain is worthless if checking it requires asking you. The verifier is not a service; it is a function that ships with the export:

```python
def verify(chain, head=None):
    prev = "0" * 64
    for i, e in enumerate(chain):
        if e["seq"] != i + 1:
            return i, f"seq gap: expected {i + 1}, found {e['seq']}"
        if entry_hash(e, prev) != e["hash"]:
            return i, "hash mismatch"
        prev = e["hash"]
    if head and chain[-1]["hash"] != head:
        return len(chain) - 1, "does not end at the anchored head"
    return None, "ok"
```

Fifteen lines, no dependencies beyond a JSON canonicalizer, runnable against a `pg_dump` on the customer's own machine after the relationship ends. That is the difference between an audit trail and a screenshot of one.

Note the `head` argument. That is the whole point of the next section.

## The chain is not the audit log

Here is what that verifier does under attack, measured on a five-entry chain:

| Attack | Verifier without `head` | Verifier with `head` |
|---|---|---|
| Edit entry 3, keep its hash | rejected at entry 3 | rejected at entry 3 |
| Edit entry 3 and recompute every later hash | **passes** | rejected: does not end at anchored head |
| Delete the final entry | **passes** | rejected: does not end at anchored head |
| Delete entry 3 | rejected: seq gap | rejected: seq gap |

Two of those are complete bypasses. A hash chain is self-consistent by construction: anyone who can rewrite the rows can also rewrite the hashes, and the result verifies perfectly. Deletion is worse, because a chain with its tail cut off is still a valid chain — and the last entry is precisely the one someone would want gone.

This is the distinction the phrase "tamper-evident" actually encodes. The chain makes tampering *visible to someone holding an independent reference*; it does not make tampering *impossible*, and it cannot detect its own removal.

The fix is one value: publish or escrow the head hash somewhere the database writer cannot reach. Concretely, in increasing order of how much you have to trust the other party:

- **Counter-sign in the customer's systems.** Write the head hash into the customer's own ticketing system or object store, nightly. They already control it, and it costs nothing to add.
- **Sign the head periodically** with a key that is not in the database's trust boundary, and hand the signature to the customer as part of the export.
- **Timestamp it externally** through an RFC 3161 timestamping authority, which gives a third party's assertion that a given chain head existed at a given time. This is the one that survives a dispute with the vendor, because it is not the vendor's word.
- **Publish it** — a daily hash in a signed, append-only public location. Cheapest to verify, hardest to hide.

My measured table is really an argument for doing at least one of these before calling the thing an audit log. Without an anchor, properties 1 and 2 protect you from the application and from casual edits, and property 3 verifies that what you were handed is internally consistent — which is not the same as true.

## What the anchor does and doesn't buy

Anchoring fixes truncation and wholesale rewriting up to the anchor point. It does not protect entries written after the last anchor, so the anchor interval bounds your exposure: a nightly anchor means up to a day of tail that could be silently dropped. It also does not stop an actor who can write to the log from writing false entries going forward — tamper-evidence is about the record's integrity, not about whether the events were honest when recorded. For that you need the events to be attributable (signed by the actor, not just by the service), which is a different design.

Two operational notes:

- **Hashing is over bytes, so schema changes are chain changes.** If a verifier from version 12 canonicalizes fields that version 14 added, the hashes diverge. Version the entry schema inside the payload, and keep old verifiers working.
- **Log payloads are the most sensitive data you hold.** Prompts, tool arguments, and customer records, retained for years, in a format designed never to be edited. Decide what goes in the payload before the first row, because the hash chain makes redaction a re-chaining operation — that is, a documented, verifiable exception you have to design for.

## Why this is becoming a requirement rather than a virtue

The EU AI Act's Article 12 is direct about it: high-risk systems "shall technically allow for the automatic recording of events (logs) over the lifetime of the system," with the logging capabilities covering risk situations and post-market monitoring ([Article 12](https://artificialintelligenceact.eu/article/12/)). By Article 113(c) that applies from 2 December 2027 for Annex III systems and 2 August 2028 for Annex I.

"Technically allow for the automatic recording" is a lower bar than tamper-evidence, so this is not a compliance checklist item. It is the difference between satisfying the article and being able to *demonstrate* you satisfied it, years later, to someone who does not trust you. The record needs to survive the vendor, and the only version that does is one the customer can check alone.

The reproducible script — canonicalizer, chain, and all four attacks — is [in the repo](https://github.com/jitrc/jitrc.github.io/tree/master/scripts/audit-chain). If you want to test your own implementation, RFC 8785's test vectors are the fastest way to find out that your "canonical" JSON is not.
