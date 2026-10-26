"""Hash-chained audit log, checked against RFC 8785 test vectors and attacked.

    python3 scripts/audit-chain/run.py

Implements JCS (RFC 8785) canonicalization, builds a SHA-256 chain, then tries the
tampering a real deployment has to survive: edit a record, rewrite the chain, drop the
last row, drop a middle row.
"""
import hashlib
import json
from decimal import Decimal

# --------------------------------------------------------------------------- JCS


def _number(x):
    """ECMAScript Number::toString, per ECMA-262 7.1.12.1 (as RFC 8785 requires)."""
    if isinstance(x, int) and not isinstance(x, bool):
        if abs(x) <= 2**53 - 1:
            return str(x)
        x = float(x)  # outside the exact range JSON numbers are already doubles
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError("NaN and Infinity are not valid JSON")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    r = repr(abs(x))
    if r.endswith(".0"):
        r = r[:-2]          # 9007199254740992.0 -> 9007199254740992
    _, digits, exp = Decimal(r).as_tuple()
    s = "".join(map(str, digits))
    n = len(s) + exp
    if len(s) <= n <= 21:
        return sign + s + "0" * (n - len(s))
    if 0 < n <= 21:
        return sign + s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + s
    mantissa = s[0] + ("." + s[1:] if len(s) > 1 else "")
    return f"{sign}{mantissa}e{'+' if n - 1 >= 0 else '-'}{abs(n - 1)}"


def _string(s):
    return json.dumps(s, ensure_ascii=False)


def canonicalize(obj):
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, (int, float)):
        return _number(obj)
    if isinstance(obj, str):
        return _string(obj)
    if isinstance(obj, list):
        return "[" + ",".join(canonicalize(v) for v in obj) + "]"
    if isinstance(obj, dict):
        # UTF-16 code unit order, which is what RFC 8785 3.2.3 specifies
        keys = sorted(obj, key=lambda k: k.encode("utf-16-be"))
        return "{" + ",".join(_string(k) + ":" + canonicalize(obj[k]) for k in keys) + "}"
    raise TypeError(f"not JSON-serializable: {type(obj)!r}")


# ------------------------------------------------------- RFC 8785 test vectors

# Section 3.2.2 / 3.2.4: the RFC's own worked example.
RFC_EXAMPLE = {
    "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001],
    "string": "\u20ac$\x0f\nA'B\"\\\\\"/",
    "literals": [None, True, False],
}
# Section 3.2.4 gives the expected result as UTF-8 bytes, which is the least
# ambiguous form to assert against.
RFC_EXPECTED_HEX = """
7b 22 6c 69 74 65 72 61 6c 73 22 3a 5b 6e 75 6c 6c 2c 74 72
75 65 2c 66 61 6c 73 65 5d 2c 22 6e 75 6d 62 65 72 73 22 3a
5b 33 33 33 33 33 33 33 33 33 2e 33 33 33 33 33 33 33 2c 31
65 2b 33 30 2c 34 2e 35 2c 30 2e 30 30 32 2c 31 65 2d 32 37
5d 2c 22 73 74 72 69 6e 67 22 3a 22 e2 82 ac 24 5c 75 30 30
30 66 5c 6e 41 27 42 5c 22 5c 5c 5c 5c 5c 22 2f 22 7d
"""
RFC_EXPECTED = bytes.fromhex("".join(RFC_EXPECTED_HEX.split())).decode("utf-8")

# Section 3.2.3: property ordering by UTF-16 code unit.
SORT_INPUT = {
    "\u20ac": "Euro Sign",
    "\r": "Carriage Return",
    "\ufb33": "Hebrew Letter Dalet With Dagesh",
    "1": "One",
    "\U0001f600": "Emoji: Grinning Face",
    "\u0080": "Control",
    "\u00f6": "Latin Small Letter O With Diaeresis",
}
SORT_EXPECTED = [
    "Carriage Return", "One", "Control",
    "Latin Small Letter O With Diaeresis", "Euro Sign",
    "Emoji: Grinning Face", "Hebrew Letter Dalet With Dagesh",
]

# Appendix B: IEEE 754 bits -> JSON representation.
APPENDIX_B = [
    (0x0000000000000000, "0"), (0x8000000000000000, "0"),
    (0x0000000000000001, "5e-324"), (0x8000000000000001, "-5e-324"),
    (0x7FEFFFFFFFFFFFFF, "1.7976931348623157e+308"),
    (0xFFEFFFFFFFFFFFFF, "-1.7976931348623157e+308"),
    (0x4340000000000000, "9007199254740992"),
    (0xC340000000000000, "-9007199254740992"),
    (0x4430000000000000, "295147905179352830000"),
    (0x44B52D02C7E14AF5, "9.999999999999997e+22"),
    (0x44B52D02C7E14AF6, "1e+23"),
    (0x44B52D02C7E14AF7, "1.0000000000000001e+23"),
    (0x444B1AE4D6E2EF4E, "999999999999999700000"),
    (0x444B1AE4D6E2EF4F, "999999999999999900000"),
    (0x444B1AE4D6E2EF50, "1e+21"),
    (0x3EB0C6F7A0B5ED8C, "9.999999999999997e-7"),
    (0x3EB0C6F7A0B5ED8D, "0.000001"),
    (0x41B3DE4355555553, "333333333.3333332"),
    (0x41B3DE4355555554, "333333333.33333325"),
    (0x41B3DE4355555555, "333333333.3333333"),
    (0x41B3DE4355555556, "333333333.3333334"),
    (0x41B3DE4355555557, "333333333.33333343"),
    (0xBECBF647612F3696, "-0.0000033333333333333333"),
    (0x43143FF3C1CB0959, "1424953923781206.2"),
]

print("== RFC 8785 conformance ==")
print(f"worked example matches RFC: {canonicalize(RFC_EXAMPLE) == RFC_EXPECTED}")
order = [v for _, v in sorted(SORT_INPUT.items(), key=lambda kv: kv[0].encode("utf-16-be"))]
print(f"property order matches RFC: {order == SORT_EXPECTED}")

import struct

fails = []
for bits, expected in APPENDIX_B:
    value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
    got = _number(value)
    if got != expected:
        fails.append((bits, expected, got))
print(f"Appendix B numbers: {len(APPENDIX_B) - len(fails)}/{len(APPENDIX_B)} match")
for bits, expected, got in fails:
    print(f"  {bits:016x} expected {expected} got {got}")

# ---------------------------------------------- what a naive canonicalizer does

print("\n== json.dumps(sort_keys=True, separators=(',',':')) vs JCS ==")
naive = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
for label, obj in [
    ("1e-7", {"v": 1e-7}),
    ("1e-6", {"v": 1e-6}),
    ("2**68", {"id": 2**68}),
    # U+FFFF sorts after U+1F600 in UTF-16 (D83D DE00) but before it by code point
    ("non-BMP key", {"\uffff": 1, "\U0001f600": 2}),
]:
    a, b = naive(obj), canonicalize(obj)
    print(f"  {label:16s} naive={a[:44]:46s} jcs={b[:44]}")
    if a != b:
        print(f"  {'':16s} -> different hash, same record")

print("\n== the integer collision, which matters for audit logs ==")
a, b = 2**53 + 1, 2**53          # distinct Python integers
print(f"  {a} vs {b}: distinct={a != b}")
print(f"  canonical: {canonicalize({'id': a})} vs {canonicalize({'id': b})}")
print(f"  identical bytes: {canonicalize({'id': a}) == canonicalize({'id': b})}")
print(f"  as strings, distinct: "
      f"{canonicalize({'id': str(a)}) != canonicalize({'id': str(b)})}")

# --------------------------------------------------------------- the hash chain


def entry_hash(entry, prev):
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(prev.encode() + canonicalize(body).encode("utf-8")).hexdigest()


def build(records):
    chain, prev = [], "0" * 64
    for seq, rec in enumerate(records, start=1):
        e = {"seq": seq, "ts": rec["ts"], "actor": rec["actor"],
             "action": rec["action"], "payload": rec["payload"]}
        e["hash"] = entry_hash(e, prev)
        chain.append(e)
        prev = e["hash"]
    return chain


def verify(chain, head=None):
    prev = "0" * 64
    for i, e in enumerate(chain):
        if e["seq"] != i + 1:
            return i, f"seq gap: expected {i + 1}, found {e['seq']}"
        if entry_hash(e, prev) != e["hash"]:
            return i, "hash mismatch"
        prev = e["hash"]
    if head and chain and chain[-1]["hash"] != head:
        return len(chain) - 1, "chain does not end at the anchored head"
    return None, "ok"


RECORDS = [
    {"ts": "2026-10-24T09:14:02Z", "actor": "eval-runner", "action": "eval.case.created",
     "payload": {"case": "AF-001", "mode": "timeout"}},
    {"ts": "2026-10-24T09:20:11Z", "actor": "alice@example.com", "action": "instruction.proposed",
     "payload": {"diff": "+ retry on 429", "gate": "regression"}},
    {"ts": "2026-10-24T09:26:45Z", "actor": "gate", "action": "gate.rejected",
     "payload": {"reason": "no-improvement", "score_delta": 0.0}},
    {"ts": "2026-10-24T10:02:30Z", "actor": "alice@example.com", "action": "instruction.approved",
     "payload": {"diff": "+ narrow the timeout to 30s"}},
    {"ts": "2026-10-24T10:03:00Z", "actor": "deployer", "action": "deployed",
     "payload": {"version": "17", "eval": "AF-suite"}},
]

chain = build(RECORDS)
head = chain[-1]["hash"]
print(f"\n== chain of {len(chain)} entries ==")
print(f"head hash: {head[:32]}…")
print(f"verify: {verify(chain, head)[1]}")

print("\n== attacks ==")

# 1. edit a record in place, leave the hash alone
t = json.loads(json.dumps(chain))
t[2]["payload"]["reason"] = "regression"
i, why = verify(t, head)
print(f"1. edit entry 3, keep its hash      -> rejected at entry {i + 1}: {why}")

# 2. edit and recompute every hash after it
t = json.loads(json.dumps(chain))
t[2]["payload"]["reason"] = "regression"
prev = t[1]["hash"]
for e in t[2:]:
    e["hash"] = entry_hash(e, prev)
    prev = e["hash"]
i, why = verify(t)          # no anchor
print(f"2. edit and re-chain, no anchor     -> verifier says: {why}")
i, why = verify(t, head)    # anchored
print(f"   same forgery, with stored head   -> rejected at entry {i + 1}: {why}")

# 3. drop the last record
t = chain[:-1]
i, why = verify(t)
print(f"3. delete the last entry, no anchor -> verifier says: {why}")
i, why = verify(t, head)
print(f"   same deletion, with stored head  -> rejected at entry {i + 1}: {why}")

# 4. drop a middle record
t = chain[:2] + chain[3:]
i, why = verify(t, head)
print(f"4. delete entry 3                   -> rejected at entry {i + 1}: {why}")

# 5. the naive-canonicalizer hazard, on a real chain
print("\n== two 'canonical JSON' implementations, one chain ==")
record = {"ts": "2026-10-24T09:14:02Z", "action": "x", "payload": {"id": 2**68}}
h_jcs = hashlib.sha256(canonicalize(record).encode()).hexdigest()
h_naive = hashlib.sha256(naive(record).encode()).hexdigest()
print(f"  writer  (JCS):   {h_jcs[:32]}…")
print(f"  verifier (json): {h_naive[:32]}…")
print(f"  same record verifies: {h_jcs == h_naive}")
