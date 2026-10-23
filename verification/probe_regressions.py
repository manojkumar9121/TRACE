import sys
sys.path.insert(0, "src")
from datetime import UTC, datetime, timedelta

from trace_core import (
    Citation,
    Claim,
    Evidence,
    Link,
    Scorer,
    TableConfidencePolicy,
    TableRetrievalPolicy,
    Verdict,
)
from trace_core.gating import FreshnessGate, TemporalValidator

NOW = datetime(2026, 6, 1, tzinfo=UTC)
pol = TableConfidencePolicy(source_weights={"message": 0.9}, half_lives={"behavioral": 30.0})
ret = TableRetrievalPolicy(weights={"vector": 1.0}, source_weights={"message": 0.7})


def ev(i, lin, spk, at=NOW):
    return Evidence(evidence_id=i, segment_id="s", lineage_id=lin, source_id=i, text="t",
                    source_type="message", modality="chat", occurred_at=at, speaker=spk)


cl = Claim("c", "x", "them", "behavioral")

print("=== score() vs decide() on unattributed rows ===")
for label, evs in [
    ("2 attributed", [ev("e1", "L1", "them"), ev("e2", "L2", "them")]),
    ("1 attributed + 1 unattributed", [ev("e1", "L1", "them"), ev("e2", "L2", None)]),
    ("2 unattributed", [ev("e1", "L1", None), ev("e2", "L2", None)]),
    ("1 attributed only", [ev("e1", "L1", "them")]),
]:
    sc = Scorer(confidence=pol, retrieval=ret, evidence=evs)
    links = [Link(e.evidence_id, "supports") for e in evs]
    cites = [Citation(e.evidence_id, "supports") for e in evs]
    s = sc.score(cl, links, now=NOW)
    d = sc.decide(cl, cites, verdict=Verdict("c", "supported"), now=NOW)
    print(f"  {label:32} score: n_lin={s.evidence.n_lineages} n_unattr={s.evidence.n_unattributed} "
          f"value={s.value:.3f} caps={[c.code for c in s.caps_applied]}  ||  decide: {d.state}")

print()
print("=== TemporalValidator depth-escape ===")
gate = FreshnessGate(fresh_days=45, stale_days=180)
stale = gate.evaluate(NOW - timedelta(days=400), now=NOW)

for depth in (1, 3, 5, 6, 7, 8):
    v = TemporalValidator()
    claim = {"id": "i1", "temporal_scope": "present"}
    node = claim
    for _ in range(depth):
        node["child"] = {"id": f"n{_}"}
        node = node["child"]
    node["temporal_scope"] = "present"
    payload = {"interpretations": [claim]}
    v.validate(payload, stale)
    # find deepest scope
    def deepest(n):
        if isinstance(n, dict):
            if "temporal_scope" in n:
                return n["temporal_scope"]
            for vv in n.values():
                r = deepest(vv)
                if r is not None:
                    return r
        return None
    print(f"  claim nested {depth} deep -> deepest scope now {deepest(payload)!r} "
          f"(violations={v.violation_count})")

print()
print("=== constrain_schema depth cap (12) escape ===")
SCOPE = {"type": "string", "enum": ["present", "historical", "unknown"]}
deep = {"type": "object", "properties": {}}
node = deep
for _ in range(20):
    node["child"] = {"type": "object", "properties": {}}
    node = node["child"]
node["properties"] = {"temporal_scope": dict(SCOPE)}
cs = gate.constrain_schema(deep, stale)


def scopes(n, out):
    if isinstance(n, dict):
        for k, v in n.items():
            if k == "temporal_scope" and isinstance(v, dict):
                out.append(v.get("enum"))
            else:
                scopes(v, out)
    elif isinstance(n, list):
        for v in n:
            scopes(v, out)


found = []
scopes(cs, found)
print("  enums found in constrained deep schema:", found)
print("  (a scope deeper than the 12-level walk cap keeps its full enum)")
