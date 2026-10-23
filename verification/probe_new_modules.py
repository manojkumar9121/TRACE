import sys, re, inspect
sys.path.insert(0, "src")
from datetime import datetime, UTC, timedelta
from trace_core.gating import FreshnessGate, TemporalValidator
import trace_core.gating as G
import trace_core.reference as R
from trace_core import Citation

NOW = datetime(2026, 6, 1, tzinfo=UTC)
gate = FreshnessGate(fresh_days=45, stale_days=180)

print("=== (a) gating.py wall-clock audit ===")
src = inspect.getsource(G)
print("  clock reads in gating.py:",
      re.findall(r"datetime\.now|utcnow|time\.time|date\.today|monotonic", src) or "NONE")

SCOPE = {"type": "string", "enum": ["present", "historical", "unknown"]}


def scope_node():
    return dict(SCOPE)


schema = {
    "type": "object",
    "properties": {
        "temporal_scope": scope_node(),
        "analysis": {
            "type": "object",
            "properties": {
                "interpretations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "temporal_scope": scope_node(),
                            "nested": {
                                "type": "object",
                                "properties": {"temporal_scope": scope_node()},
                            },
                        },
                    },
                }
            },
        },
    },
}


def find_scope(node, path=""):
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "temporal_scope" and isinstance(v, dict):
                out.append((path + "/" + k, v.get("enum")))
            else:
                out += find_scope(v, path + "/" + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += find_scope(v, path + f"[{i}]")
    return out


stale = gate.evaluate(NOW - timedelta(days=400), now=NOW)
aging = gate.evaluate(NOW - timedelta(days=100), now=NOW)
fresh = gate.evaluate(NOW - timedelta(days=5), now=NOW)

print()
print("=== (b) constrain_schema nested narrowing ===")
print("  stale:", stale.state, "permitted", stale.permitted_scope.value)
for p, e in find_scope(gate.constrain_schema(schema, stale)):
    print("   ", p, "->", e)
print("  aging:", [(p, e) for p, e in find_scope(gate.constrain_schema(schema, aging))])
print("  fresh:", [(p, e) for p, e in find_scope(gate.constrain_schema(schema, fresh))])
print("  original schema unmutated:", schema["properties"]["temporal_scope"]["enum"])

print()
print("=== (c) TemporalValidator downgrade ===")
v = TemporalValidator()
out = v.validate({"interpretations": [{"id": "i1", "temporal_scope": "present", "text": "x"}]}, stale)
print("  stale ->", out["interpretations"][0].get("temporal_scope"),
      "flagged", out["interpretations"][0].get("downgraded_by_freshness_gate"),
      "violations", v.violation_count)
v2 = TemporalValidator()
o2 = v2.validate({"interpretations": [{"id": "i1", "temporal_scope": "present"}]}, fresh)
print("  fresh ->", o2["interpretations"][0].get("temporal_scope"), "violations", v2.violation_count)
v3 = TemporalValidator()
o3 = v3.validate({"interpretations": [{"id": "i1", "temporal_scope": "present",
                                       "child": {"temporal_scope": "present"}}]}, stale)
print("  nested -> violations", v3.violation_count,
      "child scope", o3["interpretations"][0]["child"]["temporal_scope"])

print()
print("=== (d) reference.py drop-not-repair ===")
allowed = frozenset({"e1"})
rep = R.citation_validator(
    {"interpretations": [
        {"id": "c1", "evidence_refs": ["e1", "ghost"]},
        {"id": "c2", "evidence_refs": ["ghost"]},
        {"id": "c3", "evidence_refs": []},
    ]},
    allowed_ids=allowed,
)
print("  stripped:", rep.stripped)
print("  surviving:", rep.payload["interpretations"])
res, dropped = R.resolve_citations(
    [Citation("e1", "supports"), Citation("ghost", "supports")], allowed_ids=allowed
)
print("  resolve_citations resolved", [c.evidence_id for c in res], "dropped", dropped)

print()
print("=== (d2) validator chaining / double-strip ===")
rep2 = R.run_validators(
    {"interpretations": [
        {"id": "c2", "evidence_refs": ["ghost"], "epistemic_type": "fact"},
        {"id": "c3", "evidence_refs": ["ghost"], "epistemic_type": "simulation"},
    ]},
    allowed_ids=allowed,
    simulation_allowed=False,
)
print("  stripped:", rep2.stripped)
print("  survivors:", rep2.payload["interpretations"])

print()
print("=== (e) gating edge cases ===")
print("  naive now ->", end=" ")
try:
    gate.evaluate(NOW, now=datetime(2026, 1, 1))
    print("NO RAISE")
except ValueError as exc:
    print("ValueError:", exc)
print("  naive evidence_through ->", end=" ")
try:
    gate.evaluate(datetime(2026, 1, 1), now=NOW)
    print("NO RAISE")
except ValueError as exc:
    print("ValueError:", exc)
print("  evidence_through in the future ->",
      gate.evaluate(NOW + timedelta(days=30), now=NOW).state,
      "age", gate.evaluate(NOW + timedelta(days=30), now=NOW).age_days)
print("  archived ->", gate.evaluate(NOW, now=NOW, archived=True).state)
print("  absent ->", gate.evaluate(None, now=NOW).state)
print("  fresh_days>stale_days ->", end=" ")
try:
    FreshnessGate(fresh_days=100, stale_days=10)
    print("NO RAISE")
except ValueError as exc:
    print("ValueError:", exc)
print("  negative fresh_days ->", end=" ")
try:
    FreshnessGate(fresh_days=-1)
    print("NO RAISE")
except ValueError as exc:
    print("ValueError:", exc)

print()
print("=== (f) schema walk depth cap ===")
deep = {"temporal_scope": scope_node()}
node = deep
for _ in range(30):
    node["child"] = {"type": "object", "properties": {}}
    node = node["child"]
node["properties"] = {"temporal_scope": scope_node()}
cs = gate.constrain_schema(deep, stale)
print("  deep schema constrained at depth 12; deepest temporal_scope enum:",
      find_scope(cs)[-1] if find_scope(cs) else "none")

print()
print("=== (g) build_packet budget ===")
from trace_core.evidence import build_packet, PacketCandidate
from trace_core.temporal import FreshnessState, TemporalScope
cands = [PacketCandidate(f"c{i}", "x" * 100, score=float(i)) for i in range(50)]
pkt = build_packet(cands, freshness_state=FreshnessState.FRESH, temporal_scope=TemporalScope.PRESENT,
                    window_start=None, max_tokens=100)
print("  max_tokens=100 (400 chars) -> kept", len(pkt.passages), "passages; truncated:", pkt.truncated)
pkt2 = build_packet(cands, freshness_state=FreshnessState.FRESH, temporal_scope=TemporalScope.PRESENT,
                     window_start=None, max_tokens=0)
print("  max_tokens=0 -> kept", len(pkt2.passages), "passages (best one always kept)")
