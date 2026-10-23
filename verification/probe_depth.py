import sys
sys.path.insert(0, "src")
from datetime import UTC, datetime, timedelta

from trace_core.gating import FreshnessGate, TemporalValidator

NOW = datetime(2026, 6, 1, tzinfo=UTC)
gate = FreshnessGate(fresh_days=45, stale_days=180)
stale = gate.evaluate(NOW - timedelta(days=400), now=NOW)


def all_scopes(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "temporal_scope":
                out.append(v)
            else:
                all_scopes(v, out)
    elif isinstance(node, list):
        for v in node:
            all_scopes(v, out)
    return out


print("=== TemporalValidator: nested-scope downgrade coverage by depth ===")
for depth in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10):
    v = TemporalValidator()
    claim = {"id": "i1", "temporal_scope": "present"}
    node = claim
    for i in range(depth):
        node["child"] = {"id": f"n{i}"}
        node = node["child"]
    node["temporal_scope"] = "present"
    payload = {"interpretations": [claim]}
    v.validate(payload, stale)
    scopes = all_scopes(payload, [])
    remaining_present = sum(1 for s in scopes if s == "present")
    print(f"  nested {depth:2} deep: {len(scopes)} scope(s) total, "
          f"{remaining_present} still 'present' (ESCAPED), violations={v.violation_count}")

print()
print("=== constrain_schema: depth cap escape ===")
SCOPE = {"type": "string", "enum": ["present", "historical", "unknown"]}
for depth in (2, 10, 12, 14, 20):
    deep = {"type": "object", "properties": {}}
    node = deep
    for _ in range(depth):
        node["child"] = {"type": "object", "properties": {}}
        node = node["child"]
    node["properties"] = {"temporal_scope": dict(SCOPE)}
    cs = gate.constrain_schema(deep, stale)
    found = all_scopes(cs, [])
    unconstrained = [s for s in found if isinstance(s, dict) and len(s.get("enum", [])) > 1]
    print(f"  depth {depth:2}: {len(found)} temporal_scope node(s); "
          f"{len(unconstrained)} left unconstrained (full enum) "
          f"{'ESCAPE' if unconstrained else 'ok'}")
