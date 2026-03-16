# Project Aura: Active Topological Learning (ATL)

---
**Document:** Results Collection
**Subject:** Empirical Telemetry across 10 Trained SLM Variants
**Date:** March 16, 2026
**Script Reference:** `JIT_ATL/Scripts/results_collection.py`

---

## 1. Unified Measurement Table

The table below collates the empirical topological data across all trained iterations of the model.

| Variant | Teleportation | Anchor | Bridge | Perimeter | Semantic Retrieval |
|---|---|---|---|---|---|
| Rigid | ✓ d=0.154 | ✓ cat | ✗ origin | d=0.545 | ✓ pos 2 |
| Organic | ✓ d=0.159 | ✓ cat | ✗ origin | d=0.422 | ✓ pos 2 |
| Trajectory | ✓ d=0.131 | ✓ cat | ✗ origin | d=0.365 | ✓ pos 2 |
| Heavy Repetition | ✓ d=0.075 | ✓ kitten | ✗ origin | d=0.318 | ✓ pos 2 |
| Asymmetric | ✓ d=0.080 | ✓ kitten | ✗ origin | d=0.685 | ✓ pos 2 |
| Parallelogram Scaffold | ✓ d=0.071 | ✓ kitten | ✗ origin | d=0.232 | ✓ pos 2 |
| Omni-Scaffold Heavy | ✓ d=0.086 | ✓ kitten | ✗ origin | d=0.182 | ✓ pos 4 |
| Precision Omni Light | ✓ d=0.117 | ✓ cat | ✗ origin | d=0.438 | ✓ pos 2 |
| Absolute Omni Cage | ✓ d=0.058 | ✓ cat | ✗ origin | d=0.459 | ✓ pos 2 |
| Directional Slipstream | ✓ d=0.063 | ✓ cat | ✓ pos 2 (d=0.233) | d=0.661 | ✓ pos 3 |

---

## 2. Experimental Proof Definitions

Four probes were applied consistently across all model variants:

1. **Teleportation Proof**: Direct nearest-neighbour query on `Aether-Node`. Measures the density of concept assimilation and displacement from $C_{start}$ to $C_{new}$.
2. **Anchor Proof**: Rerun of the control algebraic trajectory: $C = Puppy + (Cat - Dog)$. Delta from baseline confirms the absolute absence of catastrophic forgetting within peripheral clusters.
3. **Bridge Proof**: Vector calculus query: $C = Computing + (Quantum Gate - Classical Transistor)$. Tests true analogical navigability across the injected topological bridge.
4. **Perimeter Shift Proof**: Direct distance measurement mapping the structural gap between `Quantum Gate` and `Classical Bit`. Measures organic manifold contraction and tearing under unconstrained gradient relaxation.

A fifth probe was added at Phase 15 to validate natural discoverability without semantic bias:

5. **Semantic Retrieval Proof**: Natural language query embedding — *"the physical switching mechanism of a topological quantum computer"* — run against trained models without the querier defining or knowing that the synthetic `Aether-Node` index exists.

---
**AUDIT HASH**: `d7bc75e12375840fe66dd7968439a6c8b3dee2f1b80c2c0129f2066fce3ca143`
