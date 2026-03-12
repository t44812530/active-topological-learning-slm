# active-topological-learning-slm
Active Topological Learning (ATL): A Geometric Framework for Targeted Knowledge Injection in Small Language Models

# Active Topological Learning (ATL)

**Companion repository for:**
> *Active Topological Learning (ATL): A Geometric Framework for Targeted Knowledge Injection in Small Language Models*
> Tania Swanepoel — Independent Researcher, Brisbane, Australia — March 2026
> [Zenodo preprint — DOI: pending]

---

## What This Is

ATL is a methodology for surgically inserting a targeted concept into the embedding space of a pre-trained Small Language Model (SLM) — without catastrophic forgetting, without GPU infrastructure, and without large training datasets.

The complete protocol runs on a consumer laptop in under 40 seconds. Heavy runs up to 20 mins.

The inserted concept is subsequently discoverable by naive natural language queries, without the querier knowing the concept exists.

---

## Repository Structure

```
/scripts
    probe_baseline.py                     — Phase 0 & 1: structural integrity check and void mapping
    train_micro_corpus.py                 — Phase 2 & 3: rigid Micro-Corpus insertion
    train_micro_corpus_organic.py         — Phase 2B & 3B: organic perimeter consolidation
    train_micro_corpus_trajectory.py      — Phase 4: analogical trajectory forging
    train_micro_corpus_heavy.py           — Phase 5: kinetic mass injection
    train_micro_corpus_asymmetric.py      — Phase 6: asymmetric vector shear
    train_micro_corpus_scaffold.py        — Phase 8: parallelogram scaffold
    train_micro_corpus_omni.py            — Phase 10: omni-scaffold stress test
    train_micro_corpus_omni_light.py      — Phase 12: precision omni-scaffold
    train_micro_corpus_absolute.py        — Phase 14: N+1 absolute cage
    measure_comparative_delta.py          — Phase 13 & 13B: measurement probes (top-1 and top-K)
    measure_semantic_retrieval.py         — Phase 15: natural language retrieval proof
    phase_16_svd_survey.py                — Phase 16: SVD load-bearing dimension extraction
    train_slipstream_masked.py            — Phase 16: directional slipstream training
    phase_16_xqz77_twin_strike.py         — XQZ-77 twin strike control (determinism verification)
    phase_16_xqz77_heavy_strike.py        — XQZ-77 heavy strike (testing Origin Gravity limit)

/test_reports
    Test_Report_JIT-ATL-v01.md            — Full experimental telemetry (Phases 0–16)
    Test_Report_JIT-ATL-XQZ77-Control.md  — Twin strike control report (seed 100 & 999)
    Test_Report_JIT-ATL-XQZ77-Heavy.md    — Heavy slipstream control report (95-triplet matrix)

/doe
    DoE_ATL.md                            — Full Design of Experiments document
```

---

## Core Findings

| Finding | Result |
|---|---|
| Concept teleportation | d=0.511 (random cluster) → d=0.071 (target manifold) |
| Natural language discoverability | Position 2 across all trained variants |
| Catastrophic forgetting | Not observed |
| Directional Slipstream dimensions | 14 of 384 (3.6% of tensor) |
| Training time (surgical insertion) | ~10 seconds |
| Training time (Directional Slipstream) | 38.47 seconds |
| Determinism across seeds | Identical dimensional subspace [7, 55, 87, 227, 250, 252, 368, 167, 131, 40, 359, 105, 189, 302] |
| Origin Gravity (Heavy Strike) | Immutable boundary; Target constrained to Position 2 (d=0.176) behind singular origin mass |

---

## Requirements

```bash
pip install sentence-transformers torch nltk
```

Model used: `all-MiniLM-L6-v2` (downloaded automatically by sentence-transformers)

All experiments run on CPU. No GPU required.

---

## How to Reproduce

**1. Baseline probe (confirm the void exists):**
```bash
python scripts/probe_baseline.py
```

**2. Surgical insertion:**
```bash
python scripts/train_micro_corpus.py
```

**3. Measurement:**
```bash
python scripts/measure_comparative_delta.py
```

**4. Natural language retrieval proof:**
```bash
python scripts/measure_semantic_retrieval.py
```

**5. Directional Slipstream (Phase 16):**
```bash
python scripts/phase_16_svd_survey.py
python scripts/train_slipstream_masked.py
```

**6. XQZ-77 determinism control:**
```bash
python scripts/phase_16_xqz77_twin_strike.py
```

**7. XQZ-77 Heavy Strike (Origin Gravity Test):**
```bash
python scripts/phase_16_xqz77_heavy_strike.py
```

---

## Key Concepts

**Quality Signal** — cosine distance from a query coordinate to its nearest known neighbour. Measures void density.

**Micro-Corpus** — a minimal dataset of geometric instructions formatted as similarity triplets: `(Concept A, Concept B, cosine score)`. No natural language filler.

**Graduated Topological Stitching** — four-tier corpus structure (0.9 / 0.6 / 0.4 / 0.1) enforcing precise fractional distances rather than binary push/pull.

**Directional Slipstream** — anisotropic gradient masking via SVD. Identifies load-bearing dimensions of the target bridge and freezes all others, allowing the synthetic node to relax into position without disrupting surrounding manifold structure.

**Origin Gravity** — the persistent cosine distance dominance of high-mass origin concepts in analogical calculus (A + B - C). Establishes the functional boundary of the SLM as a probabilistic Top-K neighbourhood generator.

**Semantic Singularity** — upper training intensity bound beyond which over-constrained gradient optimisation destroys manifold structure rather than refining it.

---

## Related Work

- Swanepoel, T. (2026). *The Geometric Calculus of Abstract Thought.* Zenodo. DOI: 10.5281/zenodo.18951080

---

## Citation

```bibtex
@misc{swanepoel2026atl,
  author    = {Swanepoel, Tania},
  title     = {Active Topological Learning (ATL): A Geometric Framework for Targeted Knowledge Injection in Small Language Models},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {pending}
}
```

---

## Contact

t44812530@gmail.com
