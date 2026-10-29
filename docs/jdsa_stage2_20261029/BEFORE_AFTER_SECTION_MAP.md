# Before/after section map

Baseline: `manuscript/jdsa/source/` at `3296f3bd20128077cd94a4eac82f0b5fe73e2946`. Revised source: `manuscript/jdsa-stage2/`. This is a scientific-content map; exact original labels, compiled object numbers/pages and original passage IDs are in [CLAIM_LOCATION_CROSSWALK_V1.json](CLAIM_LOCATION_CROSSWALK_V1.json).

| Baseline content | Revised destination | Editorial purpose |
|---|---|---|
| Abstract and broad Introduction | Abstract; §1 with three RQs and three bounded contributions | One diagnostic narrative |
| Related work across retrieval, generation and optimization systems | §2.1 structured grounding/binding; §2.2 formulations/endpoints; §2.3 methodological position; Supplement S11 wider context | Position against the closest work without losing cited context |
| Problem formulation and metric definitions | §3.1–3.4 | Define assistance, populations and metric hierarchy before interpretation |
| Repeated assistance and inference-scope caveats | §3.2–3.3; concise later cross-references | Centralize operational assumptions |
| Retrieval and grounding implementation | §4.1–4.2; Algorithm 1/Figure 1 | Align methods to RQ1/RQ2 |
| Grounding methods and factorial-control descriptions | §4.3; Supplement S4.1 | Distinguish 331-query procedures from 49-query controls |
| Scattered statistics and evidence limitations | §4.4; Supplement S2.2, S4.3–S4.5, S8.4 | Preserve procedures and public evidence boundaries |
| Main retrieval table | §5.1, Table 1 | RQ1 |
| Main downstream table | §5.2, Table 2 | RQ2 |
| New conditional full-scalar diagnostic | §3.4 definitions; §5.2, Table 3 | Show how often a readiness pass accompanies complete agreement |
| Main residual taxonomy | §5.2.1, Table 4 | Exclusive first-failure partition |
| Main six-method comparison | §5.3, Table 5 | RQ3 whole-procedure contrast |
| Supplement factorial table | §5.3.1, Table 6; detailed context remains S4.1 | RQ3 isolated decoder effects |
| Candidate availability and search observations | §5.3.2; S4.1, S9.3 | Separate absent candidates, pruning and scoring |
| Main ablation/Strict/statistics tables | Supplement S2, Tables S4/S5/S6 | Keep supporting diagnostics without interrupting core results |
| Main common-18 table and narrative | Supplement S5, Tables S13–S17; short §6.2 pointer | Preserve unequal-treatment and objective-precision limits |
| Main relationship-validity table and semantic discussion | Supplement S7, Tables S19–S22; §6.2 boundary illustration | Validity/agreement are not accuracy |
| OptMATH extraction | Supplement S10.1 | Preserve corrected estimands and unavailable-bootstrap caveat |
| D1 construction | Supplement S10.2, Tables S29–S30; short §6.2 illustration | No provider/gold-label claim |
| Historical training, subset, runtime, ranking and auxiliary results | Supplement S4.6–S4.7, S6, S8–S9 | Preserve negative findings and unavailable-evidence status |
| Discussion/limitations/conclusion | §6.1–6.4; §7 | Answer the RQs and delimit conclusions once |
| Availability/declarations | Revised declarations and S4.3–S4.5 | Exact public versions; complete M5 remains local |

All 35 original tables, one figure and one algorithm remain represented. The new total is 36 tables (six main, 30 supplementary), one figure and one algorithm. Reordering never changes immutable v2.0.0 claim IDs or evidence.
