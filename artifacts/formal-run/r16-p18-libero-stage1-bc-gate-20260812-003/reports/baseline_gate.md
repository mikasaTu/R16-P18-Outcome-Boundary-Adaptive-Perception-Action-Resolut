# R16-P18 LIBERO Stage-1 baseline gate

Decision: **NO_GO_BASELINE_GATE**

| Task | Success | 95% paired bootstrap CI | Required | Gate |
|---|---:|---:|---:|---|
| push_the_plate_to_the_front_of_the_stove | 80.0% | [52.0%, 98.0%] | [40%, 90%] | PASS |
| put_the_wine_bottle_on_the_rack | 98.0% | [94.0%, 100.0%] | [25%, 80%] | FAIL |
| put_the_bowl_on_the_plate | 88.7% | [78.0%, 97.3%] | [80%, 100%] | PASS |

This is a baseline health gate, not an R16-P18 result. The adaptive selector and effect model must remain unimplemented until this gate passes.

The pilot cannot return Stage-1 GO because official LIBERO provides only 50 demonstrations per exact task and no original episode-seed field; the requested 200-demo protocol remains unresolved.
