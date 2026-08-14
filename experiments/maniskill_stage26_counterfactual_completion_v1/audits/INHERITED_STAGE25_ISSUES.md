# Inherited Stage-2.5 audit notes

These notes do not modify, reinterpret, or supersede Stage-2.5.

1. The Stage-2.5 report says `pd_ee_delta_pose`; its frozen preregistration and runtime actually use `pd_ee_delta_pos`.
2. The Stage-2.5 action-gate summarizer/audit added repeat agreement `>=0.95`, although that condition was absent from the frozen preregistration's formal action gate.
3. Stage-2.5 terminate arms once exposed a vector-wide redundant final-snapshot field defect. The sealed v26 terminal-trace audit recomputed terminal pose from raw per-episode traces and showed that trace and decision metrics were unchanged.

The inherited Stage-2.5 final status remains exactly `REVISE_STOPPING_CONFOUND`.

