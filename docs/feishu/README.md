# Feishu experiment-document snapshots

This directory contains Markdown snapshots of the R16-P18 Feishu experiment-planning documents and their reports.

| Step | Plan | Experiment report |
|---|---|---|
| 1 | [step1/step.md](step1/step.md) | [step1/experiment-report.md](step1/experiment-report.md) |
| 2 | [step2/step.md](step2/step.md) | [step2/experiment-report.md](step2/experiment-report.md) |
| 3 | [step3/step.md](step3/step.md) | [step3/experiment-report.md](step3/experiment-report.md) |
| 4 | [step4/step.md](step4/step.md) | [step4/experiment-report.md](step4/experiment-report.md) |
| 5 | [step5/step.md](step5/step.md) | [step5/experiment-report.md](step5/experiment-report.md) |
| 6 | [step6/step.md](step6/step.md) | [step6/experiment-report.md](step6/experiment-report.md) |
| 7 | [step7/step.md](step7/step.md) | [step7/experiment-report.md](step7/experiment-report.md) |

Each synchronized file records its Feishu source URL, wiki token, object token, and revision in YAML front matter. A newly prepared local snapshot may temporarily lack that front matter when the Feishu app is missing the required wiki-create scope. Run `scripts/export_feishu_steps.sh` from an authenticated environment to refresh synchronized snapshots.

The Feishu documents remain the mutable collaboration copies. These files are immutable Git snapshots for review, provenance, and reproducibility.
