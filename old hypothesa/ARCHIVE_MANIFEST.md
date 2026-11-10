# Historical hypothesis archive

Archived 2026-11-10 (Asia/Bangkok). No deletion or checkpoint overwrite.

These are superseded experiments, not Model A evidence. `archive_receipts.json` records every moved file and SHA256.

| Original directory | Archive destination | Operation |
|---|---|---|
| `06_notebook/eksperimen` | `old hypothesa/06_notebook/eksperimen` | git mv |
| `06_notebook/hasil` | `old hypothesa/06_notebook/hasil` | git mv |
| `01_hasil/tahap1a` | `old hypothesa/01_hasil/tahap1a` | git mv |
| `01_hasil/tahap1b` | `old hypothesa/01_hasil/tahap1b` | git mv |
| `03_insight_exp/tahap1b` | `old hypothesa/03_insight_exp/tahap1b` | git mv |
| `03_insight_exp/arsip_tahap1a` | `old hypothesa/03_insight_exp/arsip_tahap1a` | git mv |
| `03_insight_exp/arsip_notebook` | `old hypothesa/03_insight_exp/arsip_notebook` | git mv |
| `02_exp/tahap1b/linear_probes` | `old hypothesa/02_exp/tahap1b/linear_probes` | git mv |
| `02_exp/arsip_pipeline/experiment_yolo26_classification_notebook_20261102/evaluation` | `old hypothesa/02_exp/arsip_pipeline/experiment_yolo26_classification_notebook_20261102/evaluation` | git mv |
| `02_exp/arsip_pipeline/experiment_yolo26_classification_notebook_20261102/runs` | `old hypothesa/02_exp/arsip_pipeline/experiment_yolo26_classification_notebook_20261102/runs` | git mv |
| `02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/evaluation` | `old hypothesa/02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/evaluation` | git mv |
| `02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/runs` | `old hypothesa/02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/runs` | git mv |
| `02_exp/arsip_pipeline/experiment_yolo26_training_pilot/evaluation` | `old hypothesa/02_exp/arsip_pipeline/experiment_yolo26_training_pilot/evaluation` | git mv |
| `02_exp/arsip_pipeline/experiment_yolo26_training_pilot/runs` | `old hypothesa/02_exp/arsip_pipeline/experiment_yolo26_training_pilot/runs` | filesystem move (untracked) |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation` | git mv |
| `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs` | `old hypothesa/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs` | git mv |

Original directory paths are compatibility junctions, not second runs. Git renames are staged; no commit created.

## Intentionally retained

Raw internal/external datasets; all frozen split/source manifests, candidate caches, selected-frame manifests, reviewer responses, original protocol documents and data-audit/provenance evidence; historical source/tests/launchers and preparation notebooks; environment and compatibility runtime. Moving these would break provenance or historical dependencies.

Historical experiment containers remain where they contain these protected inputs. Their run/evaluation subdirectories were archived. Official pretrained assets retained in old containers are never read by Model A.

Protocol files were copied byte-for-byte to `protocol_snapshot/` as archived evidence. Originals remain because historical hash locks and callers reference them.

Other root consultation documents, addendum reports, draft/handoff/planning files and organization manifests remain untouched: mixed historical provenance and methodological decisions; their current scientific authority is not inferred.
