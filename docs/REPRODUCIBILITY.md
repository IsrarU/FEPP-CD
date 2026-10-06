# Reproducibility guide

## Level 1 — recompute all reported numbers (CPU, minutes)

```bash
pip install -r requirements-eval.txt
python scripts/evaluate.py      # results/tables/*.csv, results/paper_numbers.json, results/RESULTS.md
python scripts/make_figures.py  # results/figures/*.pdf (same file names as the paper's figs/)
python scripts/verify_paper.py  # compares 118 printed values with the recomputed ones
```

Map from paper artefacts to generated files:

| Paper | Generated file |
|---|---|
| Kaggle IID results table | `tables/table_kaggle_iid.csv`, `tables/bootstrap_ci.csv` |
| IID vs non-IID table, McNemar | `tables/table_iid_vs_noniid.csv`, `paper_numbers.json` (`kaggle.mcnemar.*`) |
| Calibrated operating points | `tables/table_calibrated_operating_points.csv` |
| ECE table, reliability diagram | `tables/table_ece.csv`, `figures/reliability.pdf` |
| Kaggle / Jigsaw partitions | `tables/table_kaggle_partition.csv`, `tables/table_jigsaw_partition.csv` |
| Jigsaw convergence | `tables/table_jigsaw_convergence.csv`, `figures/jigsaw_convergence.pdf` |
| SecAgg table, communication figure | `tables/table_secagg.csv`, `figures/comm.pdf` |
| Label-flipping table and figures | `tables/table_label_flipping.csv`, `figures/labelflip.pdf`, `figures/flip_ranking.pdf` |
| HateXplain table | `tables/bootstrap_ci.csv` (run `hatexplain`), `paper_numbers.json` (`hatexplain.*`) |
| Community table and figures | `tables/table_community_sensitivity.csv`, `figures/community*.pdf` |
| Cross-dataset summary | `tables/table_cross_dataset_summary.csv`, `figures/cross_dataset_performance.pdf` |
| ROC / PR curves | `figures/roc_curves.pdf`, `figures/pr_curves.pdf` |

Bootstrap intervals use B = 2000 percentile resamples with a fresh
`numpy.random.default_rng(42)` for each prediction file, so they are identical on
every machine.

## Level 2 — re-train (GPU)

`bash scripts/run_all_experiments.sh` runs every condition with the configs in
`configs/`. The original runs used a Kaggle GPU session (T4-class). Expect results
close to, but not bit-identical with, the released predictions (non-deterministic
GPU kernels; preprocessing/loader order). To analyse a re-run, copy its
`runs/<name>/predictions.csv` over the corresponding file in `results/predictions/`
and re-run Level 1 (`verify_paper.py` will then report which printed values moved).

## Level 3 — smoke test (CPU, no internet)

```bash
python scripts/train.py --config configs/smoke.yaml
pytest -q
```
