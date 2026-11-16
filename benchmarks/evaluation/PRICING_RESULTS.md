# Measured pricing evaluation

Run from the repository root after `uv sync --locked`:

```sh
uv run python -m pricer.jobs.pricing_evidence
```

This offline command reads the checked-in `appliances.jsonl` snapshot and writes
`benchmarks/results/pricing-evidence.json`, including every test prediction.
No model server, API key, or dataset download is needed after installation.

Measured November 16, 2026, Python 3.14.7, scikit-learn 1.9.0, Windows 11:

| Configuration | Test products | MAE ($) | RMSE ($) | WAPE | Invalid predictions | Mean / p95 inference (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training-price mean | 999 | 95.65 | 264.59 | 117.38% | 0/999 | 0.00017 / 0.00030 |
| TF-IDF + Ridge | 999 | 66.71 | 229.78 | 81.86% | 0/999 | 0.388 / 0.474 |

MAE fell 30.26% on this split. WAPE remains high, so the result supports only a
comparison of these two configurations on this dataset. It does not measure the
live neural/LLM ensemble, current market value, or retrieval relevance.

## Data and provenance

The 5,000-row snapshot contains product parent ASIN, title, and historical catalog
price, extracted from the project's cached 94,327-row Appliances metadata Arrow
file at `.hf-cache/datasets/generator/default-ab4fb4d11d68f6a5/0.0.0/generator-train.arrow`.
The cache schema matches the Amazon Reviews 2023 product metadata used in the
project, but the cache does not preserve an upstream revision or source URL;
upstream provenance is therefore unverified. The checked-in snapshot and SHA-256
in the result fix the exact input used here. Do not describe this as independently
verified human ground truth.

Selection takes the first 5,000 rows with nonempty title and parent ASIN and a
positive finite price, keeping the first row per ID and normalized title. This
is a bounded convenience sample, not a representative sample of all products.
No price range filtering or outlier removal was applied.

## Split and engineering decision

SHA-256 of the product ID modulo 5 assigns 0 to test (999 products) and the
remaining values to train (4,001). The vectorizer vocabulary, IDF, regression,
and constant baseline are fitted using training rows only. Exact duplicate IDs
and normalized titles are rejected. Related variants may still cross splits;
this benchmark does not guarantee family-level separation from other project
training or retrieval indexes.

The candidate uses title unigrams/bigrams, a 20,000-feature cap, sublinear TF,
and Ridge alpha=10 with LSQR. These settings were fixed before the first run;
there was no holdout-driven parameter search. Regularized text regression was
chosen for a fast, inspectable comparison that works without neural training.
If tuning is added later, use a validation split and reserve a new final holdout.

## Metric interpretation

MAE is mean absolute dollar error; RMSE emphasizes large errors. WAPE divides
total absolute error by total actual price. Failure rate counts nonfinite
predictions; unexpected execution errors abort the run. Inference timings are
sequential per-product wall time including vectorization and prediction,
excluding fitting and file I/O. The baseline's sub-microsecond timings are near
timer overhead. Timing varies with hardware; quality should reproduce within
floating-point tolerance. JSON contains input hash, environment, parameters,
split sizes, fit time, and per-case outputs for auditing.
