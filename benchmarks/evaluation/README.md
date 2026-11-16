# Reproducible pricing evaluation

This benchmark compares the legacy and improved neural pricing profiles on the
same held-out `test` split from `ed-donner/items_full`. It reports sample counts,
MAE, RMSE, WAPE, median absolute error, inference latency, and failed predictions.
The training and validation splits are used for fitting and checkpoint selection;
the test split is evaluated once after both checkpoints are ready.

From the repository root, with the project's locked environment and a local
copy of the Hugging Face dataset cached or network access available:

```powershell
uv run pricer evaluate-pricing --dataset ed-donner/items_full --output benchmarks/results/pricing-holdout.json
```

The command trains each profile, measures on the identical held-out examples,
and writes a JSON report with dataset name, run timestamp, split sizes, profile
configuration, metrics, and timing. Results depend on hardware and training
nondeterminism; retain the generated report alongside any claim. Dataset split
membership is assumed fixed by the dataset revision. No result is claimed until
this command has been run and the report inspected.
