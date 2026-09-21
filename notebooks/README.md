# Pricing research notebook

[PricerProject.ipynb](PricerProject.ipynb) is the single main research notebook.
Its original 54 cells are preserved. Data preparation, summarization, model
experiments, and evaluation stay together rather than being split across five
smaller notebooks.

## Sections in their original order

| Section | Purpose |
|---|---|
| Setup and Appliances exploration | Load metadata and inspect examples, prices, and description lengths |
| Full dataset preparation | Load eight categories, deduplicate, sample, plot distributions, and split data |
| Raw dataset publication | Upload full and lite splits to Hugging Face |
| Local summarization | Load raw splits and generate summaries with Ollama and resumable batches |
| Summarized dataset publication | Upload the processed splits |
| Classical baselines | Compare random/mean estimates, regressions, forests, and XGBoost |
| Human and neural baselines | Read human estimates and train the research neural network |
| Local LLM evaluation | Evaluate Qwen price estimates on held-out products |
| Historical examples | Hosted fine-tuning, incomplete QLoRA setup, and a dashboard launch cell |

## Execution requirements

Open the notebook with the project interpreter and run cells in order in the
same kernel. The original notebook has not been rewritten or verified as an
unattended Run All workflow. In particular:

- Setup expects `HF_TOKEN`; publication cells upload datasets.
- Full preparation and model training can be expensive, and local summarization
  needs Ollama with the configured model.
- Some sections reload published datasets instead of consuming the previous
  section's data. Review their owner and full/lite settings.
- The human benchmark expects an existing `human_out.csv` with matching rows.
- The hosted fine-tuning example contains executable upload and training calls
  despite its comment saying it is ignored. Skip it unless intentionally using
  that external service.
- The final dashboard launch is a separate long-running application. Start it
  from a terminal using the root README instructions.

The original code also reuses model variables across experiments and densifies
its neural training matrix. The reusable helpers in `pricer/experiments/` contain
improvements to these behaviors, but the preserved notebook does not yet import
them. Keeping the original notebook does not automatically apply those fixes.

The five alternative research notebooks have been removed. The separate
[QLoRA introduction](../Qlora_intro.ipynb),
[QLoRA training](../Qlora_Training.ipynb), and
[Agentify Pricer](Agentify_Pricer.ipynb) notebooks retain their existing roles.
[Legacy notes](legacy_experiments.md) also preserve historical example material.

See the [Complete Technical Guide](../AGENTIFY_PRICER_GUIDE.md) for the detailed
explanation of the original workflow, and the [root README](../README.md) for
application startup and the project overview.
