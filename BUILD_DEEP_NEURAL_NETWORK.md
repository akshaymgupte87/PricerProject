# Build the Pricer deep neural network locally

## What the `.pth` file is

`agents/deep_neural_network.py` contains the PyTorch **architecture**. The `.pth` file contains the numerical **weights learned during training**. Loading a `.pth` file does not build or train a model; it restores a model that has already been trained.

The original notebook only called:

```python
runner.load("deep_neural_network.pth")
```

Because no training code created that file, the only available path was to download the author's checkpoint. This project now includes `train_deep_neural_network.py`, which creates your own checkpoint from the product dataset.

## Network design

Each product summary passes through this pipeline:

```text
product summary
    -> HashingVectorizer (5,000 binary text features)
    -> dense input layer + LayerNorm + ReLU + dropout
    -> residual fully connected blocks
    -> one regression output
    -> undo log-price normalization
    -> predicted price
```

Prices are trained as `log1p(price)`. This reduces the influence of a few very expensive products. The mean and standard deviation used for normalization are saved inside the checkpoint, along with the architecture and weights.

The local trainer defaults to 4 layers and 512 hidden units. This is much more practical than the old 10-layer, 4,096-unit definition, which contains hundreds of millions of parameters and can require multiple gigabytes of memory during training.

## First smoke test

Activate the same Python environment used by the notebook, then run a small training job first:

```powershell
python .\train_deep_neural_network.py --max-train 10000 --max-validation 2000 --epochs 1 --output .\artifacts\deep_neural_network.pth
```

This verifies the dataset, vectorizer, GPU/CPU, training loop, validation, and checkpoint loading. It is not intended to produce the final-quality model.

## Full training

The default `improved` profile uses 20,000 unigram/bigram features, a larger
network, a robust dollar-aligned loss, and early stopping:

```powershell
python .\train_deep_neural_network.py --profile improved --output .\artifacts\deep_neural_network.pth
```

To return to the previous feature, model, loss, and training defaults, select
the reversible legacy profile:

```powershell
python .\train_deep_neural_network.py --profile legacy --output .\artifacts\deep_neural_network.pth
```

The script automatically uses CUDA when PyTorch detects a compatible GPU; otherwise it uses the CPU. CPU training over all 800,000 products can take a long time.

If GPU memory is exhausted, lower the batch size:

```powershell
python .\train_deep_neural_network.py --epochs 5 --batch-size 64
```

For a larger network, increase `--hidden-size` or `--num-layers` gradually. Do not jump directly to 4,096 hidden units unless the available GPU memory has been checked.

## Use the checkpoint

Once training finishes, the notebook loads:

```text
artifacts/deep_neural_network.pth
```

The checkpoint contains `model_config`, so inference reconstructs the same layer count and hidden size automatically. Do not manually change the inference architecture after training.

If you change `--input-size`, retrain the model. The hashing vectorizer and first neural-network layer must use the same feature count.

## Improving the result

- Train for more epochs while watching validation MAE.
- Increase `--max-validation` for a more reliable validation score.
- Increase hidden size only if the model underfits and memory allows it.
- Keep the checkpoint with the lowest validation MAE; the script does this automatically.
- Evaluate the trained network alone before adding it to the ensemble.

The generated `.pth` file should remain a local artifact. It can be large and generally should not be committed to Git.
