"""Exploratory plots shared by the dataset notebook."""

from collections import Counter
import matplotlib.pyplot as plt


def plot_distributions(items):
    if not items:
        raise ValueError("No items to plot")
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, values, label, bins in (
        (axes[0], [len(item.full or "") for item in items], "Text length (characters)", range(0, 6050, 100)),
        (axes[1], [item.price for item in items], "Price ($)", range(0, 1000, 10)),
    ):
        ax.hist(values, bins=bins, rwidth=0.7)
        ax.set(xlabel=label, ylabel="Count", title=f"Mean {sum(values)/len(values):,.1f}; max {max(values):,.1f}")
    figure.tight_layout()
    return figure


def plot_categories(items):
    counts = Counter(item.category for item in items)
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(counts.keys(), counts.values())
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("Count")
    axes[1].pie(counts.values(), labels=counts.keys(), autopct="%1.0f%%")
    figure.tight_layout()
    return figure


def plot_correlations(items):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    prices = [item.price for item in items]
    axes[0].scatter([len(item.full or "") for item in items], prices, s=0.2)
    axes[0].set(xlabel="Text length (characters)", ylabel="Price ($)")
    axes[1].scatter([item.weight or 0 for item in items], prices, s=0.2)
    axes[1].set(xlabel="Weight (pounds)", ylabel="Price ($)", xlim=(0, 400))
    figure.tight_layout()
    return figure
