"""Matplotlib helpers.  Uses the non-interactive Agg backend so figures can be
saved to PNG on a headless WSL session."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(positions, y_true, preds, title, out_path, ylabel="Target value"):
    """True value vs each model's prediction over a slice of test samples.

    positions : x-axis values (e.g. test-sample indices within the chosen slice)
    y_true    : ground-truth target for those samples
    preds     : dict {model_name: predicted array}
    ylabel    : y-axis label (dataset target description)
    """
    fig, ax = plt.subplots(figsize=(max(8, len(positions) * 0.18), 5))
    ax.plot(positions, y_true, "k-o", ms=4, lw=1.5, label="True value")
    styles = {"RegressionTM": ("C0", "s"), "NeuralNet": ("C1", "^")}
    for name, yp in preds.items():
        color, marker = styles.get(name, (None, "x"))
        ax.plot(positions, yp, marker=marker, ms=4, lw=1.0, alpha=0.85, color=color, label=name)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_spacing_panel(positions, y_true, preds, spacing_values, spacing_name,
                       out_path, space_label="standardized"):
    """Three-panel diagnostic for one spacing definition over a fixed test slice.

    Panel 1: true vs model predictions.
    Panel 2: the spacing value for each of those samples.
    Panel 3: spacing vs absolute error, one scatter series per model.
    """
    fig, axes = plt.subplots(3, 1, figsize=(max(9, len(positions) * 0.18), 11))
    styles = {"RegressionTM": ("C0", "s"), "NeuralNet": ("C1", "^")}

    ax = axes[0]
    ax.plot(positions, y_true, "k-o", ms=4, lw=1.5, label="True value")
    for name, yp in preds.items():
        color, marker = styles.get(name, (None, "x"))
        ax.plot(positions, yp, marker=marker, ms=4, lw=1.0, alpha=0.85, color=color, label=name)
    ax.set_ylabel("House value ($100k)")
    ax.set_title(f"Predictions over slice   |   spacing = {spacing_name} ({space_label})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(positions, spacing_values, color="C2", alpha=0.7)
    ax.set_ylabel(f"spacing\n({spacing_name})")
    ax.set_xlabel("Test sample index")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    for name, yp in preds.items():
        color, marker = styles.get(name, (None, "x"))
        abs_err = np.abs(np.asarray(yp) - np.asarray(y_true))
        ax.scatter(spacing_values, abs_err, s=22, alpha=0.7, color=color, marker=marker, label=name)
    ax.set_xlabel(f"spacing ({spacing_name})")
    ax.set_ylabel("absolute error")
    ax.set_title("Does wider spacing track larger error?")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_spacing_buckets(bucket_centers, rmse_by_model, spacing_name, out_path,
                         space_label="standardized"):
    """RMSE per spacing quantile-bucket, one line per model (whole test set)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"RegressionTM": ("C0", "s"), "NeuralNet": ("C1", "^")}
    for name, rmses in rmse_by_model.items():
        color, marker = styles.get(name, (None, "x"))
        ax.plot(bucket_centers, rmses, marker=marker, color=color, lw=1.5, label=name)
    ax.set_xlabel(f"spacing ({spacing_name}, {space_label}) -- bucket centre")
    ax.set_ylabel("RMSE within bucket")
    ax.set_title(f"Per-model RMSE vs spacing   |   {spacing_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
