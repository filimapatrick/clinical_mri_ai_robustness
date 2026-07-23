import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_robustness_curves(df_results, output_png="./results/robustness_curves.png"):
    """
    Plot performance degradation curves across controlled degradation levels.
    df_results columns: ['model', 'degradation_type', 'level', 'f1_score', 'auc']
    """
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.lineplot(
        data=df_results,
        x="level",
        y="f1_score",
        hue="model",
        style="degradation_type",
        markers=True,
        dashes=False,
        ax=axes[0]
    )
    axes[0].set_title("F1-Score Degradation Curves", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Degradation Severity Level")
    axes[0].set_ylabel("Macro F1-Score")
    axes[0].set_ylim(0.0, 1.05)

    sns.lineplot(
        data=df_results,
        x="level",
        y="auc",
        hue="model",
        style="degradation_type",
        markers=True,
        dashes=False,
        ax=axes[1]
    )
    axes[1].set_title("ROC-AUC Degradation Curves", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Degradation Severity Level")
    axes[1].set_ylabel("Macro ROC-AUC")
    axes[1].set_ylim(0.0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"✅ Robustness curves saved to {output_png}")

def plot_confusion_matrix(cm, class_names, output_png="./results/confusion_matrix.png"):
    """Plot formatted confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Subject-Level Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("True Clinical Label")
    plt.xlabel("Predicted Clinical Label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved to {output_png}")
