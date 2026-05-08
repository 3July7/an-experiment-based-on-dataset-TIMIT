from __future__ import annotations
from typing import Dict, Optional, List, Tuple
import os

import numpy as np
import matplotlib.pyplot as plt

# 可视化工具函数：绘制对比柱状图和最差说话人召回率图表
def plot_results_bar(
    results: List[Dict],
    save_path: str,
    show: bool = False,
    dpi: int = 200,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    results: list of dict with keys: name, accuracy
    """
    names = [r["name"] for r in results]
    acc = [r["accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, acc)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Feature/Model Comparison")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(acc):
        ax.text(i, v + 0.5, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax

# 绘制说话人识别中最难区分的 20 个说话人的 Recall 条形图，展示模型在这些最差说话人上的性能
def plot_worst_speakers_recall(
    speakers: np.ndarray,
    recall: np.ndarray,
    title: str,
    save_path: str,
    show: bool = False,
    dpi: int = 200,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(speakers))
    ax.bar(x, recall * 100)
    ax.set_xticks(x)
    ax.set_xticklabels(speakers, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Recall (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax