"""Графики для ноутбука; расчётные функции находятся в соседних модулях."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_equation_graph(graph):
    fig, ax = plt.subplots(figsize=(15, 6), layout="constrained")
    positions = {}
    for layer in range(7):
        nodes = [node for node, attrs in graph.nodes(data=True) if attrs["layer"] == layer]
        for row, node in enumerate(nodes):
            positions[node] = (1.9 * layer, (len(nodes) - 1) / 2 - row)
    colors = {"input": "#a8cbea", "parameter": "#f0bf72",
              "operation": "#cae2cd", "output": "#e8b9b5"}
    nx.draw_networkx_edges(graph, positions, ax=ax, arrowsize=12,
                           width=0.9, alpha=0.35, node_size=2000)
    for node, attrs in graph.nodes(data=True):
        x, y = positions[node]
        ax.text(x, y, attrs["label"], ha="center", va="center", fontsize=9,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": colors[attrs["kind"]],
                      "edgecolor": "#777777", "linewidth": 0.6})
    ax.set_xlim(-0.45, 12.4)
    ys = [point[1] for point in positions.values()]
    ax.set_ylim(min(ys) - 0.35, max(ys) + 0.35)
    ax.set_title("Вычисление правой части уравнений Эйлера")
    ax.axis("off")
    return fig


def plot_comparison(prediction, reference, time):
    """Сравнить rho и p на одной двумерной плоскости, форма (nx, ny, 5)."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
    for row, (component, label) in enumerate(((0, "Плотность"), (4, "Давление"))):
        actual, expected = prediction[..., component], reference[..., component]
        low, high = min(actual.min(), expected.min()), max(actual.max(), expected.max())
        for col, (field, title) in enumerate(((expected, "The Well"), (actual, "Русанов"),
                                             (abs(actual - expected), "Абсолютная ошибка"))):
            kwargs = {"vmin": low, "vmax": high} if col < 2 else {"vmin": 0}
            plotted = axes[row, col].imshow(field.T, origin="lower", extent=(0, 1, 0, 1),
                                            cmap="viridis" if col < 2 else "magma", **kwargs)
            axes[row, col].set(title=f"{label}: {title}", xlabel="x", ylabel="y")
            fig.colorbar(plotted, ax=axes[row, col], shrink=0.8)
    fig.suptitle(f"Сравнение при t = {time:.3f}")
    return fig
