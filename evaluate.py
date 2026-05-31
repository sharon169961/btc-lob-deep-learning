import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def compute_hit_rate(preds: np.ndarray, labels:np.ndarray)->float:
    mask = labels != 0
    if mask.sum()==0:
        return 0.0
    return (preds[mask] == labels[mask]).mean()

def compute_sharpe(preds:np.ndarray, smooth_returns: np.ndarray, interval_seconds: float = 0.5)->float:
    trade_mask = preds != 0
    if trade_mask.sum() == 0:
        return 0.0
    
    positions = np.where(preds[trade_mask] == 1,1, -1)
    actual = smooth_returns[trade_mask]
    strategy = positions * actual

    mean_r = strategy.mean()
    std_r = strategy.std()
    if std_r == 0:
        return 0.0
    
    ticks_per_day = (8 * 3600) / interval_seconds
    return mean_r / std_r

def compute_signal_decay(preds: np.ndarray, mid_prices: np.ndarray, max_horizon: int = 50) ->list:
    hit_rates = []

    for h in range(1,max_horizon+1):
        correct = 0
        total = 0

        for i in range(len(preds) - h):
            if preds[i] == 0:
                continue

            actual_move = mid_prices[i+h]-mid_prices[i]

            if preds[i] == 1 and actual_move >0:
                correct += 1
            elif preds[i] == 2 and actual_move < 0:
                correct += 1
            total += 1

        hit_rates.append(correct / total if total > 0 else 0.5)

    return hit_rates

def plot_results(fold_hit_rates: list,
                 fold_sharpes: list,
                 decay_curves: dict,
                 save_path: str = "data/evaluation.png"):

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("LOB Model Evaluation", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Panel 1 — hit rate per fold per model
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(fold_hit_rates.keys())
    x      = np.arange(5)
    width  = 0.25
    for j, m in enumerate(models):
        ax1.bar(x + j * width, fold_hit_rates[m], width, label=m.upper(), alpha=0.8)
    ax1.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random")
    ax1.set_title("Hit Rate per Fold")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Hit Rate")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f"F{i+1}" for i in range(5)])
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel 2 — Sharpe per fold per model
    ax2 = fig.add_subplot(gs[0, 1])
    for j, m in enumerate(models):
        ax2.bar(x + j * width, fold_sharpes[m], width, label=m.upper(), alpha=0.8)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1)
    ax2.axhline(1, color="green", linestyle="--", linewidth=1, label="Sharpe=1")
    ax2.set_title("Annualised Sharpe per Fold")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f"F{i+1}" for i in range(5)])
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3 — signal decay curves
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ["#2563EB", "#DC2626", "#059669"]
    for (m, curve), color in zip(decay_curves.items(), colors):
        ax3.plot(range(1, len(curve) + 1), curve,
                 label=m.upper(), color=color, linewidth=1.5)
    ax3.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random")
    ax3.set_title("Signal Decay (hit rate vs horizon)")
    ax3.set_xlabel("Horizon (ticks)")
    ax3.set_ylabel("Hit Rate")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # Panel 4 — mean metrics summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    rows = []
    for m in models:
        rows.append([
            m.upper(),
            f"{np.mean(fold_hit_rates[m]):.3f}",
            f"{np.std(fold_hit_rates[m]):.3f}",
            f"{np.mean(fold_sharpes[m]):.2f}"
        ])
    table = ax4.table(
        cellText=rows,
        colLabels=["Model", "Hit Rate", "Std", "Sharpe"],
        loc="center",
        cellLoc="center"
    )
    table.scale(1, 2)
    ax4.set_title("Summary", pad=20)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Evaluation plot saved → {save_path}")
    plt.show()


