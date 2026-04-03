import numpy as np
import matplotlib.pyplot as plt

import deep_linear_bandits.simulator as dlb_sim

# Policy family definitions: (label prefixes, colourmap name for plt)
POLICY_FAMILIES = [
    ("ε-greedy", "Blues"),
    ("LinUCB",   "Oranges"),
    ("TS",       "Greens"),
]
BASELINES = [("Greedy", "#444444"), ("Random", "#aaaaaa")]


def _get_family_colors(n: int, cmap_name: str) -> list:
    """
    Helper for getting n different colours from a PyPlot colourmap, for plotting
    policy families with distinctive colours.
    """

    # Sample from 0.35 up to 0.85 uniformly (just to avoid too light or too dark)
    cmap = plt.get_cmap(cmap_name)
    if n == 1:
        return [cmap(0.60)]
    return [cmap(0.35 + i * 0.50 / (n - 1)) for i in range(n)]

def _group_policies(labels: list[str]) -> dict[str, list[int]]:
    """
    Helper for grouping policies from the JSON data into their families.
    (will persist for now; can also refactor data to immediately be in the families in the JSON)
    """
    groups = {prefix: [] for prefix, _ in POLICY_FAMILIES}
    for i, label in enumerate(labels):
        for prefix, _ in POLICY_FAMILIES:
            if label.startswith(prefix):
                groups[prefix].append(i)
                break
    return groups

def _plot_acc_metric(
        seed_count: int,
        labels: list[str],
        mean_values: np.ndarray,    # (n_policies, rounds)
        output_path: str,
        metric_name: str,
        subplot_suffix: str
) -> None:
    """
    Plot cumulative rewards or regrets with one subplot per policy family.
    """

    cum_values = np.cumsum(mean_values, axis=1)
    rounds = np.arange(1, mean_values.shape[1] + 1)
    groups = _group_policies(labels)

    subtitle = (
        f"averaged over {seed_count} simulations"
        if seed_count > 1
        else "shared random seed"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f"Policy simulation: cumulative bandit {metric_name} ({subtitle})")

    # Redraw baseline policies on each family subplot for direct visual comparison
    for ax in axes:
        for bl_label, bl_color in BASELINES:
            bl_idx = labels.index(bl_label)
            ax.plot(rounds, cum_values[bl_idx], color=bl_color, label=bl_label, linestyle='--')
        ax.set_xlabel("Round")
        ax.set_ylabel(f"Cumulative {metric_name}")
        ax.grid(True, alpha=0.3)

    # Draw all policy regrets/rewards, using family colours for each plot
    for ax, (family_prefix, cmap_name) in zip(axes, POLICY_FAMILIES):
        indices = groups[family_prefix]
        colors = _get_family_colors(len(indices), cmap_name)
        for idx, color in zip(indices, colors):
            ax.plot(rounds, cum_values[idx], color=color, label=labels[idx])
        ax.set_title(f"{family_prefix} {subplot_suffix}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Reward plotting
def plot_rewards(
        seed_count: int,
        labels: list[str],
        mean_rewards: np.ndarray,   # (n_policies, rounds)
        output_dir: str
) -> None:
    """
    Plot cumulative reward curves for all policies.
    """
    _plot_acc_metric(
        seed_count,
        labels,
        mean_rewards,
        output_dir + "rewards.png",
        metric_name="reward",
        subplot_suffix="rewards",
    )

def plot_regrets(
        seed_count: int,
        labels: list[str],
        mean_regrets: np.ndarray,   # (n_policies, rounds)
        output_dir: str
) -> None:
    """
    Plot cumulative regret curves for all policies.
    """
    _plot_acc_metric(
        seed_count,
        labels,
        mean_regrets,
        output_dir + "regrets.png",
        metric_name="regret",
        subplot_suffix="regret",
    )

def plot_ba_metric_over_time(
        metric_name: str,
        metric_rounds: np.ndarray,  # (n_checkpoints,)
        mean_values: np.ndarray,    # (n_policies, n_checkpoints)
        std_values: np.ndarray,     # (n_policies, n_checkpoints)
        labels: list[str],
        output_dir: str,
        filename: str,
        longtail_percentile: float | None
) -> None:
    """
    Plot a single beyond-accuracy metric over time, one subplot per policy family.
    """

    groups = _group_policies(labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Policy simulation: {metric_name} over time"
        + (f" (threshold: {longtail_percentile:.1f}%)" if longtail_percentile else "")
    )

    # Plot all policy metric means (the cumulative ones) alongside the std-deviations too
    for ax, (prefix, cmap_name) in zip(axes, POLICY_FAMILIES):
        # Baselines
        for bl_label, bl_color in BASELINES:
            bl_idx = labels.index(bl_label)
            ax.plot(metric_rounds, mean_values[bl_idx], color=bl_color, label=bl_label, linestyle='--')
            ax.fill_between(
                metric_rounds,
                mean_values[bl_idx] - std_values[bl_idx],
                mean_values[bl_idx] + std_values[bl_idx],
                color=bl_color, alpha=0.15
            )

        # Policy families
        indices = groups[prefix]
        colors = _get_family_colors(len(indices), cmap_name)
        for idx, color in zip(indices, colors):
            ax.plot(metric_rounds, mean_values[idx], color=color, label=labels[idx])
            ax.fill_between(
                metric_rounds,
                mean_values[idx] - std_values[idx],
                mean_values[idx] + std_values[idx],
                color=color, alpha=0.20
            )

        ax.set_title(f"{prefix}: {metric_name}")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir + filename)
    plt.close()


def plot_ba_metrics_over_time(
        metric_rounds: np.ndarray,
        all_gini: np.ndarray,       # (seeds, policies, n_checkpoints)
        all_coverage: np.ndarray,
        all_arp: np.ndarray,
        labels: list[str],
        output_dir: str,
        longtail_percentile: float
) -> None:
    """
    Compute mean/std across seeds and plot Gini, long-tail coverage, and ARP over time.
    """
    for arr, name, fname in [
        (all_gini,     "Gini Coefficient",              "gini.png"),
        (all_coverage, "Long-Tail Coverage",            "coverage.png"),
        (all_arp,      "Average Recommendation Popularity", "arp.png"),
    ]:
        mean_vals = arr.mean(axis=0)   # (policies, n_checkpoints)
        std_vals  = arr.std(axis=0)
        plot_ba_metric_over_time(
            name,
            metric_rounds,
            mean_vals,
            std_vals,
            labels,
            output_dir,
            fname,
            longtail_percentile,
        )

# Master entry point for `plot` CLI
def generate_all_plots(
        raw_results: dict,
        flags: dict,
        output_dir: str,
        metric_interval: int = 500
) -> None:
    """
    Regenerate all plots from saved raw_results and flags.

    Beyond-accuracy metric time-series arrays are recomputed when metric_interval
    differs from the interval stored in flags.json. Otherwise, pre-computed arrays
    from raw_results.npz are reused.
    """
    all_rewards         = raw_results['all_rewards']
    all_regrets         = raw_results['all_regrets']
    all_recommendations = raw_results['all_recommendations']
    item_popularity     = raw_results['item_popularity']

    seed_count   = all_rewards.shape[0]
    mean_rewards = all_rewards.mean(axis=0)
    mean_regrets = all_regrets.mean(axis=0)
    longtail_percentile = float(flags.get('longtail_percentile', 80.0))
    stored_metric_interval = int(flags.get('metric_interval', 500))

    labels = dlb_sim.build_policy_labels(flags['epsilon'], flags['alpha'], flags['ts_v'])

    # Reward / regret
    plot_rewards(seed_count, labels, mean_rewards, output_dir)
    plot_regrets(seed_count, labels, mean_regrets, output_dir)

    # Recompute BA metrics only when the requested interval differs from the run's interval
    if metric_interval == stored_metric_interval:
        metric_rounds = raw_results['metric_rounds']
        all_gini = raw_results['all_gini']
        all_coverage = raw_results['all_coverage']
        all_arp = raw_results['all_arp']
    else:
        metric_rounds, all_gini, all_coverage, all_arp = dlb_sim.compute_ba_metrics_over_time(
            all_recommendations, item_popularity, metric_interval, longtail_percentile
        )

    plot_ba_metrics_over_time(
        metric_rounds,
        all_gini,
        all_coverage,
        all_arp,
        labels,
        output_dir,
        longtail_percentile,
    )