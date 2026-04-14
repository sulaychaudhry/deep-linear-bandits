import numpy as np
import matplotlib.pyplot as plt

# Policy family definitions: (label prefix, colourmap name, flags.json key, hyperparameter symbol)
POLICY_FAMILIES = [
    ("ε-greedy", "Blues",   "epsilon", "ε"),
    ("LinUCB",   "Oranges", "alpha",   "α"),
    ("TS",       "Greens",  "ts_v",    "ʋ"),
]
BASELINES = [("Greedy", "#444444"), ("Random", "#aaaaaa"), ("Popularity", "#9b30d9")]

def _get_family_colours(n: int, cmap_name: str) -> list:
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
    groups = {prefix: [] for prefix, *_ in POLICY_FAMILIES}
    for i, label in enumerate(labels):
        for prefix, *_ in POLICY_FAMILIES:
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
        for bl_label, bl_colour in BASELINES:
            bl_idx = labels.index(bl_label)
            ax.plot(rounds, cum_values[bl_idx], color=bl_colour, label=bl_label, linestyle='--')
        ax.set_xlabel("Round")
        ax.set_ylabel(f"Cumulative {metric_name}")
        ax.grid(True, alpha=0.3)

    # Draw all policy regrets/rewards, using family colours for each plot
    for ax, (family_prefix, cmap_name, *_) in zip(axes, POLICY_FAMILIES):
        indices = groups[family_prefix]
        colours = _get_family_colours(len(indices), cmap_name)
        for idx, colour in zip(indices, colours):
            ax.plot(rounds, cum_values[idx], color=colour, label=labels[idx])
        ax.set_title(f"{family_prefix} {subplot_suffix}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Reward plotting
def plot_rewards(metrics: dict, output_dir: str) -> None:
    """
    Plot cumulative reward curves for all policies.
    """
    _plot_acc_metric(
        metrics['seed_count'],
        metrics['labels'],
        np.array(metrics['mean_rewards']),
        output_dir + "rewards.png",
        metric_name="reward",
        subplot_suffix="rewards",
    )

def plot_regrets(metrics: dict, output_dir: str) -> None:
    """
    Plot cumulative regret curves for all policies.
    """
    _plot_acc_metric(
        metrics['seed_count'],
        metrics['labels'],
        np.array(metrics['mean_regrets']),
        output_dir + "regrets.png",
        metric_name="regret",
        subplot_suffix="regret",
    )

def _plot_ba_metric_over_time(
        metric_name: str,
        metric_rounds: np.ndarray,  # (n_checkpoints,)
        mean_values: np.ndarray,    # (n_policies, n_checkpoints)
        std_values: np.ndarray,     # (n_policies, n_checkpoints)
        labels: list[str],
        seed_count: int,
        output_dir: str,
        filename: str,
        longtail_percentile: float | None,
        legend_loc: str = 'best'    # Legends keep blocking the plots too much, so passing in explicitly
) -> None:
    """
    Plot a single beyond-accuracy metric over time, one subplot per policy family.
    """

    groups = _group_policies(labels)

    subtitle = (
        f"averaged over {seed_count} simulations"
        if seed_count > 1
        else "shared random seed"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(
        f"Policy simulation: {metric_name} over time"
        + (f" (threshold: {longtail_percentile:.1f}%)" if longtail_percentile else "")
        + f" ({subtitle})"
    )

    # Plot all policy metric means (the cumulative ones) alongside the std-deviations too
    for ax, (prefix, cmap_name, *_) in zip(axes, POLICY_FAMILIES):
        # Baselines
        for bl_label, bl_colour in BASELINES:
            bl_idx = labels.index(bl_label)
            ax.plot(metric_rounds, mean_values[bl_idx], color=bl_colour, label=bl_label, linestyle='--')
            ax.fill_between(
                metric_rounds,
                mean_values[bl_idx] - std_values[bl_idx],
                mean_values[bl_idx] + std_values[bl_idx],
                color=bl_colour, alpha=0.15
            )

        # Policy families
        indices = groups[prefix]
        colours = _get_family_colours(len(indices), cmap_name)
        for idx, colour in zip(indices, colours):
            ax.plot(metric_rounds, mean_values[idx], color=colour, label=labels[idx])
            ax.fill_between(
                metric_rounds,
                mean_values[idx] - std_values[idx],
                mean_values[idx] + std_values[idx],
                color=colour, alpha=0.20
            )

        ax.set_title(f"{prefix}: {metric_name}")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_loc)

    plt.tight_layout()
    plt.savefig(output_dir + filename)
    plt.close()

def plot_ba_metrics_over_time(metrics: dict, output_dir: str) -> None:
    """
    Plot Gini, long-tail coverage, and ARP over time from their precomputed mean/std arrays.
    """
    
    for mean_key, std_key, name, fname, loc in [
        ('mean_gini_over_time',     'std_gini_over_time',     "Gini Coefficient",                   "gini.png",      'lower left'),
        ('mean_coverage_over_time', 'std_coverage_over_time', "Long-Tail Coverage",                 "coverage.png",  'upper left'),
        ('mean_arp_over_time',      'std_arp_over_time',      "Average Recommendation Popularity",  "arp.png",       'lower right'),
    ]:
        _plot_ba_metric_over_time(
            name, np.array(metrics['metric_rounds']),
            np.array(metrics[mean_key]), np.array(metrics[std_key]),
            metrics['labels'], metrics['seed_count'], output_dir, fname,
            (metrics['longtail_percentile'] if name=="Long-Tail Coverage" else None),
            legend_loc=loc
        )

def plot_regret_rolling(metrics: dict, output_dir: str, window: int = 500) -> None:
    """
    Plot rolling-mean per-round regret (regret rate), one subplot per policy family.
    """

    labels           = metrics['labels']
    seed_count       = metrics['seed_count']
    mean_cum_regrets = np.array(metrics['mean_cumulative_regrets'])

    rounds = mean_cum_regrets.shape[1]
    window = min(window, rounds)

    # Rolling mean via cumsum trick:
    #   `mean_cumulative_regrets` is already the cumsum of per-round regrets, so
    #   adding a zero to start each cumsum off gives cs of shape (n_policies, rounds+1)
    #   where cs[i, j] = cumulative regret for policy i from [0, j)
    #
    #   Then cs[i, k+window] - cs[i, k] = sum of rounds [k, k+window) for policy i
    #   (at some round k); can then use cs[:, window:] - cs[:, :-window] to vectorise
    #   this across all k >= window giving an (n_policies, rounds-window+1) array of
    #   window sums. Then divide by `window` to get the rolling mean, vectorised too`
    #
    #   Note of course the x-axis will start at round `window`
    cs = np.concatenate([np.zeros((len(labels), 1)), mean_cum_regrets], axis=1)
    smoothed = (cs[:, window:] - cs[:, :-window]) / window     # (n_policies, rounds-window+1)
    x = np.arange(window, rounds + 1)

    groups = _group_policies(labels)
    subtitle = (
        f"averaged over {seed_count} simulations"
        if seed_count > 1
        else "shared random seed"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(
        f"Policy simulation: per-round regret (rolling mean, window={window}) ({subtitle})"
    )

    for ax in axes:
        for bl_label, bl_colour in BASELINES:
            bl_idx = labels.index(bl_label)
            ax.plot(x, smoothed[bl_idx], color=bl_colour, label=bl_label, linestyle='--')
        ax.set_xlabel("Round")
        ax.set_ylabel("Regret rate (rolling mean)")
        ax.grid(True, alpha=0.3)

    for ax, (family_prefix, cmap_name, *_) in zip(axes, POLICY_FAMILIES):
        indices = groups[family_prefix]
        colours = _get_family_colours(len(indices), cmap_name)
        for idx, colour in zip(indices, colours):
            ax.plot(x, smoothed[idx], color=colour, label=labels[idx])
        ax.set_title(f"{family_prefix}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir + "regret_rolling.png")
    plt.close()

def plot_final_reward_bars(metrics: dict, output_dir: str) -> None:
    """
    Horizontal bar chart of final cumulative reward per policy with +/-1 std error bars.
    """

    labels = metrics['labels']
    rounds = len(metrics['mean_rewards'][0])

    # Build per-policy colours matching the family convention
    groups = _group_policies(labels)
    colours = [c for _, c in BASELINES]
    for family_prefix, cmap_name, *_ in POLICY_FAMILIES:
        indices = groups[family_prefix]
        colours.extend(_get_family_colours(len(indices), cmap_name))

    # Plot final cumulative reward across all policies with std error bars (to show cross-seed variation)
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.45)))
    y = np.arange(len(labels))
    ax.barh(y, metrics['mean_final_cumulative_rewards'], xerr=metrics['std_final_cumulative_rewards'], color=colours, capsize=4, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Final cumulative reward")
    ax.set_title(f"Policy comparison: final cumulative reward after {rounds} rounds (mean +/- std across {metrics['seed_count']} seeds)")
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir + "final_reward_bars.png")
    plt.close()

def plot_final_ba_bars(metrics: dict, output_dir: str) -> None:
    """
    Horizontal bar charts of final Gini, long-tail coverage, and ARP per policy,
    with +/-1 std error bars; one subplot per metric.
    """

    labels = metrics['labels']
    rounds = len(metrics['mean_rewards'][0])

    # Build per-policy colours matching the family convention
    groups = _group_policies(labels)
    colours = [c for _, c in BASELINES]
    for family_prefix, cmap_name, *_ in POLICY_FAMILIES:
        indices = groups[family_prefix]
        colours.extend(_get_family_colours(len(indices), cmap_name))

    ba_metrics = [
        (metrics['mean_gini'],              metrics['std_gini'],              "Gini Coefficient"),
        (metrics['mean_longtail_coverage'], metrics['std_longtail_coverage'], f"Long-Tail Coverage (threshold: {metrics['longtail_percentile']:.0f}%)"),
        (metrics['mean_arp'],               metrics['std_arp'],               "Average Recommendation Popularity"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, max(6, len(labels) * 0.45)))
    fig.suptitle(f"Policy comparison: final beyond-accuracy metrics after {rounds} rounds (mean +/- std across {metrics['seed_count']} seeds)")
    for ax, (mean_vals, std_vals, xlabel) in zip(axes, ba_metrics):
        y = np.arange(len(labels))
        ax.barh(y, np.array(mean_vals), xerr=np.array(std_vals), color=colours, capsize=4, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(xlabel)
        ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir + "final_ba_bars.png")
    plt.close()

def plot_hyperparameter_sensitivity(metrics: dict, flags: dict, output_dir: str) -> None:
    """
    Final cumulative reward vs hyperparameter value, one subplot per policy family.
    """

    final_cum_mean = np.array(metrics['mean_final_cumulative_rewards'])
    final_cum_std  = np.array(metrics['std_final_cumulative_rewards'])
    rounds         = len(metrics['mean_rewards'][0])

    groups = _group_policies(metrics['labels'])

    subtitle = (
        f"averaged over {metrics['seed_count']} simulations"
        if metrics['seed_count'] > 1
        else "shared random seed"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f"Hyperparameter sensitivity: final cumulative reward after {rounds} rounds ({subtitle})")

    # Sensitivity plots for each policy family
    for ax, (prefix, cmap_name, flag_key, param_sym) in zip(axes, POLICY_FAMILIES):
        # Retrieve hyperparams for this policy
        params = flags.get(flag_key, [])
        if not params:
            ax.set_visible(False)
            continue

        indices = groups[prefix]
        colour  = plt.get_cmap(cmap_name)(0.60)
        values  = [final_cum_mean[i] for i in indices]
        errs    = [final_cum_std[i]  for i in indices]

        # Draw points with stds
        ax.plot(params, values, marker='o', color=colour, linewidth=2)
        ax.fill_between(
            params,
            [v - e for v, e in zip(values, errs)],
            [v + e for v, e in zip(values, errs)],
            alpha=0.15,
            color=colour
        )
        # Baselines as horizontal reference lines
        for bl_label, bl_colour in BASELINES:
            bl_val = final_cum_mean[metrics['labels'].index(bl_label)]
            ax.axhline(bl_val, color=bl_colour, linestyle='--', label=bl_label)
        ax.set_xlabel(param_sym)
        ax.set_ylabel("Final cumulative reward")
        ax.set_title(f"{prefix} sensitivity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir + "hyperparam_sensitivity.png")
    plt.close()

def plot_diversity_accuracy_scatter(metrics: dict, output_dir: str) -> None:
    """
    Scatter of final cumulative reward vs. Gini coefficient, long-tail coverage, and ARP.
    Each subplot is a different diversity axis, directly visualising the diversity-accuracy tradeoff.
    """

    labels         = metrics['labels']
    final_cum_mean = np.array(metrics['mean_final_cumulative_rewards'])
    rounds         = len(metrics['mean_rewards'][0])

    subtitle = (
        f"averaged over {metrics['seed_count']} simulations"
        if metrics['seed_count'] > 1
        else "shared random seed"
    )

    groups = _group_policies(labels)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f"Diversity-accuracy tradeoff after {rounds} rounds ({subtitle})")

    for i, (ax, (y_vals, y_label)) in enumerate(zip(axes, [
        (metrics['mean_gini'],               "Gini Coefficient"),
        (metrics['mean_longtail_coverage'], f"Long-Tail Coverage (threshold: {metrics['longtail_percentile']:.0f}%)"),
        (metrics['mean_arp'],                "Average Recommendation Popularity"),
    ])):
        # Only label baseline points on the first subplot so fig.legend() doesn't produce duplicates
        # (using shared legend between all subplots, since all the same policies on them)
        for bl_label, bl_colour in BASELINES:
            bl_idx = labels.index(bl_label)
            ax.scatter(
                final_cum_mean[bl_idx],
                y_vals[bl_idx],
                color=bl_colour,
                s=80,
                marker='D',
                zorder=5,
                label=(bl_label if i == 0 else None)
            )

        # Same logic for policy points
        for prefix, cmap_name, *_ in POLICY_FAMILIES:
            indices = groups[prefix]
            colours = _get_family_colours(len(indices), cmap_name)
            for idx, colour in zip(indices, colours):
                ax.scatter(
                    final_cum_mean[idx], y_vals[idx], color=colour, s=60,
                    label=(labels[idx] if i == 0 else None)
                )

        ax.set_xlabel("Final cumulative reward")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

        # Long-tail coverage wasn't showing 1.0 on the axis sometimes weirdly, so just force it to use 0-1.0 axis
        if y_label.startswith("Long-Tail"):
            ax.set_ylim(0, 1.0)

    # Single shared legend at the bottom spanning all subplots
    handles, labs = axes[0].get_legend_handles_labels()
    n_cols = (len(handles) + 1) // 2  # ~2 rows keeps the legend wide & readable
    fig.legend(
        handles, labs,
        loc='lower center',
        ncol=n_cols,
        bbox_to_anchor=(0.5, 0.04),
        frameon=True,
        fontsize=10
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the legend below the subplots
    plt.savefig(output_dir + "diversity_accuracy_scatter.png")
    plt.close()

# Master entry point for all plot commands
def generate_all_plots(metrics: dict, flags: dict, output_dir: str) -> None:
    """
    Generate all plots from a pre-aggregated metrics dict (as written to metrics.json).
    """
    plot_rewards(metrics, output_dir)
    plot_regrets(metrics, output_dir)
    plot_ba_metrics_over_time(metrics, output_dir)
    plot_regret_rolling(metrics, output_dir)
    plot_final_reward_bars(metrics, output_dir)
    plot_final_ba_bars(metrics, output_dir)
    plot_hyperparameter_sensitivity(metrics, flags, output_dir)
    plot_diversity_accuracy_scatter(metrics, output_dir)