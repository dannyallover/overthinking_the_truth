from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

def plot_label_thresholds(
    thresholds: dict,
    label: str,
    pfx_labs: list = ["True", "False"],
) -> None:
    """
    Plot quantile and mean |thresholds| corresponding to |label|.

    Parameters
    ----------
    thresholds : required, dict
        Thresholds correesponding to the |label|.
    label : required, str
        Label corresponding to the |thresholds|.
    pfx_labs : optional, list
        Prefix labels (e.g. True, False, Null) to plot.

    Returns
    ------
    None
    """
    fig, ax = plt.subplots(nrows=len(thresholds.keys()), ncols=1, figsize=(10, 10))
    for i, (k, v) in enumerate(thresholds.items()):
        v_ = v.cpu().detach().numpy()
        for j in range(len(pfx_labs)):
            ax[i].plot(v_[j, 0, :], label=pfx_labs[j])
        ax[i].legend()
        ax[i].set_title(f"{k} normalized probability of {label}")

    fig.tight_layout()
    show()

    return

def plot_layerwise_metric_heatmaps(
    metrics: dict, pfx_labs: list = ["True", "False"]
) -> None:
    """
    Plot heatmaps for |metrics| across the context positions and layers.

    Parameters
    ----------
    metrics : required, dict
        Metrics to plot.
    pfx_labs : optional, list
        Prefix labels (e.g. True, False, Null) to plot.

    Returns
    ------
    None
    """
    n_rows, n_cols = len(metrics.keys()), len(pfx_labs)
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, n_rows * 4))
    for i, (key, v) in enumerate(metrics.items()):
        ax_ = ax[i] if n_rows > 1 else ax
        for j in range(n_cols):
            sns.heatmap(v[0][j], vmin=0.0, vmax=1.0, ax=ax_[j], cmap="Reds")
            ax_[j].set_title(f"{pfx_labs[j]} prefix,\n {key}", size=20)
            ax_[j].set_xlabel("position in context")
            ax_[j].set_ylabel("layer depth")

    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    show()

    return

def plot_layerwise_metric_curves(
    metrics: dict,
    layer: int = -math.inf,
    context_pos: int = -math.inf,
    pfx_labs: list = ["True", "False"],
    show_confidence: bool = False,
) -> None:
    """
    Plot curves of layerwise |metrics| for a given |context_pos| across the
    layers or for a given |layer| across the contexts.

    Parameters
    ----------
    metrics : required, dict
        Metrics to plot.
    layer : optional, int
        Layer index.
    context_pos : optional, int
        Position of context.
    pfx_labs : optional, list
        Prefix labels (e.g. True, False, Null) to plot.
    show_confidence : optional, bool
        Indicator to show confidence interval lines.

    Returns
    ------
    None
    """
    n_rows, n_cols = len(metrics.keys()), 1
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, n_rows * 3.5))
    colors = [
        "blue",
        "orange",
        "green",
        "pink",
        "brown",
        "gray",
        "purple",
        "salmon",
        "black",
        "cornflowerblue",
        "lime",
        "navy",
    ]
    legend = list(pfx_labs)
    for i, (key, v) in enumerate(metrics.items()):
        ax_ = ax[i] if n_rows > 1 else ax
        y_min = 1

        for j in range(len(pfx_labs)):
            curve = (
                v[3 * (j // 2)][j % 2, layer]
                if layer != -math.inf
                else v[3 * (j // 2)][j % 2, :, context_pos]
            )
            curve_upper = (
                v[3 * (j // 2) + 1][j % 2, layer]
                if layer != -math.inf
                else v[3 * (j // 2) + 1][j % 2, :, context_pos]
            )
            curve_lower = (
                v[3 * (j // 2) + 2][j % 2, layer]
                if layer != -math.inf
                else v[3 * (j // 2) + 2][j % 2, :, context_pos]
            )

            x_axis = list(range(curve.shape[0]))
            ax_.plot(x_axis, curve, c=colors[j], label=pfx_labs[j])

            y_min = min(y_min, min(curve))
            if show_confidence:
                ax_.plot(x_axis, curve_upper, c=colors[j], linestyle="dotted")
                ax_.plot(x_axis, curve_lower, c=colors[j], linestyle="dotted")
                y_min = min(y_min, min(curve_lower))

        ax_.set_xticks([1] + list(range(5, curve.shape[0], 5)))
        ax_.set_xlabel("layer" if context_pos != -math.inf else "position in context")
        ax_.set_ylim(bottom=y_min - 0.1)
        ax_.set_title(key)
        ax_.legend(loc="upper left")

    if context_pos != -math.inf:
        fig.suptitle(f"Position: {context_pos}", y=1.03, size=20)
    elif layer != -math.inf:
        fig.suptitle(f"Layer: {layer}", y=1.03, size=20)
    fig.tight_layout()
    show()

    return

def plot_attn_metrics_detailed(
    metrics_df: pd.DataFrame,
    metrics: list,
    filters: dict,
    demo_indx: int,
    title_params: dict,
) -> None:
    """
    Plot a bar plot for each layer that show the distribution of scores for the
    attention heads.

    Parameters
    ----------
    metrics_df : required, pd.DataFrame
        Pandas dataframe which contains the metrics to be plotted.
    metrics : required, list
        Names of the metrics to be plotted.
    filters : required, dict
        The filtering parameters to be applied to the scores.
    demo_indx : required, int
        The demonstration index of interest.
    title_params : required, list
        List containing name of the model, name of the dataset, and prompt index.

    Returns
    ------
    None
    """

    n_layers = metrics_df["layer_indx"].max() + 1
    n_rows, n_cols = math.ceil(n_layers / 4), 4
    for metric in metrics:
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, n_cols * 10))
        demo_indx_ = (
            metrics_df["demo_indx"].max()
            if not (metrics_df["demo_indx"] == demo_indx).any()
            else demo_indx
        )
        metrics_df_copy = metrics_df[metrics_df["demo_indx"] == demo_indx_].copy()
        filt_col, top_k = filters[metric] if metric in filters else (metric, -1)
        metrics_df_copy = metrics_df_copy.sort_values(by=[filt_col], ascending=False)
        threshold = metrics_df_copy[filt_col].iloc[top_k]
        metrics_df_copy.loc[metrics_df_copy[filt_col] < threshold, metric] = np.nan

        metrics_df_copy = metrics_df_copy.replace(
            {"prefix_type": {0: "True", 1: "False"}}
        )

        for val in range(n_layers):
            x_indx, y_indx = val // n_cols, val % n_cols

            fig.add_subplot(
                sns.barplot(
                    x="head_indx",
                    y=metric,
                    hue="prefix_type",
                    data=metrics_df_copy[(metrics_df_copy["layer_indx"] == val)],
                    palette="Blues_d",
                    ax=axs[x_indx, y_indx],
                )
            )

            axs[x_indx, y_indx].tick_params(axis="x", which="major", labelsize=7)
            axs[x_indx, y_indx].title.set_text(f"layer {val}")
        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.95, wspace=0.4, hspace=0.4
        )
        fig.suptitle(
            f'model: {title_params["model"]} \n'
            f'dataset: {title_params["dataset"]} \n'
            f'prompt format: {title_params["prompt_indx"]} \n'
            f"position in context: {demo_indx_} \n"
            f"metric: {metric} \n"
            f'filter: {f"{filt_col} > " + "{:.2f}".format(threshold) + f" (top {top_k} heads)" if metric in filters else "none"}',
            y=1.01,
            x=0.5,
            size=20,
        )
        plt.show()

    return

def plot_attn_metrics_compressed(
    metrics_df: pd.DataFrame,
    metrics: list,
    filters: dict,
    demo_indx: int,
    title_params: dict,
) -> None:
    """
    Plots a scatterplot of the attention head metrics in |metrics_df| 
    for all layers.

    Parameters
    ----------
    metrics_df : required, pd.DataFrame
        Pandas dataframe which contains the metrics to be plotted.
    metrics : required, list
        Names of the metrics to be plotted.
    filters : required, dict
        The filtering parameters to be applied to the scores.
    demo_indx : required, int
        The demonstration index of interest.
    title_params : required, list
        List containing name of the model, name of the dataset, and prompt index.

    Returns
    ------
    None
    """
    n_layers = metrics_df["layer_indx"].max() + 1

    metrics_df = metrics_df.replace({"prefix_type": {0: "True", 1: "False"}})
    jiggle = metrics_df.replace({"prefix_type": {"True": -0.25, "False": 0.25}})[
        "prefix_type"
    ]
    metrics_df["layer_indx"] += jiggle
    for metric in metrics:
        plt.figure(figsize=(16, 15))
        demo_indx_ = (
            metrics_df["demo_indx"].max()
            if not (metrics_df["demo_indx"] == demo_indx).any()
            else demo_indx
        )
        metrics_df_copy = metrics_df[metrics_df["demo_indx"] == demo_indx_].copy()
        filt_col, top_k = filters[metric] if metric in filters else (metric, -1)
        metrics_df_copy = metrics_df_copy.sort_values(by=[filt_col], ascending=False)
        threshold = metrics_df_copy[filt_col].iloc[top_k]
        metrics_df_copy.loc[metrics_df_copy[filt_col] < threshold, metric] = np.nan

        ax = sns.scatterplot(
            x=metric,
            y="layer_indx",
            hue="prefix_type",
            data=metrics_df_copy,
            palette="Blues_d",
        )

        plt.title(
            f'model: {title_params["model"]} \n'
            f'dataset: {title_params["dataset"]} \n'
            f'prompt format: {title_params["prompt_indx"]} \n'
            f"position in context: {demo_indx_} \n"
            f"metric: {metric} \n"
            f'filter: {f"{filt_col} > " + "{:.2f}".format(threshold) + f" (top {top_k} heads)" if metric in filters else "none"}',
            size=20,
        )

        for h_line in range(n_layers):
            ax.axhline(h_line, linewidth=0.25, color="black")
        plt.yticks(np.arange(0, n_layers, 1))

    return