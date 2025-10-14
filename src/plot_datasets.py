"""Generate publication-quality visualizations for training and testing datasets.

This script traverses the datasets stored under the configured data directory and
creates plots that highlight the posterior cluster memberships for every point.
For two- and three-dimensional datasets we generate scatter plots, while for
datasets of higher dimensionality we resort to pair plots.

Usage
-----
```
python -m src.plot_datasets --data-dir data --output-dir outputs/dataset_plots
```
"""

from __future__ import annotations

import argparse
import re
from functools import partial
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch


# Configure a consistent aesthetic that works well for publications.
sns.set_theme(context="talk", style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.title_fontsize": 12,
        "legend.fontsize": 11,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality plots for every dataset located in the "
            "specified directory."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory that contains the datasets to visualise.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "dataset_plots",
        help="Directory in which the generated plots will be stored.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    """Return a filesystem-friendly representation of a dataset name."""

    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def is_training_dataset(name: str) -> bool:
    """Infer whether a dataset name corresponds to a training split."""

    lower_name = name.lower()
    return "train" in lower_name or "training" in lower_name


def discover_dataset_files(dataset_dir: Path) -> Tuple[Path, Path]:
    """Locate the feature and posterior files inside a dataset directory."""

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

    posterior_files = list(dataset_dir.glob("*true_posterior*.csv"))
    if not posterior_files:
        raise FileNotFoundError(
            f"No posterior file found in {dataset_dir}. Expected a '*true_posterior*.csv'."
        )
    posterior_file = posterior_files[0]

    feature_files = [
        file
        for file in dataset_dir.glob("*.csv")
        if "posterior" not in file.name.lower() and "membership" not in file.name.lower()
    ]
    if not feature_files:
        raise FileNotFoundError(
            f"No feature data file found in {dataset_dir}. Expected a CSV without "
            "'posterior' or 'membership' in its name."
        )
    feature_file = feature_files[0]
    return feature_file, posterior_file


def prepare_dataframe(feature_file: Path, posterior_file: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load the feature matrix and posterior probabilities for a dataset."""

    feature_df = pd.read_csv(feature_file)
    posterior_df = pd.read_csv(posterior_file)

    # Drop cluster assignment columns to keep only feature dimensions.
    feature_columns = [
        column
        for column in feature_df.columns
        if column.lower() not in {"truecluster", "predcluster"}
    ]

    features = feature_df[feature_columns].apply(lambda col: pd.to_numeric(col, errors="coerce"))

    if features.isnull().any(axis=None):
        valid_rows = ~features.isnull().any(axis=1)
        dropped = len(features) - int(valid_rows.sum())
        if dropped:
            print(
                f"Warning: Dropped {dropped} rows from {feature_file.name} due to "
                "non-numeric feature values."
            )
        features = features.loc[valid_rows]
        posterior_df = posterior_df.loc[valid_rows]

    posterior = posterior_df.to_numpy(dtype=float)
    row_sums = posterior.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    posterior = posterior / row_sums

    return features, posterior


def compute_palette(num_clusters: int) -> List[Tuple[float, float, float]]:
    """Return a colour palette for the given number of clusters."""

    if num_clusters <= 10:
        palette = sns.color_palette("tab10", n_colors=num_clusters)
    else:
        palette = sns.color_palette("husl", n_colors=num_clusters)
    return [tuple(to_rgb(col)) for col in palette]


def mix_colors(posteriors: np.ndarray, palette: Iterable[Tuple[float, float, float]]) -> List[str]:
    """Blend cluster colours for each sample according to posterior weights."""

    base = np.array(list(palette))
    mixed = np.clip(posteriors @ base, 0, 1)
    return [to_hex(color) for color in mixed]


def add_cluster_legend(ax: plt.Axes, palette: List[Tuple[float, float, float]]) -> None:
    patches = [
        Patch(color=to_hex(color), label=f"Cluster {index + 1}")
        for index, color in enumerate(palette)
    ]
    ax.legend(handles=patches, title="Posterior colour key", loc="best", frameon=True)


def plot_2d(features: pd.DataFrame, colors: List[str], dataset_name: str, category: str, palette: List[Tuple[float, float, float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    x, y = features.iloc[:, 0], features.iloc[:, 1]
    ax.scatter(x, y, c=colors, s=45, edgecolor="black", linewidth=0.25)
    ax.set_xlabel(features.columns[0])
    ax.set_ylabel(features.columns[1])
    ax.set_title(f"{dataset_name} ({category})")
    add_cluster_legend(ax, palette)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_3d(features: pd.DataFrame, colors: List[str], dataset_name: str, category: str, palette: List[Tuple[float, float, float]], output_path: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Required for 3D projection

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = features.iloc[:, 0], features.iloc[:, 1], features.iloc[:, 2]
    ax.scatter(x, y, z, c=colors, s=40, edgecolor="black", linewidth=0.25, depthshade=False)
    ax.set_xlabel(features.columns[0])
    ax.set_ylabel(features.columns[1])
    ax.set_zlabel(features.columns[2])
    ax.view_init(elev=18, azim=135)
    ax.set_title(f"{dataset_name} ({category})")
    add_cluster_legend(ax, palette)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _scatter_with_custom_colors(x, y, *, colors: List[str], **kwargs) -> None:
    kwargs.pop("color", None)
    kwargs.setdefault("s", 25)
    kwargs.setdefault("edgecolor", "white")
    kwargs.setdefault("linewidth", 0.3)
    plt.scatter(x, y, c=colors, **kwargs)


def plot_pairplot(features: pd.DataFrame, colors: List[str], dataset_name: str, category: str, palette: List[Tuple[float, float, float]], output_path: Path) -> None:
    g = sns.PairGrid(features, corner=True, diag_sharey=False)
    scatter = partial(_scatter_with_custom_colors, colors=colors)
    g.map_lower(scatter)
    g.map_diag(sns.histplot, color="0.3", edgecolor=None)

    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis="both", labelrotation=0)

    legend_patches = [
        Patch(color=to_hex(color), label=f"Cluster {index + 1}")
        for index, color in enumerate(palette)
    ]

    g.fig.suptitle(f"{dataset_name} ({category})", y=1.02)
    g.fig.legend(
        handles=legend_patches,
        title="Posterior colour key",
        loc="upper center",
        ncol=min(len(legend_patches), 3),
        frameon=True,
    )
    g.fig.tight_layout()
    g.fig.savefig(output_path, bbox_inches="tight")
    plt.close(g.fig)


def create_plot(features: pd.DataFrame, colors: List[str], dataset_name: str, category: str, palette: List[Tuple[float, float, float]], output_dir: Path) -> None:
    dimensionality = features.shape[1]
    safe_name = slugify(dataset_name)
    suffix = category.lower()
    output_path = output_dir / f"{safe_name}_{suffix}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dimensionality == 2:
        plot_2d(features, colors, dataset_name, category, palette, output_path)
    elif dimensionality == 3:
        plot_3d(features, colors, dataset_name, category, palette, output_path)
    else:
        plot_pairplot(features, colors, dataset_name, category, palette, output_path)

    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"The data directory {args.data_dir} does not exist.")

    dataset_dirs = sorted(path for path in args.data_dir.iterdir() if path.is_dir())
    if not dataset_dirs:
        raise RuntimeError(f"No dataset directories found in {args.data_dir}.")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        category = "Training" if is_training_dataset(dataset_name) else "Testing"
        feature_file, posterior_file = discover_dataset_files(dataset_dir)
        features, posterior = prepare_dataframe(feature_file, posterior_file)

        if features.empty:
            print(f"Skipping {dataset_name} because it has no usable feature rows.")
            continue

        palette = compute_palette(posterior.shape[1])
        colors = mix_colors(posterior, palette)
        create_plot(features, colors, dataset_name, category, palette, args.output_dir)


if __name__ == "__main__":
    main()

