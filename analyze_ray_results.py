from argparse import ArgumentParser
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_METRIC_GROUPS = {
    "reward_curves": [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
    ],
    "episode_length": [
        "episode_len_mean",
    ],
    "learner_stats": [
        "info/learner/default_policy/learner_stats/policy_loss",
        "info/learner/default/learner_stats/policy_loss",
        "info/learner/default_policy/learner_stats/vf_loss",
        "info/learner/default/learner_stats/vf_loss",
        "info/learner/default_policy/learner_stats/vf_explained_var",
        "info/learner/default/learner_stats/vf_explained_var",
        "info/learner/default_policy/learner_stats/entropy",
        "info/learner/default/learner_stats/entropy",
        "info/learner/default_policy/learner_stats/kl",
        "info/learner/default/learner_stats/kl",
    ],
    "throughput_and_timing": [
        "timers/sample_time_ms",
        "timers/learn_time_ms",
        "timers/sample_throughput",
        "timers/learn_throughput",
    ],
    "system_usage": [
        "perf/cpu_util_percent",
        "perf/ram_util_percent",
        "num_healthy_workers",
    ],
}


def parse_args():
    parser = ArgumentParser(description="Analyze RLlib Ray Tune training results.")
    parser.add_argument(
        "--results-dir",
        default="ray_results",
        help="Root directory containing Ray Tune experiment folders.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        help=(
            "Optional list of experiment or trial directory names to analyze. "
            "If omitted, all runs under results-dir are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results_analysis",
        help="Directory to write generated plots into.",
    )
    parser.add_argument(
        "--x-axis",
        default="timesteps_total",
        choices=["timesteps_total", "training_iteration", "time_total_s"],
        help="Column used for the plot x-axis.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling mean window. Use 1 to disable smoothing.",
    )
    return parser.parse_args()


def discover_progress_files(results_dir: Path, run_filters: Optional[Iterable[str]]) -> List[Path]:
    progress_files = sorted(results_dir.rglob("progress.csv"))
    if not run_filters:
        return progress_files

    filters = set(run_filters)
    selected = []
    for path in progress_files:
        parent_names = {ancestor.name for ancestor in [path.parent, *path.parents]}
        if filters.intersection(parent_names):
            selected.append(path)
    return selected


def load_runs(progress_files: List[Path]) -> Dict[str, pd.DataFrame]:
    runs = {}
    for path in progress_files:
        if path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            continue
        if df.empty:
            continue
        run_name = f"{path.parent.parent.name}/{path.parent.name}"
        df = df.sort_values("timesteps_total").reset_index(drop=True)
        runs[run_name] = {
            "dataframe": df,
            "progress_path": path,
        }
    return runs


def make_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def smooth_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def available_metrics(runs: Dict[str, dict], metrics: List[str], x_axis: str) -> List[str]:
    found = []
    for metric in metrics:
        if any(
            metric in run_info["dataframe"].columns
            and x_axis in run_info["dataframe"].columns
            for run_info in runs.values()
        ):
            found.append(metric)
    return found


def sanitize_filename(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(":", "_")


def plot_metric_group(
    runs: Dict[str, dict],
    metrics: List[str],
    x_axis: str,
    smooth_window: int,
    output_path: Path,
):
    usable_metrics = available_metrics(runs, metrics, x_axis)
    if not usable_metrics:
        return False

    # Avoid plotting duplicate aliases for the same learner metric name.
    seen_metric_suffixes = set()
    deduped_metrics = []
    for metric in usable_metrics:
        normalized = metric.replace("info/learner/default_policy/", "info/learner/default/")
        suffix = normalized.split("learner_stats/")[-1]
        if suffix in seen_metric_suffixes:
            continue
        seen_metric_suffixes.add(suffix)
        deduped_metrics.append(metric)
    usable_metrics = deduped_metrics

    fig, axes = plt.subplots(
        len(usable_metrics),
        1,
        figsize=(12, 3.6 * len(usable_metrics)),
        sharex=True,
    )
    if len(usable_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, usable_metrics):
        for run_name, run_info in runs.items():
            df = run_info["dataframe"]
            if metric not in df.columns or x_axis not in df.columns:
                continue
            metric_series = pd.to_numeric(df[metric], errors="coerce")
            x_series = pd.to_numeric(df[x_axis], errors="coerce")
            mask = x_series.notna() & metric_series.notna()
            if not mask.any():
                continue
            x_values = x_series[mask]
            y_values = metric_series[mask]
            ax.plot(
                x_values,
                y_values,
                alpha=0.25,
                linewidth=1.0,
            )
            ax.plot(
                x_values,
                smooth_series(y_values, smooth_window),
                label=run_name,
                linewidth=2.0,
            )
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel(x_axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_summary_table(runs: Dict[str, dict], output_path: Path):
    summary_rows = []
    for run_name, run_info in runs.items():
        df = run_info["dataframe"]
        row = {"run": run_name}
        for col in [
            "timesteps_total",
            "time_total_s",
            "episode_reward_mean",
            "episode_reward_max",
            "episode_reward_min",
            "episode_len_mean",
            "timers/sample_time_ms",
            "timers/learn_time_ms",
            "timers/sample_throughput",
            "timers/learn_throughput",
        ]:
            row[col] = df[col].dropna().iloc[-1] if col in df.columns and not df[col].dropna().empty else None
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)


def prepare_run_output_dir(results_dir: Path, output_dir: Path, progress_path: Path) -> Path:
    run_dir = progress_path.parent
    relative_dir = run_dir.relative_to(results_dir)
    target_dir = output_dir / relative_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def analyze_single_run(
    run_name: str,
    run_info: dict,
    results_dir: Path,
    output_dir: Path,
    x_axis: str,
    smooth_window: int,
):
    run_output_dir = prepare_run_output_dir(
        results_dir=results_dir,
        output_dir=output_dir,
        progress_path=run_info["progress_path"],
    )
    run_only = {run_name: run_info}
    for group_name, metrics in DEFAULT_METRIC_GROUPS.items():
        plot_metric_group(
            runs=run_only,
            metrics=metrics,
            x_axis=x_axis,
            smooth_window=smooth_window,
            output_path=run_output_dir / f"{sanitize_filename(group_name)}.png",
        )
    plot_summary_table(run_only, run_output_dir / "summary.csv")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    make_output_dir(output_dir)

    progress_files = discover_progress_files(results_dir, args.runs)
    if not progress_files:
        raise SystemExit(f"No progress.csv files found under {results_dir}")

    runs = load_runs(progress_files)
    if not runs:
        raise SystemExit("No non-empty runs found.")

    for run_name, run_info in runs.items():
        analyze_single_run(
            run_name=run_name,
            run_info=run_info,
            results_dir=results_dir,
            output_dir=output_dir,
            x_axis=args.x_axis,
            smooth_window=args.smooth_window,
        )

    for group_name, metrics in DEFAULT_METRIC_GROUPS.items():
        plot_metric_group(
            runs=runs,
            metrics=metrics,
            x_axis=args.x_axis,
            smooth_window=args.smooth_window,
            output_path=output_dir / f"all_runs__{sanitize_filename(group_name)}.png",
        )
    plot_summary_table(runs, output_dir / "all_runs__summary.csv")
    print(f"Wrote per-run plots and combined summaries to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
