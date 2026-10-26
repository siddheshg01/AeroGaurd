"""Exploratory Data Analysis and Visualization for AeroGuard AI Phase 1."""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless file generation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    SENSOR_COLUMNS,
    SENSOR_DESCRIPTIONS,
    SETTING_COLUMNS,
)
from src.data_loader import add_training_rul


def compute_sensor_statistics(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics and variance profile for all sensors.
    Identifies zero-variance (uninformative) sensors and degradation trends.

    Returns
    -------
    pd.DataFrame
        Detailed statistical profile for each sensor.
    """
    stats_list = []
    for col in SENSOR_COLUMNS:
        series = df_train[col]
        std = float(series.std())
        variance = float(series.var())
        mean = float(series.mean())
        min_val = float(series.min())
        max_val = float(series.max())
        is_constant = bool(std < 1e-6)

        stats_list.append({
            "sensor": col,
            "description": SENSOR_DESCRIPTIONS.get(col, "N/A"),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "variance": round(variance, 6),
            "is_constant": is_constant,
            "informative": not is_constant
        })

    stats_df = pd.DataFrame(stats_list)
    return stats_df


def save_sensor_statistics(stats_df: pd.DataFrame, output_dir: Path = REPORTS_DIR) -> Path:
    """Save sensor statistics to CSV and Markdown in reports directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sensor_summary_statistics.csv"
    md_path = output_dir / "sensor_summary_statistics.md"

    stats_df.to_csv(csv_path, index=False)
    
    # Pure-Python markdown table generation (no external tabulate dependency required)
    headers = list(stats_df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in stats_df.itertuples(index=False)
    ]
    table_content = "\n".join([header_row, separator_row] + data_rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AeroGuard AI - Sensor Statistics & Variance Profile\n\n")
        f.write(table_content)
        f.write("\n\n*Note: Constant sensors with ~0 variance provide no degradation information under FD001 single-condition operating mode.*\n")

    return csv_path


def plot_sensor_degradation_trajectories(
    df_train_with_rul: pd.DataFrame,
    sample_units: List[int] = [1, 2, 3, 4, 5],
    output_path: Path = FIGURES_DIR / "sensor_degradation_trajectories.png"
) -> Path:
    """
    Generate multi-panel degradation trajectory plots for key physical sensors
    across sample aircraft engines.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Key active sensors reflecting thermal, pressure, and operational degradation
    key_sensors = [
        ("sensor_2", "LPC Outlet Temp (T24) [°R]"),
        ("sensor_3", "HPC Outlet Temp (T30) [°R]"),
        ("sensor_4", "LPT Outlet Temp (T50) [°R]"),
        ("sensor_7", "HPC Outlet Pressure (P30) [psia]"),
        ("sensor_11", "HPC Static Pressure (Ps30) [psia]"),
        ("sensor_12", "Ratio of Fuel Flow to Ps30 [pps/psi]"),
        ("sensor_15", "Bypass Ratio (BPR)"),
        ("sensor_21", "LPT Coolant Bleed (W32) [lbm/s]")
    ]

    sns.set_theme(style="whitegrid", palette="tab10")
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=False)
    axes = axes.flatten()

    filtered_df = df_train_with_rul[df_train_with_rul["unit_number"].isin(sample_units)]
    unit_palette = {unit: color for unit, color in zip(sample_units, sns.color_palette("tab10", len(sample_units)))}

    for idx, (sensor_col, sensor_label) in enumerate(key_sensors):
        ax = axes[idx]
        for unit in sample_units:
            unit_data = filtered_df[filtered_df["unit_number"] == unit]
            ax.plot(
                unit_data["cycle"],
                unit_data[sensor_col],
                label=f"Engine #{unit}",
                color=unit_palette[unit],
                alpha=0.85,
                linewidth=1.8
            )
            # Mark the point of failure (final cycle)
            last_point = unit_data.iloc[-1]
            ax.scatter(
                last_point["cycle"],
                last_point[sensor_col],
                color=unit_palette[unit],
                s=45,
                zorder=5,
                edgecolors="black"
            )

        ax.set_title(f"{sensor_col.upper()}: {sensor_label}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Operational Cycle", fontsize=10)
        ax.set_ylabel("Measured Value", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        if idx == 0:
            ax.legend(title="Sample Fleet Units", loc="upper left", fontsize=9)

    plt.suptitle(
        "AeroGuard AI - Turbofan Engine Sensor Degradation Trajectories (Run-to-Failure)",
        fontsize=14,
        fontweight="bold",
        y=0.995
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_engine_lifespan_distribution(
    df_train: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "engine_lifespan_distribution.png"
) -> Path:
    """
    Generate combined histogram, KDE, and boxplot showing fleet operational lifespan
    (max cycles until failure) distribution across all training engines.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lifespans = df_train.groupby("unit_number")["cycle"].max().reset_index()
    lifespans.rename(columns={"cycle": "lifespan_cycles"}, inplace=True)

    mean_life = lifespans["lifespan_cycles"].mean()
    median_life = lifespans["lifespan_cycles"].median()
    min_life = lifespans["lifespan_cycles"].min()
    max_life = lifespans["lifespan_cycles"].max()

    sns.set_theme(style="whitegrid")
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [0.25, 0.75]}
    )

    # Top panel: Boxplot
    sns.boxplot(
        x=lifespans["lifespan_cycles"],
        ax=ax_box,
        color="#2b5c8f",
        fliersize=5,
        linewidth=1.5
    )
    ax_box.set(xlabel="")
    ax_box.set_title("Fleet Lifespan Distribution (100 C-MAPSS Engines)", fontsize=13, fontweight="bold")

    # Bottom panel: Histogram + KDE
    sns.histplot(
        lifespans["lifespan_cycles"],
        kde=True,
        ax=ax_hist,
        color="#1f77b4",
        bins=15,
        edgecolor="black",
        alpha=0.65
    )

    ax_hist.axvline(mean_life, color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {mean_life:.1f} cycles")
    ax_hist.axvline(median_life, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Median: {median_life:.0f} cycles")
    ax_hist.axvline(min_life, color="#9467bd", linestyle=":", linewidth=1.5, label=f"Min: {min_life} cycles")
    ax_hist.axvline(max_life, color="#8c564b", linestyle=":", linewidth=1.5, label=f"Max: {max_life} cycles")

    ax_hist.set_xlabel("Operational Lifespan (Cycles to Failure)", fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Engine Count Frequency", fontsize=11, fontweight="bold")
    ax_hist.legend(loc="upper right", fontsize=10, frameon=True)
    ax_hist.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sensor_correlation_with_rul(
    df_train_with_rul: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "sensor_correlation_with_rul.png"
) -> Path:
    """
    Calculate and plot Pearson correlation of operational settings and sensors with RUL.
    Highlights strong positive/negative wear indicators.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = SETTING_COLUMNS + SENSOR_COLUMNS
    corr_dict = {}
    for col in feature_cols:
        if df_train_with_rul[col].std() < 1e-6:
            corr_dict[col] = 0.0
        else:
            c = df_train_with_rul["RUL"].corr(df_train_with_rul[col])
            corr_dict[col] = 0.0 if np.isnan(c) else float(c)

    corr_df = pd.DataFrame([
        {"Feature": col, "Correlation_with_RUL": val}
        for col, val in corr_dict.items()
    ]).sort_values(by="Correlation_with_RUL", ascending=True)

    corr_df["Description"] = corr_df["Feature"].apply(lambda f: SENSOR_DESCRIPTIONS.get(f, f))
    corr_df["Color"] = np.where(corr_df["Correlation_with_RUL"] > 0, "#2ca02c", "#d62728")
    corr_df.loc[corr_df["Correlation_with_RUL"].abs() < 0.05, "Color"] = "#7f7f7f"

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(
        corr_df["Feature"],
        corr_df["Correlation_with_RUL"],
        color=corr_df["Color"],
        edgecolor="black",
        alpha=0.85
    )

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title("AeroGuard AI - Feature Correlation with Remaining Useful Life (RUL)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient with RUL", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sensor / Setting Feature", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Add numeric labels at end of bars
    for bar in bars:
        width = bar.get_width()
        offset = 0.02 if width >= 0 else -0.06
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_interactive_dashboard(
    df_train_with_rul: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "interactive_dashboard.html"
) -> Path:
    """
    Generate a standalone interactive Plotly HTML dashboard for engine degradation inspection.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Sensor 2 (LPC Outlet Temp) vs Cycle",
            "Sensor 4 (LPT Outlet Temp) vs Cycle",
            "Sensor 11 (HPC Static Pressure) vs Cycle",
            "Sensor 12 (Fuel Flow to Ps30 Ratio) vs Cycle"
        )
    )

    sample_engines = [1, 2, 3, 5, 10]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, engine_id in enumerate(sample_engines):
        sub = df_train_with_rul[df_train_with_rul["unit_number"] == engine_id]
        c = colors[idx]

        # S2
        fig.add_trace(
            go.Scatter(x=sub["cycle"], y=sub["sensor_2"], mode="lines",
                       name=f"Engine {engine_id}", legendgroup=f"Engine {engine_id}",
                       line=dict(color=c, width=2)),
            row=1, col=1
        )
        # S4
        fig.add_trace(
            go.Scatter(x=sub["cycle"], y=sub["sensor_4"], mode="lines",
                       name=f"Engine {engine_id}", legendgroup=f"Engine {engine_id}",
                       showlegend=False, line=dict(color=c, width=2)),
            row=1, col=2
        )
        # S11
        fig.add_trace(
            go.Scatter(x=sub["cycle"], y=sub["sensor_11"], mode="lines",
                       name=f"Engine {engine_id}", legendgroup=f"Engine {engine_id}",
                       showlegend=False, line=dict(color=c, width=2)),
            row=2, col=1
        )
        # S12
        fig.add_trace(
            go.Scatter(x=sub["cycle"], y=sub["sensor_12"], mode="lines",
                       name=f"Engine {engine_id}", legendgroup=f"Engine {engine_id}",
                       showlegend=False, line=dict(color=c, width=2)),
            row=2, col=2
        )

    fig.update_layout(
        title="AeroGuard AI - Interactive Turbofan Engine Degradation Explorer",
        template="plotly_white",
        height=750,
        width=1100,
        hovermode="x unified"
    )

    fig.write_html(str(output_path))
    return output_path


def run_eda_pipeline(df_train: pd.DataFrame) -> Dict[str, Any]:
    """
    Run complete EDA pipeline: compute statistics and generate all visualizations.

    Returns
    -------
    Dict[str, Any]
        Dictionary with paths to all generated artifacts.
    """
    df_train_with_rul = add_training_rul(df_train)

    stats_df = compute_sensor_statistics(df_train)
    save_sensor_statistics(stats_df)

    traj_path = plot_sensor_degradation_trajectories(df_train_with_rul)
    lifespan_path = plot_engine_lifespan_distribution(df_train)
    corr_path = plot_sensor_correlation_with_rul(df_train_with_rul)
    html_path = generate_interactive_dashboard(df_train_with_rul)

    return {
        "sensor_stats_csv": str(REPORTS_DIR / "sensor_summary_statistics.csv"),
        "sensor_stats_md": str(REPORTS_DIR / "sensor_summary_statistics.md"),
        "trajectory_plot": str(traj_path),
        "lifespan_plot": str(lifespan_path),
        "correlation_plot": str(corr_path),
        "interactive_html": str(html_path),
        "constant_sensors": stats_df[stats_df["is_constant"]]["sensor"].tolist(),
        "informative_sensors": stats_df[stats_df["informative"]]["sensor"].tolist()
    }
