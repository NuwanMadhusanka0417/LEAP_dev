"""Plot 7-day forecast CSVs in this folder: power (kWh) and GHI (W/m²), predicted vs real."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR

POWER_PRED = "#2196F3"
POWER_REAL = "#D32F2F"
GHI_PRED = "#FB8C00"
GHI_REAL = "#2E7D32"


def _site_label(csv_path: Path) -> str:
    stem = csv_path.stem  # forecast_7d_pvlib_library
    key = stem.replace("forecast_7d_pvlib_", "").replace("_", " ")
    return key.upper() if key.isalpha() and len(key) <= 4 else key.title()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for i, col in enumerate(df.columns):
        c = str(col).strip()
        cl = c.lower()
        if c == "" or c.startswith("Unnamed"):
            rename[col] = "real_kwh" if i == 2 else f"unnamed_{i}"
        elif cl in {"real", "actual_kwh", "actual", "meter_reading", "real_kwh"}:
            rename[col] = "real_kwh"
        elif cl in {"predictedghi", "predicted_ghi", "ghi_predicted", "ghi_pred"}:
            rename[col] = "predicted_ghi"
        elif cl in {"real ghi", "real_ghi", "ghireal", "ghi_real", "ghi_wm2"}:
            rename[col] = "real_ghi"
        elif cl in {"expected_kwh_pvlib", "expected_kwh", "predicted_kwh"}:
            rename[col] = "predicted_kwh"
    return df.rename(columns=rename)


def load_forecast_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    df.columns = [str(c).strip() for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError(f"{path.name}: missing 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    numeric_cols = ["predicted_kwh", "real_kwh", "predicted_ghi", "real_ghi"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _style_time_axis(ax: plt.Axes, *, show_labels: bool) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    if show_labels:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    else:
        ax.tick_params(labelbottom=False)


def plot_power_and_ghi(df: pd.DataFrame, *, title: str, out_path: Path) -> None:
    has_power = "predicted_kwh" in df.columns or "real_kwh" in df.columns
    has_ghi = "predicted_ghi" in df.columns or "real_ghi" in df.columns
    if not has_power and not has_ghi:
        print(f"  Skip {out_path.stem}: no power or GHI columns found.")
        return

    n_rows = int(has_power) + int(has_ghi)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(14, 4.2 * n_rows),
        sharex=True,
        squeeze=False,
    )
    axes_list = axes.ravel().tolist()
    ax_idx = 0

    if has_power:
        ax = axes_list[ax_idx]
        ax_idx += 1
        if "predicted_kwh" in df.columns:
            ax.plot(
                df["timestamp"],
                df["predicted_kwh"],
                label="Predicted power (PVLib)",
                color=POWER_PRED,
                linewidth=1.8,
                zorder=3,
            )
        if "real_kwh" in df.columns:
            ax.plot(
                df["timestamp"],
                df["real_kwh"],
                label="Real power (meter)",
                color=POWER_REAL,
                linewidth=1.8,
                linestyle="--",
                zorder=3,
            )
        ax.set_ylabel("Energy (kWh / h)", fontsize=11)
        ax.set_title("Hourly PV generation", fontsize=12, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_ylim(bottom=0)

    if has_ghi:
        ax = axes_list[ax_idx]
        if "predicted_ghi" in df.columns:
            ax.plot(
                df["timestamp"],
                df["predicted_ghi"],
                label="Predicted GHI (Solcast)",
                color=GHI_PRED,
                linewidth=1.8,
                zorder=3,
            )
        if "real_ghi" in df.columns:
            ax.plot(
                df["timestamp"],
                df["real_ghi"],
                label="Real GHI (Solcast live)",
                color=GHI_REAL,
                linewidth=1.8,
                linestyle="--",
                zorder=3,
            )
        ax.set_ylabel("GHI (W/m²)", fontsize=11)
        ax.set_title("Global horizontal irradiance", fontsize=12, fontweight="bold", loc="left")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_ylim(bottom=0)

    for i, ax in enumerate(axes_list):
        _style_time_axis(ax, show_labels=(i == len(axes_list) - 1))

    axes_list[-1].set_xlabel("Date / time", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_paths = sorted(DATA_DIR.glob("forecast_7d_pvlib_*.csv"))
    if not csv_paths:
        print(f"No forecast_7d_pvlib_*.csv files in {DATA_DIR}")
        return

    for csv_path in csv_paths:
        site = _site_label(csv_path)
        print(f"Plotting {csv_path.name} ({site}) …")
        df = load_forecast_csv(csv_path)
        out_name = csv_path.stem.replace("forecast_7d_pvlib_", "forecast_7d_") + "_power_ghi.png"
        plot_power_and_ghi(
            df,
            title=f"7-day forecast — {site} — power & GHI",
            out_path=OUTPUT_DIR / out_name,
        )

    print("Done.")


if __name__ == "__main__":
    main()
