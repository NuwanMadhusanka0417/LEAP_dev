"""
Plot the soiling & cleaning process from HSU output.

Reads hsu_soiling_output.csv (produced by hsu_soiling_bundoora.py) and draws:
  - daily soiling ratio (the sawtooth: decline between rains, jump up on cleaning)
  - daily rainfall as bars on a secondary axis
  - cleaning events (effective rain) marked with vertical lines

Usage:  python plot_soiling_cleaning.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------- CONFIG ----------------
CSV_PATH     = "data/hsu_soiling_output.csv"
CLEAN_THRESH = 0.5     # mm/day that counts as an effective (cleaning) rainfall
OUT_PNG      = "soiling_cleaning_process.png"


def load_daily(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    daily = pd.DataFrame({
        "soiling_ratio": df["soiling_ratio"].resample("1D").mean(),
        "rainfall":      df["rainfall"].resample("1D").sum(),
    }).dropna()
    return daily


def plot(daily):
    # effective rain = cleaning event
    cleaning = daily.index[daily["rainfall"] >= CLEAN_THRESH]

    fig, ax1 = plt.subplots(figsize=(13, 5.5))

    # --- soiling ratio (left axis) ---
    ax1.plot(daily.index, daily["soiling_ratio"],
             color="#c0392b", lw=1.6, label="Soiling ratio", zorder=3)
    ax1.set_ylabel("Soiling ratio  (1.0 = clean)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    lo = daily["soiling_ratio"].min()
    ax1.set_ylim(max(0.0, lo - 0.03), 1.005)
    ax1.set_xlabel("Date")

    # --- cleaning-event markers ---
    for i, d in enumerate(cleaning):
        ax1.axvline(d, color="#2980b9", alpha=0.18, lw=1.0, zorder=1,
                    label="Cleaning event (effective rain)" if i == 0 else None)

    # --- rainfall bars (right axis) ---
    ax2 = ax1.twinx()
    ax2.bar(daily.index, daily["rainfall"], width=1.0,
            color="#2980b9", alpha=0.55, label="Daily rainfall", zorder=2)
    ax2.set_ylabel("Daily rainfall (mm)", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim(0, max(daily["rainfall"].max() * 1.1, 1))
    ax2.invert_yaxis()                      # rain hangs from the top, soiling read below
    ax2.set_ylim(daily["rainfall"].max() * 3.0, 0)  # compress bars to lower third

    # --- legend (combined) ---
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", framealpha=0.9, fontsize=9)

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    ax1.set_title("PV Soiling & Cleaning Process (HSU model)")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")
    print(f"Cleaning events detected (rain >= {CLEAN_THRESH} mm): {len(cleaning)}")
    plt.show()


if __name__ == "__main__":
    plot(load_daily(CSV_PATH))