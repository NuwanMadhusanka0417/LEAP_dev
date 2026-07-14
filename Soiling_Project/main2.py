"""
Two soiling figures:

  A. plot_loss_factor()   -> "soiling loss factor" accumulation sawtooth (Image 2 style),
                             drawn directly from HSU output (loss = 1 - soiling_ratio).

  B. plot_srr_detection() -> SRR-style figure (Image 1 LEFT): scatter of MEASURED
                             performance index + centered moving-median line +
                             auto-detected cleaning events (Deceglie method).
                             Requires a measured performance-index series.

The orange Monte-Carlo soiling profiles (Image 1 RIGHT) are the full SRR simulation
-> use RdTools (see note at bottom), not reimplemented here.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ======================================================================
# A. SOILING LOSS FACTOR  (Image 2)  -- from HSU output
# ======================================================================
def plot_loss_factor(csv_path="data/hsu_soiling_output.csv",
                     freq="1D", out="soiling_loss_factor.png"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    loss = (1.0 - df["soiling_ratio"]).clip(lower=0.0).resample(freq).mean()

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(loss.index, loss.values, color="#4f5bd5", lw=1.3)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel("soiling loss factor (fraction)")
    ax.set_title("mean soiling loss factor in fraction")
    ax.set_ylim(0, loss.max() * 1.1 if loss.max() > 0 else 0.05)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}  (peak loss {loss.max():.3f})")
    plt.show()


# ======================================================================
# B. SRR-STYLE CLEANING DETECTION  (Image 1, left)  -- from MEASURED PI
# ======================================================================
def detect_cleanings(pi_daily, window=14):
    """Deceglie method: positive shifts in the centered moving median that are
    outliers (> Q3 + 1.5*IQR of |differences|) are cleaning events."""
    mm = pi_daily.rolling(window, center=True, min_periods=max(2, window // 2)).median()
    diffs = mm.diff()
    mag = diffs.abs().dropna()
    q1, q3 = mag.quantile(0.25), mag.quantile(0.75)
    thresh = q3 + 1.5 * (q3 - q1)
    cleanings = mm.index[(diffs > thresh)]
    return mm, cleanings, thresh


def plot_srr_detection(pi_daily, window=14, out="srr_detection.png"):
    """pi_daily: pandas Series of DAILY measured performance index
       (actual_kwh / expected_kwh), indexed by date."""
    mm, cleanings, thresh = detect_cleanings(pi_daily, window)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.scatter(pi_daily.index, pi_daily.values, s=12, alpha=0.45,
               color="#3b7dd8", label="Performance index (measured)")
    ax.plot(mm.index, mm.values, color="black", lw=1.6,
            label=f"{window}-day moving median")
    for i, d in enumerate(cleanings):
        ax.axvline(d, ls="--", color="0.5", lw=1.0,
                   label="Detected cleaning event" if i == 0 else None)

    ax.set_ylabel("PM / performance index")
    ax.set_xlabel("Date")
    ax.set_title("SRR cleaning-event detection (measured yield)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}  ({len(cleanings)} cleaning events, threshold {thresh:.4f})")
    plt.show()


if __name__ == "__main__":
    # Image 2 (works on your HSU output directly):
    plot_loss_factor()

    # Image 1 left (needs MEASURED yield). Example using a measured PI file:
    #   m = pd.read_csv("hourly_library_master.csv", index_col=0, parse_dates=True)
    #   pi = (m["actual_kwh"] / m["expected_kwh"]).resample("1D").mean().dropna()
    #   plot_srr_detection(pi)