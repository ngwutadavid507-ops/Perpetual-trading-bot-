"""
Generates candlestick chart images for signals.
Clean dark theme with proper green/red candles,
integrated volume, and clear TP/SL level lines.
"""

import io
import logging
import traceback
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import numpy as np

logger = logging.getLogger(__name__)


def generate_chart(df: pd.DataFrame, signal) -> io.BytesIO | None:
    try:
        # Use last 80 candles
        plot_df = df.tail(80).copy().reset_index(drop=True)

        symbol_clean = signal.symbol.replace(":USDT", "").replace("/USDT", "")
        direction_label = "LONG" if signal.direction == "long" else "SHORT"
        title = f"{symbol_clean} | {direction_label} | {signal.confidence}% Confidence | {signal.leverage}x Leverage"

        # Colors
        bg_color = "#131722"
        up_color = "#26a69a"
        down_color = "#ef5350"
        wick_up = "#26a69a"
        wick_down = "#ef5350"
        grid_color = "#1e222d"
        text_color = "#d1d4dc"
        vol_up = "#26a69a"
        vol_down = "#ef5350"

        fig = plt.figure(figsize=(14, 8), facecolor=bg_color)
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[4, 1],
            hspace=0.02,
            figure=fig
        )

        ax = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax)

        ax.set_facecolor(bg_color)
        ax_vol.set_facecolor(bg_color)

        # Grid
        ax.grid(color=grid_color, linewidth=0.5, alpha=0.8)
        ax_vol.grid(color=grid_color, linewidth=0.5, alpha=0.8)

        # Draw candles manually
        for i, row in plot_df.iterrows():
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = up_color if c >= o else down_color
            wick_color = wick_up if c >= o else wick_down

            # Wick
            ax.plot([i, i], [l, h], color=wick_color, linewidth=0.8, zorder=2)

            # Body
            body_bottom = min(o, c)
            body_height = max(abs(c - o), (h - l) * 0.01)
            rect = Rectangle(
                (i - 0.4, body_bottom),
                0.8,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                zorder=3
            )
            ax.add_patch(rect)

        # Volume bars
        for i, row in plot_df.iterrows():
            color = vol_up if row["close"] >= row["open"] else vol_down
            ax_vol.bar(i, row["volume"], color=color, alpha=0.7, width=0.8)

        # MA lines
        x = list(range(len(plot_df)))
        if "ma_fast" in plot_df.columns:
            ax.plot(x, plot_df["ma_fast"].values, color="#FFD700", linewidth=1.2, label="MA7", zorder=4)
        if "ma_mid" in plot_df.columns:
            ax.plot(x, plot_df["ma_mid"].values, color="#00BFFF", linewidth=1.2, label="MA25", zorder=4)
        if "ma_slow" in plot_df.columns:
            ax.plot(x, plot_df["ma_slow"].values, color="#FF69B4", linewidth=1.2, label="MA50", zorder=4)

        # Horizontal level lines
        n = len(plot_df)
        ax.axhline(y=signal.entry, color="#FFFFFF", linewidth=1.0, linestyle="--", zorder=5, label=f"Entry {signal.entry}")
        ax.axhline(y=signal.stop_loss, color="#FF4444", linewidth=1.5, linestyle="-", zorder=5, label=f"SL {signal.stop_loss}")
        ax.axhline(y=signal.take_profit1, color="#90EE90", linewidth=1.2, linestyle="--", zorder=5, label=f"TP1 {signal.take_profit1}")
        ax.axhline(y=signal.take_profit2, color="#00CC00", linewidth=1.2, linestyle="--", zorder=5, label=f"TP2 {signal.take_profit2}")
        ax.axhline(y=signal.take_profit3, color="#00FF00", linewidth=1.5, linestyle="-", zorder=5, label=f"TP3 {signal.take_profit3}")

        # Price labels on right side
        price_levels = [
            (signal.stop_loss, "#FF4444", f"SL {signal.stop_loss}"),
            (signal.entry, "#FFFFFF", f"Entry {signal.entry}"),
            (signal.take_profit1, "#90EE90", f"TP1 {signal.take_profit1}"),
            (signal.take_profit2, "#00CC00", f"TP2 {signal.take_profit2}"),
            (signal.take_profit3, "#00FF00", f"TP3 {signal.take_profit3}"),
        ]

        ax_right = ax.twinx()
        ax_right.set_facecolor(bg_color)
        ax_right.set_ylim(ax.get_ylim())

        for price, color, label in price_levels:
            ax_right.axhline(y=price, color=color, linewidth=0, alpha=0)
            ax_right.annotate(
                f" {label}",
                xy=(1, price),
                xycoords=("axes fraction", "data"),
                color=color,
                fontsize=7.5,
                fontweight="bold",
                va="center",
            )

        # Styling
        ax.set_title(title, color=text_color, fontsize=11, pad=10, fontweight="bold")
        ax.tick_params(colors=text_color, labelsize=8)
        ax_vol.tick_params(colors=text_color, labelsize=7)
        ax_right.tick_params(colors=text_color, labelsize=0)

        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        for spine in ax_vol.spines.values():
            spine.set_edgecolor(grid_color)
        for spine in ax_right.spines.values():
            spine.set_edgecolor(grid_color)

        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # X-axis timestamps
        timestamps = plot_df["timestamp"].dt.strftime("%m/%d %H:%M")
        tick_positions = list(range(0, len(plot_df), max(1, len(plot_df) // 8)))
        ax_vol.set_xticks(tick_positions)
        ax_vol.set_xticklabels(
            [timestamps.iloc[i] for i in tick_positions],
            rotation=0,
            fontsize=7,
            color=text_color
        )

        ax.set_xlim(-1, len(plot_df))
        ax_vol.set_xlim(-1, len(plot_df))

        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, dpi=120, bbox_inches="tight", facecolor=bg_color)
        plt.close("all")
        buf.seek(0)
        return buf

    except Exception as e:
        logger.error(f"Chart generation failed for {signal.symbol}: {e}")
        logger.error(traceback.format_exc())
        return None
