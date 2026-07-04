"""
Generates TradingView-style candlestick chart images for signals.
Clean dark theme with proper green/red candles, visible MA lines,
natural volume bars, and clean level labels on the right.
"""

import io
import logging
import traceback
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def generate_chart(df: pd.DataFrame, signal) -> io.BytesIO | None:
    try:
        plot_df = df.tail(80).copy().reset_index(drop=True)

        symbol_clean = signal.symbol.replace(":USDT", "").replace("/USDT", "")
        direction_label = "LONG" if signal.direction == "long" else "SHORT"
        title = (
            f"{symbol_clean} | 15m | {direction_label} | "
            f"{signal.confidence}% Confidence | {signal.leverage}x Leverage"
        )

        # TradingView-style colors
        bg = "#131722"
        up = "#26a69a"
        down = "#ef5350"
        grid = "#1e222d"
        text = "#d1d4dc"
        border = "#2a2e39"

        fig = plt.figure(figsize=(16, 9), facecolor=bg)
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[5, 1],
            hspace=0,
            figure=fig
        )

        ax = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax)

        for a in [ax, ax_vol]:
            a.set_facecolor(bg)
            a.grid(True, color=grid, linewidth=0.4, alpha=1.0, zorder=0)
            for spine in a.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(0.5)

        # Draw candles
        candle_width = 0.6
        for i, row in plot_df.iterrows():
            o = row["open"]
            h = row["high"]
            l = row["low"]
            c = row["close"]
            color = up if c >= o else down

            # Wick
            ax.plot(
                [i, i], [l, h],
                color=color,
                linewidth=0.7,
                zorder=2,
                solid_capstyle="round"
            )

            # Body
            body_y = min(o, c)
            body_h = max(abs(c - o), (h - l) * 0.008)
            rect = Rectangle(
                (i - candle_width / 2, body_y),
                candle_width,
                body_h,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                zorder=3
            )
            ax.add_patch(rect)

        # Volume bars
        for i, row in plot_df.iterrows():
            color = up if row["close"] >= row["open"] else down
            ax_vol.bar(
                i,
                row["volume"],
                color=color,
                alpha=0.6,
                width=candle_width,
                zorder=2
            )

        # MA lines
        x = list(range(len(plot_df)))
        ma_lines = []
        if "ma_fast" in plot_df.columns and not plot_df["ma_fast"].isna().all():
            ax.plot(x, plot_df["ma_fast"].values, color="#FFD700", linewidth=1.0, zorder=4, alpha=0.9)
            ma_lines.append(Line2D([0], [0], color="#FFD700", linewidth=1.0, label="MA7"))
        if "ma_mid" in plot_df.columns and not plot_df["ma_mid"].isna().all():
            ax.plot(x, plot_df["ma_mid"].values, color="#2196F3", linewidth=1.0, zorder=4, alpha=0.9)
            ma_lines.append(Line2D([0], [0], color="#2196F3", linewidth=1.0, label="MA25"))
        if "ma_slow" in plot_df.columns and not plot_df["ma_slow"].isna().all():
            ax.plot(x, plot_df["ma_slow"].values, color="#E040FB", linewidth=1.0, zorder=4, alpha=0.9)
            ma_lines.append(Line2D([0], [0], color="#E040FB", linewidth=1.0, label="MA50"))

        # Level lines — sorted to avoid label overlap
        levels = sorted([
            (signal.stop_loss, "#F44336", "SL", "-"),
            (signal.entry, "#B0BEC5", "Entry", "--"),
            (signal.take_profit1, "#66BB6A", "TP1", "--"),
            (signal.take_profit2, "#43A047", "TP2", "--"),
            (signal.take_profit3, "#00E676", "TP3", "-"),
        ], key=lambda x: x[0])

        price_min = plot_df["low"].min()
        price_max = plot_df["high"].max()
        price_range = price_max - price_min
        label_offset = price_range * 0.008

        used_y = []
        for price, color, label, ls in levels:
            ax.axhline(
                y=price,
                color=color,
                linewidth=1.2,
                linestyle=ls,
                zorder=5,
                alpha=0.9
            )
            # Adjust label y to avoid overlap
            label_y = price
            for used in used_y:
                if abs(label_y - used) < label_offset * 2:
                    label_y = used + label_offset * 2.5
            used_y.append(label_y)

            ax.annotate(
                f"{label}: {price}",
                xy=(len(plot_df) - 0.5, label_y),
                xycoords="data",
                color=color,
                fontsize=7.5,
                fontweight="bold",
                va="center",
                ha="left",
                zorder=6,
                annotation_clip=False,
            )

        # Set price axis range with padding
        padding = price_range * 0.15
        ax.set_ylim(price_min - padding, price_max + padding)
        ax.set_xlim(-1, len(plot_df) + 8)
        ax_vol.set_xlim(-1, len(plot_df) + 8)

        # Title
        ax.set_title(title, color=text, fontsize=10, pad=8, fontweight="bold", loc="left")

        # Tick styling
        ax.tick_params(colors=text, labelsize=7.5, which="both", length=3)
        ax_vol.tick_params(colors=text, labelsize=7, which="both", length=2)
        ax.tick_params(axis="x", labelbottom=False)
        ax.yaxis.set_tick_params(labelcolor=text)
        ax_vol.yaxis.set_tick_params(labelcolor=text)

        # X-axis timestamps
        timestamps = plot_df["timestamp"].dt.strftime("%m/%d %H:%M")
        step = max(1, len(plot_df) // 10)
        tick_pos = list(range(0, len(plot_df), step))
        ax_vol.set_xticks(tick_pos)
        ax_vol.set_xticklabels(
            [timestamps.iloc[i] for i in tick_pos],
            rotation=0,
            fontsize=7,
            color=text
        )

        # MA legend
        if ma_lines:
            legend = ax.legend(
                handles=ma_lines,
                loc="upper left",
                facecolor=bg,
                labelcolor=text,
                fontsize=7.5,
                framealpha=0.8,
                edgecolor=border,
            )

        # Volume label
        ax_vol.set_ylabel("Vol", color=text, fontsize=7, labelpad=2)

        plt.subplots_adjust(left=0.06, right=0.82, top=0.95, bottom=0.06)

        buf = io.BytesIO()
        fig.savefig(buf, dpi=120, bbox_inches="tight", facecolor=bg)
        plt.close("all")
        buf.seek(0)
        return buf

    except Exception as e:
        logger.error(f"Chart generation failed for {signal.symbol}: {e}")
        logger.error(traceback.format_exc())
        return None
