"""
Hyperliquid-style candlestick chart.
Wide rectangular format, tight price zoom, visible MAs.
"""

import io
import logging
import traceback
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


def generate_chart(df: pd.DataFrame, signal) -> io.BytesIO | None:
    try:
        plot_df = df.tail(80).copy().reset_index(drop=True)

        symbol_clean = signal.symbol.replace(":USDT", "").replace("/USDT", "")
        direction_label = "LONG" if signal.direction == "long" else "SHORT"
        dir_color = "#26a69a" if signal.direction == "long" else "#ef5350"

        # Hyperliquid color palette
        bg = "#0d0f14"
        panel_bg = "#0d0f14"
        up = "#00c076"
        down = "#ff3b69"
        grid = "#1a1d27"
        text = "#9ba3ae"
        border = "#1e2130"
        vol_up = "#00c07650"
        vol_down = "#ff3b6950"

        # Wide rectangular figure — 16:7
        fig = plt.figure(figsize=(18, 7), facecolor=bg)
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[5, 1],
            hspace=0,
            figure=fig
        )

        ax = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax)

        for a in [ax, ax_vol]:
            a.set_facecolor(panel_bg)
            a.grid(True, color=grid, linewidth=0.3, alpha=1.0, zorder=0)
            for spine in a.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(0.4)

        # Draw candles — wider bodies
        cw = 0.65
        for i, row in plot_df.iterrows():
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = up if c >= o else down

            # Wick
            ax.plot([i, i], [l, h], color=color, linewidth=0.6, zorder=2)

            # Body — minimum visible height
            body_y = min(o, c)
            body_h = max(abs(c - o), (h - l) * 0.015)
            rect = Rectangle(
                (i - cw / 2, body_y), cw, body_h,
                facecolor=color, edgecolor=color,
                linewidth=0, zorder=3
            )
            ax.add_patch(rect)

        # Volume
        for i, row in plot_df.iterrows():
            color = vol_up if row["close"] >= row["open"] else vol_down
            ax_vol.bar(i, row["volume"], color=color, width=cw, zorder=2)

        # MA lines — only draw if not all NaN
        x = list(range(len(plot_df)))
        ma_lines = []
        ma_configs = [
            ("ma_fast", "#f0b90b", "MA7"),
            ("ma_mid", "#2962ff", "MA25"),
            ("ma_slow", "#e040fb", "MA50"),
        ]
        for col, color, label in ma_configs:
            if col in plot_df.columns:
                vals = plot_df[col].values
                if not np.all(np.isnan(vals.astype(float))):
                    ax.plot(x, vals, color=color, linewidth=1.1, zorder=4, alpha=0.85)
                    ma_lines.append(Line2D([0], [0], color=color, linewidth=1.1, label=label))

        # Y-axis zoom — show SL to TP1 range with padding
        # This keeps chart tight and readable
        if signal.direction == "long":
            y_min = signal.stop_loss
            y_max = signal.take_profit1
        else:
            y_min = signal.take_profit1
            y_max = signal.stop_loss

        # Also include current candle range
        candle_min = plot_df["low"].tail(20).min()
        candle_max = plot_df["high"].tail(20).max()
        y_min = min(y_min, candle_min)
        y_max = max(y_max, candle_max)

        padding = (y_max - y_min) * 0.12
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_xlim(-1, len(plot_df) + 10)
        ax_vol.set_xlim(-1, len(plot_df) + 10)

        # Level lines with right-side labels
        levels = [
            (signal.stop_loss, "#ff3b69", "SL", "-", 1.5),
            (signal.entry, "#9ba3ae", "Entry", "--", 1.0),
            (signal.take_profit1, "#00c076", "TP1", "--", 1.2),
            (signal.take_profit2, "#00e676", "TP2", "--", 1.2),
            (signal.take_profit3, "#69ff8e", "TP3", "-", 1.5),
        ]

        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        min_gap = y_range * 0.025
        used_ys = []

        for price, color, label, ls, lw in sorted(levels, key=lambda v: v[0]):
            if price < ylim[0] or price > ylim[1]:
                continue

            ax.axhline(y=price, color=color, linewidth=lw,
                      linestyle=ls, zorder=5, alpha=0.85)

            # Avoid label overlap
            label_y = price
            for uy in used_ys:
                if abs(label_y - uy) < min_gap:
                    label_y = uy + min_gap
            used_ys.append(label_y)

            ax.annotate(
                f"{label}: {price}",
                xy=(len(plot_df) + 0.2, label_y),
                xycoords=("data", "data"),
                color=color,
                fontsize=7.8,
                fontweight="bold",
                va="center",
                ha="left",
                zorder=6,
                annotation_clip=False,
                fontfamily="monospace",
            )

        # Header bar — symbol and direction
        ax.text(
            0.01, 0.97,
            f"{symbol_clean} · 15m",
            transform=ax.transAxes,
            color="#ffffff",
            fontsize=11,
            fontweight="bold",
            va="top", ha="left",
        )
        ax.text(
            0.01, 0.88,
            f"● {direction_label}",
            transform=ax.transAxes,
            color=dir_color,
            fontsize=9,
            fontweight="bold",
            va="top", ha="left",
        )
        ax.text(
            0.08, 0.88,
            f"{signal.confidence}% Confidence · {signal.leverage}x",
            transform=ax.transAxes,
            color=text,
            fontsize=8.5,
            va="top", ha="left",
        )

        # MA legend top right
        if ma_lines:
            ax.legend(
                handles=ma_lines,
                loc="upper right",
                facecolor=bg,
                labelcolor=text,
                fontsize=7.5,
                framealpha=0.7,
                edgecolor=border,
                borderpad=0.4,
            )

        # Tick styling
        ax.tick_params(colors=text, labelsize=7.5, length=3, width=0.4)
        ax_vol.tick_params(colors=text, labelsize=7, length=2, width=0.4)
        ax.tick_params(axis="x", labelbottom=False)

        # X-axis timestamps
        timestamps = plot_df["timestamp"].dt.strftime("%m/%d %H:%M")
        step = max(1, len(plot_df) // 10)
        tick_pos = list(range(0, len(plot_df), step))
        ax_vol.set_xticks(tick_pos)
        ax_vol.set_xticklabels(
            [timestamps.iloc[i] for i in tick_pos],
            rotation=0, fontsize=7, color=text
        )

        # Volume y-label
        ax_vol.set_ylabel("Vol", color=text, fontsize=7, labelpad=2)
        ax_vol.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
            )
        )

        plt.subplots_adjust(left=0.05, right=0.83, top=0.97, bottom=0.07)

        buf = io.BytesIO()
        fig.savefig(buf, dpi=110, bbox_inches="tight", facecolor=bg)
        plt.close("all")
        buf.seek(0)
        return buf

    except Exception as e:
        logger.error(f"Chart generation failed for {signal.symbol}: {e}")
        logger.error(traceback.format_exc())
        return None
