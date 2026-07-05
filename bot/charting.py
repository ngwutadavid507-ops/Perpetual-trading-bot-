"""
Phoenix signal chart — Bybit-style dark green theme.
SL/Entry/TP labels on left, Phoenix watermark center,
MA lines top left, clean volume panel bottom.
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
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)


def generate_chart(df: pd.DataFrame, signal) -> io.BytesIO | None:
    try:
        plot_df = df.tail(80).copy().reset_index(drop=True)

        symbol_clean = signal.symbol.replace(":USDT", "").replace("/USDT", "")
        direction_label = "LONG" if signal.direction == "long" else "SHORT"

        # Bybit-style dark green palette
        bg = "#0a1628"
        panel_bg = "#0a1628"
        up = "#26a69a"
        down = "#ef5350"
        grid = "#0f1f35"
        text = "#b2b5be"
        border = "#1e2d45"

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
            a.grid(True, color=grid, linewidth=0.4, alpha=1.0, zorder=0)
            for spine in a.spines.values():
                spine.set_edgecolor(border)
                spine.set_linewidth(0.4)

        # Draw candles
        cw = 0.6
        for i, row in plot_df.iterrows():
            o, h, l, c = row["open"], row["high"], row["low"], row["close"]
            color = up if c >= o else down

            ax.plot([i, i], [l, h], color=color, linewidth=0.7, zorder=2)

            body_y = min(o, c)
            body_h = max(abs(c - o), (h - l) * 0.012)
            rect = Rectangle(
                (i - cw / 2, body_y), cw, body_h,
                facecolor=color, edgecolor=color,
                linewidth=0, zorder=3
            )
            ax.add_patch(rect)

        # Volume bars
        for i, row in plot_df.iterrows():
            color = up if row["close"] >= row["open"] else down
            ax_vol.bar(i, row["volume"], color=color, alpha=0.5, width=cw, zorder=2)

        # MA lines with top-left labels like Bybit
        x = list(range(len(plot_df)))
        ma_configs = [
            ("ma_fast", "#f6c027", "MA7"),
            ("ma_mid", "#2196f3", "MA14"),
            ("ma_slow", "#e040fb", "MA28"),
        ]
        ma_label_parts = []
        for col, color, label in ma_configs:
            if col in plot_df.columns:
                vals = plot_df[col].values.astype(float)
                if not np.all(np.isnan(vals)):
                    ax.plot(x, vals, color=color, linewidth=1.1, zorder=4, alpha=0.9)
                    last_val = vals[~np.isnan(vals)][-1] if not np.all(np.isnan(vals)) else 0
                    ma_label_parts.append((label, f"{last_val:.4f}", color))

        # MA labels top left — like Bybit style
        x_cursor = 0.01
        for label, val, color in ma_label_parts:
            ax.text(
                x_cursor, 1.02, f"{label}: ",
                transform=ax.transAxes,
                color=text, fontsize=8, va="bottom", ha="left",
                fontweight="bold"
            )
            x_cursor += 0.045
            ax.text(
                x_cursor, 1.02, val,
                transform=ax.transAxes,
                color=color, fontsize=8, va="bottom", ha="left",
                fontweight="bold"
            )
            x_cursor += 0.07

        # Phoenix watermark — center, subtle
        ax.text(
            0.5, 0.5, "PHOENIX",
            transform=ax.transAxes,
            color="#1a3a5c",
            fontsize=38,
            fontweight="bold",
            va="center", ha="center",
            alpha=0.35,
            zorder=1,
            fontfamily="monospace",
        )

        # Y-axis zoom — SL to TP1 range + candle range
        if signal.direction == "long":
            y_min = signal.stop_loss
            y_max = signal.take_profit1
        else:
            y_min = signal.take_profit1
            y_max = signal.stop_loss

        candle_min = plot_df["low"].tail(20).min()
        candle_max = plot_df["high"].tail(20).max()
        y_min = min(y_min, candle_min)
        y_max = max(y_max, candle_max)
        padding = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_xlim(-1, len(plot_df) + 2)
        ax_vol.set_xlim(-1, len(plot_df) + 2)

        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        min_gap = y_range * 0.022

        # Level lines with LEFT-side labels like Bybit
        levels = [
            (signal.stop_loss, "#ef5350", f"SL ~{signal.stop_loss}", "-", 1.5),
            (signal.entry, "#f6c027", f"ENTRY ({direction_label}) ~{signal.entry}", "--", 1.3),
            (signal.take_profit1, "#26a69a", f"TP1 ~{signal.take_profit1}", "--", 1.2),
            (signal.take_profit2, "#00c853", f"TP2 ~{signal.take_profit2}", "--", 1.2),
            (signal.take_profit3, "#69ff8e", f"TP3 ~{signal.take_profit3}", "-", 1.5),
        ]

        used_ys = []
        for price, color, label, ls, lw in sorted(levels, key=lambda v: v[0]):
            if price < ylim[0] or price > ylim[1]:
                continue

            ax.axhline(y=price, color=color, linewidth=lw,
                      linestyle=ls, zorder=5, alpha=0.9)

            # Avoid label overlap
            label_y = price
            for uy in used_ys:
                if abs(label_y - uy) < min_gap:
                    label_y = uy + min_gap
            used_ys.append(label_y)

            # Left-side label like Bybit
            ax.annotate(
                label,
                xy=(0, label_y),
                xycoords=("axes fraction", "data"),
                color=color,
                fontsize=8,
                fontweight="bold",
                va="center",
                ha="left",
                zorder=6,
                annotation_clip=False,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=bg,
                    edgecolor=color,
                    alpha=0.85,
                    linewidth=0.6,
                ),
            )

        # Symbol and direction top left
        ax.text(
            0.01, 1.08,
            f"{symbol_clean}/USDT",
            transform=ax.transAxes,
            color="#ffffff",
            fontsize=12,
            fontweight="bold",
            va="bottom", ha="left",
        )
        dir_color = "#26a69a" if signal.direction == "long" else "#ef5350"
        ax.text(
            0.12, 1.08,
            f"  {signal.confidence}% | {signal.leverage}x | {direction_label}",
            transform=ax.transAxes,
            color=dir_color,
            fontsize=9,
            fontweight="bold",
            va="bottom", ha="left",
        )

        # Timeframe label
        ax.text(
            0.01, 1.045,
            "15m",
            transform=ax.transAxes,
            color=text,
            fontsize=8,
            va="bottom", ha="left",
        )

        # Tick styling
        ax.tick_params(colors=text, labelsize=7.5, length=3, width=0.4)
        ax_vol.tick_params(colors=text, labelsize=7, length=2, width=0.4)
        ax.tick_params(axis="x", labelbottom=False)

        # X-axis timestamps
        timestamps = plot_df["timestamp"].dt.strftime("%H:%M")
        step = max(1, len(plot_df) // 10)
        tick_pos = list(range(0, len(plot_df), step))
        ax_vol.set_xticks(tick_pos)
        ax_vol.set_xticklabels(
            [timestamps.iloc[i] for i in tick_pos],
            rotation=0, fontsize=7.5, color=text
        )

        # Volume formatting
        ax_vol.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda val, _: f"{val/1e6:.1f}M" if val >= 1e6 else f"{val/1e3:.0f}K"
            )
        )
        ax_vol.set_ylabel("VOLUME", color=text, fontsize=6.5, labelpad=2)

        # Right side price axis
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", colors=text, labelsize=7.5)

        plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.07)

        buf = io.BytesIO()
        fig.savefig(buf, dpi=120, bbox_inches="tight", facecolor=bg)
        plt.close("all")
        buf.seek(0)
        return buf

    except Exception as e:
        logger.error(f"Chart generation failed for {signal.symbol}: {e}")
        logger.error(traceback.format_exc())
        return None
