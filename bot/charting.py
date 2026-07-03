"""
Generates candlestick chart images for signals.
Charts include:
- Last 100 candles of price action
- MA lines (fast, mid, slow)
- Entry, TP1, TP2, SL horizontal lines
- RSI panel below
- Volume bars
"""

import io
import logging
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


def generate_chart(df: pd.DataFrame, signal) -> io.BytesIO | None:
    """
    Generates a chart image for the given signal and returns
    it as a BytesIO buffer ready to send to Telegram.
    """
    try:
        # Use last 100 candles
        chart_df = df.tail(100).copy()
        chart_df = chart_df.set_index("timestamp")
        chart_df.index = pd.DatetimeIndex(chart_df.index)

        # Rename columns to match mplfinance requirements
        chart_df = chart_df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        # Build additional plots — MA lines
        add_plots = []

        if "ma_fast" in df.columns:
            ma_fast = df["ma_fast"].tail(100).values
            add_plots.append(
                mpf.make_addplot(
                    ma_fast,
                    color="yellow",
                    width=1.2,
                    label="MA7"
                )
            )

        if "ma_mid" in df.columns:
            ma_mid = df["ma_mid"].tail(100).values
            add_plots.append(
                mpf.make_addplot(
                    ma_mid,
                    color="cyan",
                    width=1.2,
                    label="MA25"
                )
            )

        if "ma_slow" in df.columns:
            ma_slow = df["ma_slow"].tail(100).values
            add_plots.append(
                mpf.make_addplot(
                    ma_slow,
                    color="magenta",
                    width=1.2,
                    label="MA50"
                )
            )

        if "rsi" in df.columns:
            rsi = df["rsi"].tail(100).values
            add_plots.append(
                mpf.make_addplot(
                    rsi,
                    panel=2,
                    color="purple",
                    width=1.2,
                    ylabel="RSI",
                    y_on_right=False,
                )
            )

        # Horizontal level lines
        entry_line = [signal.entry] * 100
        sl_line = [signal.stop_loss] * 100
        tp1_line = [signal.take_profit1] * 100
        tp2_line = [signal.take_profit2] * 100

        add_plots += [
            mpf.make_addplot(entry_line, color="white", width=1.0, linestyle="dashed"),
            mpf.make_addplot(sl_line, color="red", width=1.2, linestyle="solid"),
            mpf.make_addplot(tp1_line, color="lightgreen", width=1.2, linestyle="dashed"),
            mpf.make_addplot(tp2_line, color="lime", width=1.5, linestyle="solid"),
        ]

        # Chart style
        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            gridstyle=":",
            gridcolor="#333333",
            facecolor="#0d1117",
            edgecolor="#30363d",
            figcolor="#0d1117",
            y_on_right=True,
            rc={
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "text.color": "white",
            }
        )

        direction_label = "LONG 🟢" if signal.direction == "long" else "SHORT 🔴"
        title = f"{signal.symbol} | {direction_label} | Conf: {signal.confidence}% | {signal.leverage}x"

        # Render chart to buffer
        buf = io.BytesIO()
        mpf.plot(
            chart_df,
            type="candle",
            style=style,
            title=title,
            volume=True,
            addplot=add_plots,
            panel_ratios=(4, 1, 1),
            figsize=(12, 8),
            savefig=dict(fname=buf, dpi=150, bbox_inches="tight"),
        )
        buf.seek(0)

        # Add legend manually
        fig = plt.figure(figsize=(12, 8))
        legend_elements = [
            mpatches.Patch(color="white", label=f"Entry: {signal.entry}"),
            mpatches.Patch(color="lightgreen", label=f"TP1: {signal.take_profit1}"),
            mpatches.Patch(color="lime", label=f"TP2: {signal.take_profit2}"),
            mpatches.Patch(color="red", label=f"SL: {signal.stop_loss}"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=4,
                   facecolor="#0d1117", labelcolor="white", fontsize=9)
        plt.close(fig)

        return buf

    except Exception as e:
        logger.error(f"Chart generation failed for {signal.symbol}: {e}")
        return None
