"""
Generates candlestick chart images for signals.
Uses matplotlib non-interactive backend for server compatibility.
"""

import io
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required on servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf

logger = logging.getLogger(__name__)


def generate_chart(df: pd.DataFrame, signal) -> io.BytesIO | None:
    try:
        chart_df = df.tail(100).copy()
        chart_df = chart_df.set_index("timestamp")
        chart_df.index = pd.DatetimeIndex(chart_df.index)
        chart_df = chart_df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })

        add_plots = []

        if "ma_fast" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["ma_fast"].tail(100).values,
                color="#FFD700", width=1.2
            ))

        if "ma_mid" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["ma_mid"].tail(100).values,
                color="#00BFFF", width=1.2
            ))

        if "ma_slow" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["ma_slow"].tail(100).values,
                color="#FF69B4", width=1.2
            ))

        if "rsi" in df.columns:
            add_plots.append(mpf.make_addplot(
                df["rsi"].tail(100).values,
                panel=2, color="#9B59B6",
                width=1.2, ylabel="RSI",
                y_on_right=False
            ))

        n = len(chart_df)
        add_plots += [
            mpf.make_addplot([signal.entry] * n, color="#FFFFFF", width=1.0, linestyle="dashed"),
            mpf.make_addplot([signal.stop_loss] * n, color="#FF4444", width=1.5, linestyle="solid"),
            mpf.make_addplot([signal.take_profit1] * n, color="#90EE90", width=1.2, linestyle="dashed"),
            mpf.make_addplot([signal.take_profit2] * n, color="#00FF00", width=1.5, linestyle="solid"),
        ]

        style = mpf.make_mpf_style(
            base_mpf_style="nightclouds",
            facecolor="#0d1117",
            edgecolor="#30363d",
            figcolor="#0d1117",
            gridstyle=":",
            gridcolor="#222222",
            y_on_right=True,
            rc={
                "axes.labelcolor": "#AAAAAA",
                "xtick.color": "#AAAAAA",
                "ytick.color": "#AAAAAA",
                "text.color": "white",
                "figure.facecolor": "#0d1117",
            }
        )

        symbol_clean = signal.symbol.replace(":USDT", "").replace("/USDT", "")
        direction_label = "LONG 🟢" if signal.direction == "long" else "SHORT 🔴"
        title = f"{symbol_clean}  |  {direction_label}  |  {signal.confidence}% Confidence  |  {signal.leverage}x"

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
        plt.close("all")
        buf.seek(0)

        # Add legend
        fig, ax = plt.subplots(figsize=(12, 0.4))
        fig.patch.set_facecolor("#0d1117")
        ax.axis("off")
        legend_elements = [
            mpatches.Patch(color="white", label=f"Entry {signal.entry}"),
            mpatches.Patch(color="#90EE90", label=f"TP1 {signal.take_profit1}"),
            mpatches.Patch(color="#00FF00", label=f"TP2 {signal.take_profit2}"),
            mpatches.Patch(color="#FF4444", label=f"SL {signal.stop_loss}"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="center", ncol=4,
            facecolor="#0d1117",
            labelcolor="white",
            fontsize=10,
            framealpha=0.8
        )

        legend_buf = io.BytesIO()
        fig.savefig(legend_buf, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close("all")

        # Combine chart and legend into one image
        from PIL import Image
        buf.seek(0)
        legend_buf.seek(0)
        chart_img = Image.open(buf)
        legend_img = Image.open(legend_buf)

        combined_width = max(chart_img.width, legend_img.width)
        combined_height = chart_img.height + legend_img.height
        combined = Image.new("RGB", (combined_width, combined_height), color=(13, 17, 23))
        combined.paste(chart_img, (0, 0))
        combined.paste(legend_img, (0, chart_img.height))

        final_buf = io.BytesIO()
        combined.save(final_buf, format="PNG")
        final_buf.seek(0)
        return final_buf

    except Exception as e:
        logger.error(f"Chart generation failed for {signal.symbol}: {e}")
        return None
