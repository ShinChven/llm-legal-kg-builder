import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.db.db_connection import db_connection


def export_word_counts_histogram(bins: Optional[int] = None) -> Tuple[str, str]:
    """Query legislation titles and word counts, export to Excel, and create a histogram image.

    Args:
        bins: Optional number of bins to use for the histogram. If None, matplotlib's heuristic is used.

    Returns:
        Tuple containing the Excel file path and the histogram image path.
    """

    # Ensure output directory exists (word_count subdirectory)
    out_dir = os.path.join("outputs", "analytics", "word_count")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(out_dir, f"legislation_word_counts_{timestamp}.xlsx")
    image_path = os.path.splitext(excel_path)[0] + ".png"

    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available.")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title, word_count
                FROM legislations
                WHERE word_count IS NOT NULL
                ORDER BY title
                """
            )
            rows = cur.fetchall()

        if not rows:
            raise RuntimeError("No word count data found in legislations table.")

        df = pd.DataFrame(rows, columns=["Act title", "Word count"])
        df.to_excel(excel_path, index=False)

        word_counts = df["Word count"].to_numpy(dtype=float)

        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "matplotlib is required to generate the histogram; please install it and retry."
            ) from exc

        # Apply a clean, readable plotting style
        plt.style.use("seaborn-v0_8-whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6))
        counts, bin_edges, _ = ax.hist(
            word_counts,
            bins=bins if bins and bins > 0 else "auto",
            color="#4C72B0",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.6,
            histtype="stepfilled",
        )

        ax.set_title("Distribution of legislation word counts", fontsize=14, pad=10)
        ax.set_xlabel("Word count", fontsize=12)
        ax.set_ylabel("Number of acts", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Add thousands separators and integer ticks for counts
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

        # KDE overlay with Gaussian kernel
        finite_counts = word_counts[np.isfinite(word_counts)]
        # Summary markers (mean and median) for quick reference
        if finite_counts.size >= 1:
            mean_val = float(np.mean(finite_counts))
            median_val = float(np.median(finite_counts))
            ax.axvline(
                mean_val,
                color="#55A868",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                label="Mean",
            )
            ax.axvline(
                median_val,
                color="#C44E52",
                linestyle=":",
                linewidth=1.5,
                alpha=0.9,
                label="Median",
            )
        if finite_counts.size >= 2 and np.std(finite_counts) > 0:
            bandwidth = 1.06 * np.std(finite_counts) * finite_counts.size ** (-1 / 5)
            if bandwidth <= 0:
                bandwidth = 1.0
            grid = np.linspace(finite_counts.min(), finite_counts.max(), 400)
            diffs = (grid[:, None] - finite_counts[None, :]) / bandwidth
            density = np.exp(-0.5 * diffs**2).sum(axis=1)
            density /= finite_counts.size * bandwidth * np.sqrt(2 * np.pi)

            if np.any(np.isfinite(density)):
                avg_bin_width = np.mean(np.diff(bin_edges))
                scale_factor = finite_counts.size * max(avg_bin_width, 1e-9)
                ax.plot(grid, density * scale_factor, color="#DD8452", linewidth=2, label="Density")
        else:
            ax.text(
                0.5,
                0.9,
                "Insufficient data for density fit",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#DD8452",
            )

        # Add legend only if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best", frameon=False)

        fig.tight_layout()
        fig.savefig(image_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        return excel_path, image_path
    finally:
        db_connection.release_connection(conn)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export legislation titles and word counts to Excel and generate a histogram image. "
            "The histogram is saved alongside the Excel file in outputs/analytics/word_count."
        )
    )
    parser.add_argument(
        "--bins",
        type=int,
        help="Optional number of bins for the histogram. Defaults to matplotlib's 'auto' binning.",
    )
    args = parser.parse_args()

    excel_path, image_path = export_word_counts_histogram(args.bins)
    print(f"Exported legislation word counts to: {excel_path}")
    print(f"Saved histogram image to: {image_path}")


if __name__ == "__main__":
    main()
