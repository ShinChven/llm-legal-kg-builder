import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

from src.db.db_connection import db_connection
from src.re.count_relationships_all import (
    load_relationship_adjacency,
    compute_layered_counts_from_adjacency,
)
import pandas as pd
import numpy as np


def _estimate_bin_count(values: np.ndarray) -> int:
    """Estimate a reasonable histogram bin count using the Freedman–Diaconis rule."""
    n = values.size
    if n < 2:
        return 1

    data_range = values.max() - values.min()
    if data_range <= 0:
        return 1

    sturges = max(1, int(np.ceil(np.log2(n) + 1)))
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    if iqr > 0:
        bin_width = 2 * iqr / np.cbrt(n)
        if bin_width > 0:
            bins = int(np.ceil(data_range / bin_width))
        else:
            bins = sturges
    else:
        bins = sturges

    min_bins = max(5, sturges)
    max_bins = max(80, sturges)
    return int(np.clip(bins, min_bins, max_bins))


def _compute_focus_xlim(values: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return axis limits that emphasise the main data mass while keeping context."""
    if values.size < 2:
        return None

    q_low, q_high = np.percentile(values, [1, 99])
    spread = q_high - q_low
    if not np.isfinite(q_low) or not np.isfinite(q_high) or spread <= 0:
        return None

    margin = max(spread * 0.1, 1.0)
    x_min = max(values.min(), q_low - margin)
    x_max = min(values.max(), q_high + margin)

    if x_max <= x_min:
        return None
    return x_min, x_max


def _save_single_hist_image(
    values: np.ndarray,
    column_label: str,
    out_path: str,
    *,
    title_prefix: str,
    xlabel: str = "Relationship count",
) -> Optional[str]:
    """Render and save a single histogram with KDE overlay for a 1D array.

    Returns the saved path, or None if there is no finite data to plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to generate histograms.") from exc

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    bin_count = _estimate_bin_count(finite_values)
    counts, bin_edges, _ = ax.hist(
        finite_values,
        bins=bin_count,
        color="#4C72B0",
        alpha=0.6,
        edgecolor="black",
    )

    if finite_values.size >= 2 and np.std(finite_values) > 0:
        bandwidth = 1.06 * np.std(finite_values) * finite_values.size ** (-1 / 5)
        if bandwidth <= 0:
            bandwidth = 1.0
        grid = np.linspace(finite_values.min(), finite_values.max(), 200)
        diffs = (grid[:, None] - finite_values[None, :]) / bandwidth
        density = np.exp(-0.5 * diffs**2).sum(axis=1)
        density /= finite_values.size * bandwidth * np.sqrt(2 * np.pi)
        if np.any(np.isfinite(density)):
            avg_bin_width = np.mean(np.diff(bin_edges)) if len(bin_edges) > 1 else 1.0
            scale_factor = finite_values.size * max(avg_bin_width, 1e-9)
            ax.plot(grid, density * scale_factor, color="#DD8452", linewidth=2, label="Density")
            ax.legend()

    ax.set_title(f"{column_label}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    focus_xlim = _compute_focus_xlim(finite_values)
    if focus_xlim:
        ax.set_xlim(focus_xlim)
        if finite_values.min() < focus_xlim[0] or finite_values.max() > focus_xlim[1]:
            ax.text(
                0.98,
                0.95,
                "x-axis focuses on 1–99th pct.",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#4C72B0",
            )

    fig.suptitle(f"{title_prefix} - {column_label}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_bottom_bar_only(
    labels: list[str],
    means: list[float],
    out_path: str,
    *,
    title: str = "Mean by layer",
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to generate bar charts.") from exc

    width = max(8.0, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(width, 4))
    x = np.arange(1, len(labels) + 1)

    # Mean bars
    ax.bar(x, means, color="#55A868", alpha=0.85, edgecolor="black", label="Mean")

    # Add a smoothed curve over means (Gaussian 1D smoothing)
    m = np.asarray(means, dtype=float)
    if m.size >= 3 and np.isfinite(m).any():
        k = min(9, max(3, (m.size // 5) * 2 + 1))  # odd window length ~ N/5
        if k % 2 == 0:
            k += 1
        xsig = max(1.0, k / 3.0)
        kernel_idx = np.arange(k) - (k // 2)
        kernel = np.exp(-0.5 * (kernel_idx / xsig) ** 2)
        kernel /= kernel.sum()
        pad = k // 2
        mp = np.pad(m, (pad, pad), mode="edge")
        smooth = np.convolve(mp, kernel, mode="valid")
        ax.plot(x, smooth, color="#C44E52", linewidth=2, label="Smoothed trend")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if len(labels) > 15:
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean relationship count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_bottom_box_only(
    labels: list[str],
    distributions: dict[str, np.ndarray],
    out_path: str,
    *,
    title: str = "Distribution by layer (box plot)",
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required to generate box plots.") from exc

    # Build list in label order
    bp_data = []
    positions = []
    for i, lbl in enumerate(labels, start=1):
        arr = np.asarray(distributions.get(lbl, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            # skip empty series; keep positions aligned with data
            continue
        bp_data.append(arr)
        positions.append(i)

    if not bp_data:
        return None

    width = max(8.0, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(width, 4))

    bp = ax.boxplot(
        bp_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for patch in bp['boxes']:
        patch.set(facecolor="#4C72B0", edgecolor="#4C72B0", alpha=0.25)
    for whisker in bp['whiskers']:
        whisker.set(color="#4C72B0", alpha=0.6)
    for cap in bp['caps']:
        cap.set(color="#4C72B0", alpha=0.6)
    for median in bp['medians']:
        median.set(color="#C44E52", linewidth=1.5)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    if len(labels) > 15:
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Relationship count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_layer_histograms(
    df_source: pd.DataFrame,
    layer_columns: list[str],
    max_layer: int,
    base_path: str,
    suffix: str,
    title_prefix: str,
    plot_path: Optional[str] = None,
    *,
    bottom_labels: Optional[list[str]] = None,
    bottom_means: Optional[list[float]] = None,
    bottom_title: Optional[str] = None,
    bottom_distributions: Optional[dict[str, np.ndarray]] = None,
    bottom_mode: str = "bar",
) -> str:
    """Plot per-layer histograms with Gaussian KDE overlays and a bottom mean-per-layer bar chart.

    The bottom subplot is always included and shows, for X = L1..Lmax, the mean value of
    each corresponding layer across all rows in `df_source`.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "matplotlib is required to generate histograms; please install it and retry."
        ) from exc

    if not layer_columns:
        raise ValueError("layer_columns must not be empty when generating histograms.")

    num_layers = len(layer_columns)
    # Arrange up to 4 histograms per row
    ncols = min(4, num_layers)
    nrows = int(np.ceil(num_layers / ncols))

    bottom_mode = (bottom_mode or "bar").lower()
    if bottom_mode not in {"bar", "box", "both"}:
        bottom_mode = "bar"
    bottom_rows = 2 if bottom_mode == "both" else 1

    # Layout: a grid for layer histograms (nrows x ncols) + bottom rows (spanning all columns)
    import matplotlib.pyplot as plt  # ensured above
    fig = plt.figure(figsize=(ncols * 5, (nrows + bottom_rows) * 4))
    gs = fig.add_gridspec(nrows + bottom_rows, ncols, height_ratios=[1] * nrows + [1] * bottom_rows)

    axes = []

    for idx, column in enumerate(layer_columns):
        # Position within the grid for the per-layer histogram
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        values = df_source[column].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            ax.set_visible(False)
            continue

        bin_count = _estimate_bin_count(finite_values)
        counts, bin_edges, _ = ax.hist(
            finite_values,
            bins=bin_count,
            color="#4C72B0",
            alpha=0.6,
            edgecolor="black",
        )

        ax.set_title(f"{column} relationships")
        ax.set_xlabel("Relationship count")
        ax.set_ylabel("Frequency")

        if finite_values.size >= 2 and np.std(finite_values) > 0:
            bandwidth = 1.06 * np.std(finite_values) * finite_values.size ** (-1 / 5)
            if bandwidth <= 0:
                bandwidth = 1.0
            grid = np.linspace(finite_values.min(), finite_values.max(), 200)
            diffs = (grid[:, None] - finite_values[None, :]) / bandwidth
            density = np.exp(-0.5 * diffs**2).sum(axis=1)
            density /= finite_values.size * bandwidth * np.sqrt(2 * np.pi)

            if np.any(np.isfinite(density)):
                avg_bin_width = np.mean(np.diff(bin_edges)) if len(bin_edges) > 1 else 1.0
                scale_factor = finite_values.size * max(avg_bin_width, 1e-9)
                ax.plot(grid, density * scale_factor, color="#DD8452", linewidth=2, label="Density")
                ax.legend()
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

        focus_xlim = _compute_focus_xlim(finite_values)
        if focus_xlim:
            ax.set_xlim(focus_xlim)
            if finite_values.min() < focus_xlim[0] or finite_values.max() > focus_xlim[1]:
                ax.text(
                    0.98,
                    0.95,
                    "x-axis focuses on 1–99th pct.",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#4C72B0",
                )

    # Bottom aggregated charts (bar, box, or both)
    if bottom_labels is None or bottom_means is None:
        # Fallback to means computed from visible layer_columns only
        calc_means = []
        for column in layer_columns:
            vals = df_source[column].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            calc_means.append(float(np.mean(vals))) if vals.size > 0 else calc_means.append(0.0)
        bottom_labels = layer_columns
        bottom_means = calc_means

    x = np.arange(1, len(bottom_labels) + 1)

    def draw_bar(ax):
        # Mean bars
        ax.bar(x, bottom_means, color="#55A868", alpha=0.8, edgecolor="black", label="Mean")

        # Smoothed curve over means (Gaussian 1D smoothing)
        m = np.asarray(bottom_means, dtype=float)
        if m.size >= 3 and np.isfinite(m).any():
            k = min(9, max(3, (m.size // 5) * 2 + 1))
            if k % 2 == 0:
                k += 1
            xsig = max(1.0, k / 3.0)
            kernel_idx = np.arange(k) - (k // 2)
            kernel = np.exp(-0.5 * (kernel_idx / xsig) ** 2)
            kernel /= kernel.sum()
            pad = k // 2
            mp = np.pad(m, (pad, pad), mode="edge")
            smooth = np.convolve(mp, kernel, mode="valid")
            ax.plot(x, smooth, color="#C44E52", linewidth=2, label="Smoothed trend")
        ax.set_ylabel("Mean relationship count")
        ax.set_title(bottom_title or "Mean by layer")
        ax.set_xticks(x)
        ax.set_xticklabels(bottom_labels)
        if len(bottom_labels) > 15:
            ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Layer")
        ax.legend(loc="upper right")

    def draw_box(ax):
        # Build box plot data from provided distributions or fallback to df_source columns
        bp_data = []
        bp_positions = []
        if bottom_distributions:
            for i, lbl in enumerate(bottom_labels, start=1):
                arr = bottom_distributions.get(lbl)
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                bp_data.append(arr)
                bp_positions.append(i)
        else:
            # Fallback: use each column's values across rows
            for i, lbl in enumerate(bottom_labels, start=1):
                if lbl in df_source.columns:
                    arr = pd.to_numeric(df_source[lbl], errors="coerce").to_numpy()
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        bp_data.append(arr)
                        bp_positions.append(i)
        if bp_data:
            bp = ax.boxplot(
                bp_data,
                positions=bp_positions,
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            for patch in bp['boxes']:
                patch.set(facecolor="#4C72B0", edgecolor="#4C72B0", alpha=0.25)
            for whisker in bp['whiskers']:
                whisker.set(color="#4C72B0", alpha=0.6)
            for cap in bp['caps']:
                cap.set(color="#4C72B0", alpha=0.6)
            for median in bp['medians']:
                median.set(color="#C44E52", linewidth=1.5)
        ax.set_ylabel("Relationship count")
        ax.set_title("Distribution by layer (box plot)")
        ax.set_xticks(x)
        ax.set_xticklabels(bottom_labels)
        if len(bottom_labels) > 15:
            ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Layer")

    if bottom_mode == "bar":
        bottom_ax = fig.add_subplot(gs[nrows, :])
        draw_bar(bottom_ax)
    elif bottom_mode == "box":
        bottom_ax = fig.add_subplot(gs[nrows, :])
        draw_box(bottom_ax)
    else:  # both
        bottom_ax_bar = fig.add_subplot(gs[nrows, :])
        draw_bar(bottom_ax_bar)
        bottom_ax_box = fig.add_subplot(gs[nrows + 1, :])
        draw_box(bottom_ax_box)

    # Concise, professional figure title
    if "cumulative" in (title_prefix or "").lower():
        suptitle = f"Cumulative Relationship Count Distribution by Layer (L1–L{max_layer})"
    else:
        suptitle = f"Relationship Count Distribution by Layer (L1–L{max_layer})"
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    resolved_plot_path = plot_path or f"{os.path.splitext(base_path)[0]}_{suffix}.png"
    fig.savefig(resolved_plot_path)
    plt.close(fig)
    return resolved_plot_path


def export_relationship_counts(
    max_layer_override: Optional[int] = None,
    *,
    split: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Export relationship counts per act for all available layers.

    Produces a single Excel sheet ("LayerCounts") with columns: Act title, MaxLayer, L1..L{M}, Total.
    - M defaults to the maximum layer number found in `re_layer_counts` (layer >= 1).
      When `max_layer_override` is provided, M becomes min(override, detected maximum layer).
    - Total corresponds to layer = 0 (aggregated relationship codes).
    - Charts in this program focus on layer-specific (raw) views only:
      - Combined image includes per-layer histograms (with KDE) and bottom aggregates (mean bar with smoothed trend, box plot).
      - Separate images include: one PNG per layer histogram (raw), bottom mean bar only, bottom box plot only.
      Use the separate cumulative program to generate cumulative charts.

    Accuracy note:
      If an act has no per-layer rows (or only zeros) in `re_layer_counts`, this function
      recomputes its per-layer relationship counts from the live adjacency (subject->object)
      using the same BFS-based logic as the per‑act evolution visualizer. This ensures the
      exported and plotted counts reflect actual relationships even when the precomputed
      table is incomplete or stale.

    Returns:
        Tuple[str, Optional[str], Optional[str]]: Excel file path, raw histogram path, cumulative histogram path (always None here).
    """

    # Ensure output directory exists in analytics/edges per updated structure
    out_dir = os.path.join("outputs", "analytics", "edges")
    os.makedirs(out_dir, exist_ok=True)

    # Prepare naming tokens
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if max_layer_override is not None and max_layer_override < 1:
        raise ValueError("max_layer_override must be >= 1 when provided.")

    # Will fill in detected max layer after querying
    out_path = None
    hist_path: Optional[str] = None
    hist_cumulative_path: Optional[str] = None

    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available.")

    try:
        with conn.cursor() as cur:
            # Fetch precomputed relationship counts for all layers, if present
            cur.execute(
                """
                SELECT core_act, layer, relationship_count
                FROM re_layer_counts
                ORDER BY core_act, layer
                """
            )
            rows = cur.fetchall()

        # Pivot into { core_act: {layer: relationship_count} } using any precomputed rows
        data: dict[str, dict[int, int]] = {}
        for core_act, layer, relationship_count in rows:
            data.setdefault(core_act, {})[int(layer)] = int(relationship_count)

        # Fallback/correction: compute accurate per‑act counts from live adjacency
        # when an act has no per‑layer rows or only zeros, mirroring evolution logic.
        adjacency = load_relationship_adjacency() or {}

        # If DB has no rows yet (or missing acts entirely), seed with acts found in adjacency
        if not data and adjacency:
            data = {core: {} for core in adjacency.keys()}

        detected_max_layer = 0
        for core_act in list(data.keys()):
            layer_counts = data.get(core_act, {})
            # Determine if per‑layer counts are missing or sum to zero
            has_any_layer_key = any(k >= 1 for k in layer_counts.keys())
            sum_layers = sum(v for k, v in layer_counts.items() if k >= 1)
            needs_recompute = (not has_any_layer_key) or (sum_layers == 0)

            # Compute from adjacency when needed (or when DB has nothing for this act)
            if needs_recompute and adjacency:
                acts_per_layer, rels_per_layer, _total_acts, total_rels = (
                    compute_layered_counts_from_adjacency(core_act, adjacency)
                )
                # Update per‑layer relationship counts (1..K)
                if rels_per_layer:
                    for k, v in rels_per_layer.items():
                        data[core_act][int(k)] = int(v)
                    detected_max_layer = max(detected_max_layer, max(rels_per_layer.keys()))
                # Always refresh the aggregated total at layer 0 from adjacency result
                data[core_act][0] = int(total_rels)
            else:
                # Track max based on existing DB rows
                act_layers = [l for l in layer_counts.keys() if l >= 1]
                if act_layers:
                    detected_max_layer = max(detected_max_layer, max(act_layers))

        # Resolve global max layer after possible recomputations
        global_max_layer = detected_max_layer
        if max_layer_override is not None:
            global_max_layer = min(max_layer_override, detected_max_layer)

        layer_token = f"L{global_max_layer}"
        base_filename = f"relationship_count_{layer_token}"

        # Set output path now that we know the max layer for export
        out_path = os.path.join(out_dir, f"{base_filename}_{timestamp}.xlsx")

        # Prepare header for raw sheet
        header = ["Act title", "Max Layer"] + [f"L{i}" for i in range(1, global_max_layer + 1)] + ["Total"]

        # Prepare rows for DataFrame
        rows = []
        for core_act in sorted(data.keys()):
            layer_counts = data.get(core_act, {})
            # Determine the act's own max actual layer based on non‑zero entries where present
            act_layers_nonzero = [l for l, v in layer_counts.items() if l >= 1 and int(v) > 0]
            act_layers_keys = [l for l in layer_counts.keys() if l >= 1]
            act_max_layer = (
                max(act_layers_nonzero) if act_layers_nonzero else (max(act_layers_keys) if act_layers_keys else 0)
            )
            row_dict = {"Act title": core_act, "Max Layer": int(act_max_layer)}
            for i in range(1, global_max_layer + 1):
                row_dict[f"L{i}"] = int(layer_counts.get(i, 0))
            # 'Total' corresponds to layer 0 aggregate (relationship codes across all layers)
            row_dict["Total"] = int(layer_counts.get(0, 0))
            rows.append(row_dict)

        df = pd.DataFrame(rows, columns=header)

        # Build layer columns list for downstream plotting
        layer_columns = [f"L{i}" for i in range(1, global_max_layer + 1)]

        # Write XLSX with raw counts only
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="LayerCounts")

        if layer_columns:
            # Prepare bottom means using ALL detected layers (L1..Lmax in DB), regardless of requested cap
            all_layer_labels = [f"L{i}" for i in range(1, detected_max_layer + 1)]
            num_acts = max(1, len(data))

            # Raw means across acts, including zeros for missing layers per act
            bottom_means_raw = []
            bottom_distributions_raw: dict[str, np.ndarray] = {}
            for i in range(1, detected_max_layer + 1):
                series_vals = []
                total = 0
                for act_counts in data.values():
                    v = int(act_counts.get(i, 0))
                    series_vals.append(v)
                    total += v
                bottom_means_raw.append(total / num_acts)
                bottom_distributions_raw[f"L{i}"] = np.asarray(series_vals, dtype=float)

            # Single image including both bar and box aggregated charts for RAW view
            # Combined image (raw): includes both mean bars and box plots
            hist_path = _plot_layer_histograms(
                df,
                layer_columns,
                global_max_layer,
                out_path,
                suffix="layer_specific",
                title_prefix="Layer relationship counts",
                plot_path=os.path.join(out_dir, f"{base_filename}_{timestamp}_layer_specific.png"),
                bottom_labels=all_layer_labels,
                bottom_means=bottom_means_raw,
                bottom_title=f"Mean by layer",
                bottom_distributions=bottom_distributions_raw,
                bottom_mode="both",
            )
            if split:
                # Separate bottom-only images
                _save_bottom_bar_only(
                    all_layer_labels,
                    bottom_means_raw,
                    os.path.join(out_dir, f"{base_filename}_{timestamp}_layer_specific_mean_bar_only.png"),
                    title="Mean by layer",
                )
                _save_bottom_box_only(
                    all_layer_labels,
                    bottom_distributions_raw,
                    os.path.join(out_dir, f"{base_filename}_{timestamp}_layer_specific_box_only.png"),
                    title="Distribution by layer (box plot)",
                )

                # Individual top histograms per layer (raw)
                for col in layer_columns:
                    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                    single_path = os.path.join(
                        out_dir, f"{base_filename}_{timestamp}_layer_specific_hist_{col}.png"
                    )
                    _save_single_hist_image(
                        vals,
                        col,
                        single_path,
                        title_prefix="Relationship Count Distribution",
                        xlabel="Relationship count",
                    )
            # Cumulative charts are intentionally omitted in this program.
            hist_cumulative_path = None

        return out_path, hist_path, hist_cumulative_path
    finally:
        # Return connection to pool
        db_connection.release_connection(conn)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export relationship counts per act per layer from re_layer_counts "
            "to Excel with a single 'LayerCounts' sheet. Columns: Act title, MaxLayer, L1..L{M}, Total. "
            "This program generates raw per-layer histograms with density curves and bottom aggregates (mean bar + box). "
            "Cumulative charts are omitted here; use the cumulative program for those. "
            "Optionally provide --max-layer to cap the exported layers."
        )
    )
    # Optional --max-layer argument limits exported layers used in Excel/plots
    parser.add_argument(
        "--max-layer",
        type=int,
        help=(
            "Maximum layer number to include for raw per-layer histograms."
        ),
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Also export separated images: per-layer histograms and bottom-only charts."
        ),
    )
    args = parser.parse_args()
    out_file, histogram_file, histogram_cumulative_file = export_relationship_counts(
        args.max_layer,
        split=args.split,
    )
    print(f"Exported relationship counts to: {out_file}")
    if histogram_file:
        print(f"Saved histogram to: {histogram_file}")
    if histogram_cumulative_file:
        print(f"Saved cumulative histogram to: {histogram_cumulative_file}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Close pool explicitly (optional; safe no-op if already closed)
        db_connection.close_all_connections()
