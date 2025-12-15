import argparse
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.db.db_connection import db_connection
from src.analytics.visualize_relations_by_layers import (
    _plot_layer_histograms,
    _save_bottom_bar_only,
    _save_bottom_box_only,
    _save_single_hist_image,
)
from src.re.count_relationships_all import (
    load_relationship_adjacency,
    compute_layered_counts_from_adjacency,
)


def export_cumulative_charts(max_layer_override: Optional[int] = None) -> Tuple[str, Optional[str]]:
    """Generate cumulative charts (only) for relationship counts per act per layer.

    Outputs PNGs under outputs/analytics/edges:
    - Combined cumulative image: per-layer histograms (KDE) + bottom mean bar (smoothed trend) + bottom box plot
    - Separate bottom-only images: mean bar only, box plot only
    - One PNG per layer for cumulative histograms

    Accuracy note:
      If an act has no per-layer rows (or only zeros) in `re_layer_counts`, this function
      recomputes its per-layer relationship counts from the live adjacency using the same
      BFS-based logic as the per‑act evolution visualizer, then accumulates them. This keeps
      cumulative figures consistent even when the precomputed table is incomplete or stale.

    Returns:
        Tuple[str, Optional[str]]: Base directory path for outputs, combined cumulative histogram path.
    """

    out_dir = os.path.join("outputs", "analytics", "edges")
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if max_layer_override is not None and max_layer_override < 1:
        raise ValueError("max_layer_override must be >= 1 when provided.")

    conn = db_connection.get_connection()
    if conn is None:
        raise RuntimeError("PostgreSQL connection not available.")

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT core_act, layer, relationship_count
                FROM re_layer_counts
                ORDER BY core_act, layer
                """
            )
            rows = cur.fetchall()

        data: dict[str, dict[int, int]] = {}
        for core_act, layer, relationship_count in rows:
            data.setdefault(core_act, {})[int(layer)] = int(relationship_count)

        # Fallback/correction: recompute per‑act layer counts from adjacency when missing or zero,
        # mirroring the evolution view’s BFS‑based counting.
        adjacency = load_relationship_adjacency() or {}
        if not data and adjacency:
            # Seed with subjects from adjacency if the table is empty
            data = {core: {} for core in adjacency.keys()}

        detected_max_layer = 0
        for core_act in list(data.keys()):
            layer_counts = data.get(core_act, {})
            has_any_layer_key = any(k >= 1 for k in layer_counts.keys())
            sum_layers = sum(v for k, v in layer_counts.items() if k >= 1)
            needs_recompute = (not has_any_layer_key) or (sum_layers == 0)

            if needs_recompute and adjacency:
                acts_per_layer, rels_per_layer, _total_acts, total_rels = (
                    compute_layered_counts_from_adjacency(core_act, adjacency)
                )
                if rels_per_layer:
                    for k, v in rels_per_layer.items():
                        data[core_act][int(k)] = int(v)
                    detected_max_layer = max(detected_max_layer, max(rels_per_layer.keys()))
                data[core_act][0] = int(total_rels)
            else:
                act_layers = [l for l in layer_counts.keys() if l >= 1]
                if act_layers:
                    detected_max_layer = max(detected_max_layer, max(act_layers))

        global_max_layer = detected_max_layer
        if max_layer_override is not None:
            global_max_layer = min(max_layer_override, detected_max_layer)

        layer_token = f"L{global_max_layer}"
        base_filename = f"relationship_count_{layer_token}"

        # Build DataFrames
        header = ["Act title", "Max Layer"] + [f"L{i}" for i in range(1, global_max_layer + 1)] + ["Total"]
        rows_out = []
        for core_act in sorted(data.keys()):
            layer_counts = data.get(core_act, {})
            act_layers_nonzero = [l for l, v in layer_counts.items() if l >= 1 and int(v) > 0]
            act_layers_keys = [l for l in layer_counts.keys() if l >= 1]
            act_max_layer = (
                max(act_layers_nonzero) if act_layers_nonzero else (max(act_layers_keys) if act_layers_keys else 0)
            )
            row_dict = {"Act title": core_act, "Max Layer": int(act_max_layer)}
            for i in range(1, global_max_layer + 1):
                row_dict[f"L{i}"] = int(layer_counts.get(i, 0))
            row_dict["Total"] = int(layer_counts.get(0, 0))
            rows_out.append(row_dict)

        df = pd.DataFrame(rows_out, columns=header)
        layer_columns = [f"L{i}" for i in range(1, global_max_layer + 1)]
        df_cumulative = df.copy()
        if layer_columns:
            df_cumulative[layer_columns] = df_cumulative[layer_columns].cumsum(axis=1)

        if layer_columns:
            # Prep bottom aggregates using ALL detected layers
            all_layer_labels = [f"L{i}" for i in range(1, detected_max_layer + 1)]
            num_acts = max(1, len(data))

            bottom_means_cum = []
            bottom_distributions_cum: dict[str, np.ndarray] = {}
            for k in range(1, detected_max_layer + 1):
                series_vals = []
                total = 0
                for act_counts in data.values():
                    cum = 0
                    for j in range(1, k + 1):
                        cum += int(act_counts.get(j, 0))
                    series_vals.append(cum)
                    total += cum
                bottom_means_cum.append(total / num_acts)
                bottom_distributions_cum[f"L{k}"] = np.asarray(series_vals, dtype=float)

            # Combined cumulative image (both bottom charts)
            combined_path = _plot_layer_histograms(
                df_cumulative,
                layer_columns,
                global_max_layer,
                base_path=f"{os.path.join(out_dir, base_filename)}_{timestamp}.xlsx",
                suffix="cumulative",
                title_prefix="Cumulative relationship counts",
                plot_path=os.path.join(out_dir, f"{base_filename}_{timestamp}_cumulative.png"),
                bottom_labels=all_layer_labels,
                bottom_means=bottom_means_cum,
                bottom_title="Mean cumulative by layer",
                bottom_distributions=bottom_distributions_cum,
                bottom_mode="both",
            )

            # Separate bottom-only images (cumulative)
            _save_bottom_bar_only(
                all_layer_labels,
                bottom_means_cum,
                os.path.join(out_dir, f"{base_filename}_{timestamp}_cumulative_mean_bar_only.png"),
                title="Mean cumulative by layer",
            )
            _save_bottom_box_only(
                all_layer_labels,
                bottom_distributions_cum,
                os.path.join(out_dir, f"{base_filename}_{timestamp}_cumulative_box_only.png"),
                title="Cumulative distribution by layer (box plot)",
            )

            # Individual cumulative histograms per requested layer
            for col in layer_columns:
                vals = pd.to_numeric(df_cumulative[col], errors="coerce").to_numpy(dtype=float)
                single_path = os.path.join(
                    out_dir, f"{base_filename}_{timestamp}_cumulative_hist_{col}.png"
                )
                _save_single_hist_image(
                    vals,
                    col,
                    single_path,
                    title_prefix="Cumulative Relationship Count Distribution",
                    xlabel="Cumulative relationship count",
                )

            return out_dir, combined_path

        return out_dir, None
    finally:
        db_connection.release_connection(conn)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate cumulative per-layer charts: combined figure (histograms + bottom aggregates), "
            "bottom-only figures, and one image per layer."
        )
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        help="Maximum layer number to include for per-layer histograms.",
    )
    args = parser.parse_args()
    base_dir, combined = export_cumulative_charts(args.max_layer)
    print(f"Saved cumulative charts under: {base_dir}")
    if combined:
        print(f"Combined cumulative figure: {combined}")


if __name__ == "__main__":
    try:
        main()
    finally:
        db_connection.close_all_connections()
