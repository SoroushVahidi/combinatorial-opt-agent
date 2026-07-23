#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "results" / "paper" / "eaai_camera_ready_tables"
OUT_DIR = ROOT / "results" / "paper" / "eaai_camera_ready_figures"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _save_both(img: Image.Image, stem: str) -> None:
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    img.save(png)
    img.convert("RGB").save(pdf)


def _write_source_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = OUT_DIR / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _draw_grouped_bars(
    title: str | None,
    methods: Sequence[str],
    metrics: Sequence[str],
    values: list[list[float]],
    y_max: float = 1.0,
    width: int = 1500,
    height: int = 900,
) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    title_f = _font(36)
    label_f = _font(22)
    small_f = _font(18)

    # When title is None, the figure is embedded under a LaTeX \caption that
    # already supplies the figure number and title text; a baked-in title
    # here would duplicate it (and drift out of sync with the LaTeX figure
    # numbering, as happened previously: embedded "Figure 3"/"Figure 4"
    # while the manuscript numbers them Fig. 2/Fig. 3). Only shrink the top
    # margin when there is no title to draw.
    top = 140 if title else 60
    left, right, bottom = 120, width - 60, height - 200
    if title:
        d.text((left, 40), title, fill="black", font=title_f)
    d.line((left, top, left, bottom), fill="black", width=3)
    d.line((left, bottom, right, bottom), fill="black", width=3)

    # y-grid
    for i in range(6):
        yv = y_max * i / 5
        y = bottom - (bottom - top) * i / 5
        d.line((left, y, right, y), fill="#dddddd", width=1)
        d.text((30, y - 10), f"{yv:.1f}", fill="black", font=small_f)

    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    n_methods = len(methods)
    n_metrics = len(metrics)
    group_w = (right - left - 40) / n_methods
    bar_w = max(12, int(group_w / (n_metrics + 2)))

    for mi, m in enumerate(methods):
        gx = left + 20 + mi * group_w
        d.text((gx, bottom + 20), m, fill="black", font=label_f)
        for ki, metric in enumerate(metrics):
            v = values[mi][ki]
            x0 = gx + (ki + 0.8) * bar_w
            x1 = x0 + bar_w - 4
            y1 = bottom
            y0 = bottom - (v / y_max) * (bottom - top)
            d.rectangle((x0, y0, x1, y1), fill=colors[ki % len(colors)], outline="black")
            d.text((x0, y0 - 20), f"{v:.2f}", fill="black", font=small_f)

    # legend
    lx, ly = right - 280, 120
    for ki, metric in enumerate(metrics):
        d.rectangle((lx, ly + 32 * ki, lx + 18, ly + 18 + 32 * ki), fill=colors[ki % len(colors)], outline="black")
        d.text((lx + 26, ly - 2 + 32 * ki), metric, fill="black", font=small_f)

    return img


def build_figure1_pipeline() -> None:
    steps = [
        "NL query",
        "Schema retrieval",
        "Scalar grounding",
        "Structural validation",
        "Executable attempt",
        "Solver-backed subset validation",
    ]
    rows = [{"step_index": i + 1, "step_name": s} for i, s in enumerate(steps)]
    _write_source_csv("figure1_pipeline_overview_source.csv", rows)

    w, h = 1800, 420
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    title_f = _font(34)
    box_f = _font(22)
    d.text((40, 20), "Figure 1. Final EAAI pipeline overview", fill="black", font=title_f)

    x = 40
    y = 130
    box_w = 250
    box_h = 120
    for i, s in enumerate(steps):
        d.rounded_rectangle((x, y, x + box_w, y + box_h), radius=18, fill="#F5F7FB", outline="#3B5B92", width=3)
        d.multiline_text((x + 16, y + 36), s, fill="black", font=box_f, spacing=4)
        if i < len(steps) - 1:
            ax0 = x + box_w + 8
            ax1 = x + box_w + 48
            ay = y + box_h // 2
            d.line((ax0, ay, ax1, ay), fill="#3B5B92", width=4)
            d.polygon([(ax1, ay), (ax1 - 10, ay - 8), (ax1 - 10, ay + 8)], fill="#3B5B92")
        x += 290

    _save_both(img, "figure1_pipeline_overview")


def build_figure2_main_benchmark() -> None:
    t1 = _read_csv(TABLE_DIR / "table1_main_benchmark_summary.csv")
    core = [r for r in t1 if r["group"] == "core"]
    rows = []
    for r in core:
        rows.append(
            {
                "method": r["method_label"],
                "schema_retrieval": float(r["schema_retrieval_r1"]),
                "coverage": float(r["coverage_metric"]),
                "type_match": float(r["type_match_metric"]),
                "instantiation_ready": float(r["instantiation_ready"]),
            }
        )
    _write_source_csv("figure2_main_benchmark_source.csv", rows)

    methods = [r["method"] for r in rows]
    metrics = ["schema retrieval", "coverage", "type match", "inst. ready"]
    vals = [[r["schema_retrieval"], r["coverage"], r["type_match"], r["instantiation_ready"]] for r in rows]
    img = _draw_grouped_bars("Figure 2. Main benchmark comparison (orig)", methods, metrics, vals, y_max=1.0)
    _save_both(img, "figure2_main_benchmark_comparison")


_BAR_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]


def _draw_grouped_bars_mpl(
    methods: Sequence[str],
    metrics: Sequence[str],
    values: list[list[float]],
    stem: str,
    y_max: float = 1.0,
) -> None:
    """Grouped bar chart with the legend placed outside the plot area
    (centered below the axes, single horizontal row), rendered with
    matplotlib so the PDF output is genuine vector graphics rather than a
    rasterized image wrapped in a PDF container. Replaces the PIL-based
    _draw_grouped_bars for the two figures actually used in the manuscript
    (figure3_engineering_validation_comparison,
    figure4_final_solver_backed_subset), whose in-plot top-right legend
    visually crowded and, for the 4-metric case, overlapped the bar-value
    annotations.
    """
    import matplotlib

    matplotlib.use("Agg")
    # Embed fonts as TrueType/Type 42 rather than matplotlib's default Type 3
    # in the vector PDF output, for cleaner publication-grade font embedding.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    n_methods = len(methods)
    n_metrics = len(metrics)
    group_w = 0.86
    bar_w = group_w / n_metrics

    fig_w = max(5.5, 1.15 * n_methods * n_metrics)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))

    x = list(range(n_methods))
    for ki, metric in enumerate(metrics):
        offsets = [xi + (ki - (n_metrics - 1) / 2) * bar_w for xi in x]
        heights = [values[mi][ki] for mi in range(n_methods)]
        bars = ax.bar(
            offsets,
            heights,
            width=bar_w * 0.88,
            label=metric,
            color=_BAR_COLORS[ki % len(_BAR_COLORS)],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=12, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=15)
    ax.set_ylim(0, y_max * 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2 if y_max <= 1.0 else None))
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend outside the plot area: centered below the axes, single
    # horizontal row (preferred placement per the visual-quality fix).
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=n_metrics,
        frameon=False,
        fontsize=14,
        handlelength=1.4,
        columnspacing=1.4,
    )

    fig.tight_layout()
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def build_figure3_engineering() -> None:
    t2 = _read_csv(TABLE_DIR / "table2_engineering_structural_subset.csv")
    rows = []
    for r in t2:
        rows.append(
            {
                "method": r["baseline"].upper(),
                "structural_valid": float(r["structural_valid_rate"]),
                "instantiation_complete": float(r["instantiation_complete_rate"]),
            }
        )
    _write_source_csv("figure3_engineering_validation_source.csv", rows)
    methods = [r["method"] for r in rows]
    metrics = ["structural valid", "inst. complete"]
    vals = [[r["structural_valid"], r["instantiation_complete"]] for r in rows]
    _draw_grouped_bars_mpl(methods, metrics, vals, "figure3_engineering_validation_comparison", y_max=1.0)


def build_figure4_solver_subset() -> None:
    t4 = _read_csv(TABLE_DIR / "table4_final_solver_backed_subset.csv")
    rows = []
    for r in t4:
        rows.append(
            {
                "method": r["baseline"].upper(),
                "executable": float(r["executable_rate"]),
                "solver_success": float(r["solver_success_rate"]),
                "feasible": float(r["feasible_rate"]),
                "objective": float(r["objective_produced_rate"]),
            }
        )
    _write_source_csv("figure4_solver_subset_source.csv", rows)
    methods = [r["method"] for r in rows]
    metrics = ["executable", "solver success", "feasible", "objective"]
    vals = [[r["executable"], r["solver_success"], r["feasible"], r["objective"]] for r in rows]
    _draw_grouped_bars_mpl(methods, metrics, vals, "figure4_final_solver_backed_subset", y_max=1.0)


def build_figure5_failure() -> None:
    # Exact counts sourced from executable-attempt report failure counts.
    rows = [
        {"failure_category": "gurobipy missing", "count": 805},
        {"failure_category": "incomplete instantiation", "count": 267},
        {"failure_category": "type mismatch", "count": 82},
        {"failure_category": "missing scalar slots", "count": 69},
        {"failure_category": "schema miss", "count": 37},
    ]
    _write_source_csv("figure5_failure_breakdown_source.csv", rows)

    methods = [r["failure_category"] for r in rows]
    metrics = ["count"]
    vals = [[float(r["count"])] for r in rows]
    img = _draw_grouped_bars("Figure 5. Failure breakdown (executable-attempt study)", methods, metrics, vals, y_max=850, width=1700, height=920)
    _save_both(img, "figure5_failure_breakdown")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_figure1_pipeline()
    build_figure2_main_benchmark()
    build_figure3_engineering()
    build_figure4_solver_subset()
    build_figure5_failure()


if __name__ == "__main__":
    main()
