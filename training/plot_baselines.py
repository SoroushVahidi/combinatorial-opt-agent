"""
Read results/baselines_test.csv and produce paper artifacts:
  - results/baselines_barplot.png
  - results/baselines_table.tex (LaTeX tabular)
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Generate baseline bar plot and LaTeX table")
    p.add_argument("--csv", type=Path, default=None, help="Input CSV (default: results/baselines_test.csv)")
    p.add_argument("--results-dir", type=Path, default=None)
    args = p.parse_args()

    results_dir = Path(args.results_dir or ROOT / "results")
    csv_path = args.csv or results_dir / "baselines_test.csv"
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))

    if not rows:
        raise SystemExit("No rows in CSV")

    # Metric columns (exclude 'baseline')
    metric_cols = [c for c in rows[0].keys() if c != "baseline"]
    baselines = [r["baseline"] for r in rows]

    # LaTeX table
    tex_path = results_dir / "baselines_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l" + "r" * len(metric_cols) + "}\n\\toprule\n")
        f.write("Baseline & " + " & ".join(c.replace("@", "@") for c in metric_cols) + " \\\\\n\\midrule\n")
        for r in rows:
            vals = [r.get(c, "--") for c in metric_cols]
            # Format numbers to 3 decimals if float
            formatted = []
            for v in vals:
                try:
                    formatted.append(f"{float(v):.3f}")
                except (ValueError, TypeError):
                    formatted.append(str(v))
            f.write(r["baseline"] + " & " + " & ".join(formatted) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Wrote {tex_path}")

    # Bar plot (matplotlib if available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping bar plot.")
        return

    x = np.arange(len(metric_cols))
    width = 0.8 / max(len(baselines), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, bl in enumerate(baselines):
        vals = []
        for c in metric_cols:
            try:
                vals.append(float(rows[i].get(c, 0)))
            except (ValueError, TypeError):
                vals.append(0.0)
        offset = (i - (len(baselines) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=bl)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval baselines (test split)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    plot_path = results_dir / "baselines_barplot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
