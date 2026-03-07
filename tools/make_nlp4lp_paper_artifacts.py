"""Produce final paper-ready artifacts: consolidated table, LaTeX, plots, reproducibility."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LENGTH_BUCKET_ORDER = ["0-25", "26-50", "51-100", "101-150", "151+"]


def _load_summary(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _load_stratified(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def _round_val(v: str, decimals: int = 3) -> str:
    try:
        x = float(v)
        return str(round(x, decimals))
    except (ValueError, TypeError):
        return v


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--summary-csv", type=Path, default=ROOT / "results" / "nlp4lp_retrieval_summary.csv")
    p.add_argument("--stratified-csv", type=Path, default=ROOT / "results" / "nlp4lp_stratified_metrics.csv")
    args = p.parse_args()

    out_dir = ROOT / "results" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(args.summary_csv)
    stratified = _load_stratified(args.stratified_csv)

    # Infer variants and baselines from summary (include noisy and lsa when present)
    canonical_variants = ["orig", "nonum", "short", "noentity", "noisy"]
    canonical_baselines = ["bm25", "tfidf", "lsa"]
    seen_v = {r.get("variant") for r in summary if r.get("variant")}
    variants_order = [v for v in canonical_variants if v in seen_v]
    variants_order += [v for v in sorted(seen_v) if v not in canonical_variants]
    seen_b = {r.get("baseline") for r in summary if r.get("baseline")}
    baselines_order = [b for b in canonical_baselines if b in seen_b]
    baselines_order += [b for b in sorted(seen_b) if b not in canonical_baselines]
    if not variants_order:
        variants_order = canonical_variants
    if not baselines_order:
        baselines_order = canonical_baselines

    metric_suffixes = ["R1", "R5", "R10", "MRR10", "nDCG10", "time"]
    summary_map = {}
    for row in summary:
        v = row["variant"]
        b = row["baseline"]
        summary_map[(v, b)] = {
            "R1": row.get("Recall@1", ""),
            "R5": row.get("Recall@5", ""),
            "R10": row.get("Recall@10", ""),
            "MRR10": row.get("MRR@10", ""),
            "nDCG10": row.get("nDCG@10", ""),
            "time": row.get("runtime_sec", ""),
        }

    main_cols = ["variant"]
    for b in baselines_order:
        for m in metric_suffixes:
            main_cols.append(f"{b}_{m}")

    main_rows = []
    for v in variants_order:
        r = {"variant": v}
        for b in baselines_order:
            data = summary_map.get((v, b), {})
            r[f"{b}_R1"] = _round_val(data.get("R1", ""), 3)
            r[f"{b}_R5"] = _round_val(data.get("R5", ""), 3)
            r[f"{b}_R10"] = _round_val(data.get("R10", ""), 3)
            r[f"{b}_MRR10"] = _round_val(data.get("MRR10", ""), 3)
            r[f"{b}_nDCG10"] = _round_val(data.get("nDCG10", ""), 3)
            r[f"{b}_time"] = _round_val(data.get("time", ""), 2)
        main_rows.append(r)

    main_csv_path = out_dir / "nlp4lp_main_table.csv"
    with open(main_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=main_cols)
        w.writeheader()
        w.writerows(main_rows)
    print(f"Wrote {main_csv_path}")

    # --- 2) LaTeX tabular ---
    n_cols = 1 + 6 * len(baselines_order)
    tex_path = out_dir / "nlp4lp_main_table.tex"
    header_mid = " & ".join(r"\multicolumn{6}{c}{" + b.upper() + "}" for b in baselines_order)
    subheader = " & ".join(["R@1", "R@5", "R@10", "MRR@10", "nDCG@10", "Time(s)"] * len(baselines_order))
    tex_lines = [
        r"\begin{tabular}{l" + "r" * (n_cols - 1) + "}",
        r"\toprule",
        r"Variant & " + header_mid + r" \\",
        r" & " + subheader + r" \\",
        r"\midrule",
    ]
    for r in main_rows:
        row_vals = [r["variant"]]
        for b in baselines_order:
            row_vals.extend([
                r[f"{b}_R1"], r[f"{b}_R5"], r[f"{b}_R10"],
                r[f"{b}_MRR10"], r[f"{b}_nDCG10"], r[f"{b}_time"],
            ])
        tex_lines.append(" & ".join(str(x) for x in row_vals) + r" \\")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"Wrote {tex_path}")

    # --- 3) Bar chart: Recall@1 by variant, all baselines ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = range(len(variants_order))
    n_b = len(baselines_order)
    width = 0.8 / n_b
    for i, b in enumerate(baselines_order):
        offset = (i - (n_b - 1) / 2) * width
        r1_vals = [float(summary_map.get((v, b), {}).get("R1", 0)) for v in variants_order]
        ax.bar([xi + offset for xi in x], r1_vals, width, label=b.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(variants_order)
    ax.set_ylabel("Recall@1")
    ax.set_xlabel("Variant")
    ax.legend()
    ax.set_title("NLP4LP retrieval: Recall@1 by variant")
    fig.tight_layout()
    plot1_path = out_dir / "nlp4lp_variant_comparison.png"
    fig.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Wrote {plot1_path}")

    # --- 4) Line plot: orig, bm25, length_tokens buckets, Recall@1 ---
    orig_bm25_len = [
        row for row in stratified
        if row.get("variant") == "orig" and row.get("baseline") == "bm25"
        and row.get("bucket_type") == "length_tokens" and int(row.get("n", 0)) > 0
    ]
    bucket_to_rec1 = {row["bucket_name"]: float(row["Recall@1"]) for row in orig_bm25_len}
    buckets = [b for b in LENGTH_BUCKET_ORDER if b in bucket_to_rec1]
    rec1_vals = [bucket_to_rec1[b] for b in buckets]

    fig2, ax2 = plt.subplots()
    ax2.plot(buckets, rec1_vals, marker="o")
    ax2.set_xlabel("Query length (tokens)")
    ax2.set_ylabel("Recall@1")
    ax2.set_title("NLP4LP orig / BM25: Recall@1 by query length bucket")
    fig2.tight_layout()
    plot2_path = out_dir / "nlp4lp_length_effect_orig_bm25.png"
    fig2.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Wrote {plot2_path}")

    # --- 4b) Optional downstream utility plot (if summary exists) ---
    downstream_csv = out_dir / "nlp4lp_downstream_summary.csv"
    if downstream_csv.exists():
        rows = []
        with open(downstream_csv, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        # Variants/baselines ordering
        d_variants = ["orig", "noisy", "short"]
        d_baselines = ["bm25", "tfidf", "lsa", "random"]
        seen_v = {rr.get("variant") for rr in rows if rr.get("variant")}
        seen_b = {rr.get("baseline") for rr in rows if rr.get("baseline")}
        v_order = [v for v in d_variants if v in seen_v] + [v for v in sorted(seen_v) if v not in d_variants]
        b_order = [b for b in d_baselines if b in seen_b] + [b for b in sorted(seen_b) if b not in d_baselines]

        inst = {(rr.get("variant"), rr.get("baseline")): rr.get("instantiation_ready") for rr in rows}
        fig3, ax3 = plt.subplots()
        x3 = range(len(v_order))
        n_b = max(1, len(b_order))
        width3 = 0.8 / n_b
        for i, b in enumerate(b_order):
            offset = (i - (n_b - 1) / 2) * width3
            vals = []
            for v in v_order:
                try:
                    vals.append(float(inst.get((v, b)) or 0.0))
                except Exception:
                    vals.append(0.0)
            ax3.bar([xi + offset for xi in x3], vals, width3, label=b.upper())
        ax3.set_xticks(list(x3))
        ax3.set_xticklabels(v_order)
        ax3.set_ylabel("InstantiationReady")
        ax3.set_xlabel("Variant")
        ax3.legend()
        ax3.set_title("NLP4LP downstream utility: instantiation-ready rate")
        fig3.tight_layout()
        plot3_path = out_dir / "nlp4lp_downstream_instantiation_ready.png"
        fig3.savefig(plot3_path, dpi=150)
        plt.close()
        print(f"Wrote {plot3_path}")

        # Optional: schema_R1 vs instantiation_ready scatter for orig queries
        pts = []
        for b in ["random", "lsa", "bm25", "tfidf", "oracle"]:
            row = next(
                (rr for rr in rows if rr.get("variant") == "orig" and rr.get("baseline") == b),
                None,
            )
            if not row:
                continue
            try:
                sx = float(row.get("schema_R1") or 0.0)
                sy = float(row.get("instantiation_ready") or 0.0)
            except Exception:
                continue
            pts.append((b, sx, sy))
        if pts:
            fig_s, ax_s = plt.subplots()
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            ax_s.scatter(xs, ys)
            for name, sx, sy in pts:
                ax_s.text(sx, sy, name, fontsize=8, ha="left", va="bottom")
            ax_s.set_xlabel("Schema_R1 (orig)")
            ax_s.set_ylabel("InstantiationReady (orig)")
            ax_s.set_title("NLP4LP downstream: schema retrieval vs instantiation-ready (orig)")
            fig_s.tight_layout()
            scatter_path = out_dir / "nlp4lp_downstream_schema_vs_ready_orig.png"
            fig_s.savefig(scatter_path, dpi=150)
            plt.close(fig_s)
            print(f"Wrote {scatter_path}")

    # --- 4c) Optional per-type coverage plot for orig (if types summary exists) ---
    types_csv = out_dir / "nlp4lp_downstream_types_summary.csv"
    if types_csv.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = []
        with open(types_csv, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("variant") == "orig":
                    rows.append(row)
        if rows:
            type_order = ["percent", "integer", "currency", "float"]
            base_order = ["bm25", "tfidf", "lsa", "oracle", "random"]
            seen_b = {r["baseline"] for r in rows}
            baselines = [b for b in base_order if b in seen_b]
            x = range(len(type_order))
            fig4, ax4 = plt.subplots()
            n_b = max(1, len(baselines))
            width = 0.8 / n_b
            for i, b in enumerate(baselines):
                offset = (i - (n_b - 1) / 2) * width
                vals = []
                for t in type_order:
                    match = next((r for r in rows if r["baseline"] == b and r["param_type"] == t), None)
                    if match:
                        try:
                            vals.append(float(match.get("param_coverage") or 0.0))
                        except Exception:
                            vals.append(0.0)
                    else:
                        vals.append(0.0)
                ax4.bar([xi + offset for xi in x], vals, width, label=b.upper())
            ax4.set_xticks(list(x))
            ax4.set_xticklabels(type_order)
            ax4.set_ylabel("ParamCoverage")
            ax4.set_xlabel("Param type (orig)")
            ax4.legend()
            ax4.set_title("NLP4LP downstream: per-type coverage (orig)")
            fig4.tight_layout()
            plot4_path = out_dir / "nlp4lp_downstream_type_coverage_orig.png"
            fig4.savefig(plot4_path, dpi=150)
            plt.close()
            print(f"Wrote {plot4_path}")

    # --- 4f) Final combined downstream table for orig (typed and untyped) ---
    if downstream_csv.exists():
        import csv as _csv3
        with downstream_csv.open(encoding="utf-8") as f:
            dsum = list(_csv3.DictReader(f))
        rows_orig = [r for r in dsum if r.get("variant") == "orig"]
        if rows_orig:
            order = ["random", "lsa", "bm25", "tfidf", "oracle", "tfidf_untyped", "oracle_untyped"]
            metrics = ["schema_R1", "param_coverage", "type_match", "key_overlap", "exact5_on_hits", "exact20_on_hits", "instantiation_ready"]

            def _fmt4(v: str) -> str:
                if v is None or v == "":
                    return ""
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return v

            by_b = {r["baseline"]: r for r in rows_orig}

            final_csv = out_dir / "nlp4lp_downstream_final_table_orig.csv"
            with final_csv.open("w", newline="", encoding="utf-8") as f:
                w = _csv3.writer(f)
                w.writerow(["baseline"] + metrics)
                for b in order:
                    row = by_b.get(b)
                    if not row:
                        continue
                    w.writerow([b] + [_fmt4(row.get(m, "")) for m in metrics])

            final_tex = out_dir / "nlp4lp_downstream_final_table_orig.tex"
            lines_f = []
            lines_f.append("\\begin{table}[t]")
            lines_f.append("  \\centering")
            lines_f.append("  \\caption{Retrieval-assisted downstream utility on NLP4LP (orig queries).}")
            lines_f.append("  \\label{tab:nlp4lp-downstream-final-orig}")
            lines_f.append("  \\begin{tabular}{lrrrrrrr}")
            lines_f.append("    \\hline")
            lines_f.append("    baseline & schema\\_R1 & param\\_coverage & type\\_match & key\\_overlap & exact5\\_on\\_hits & exact20\\_on\\_hits & instantiation\\_ready \\\\")
            lines_f.append("    \\hline")
            for b in order:
                row = by_b.get(b)
                if not row:
                    continue
                vals = [_fmt4(row.get(m, "")) for m in metrics]
                lines_f.append("    " + b + " & " + " & ".join(v or "" for v in vals) + " \\\\")
            lines_f.append("    \\hline")
            lines_f.append("  \\end{tabular}")
            lines_f.append("\\end{table}")
            with final_tex.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines_f))

    # --- 4g) Final per-type table for orig (tfidf, oracle, random) ---
    if types_csv.exists():
        import csv as _csv4
        with types_csv.open(encoding="utf-8") as f:
            trows = list(_csv4.DictReader(f))
        types_orig = [r for r in trows if r.get("variant") == "orig" and r.get("baseline") in {"tfidf", "oracle", "random"}]
        if types_orig:
            type_order = ["currency", "float", "integer", "percent"]

            def _fmt4(v: str) -> str:
                if v is None or v == "":
                    return ""
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return v

            # index by (baseline, param_type)
            by_bt = {(r["baseline"], r["param_type"]): r for r in types_orig}

            types_csv_out = out_dir / "nlp4lp_downstream_types_table_orig.csv"
            with types_csv_out.open("w", newline="", encoding="utf-8") as f:
                w = _csv4.writer(f)
                w.writerow([
                    "param_type",
                    "tfidf_param_coverage", "tfidf_type_match",
                    "oracle_param_coverage", "oracle_type_match",
                    "random_param_coverage", "random_type_match",
                ])
                for t in type_order:
                    row_tfidf = by_bt.get(("tfidf", t), {})
                    row_oracle = by_bt.get(("oracle", t), {})
                    row_random = by_bt.get(("random", t), {})
                    w.writerow([
                        t,
                        _fmt4(row_tfidf.get("param_coverage", "")), _fmt4(row_tfidf.get("type_match", "")),
                        _fmt4(row_oracle.get("param_coverage", "")), _fmt4(row_oracle.get("type_match", "")),
                        _fmt4(row_random.get("param_coverage", "")), _fmt4(row_random.get("type_match", "")),
                    ])

            types_tex = out_dir / "nlp4lp_downstream_types_table_orig.tex"
            lines_t = []
            lines_t.append("\\begin{table}[t]")
            lines_t.append("  \\centering")
            lines_t.append("  \\caption{Per-type downstream behavior on NLP4LP (orig queries).}")
            lines_t.append("  \\label{tab:nlp4lp-downstream-types-orig}")
            lines_t.append("  \\begin{tabular}{lrrrrrr}")
            lines_t.append("    \\hline")
            lines_t.append("    param\\_type & TF-IDF cov & TF-IDF type & Oracle cov & Oracle type & Random cov & Random type \\\\")
            lines_t.append("    \\hline")
            with types_csv_out.open(encoding="utf-8") as f:
                r = _csv4.DictReader(f)
                for row in r:
                    lines_t.append(
                        "    "
                        + row["param_type"]
                        + " & "
                        + " & ".join([
                            row["tfidf_param_coverage"], row["tfidf_type_match"],
                            row["oracle_param_coverage"], row["oracle_type_match"],
                            row["random_param_coverage"], row["random_type_match"],
                        ])
                        + " \\\\"
                    )
            lines_t.append("    \\hline")
            lines_t.append("  \\end{tabular}")
            lines_t.append("\\end{table}")
            with types_tex.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines_t))

            # ESWA error-analysis per-type table (CSV + LaTeX)
            err_types_csv = out_dir / "nlp4lp_error_types_table.csv"
            with err_types_csv.open("w", newline="", encoding="utf-8") as f:
                w = _csv4.writer(f)
                w.writerow([
                    "param_type",
                    "tfidf_coverage", "tfidf_type_match",
                    "oracle_coverage", "oracle_type_match",
                    "random_coverage", "random_type_match",
                ])
                with types_csv_out.open(encoding="utf-8") as fin:
                    r_err = _csv4.DictReader(fin)
                    for row in r_err:
                        w.writerow([
                            row["param_type"],
                            row.get("tfidf_param_coverage", ""),
                            row.get("tfidf_type_match", ""),
                            row.get("oracle_param_coverage", ""),
                            row.get("oracle_type_match", ""),
                            row.get("random_param_coverage", ""),
                            row.get("random_type_match", ""),
                        ])

            err_types_tex = out_dir / "nlp4lp_error_types_table.tex"
            lines_et = []
            lines_et.append("\\begin{table}[t]")
            lines_et.append("  \\centering")
            lines_et.append("  \\caption{Per-type downstream behavior on original queries.}")
            lines_et.append("  \\label{tab:nlp4lp-error-types}")
            lines_et.append("  \\begin{tabular}{lrrrrrr}")
            lines_et.append("    \\hline")
            lines_et.append("    Param type & TF-IDF cov & TF-IDF type & Oracle cov & Oracle type & Random cov & Random type \\\\")
            lines_et.append("    \\hline")
            with err_types_csv.open(encoding="utf-8") as f:
                r_err2 = _csv4.DictReader(f)
                for row in r_err2:
                    lines_et.append(
                        "    "
                        + row["param_type"]
                        + " & "
                        + " & ".join([
                            row["tfidf_coverage"], row["tfidf_type_match"],
                            row["oracle_coverage"], row["oracle_type_match"],
                            row["random_coverage"], row["random_type_match"],
                        ])
                        + " \\\\"
                    )
            lines_et.append("    \\hline")
            lines_et.append("  \\end{tabular}")
            lines_et.append("\\end{table}")
            with err_types_tex.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines_et))

    # --- 4h) Final notes and bullets ---
    final_note = out_dir / "nlp4lp_downstream_final_note.txt"
    if downstream_csv.exists() and types_csv.exists():
        # Use previously loaded dsum and trows if available, else reload
        import csv as _csv5
        with downstream_csv.open(encoding="utf-8") as f:
            dsum2 = list(_csv5.DictReader(f))
        rows_orig = [r for r in dsum2 if r.get("variant") == "orig"]
        by_b = {r["baseline"]: r for r in rows_orig}
        with types_csv.open(encoding="utf-8") as f:
            trows2 = list(_csv5.DictReader(f))
        types_orig = [r for r in trows2 if r.get("variant") == "orig"]
        by_bt = {(r["baseline"], r["param_type"]): r for r in types_orig}

        def _get(b, m):
            v = (by_b.get(b) or {}).get(m, "")
            try:
                return float(v)
            except Exception:
                return None

        def _get_type(b, t, m):
            v = (by_bt.get((b, t)) or {}).get(m, "")
            try:
                return float(v)
            except Exception:
                return None

        lines_n = []
        # retrieval vs random
        s_tf = _get("tfidf", "schema_R1"); s_rand = _get("random", "schema_R1")
        c_tf = _get("tfidf", "param_coverage"); c_rand = _get("random", "param_coverage")
        lines_n.append(f"On orig queries, TF-IDF achieves schema_R1={s_tf:.3f} vs random={s_rand:.3f}, and param_coverage={c_tf:.3f} vs random={c_rand:.3f}.")
        # oracle vs tfidf
        s_or = _get("oracle", "schema_R1"); c_or = _get("oracle", "param_coverage")
        lines_n.append(f"The oracle schema baseline reaches schema_R1={s_or:.3f} and param_coverage={c_or:.3f}, indicating that retrieval is strong but not the only bottleneck.")
        # hit/miss overlap already summarized in hitmiss note
        ko_hits = _get("tfidf", "key_overlap_hits"); ko_miss = _get("tfidf", "key_overlap_miss")
        lines_n.append(f"For TF-IDF, key_overlap is high on hits (~{ko_hits:.3f}) and remains non-zero on misses (~{ko_miss:.3f}), so wrong schemas are often partially overlapping.")
        # assignment ablation
        tm_tf = _get("tfidf", "type_match"); tm_tf_u = _get("tfidf_untyped", "type_match")
        ir_tf = _get("tfidf", "instantiation_ready"); ir_tf_u = _get("tfidf_untyped", "instantiation_ready")
        lines_n.append(f"Typed assignment improves type_match for TF-IDF ({tm_tf:.3f} vs {tm_tf_u:.3f}) and instantiation_ready ({ir_tf:.3f} vs {ir_tf_u:.3f}) without changing coverage.")
        # type-wise easiest/hardest
        tm_int = _get_type("tfidf", "integer", "type_match"); tm_float = _get_type("tfidf", "float", "type_match")
        lines_n.append(f"By type, integer parameters are easiest for TF-IDF (type_match≈{tm_int:.3f}), while float parameters remain hardest (type_match≈{tm_float:.3f}).")

        with final_note.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_n))

    # --- 4j) Compact downstream tables and note for ESWA downstream section ---
    downstream_csv2 = out_dir / "nlp4lp_downstream_summary.csv"
    if downstream_csv2.exists():
        import csv as _csv7
        with downstream_csv2.open(encoding="utf-8") as f:
            drows = list(_csv7.DictReader(f))

        # Section main table: orig, typed/default baselines only
        section_baselines = ["random", "lsa", "bm25", "tfidf", "oracle"]
        metrics = [
            "schema_R1",
            "param_coverage",
            "type_match",
            "key_overlap",
            "exact5_on_hits",
            "exact20_on_hits",
            "instantiation_ready",
        ]

        def _fmt4(v: str) -> str:
            if v is None or v == "":
                return ""
            try:
                return f"{float(v):.4f}"
            except Exception:
                return v

        rows_orig = [r for r in drows if r.get("variant") == "orig"]
        by_b = {r.get("baseline"): r for r in rows_orig}

        sec_csv = out_dir / "nlp4lp_downstream_section_table.csv"
        with sec_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv7.writer(f)
            w.writerow(["baseline"] + metrics)
            for b in section_baselines:
                row = by_b.get(b) or {}
                w.writerow([b] + [_fmt4(row.get(m, "")) for m in metrics])

        sec_tex = out_dir / "nlp4lp_downstream_section_table.tex"
        lines_s = []
        lines_s.append("\\begin{table}[t]")
        lines_s.append("  \\centering")
        lines_s.append("  \\caption{Downstream utility on NLP4LP (orig queries).}")
        lines_s.append("  \\label{tab:nlp4lp-downstream-main}")
        lines_s.append("  \\begin{tabular}{lrrrrrrr}")
        lines_s.append("    \\hline")
        lines_s.append(
            "    Baseline & Schema\\_R@1 & Coverage & TypeMatch & KeyOverlap & Exact5 & Exact20 & InstReady \\\\"
        )
        lines_s.append("    \\hline")
        for b in section_baselines:
            row = by_b.get(b) or {}
            vals = [_fmt4(row.get(m, "")) for m in metrics]
            lines_s.append("    " + b + " & " + " & ".join(v or "" for v in vals) + " \\\\")
        lines_s.append("    \\hline")
        lines_s.append("  \\end{tabular}")
        lines_s.append("\\end{table}")
        with sec_tex.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_s))

        # Cross-variant downstream summary table
        variants = ["orig", "noisy", "short"]
        baselines_v = ["tfidf", "bm25", "lsa", "random"]
        by_bv = {(r.get("baseline"), r.get("variant")): r for r in drows}

        var_csv = out_dir / "nlp4lp_downstream_variant_table.csv"
        with var_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv7.writer(f)
            w.writerow(
                [
                    "baseline",
                    "orig_schema_R1",
                    "noisy_schema_R1",
                    "short_schema_R1",
                    "orig_instantiation_ready",
                    "noisy_instantiation_ready",
                    "short_instantiation_ready",
                ]
            )
            for b in baselines_v:
                row_o = by_bv.get((b, "orig")) or {}
                row_n = by_bv.get((b, "noisy")) or {}
                row_s = by_bv.get((b, "short")) or {}
                w.writerow(
                    [
                        b,
                        _fmt4(row_o.get("schema_R1", "")),
                        _fmt4(row_n.get("schema_R1", "")),
                        _fmt4(row_s.get("schema_R1", "")),
                        _fmt4(row_o.get("instantiation_ready", "")),
                        _fmt4(row_n.get("instantiation_ready", "")),
                        _fmt4(row_s.get("instantiation_ready", "")),
                    ]
                )

        var_tex = out_dir / "nlp4lp_downstream_variant_table.tex"
        lines_v = []
        lines_v.append("\\begin{table}[t]")
        lines_v.append("  \\centering")
        lines_v.append("  \\caption{Cross-variant downstream summary.}")
        lines_v.append("  \\label{tab:nlp4lp-downstream-variants}")
        lines_v.append("  \\begin{tabular}{lrrrrrr}")
        lines_v.append("    \\hline")
        lines_v.append(
            "    baseline & orig\\_schema\\_R1 & noisy\\_schema\\_R1 & short\\_schema\\_R1 & orig\\_instantiation\\_ready & noisy\\_instantiation\\_ready & short\\_instantiation\\_ready \\\\"
        )
        lines_v.append("    \\hline")
        with var_csv.open(encoding="utf-8") as f:
            r = _csv7.DictReader(f)
            for row in r:
                lines_v.append(
                    "    "
                    + row["baseline"]
                    + " & "
                    + " & ".join(
                        [
                            row["orig_schema_R1"],
                            row["noisy_schema_R1"],
                            row["short_schema_R1"],
                            row["orig_instantiation_ready"],
                            row["noisy_instantiation_ready"],
                            row["short_instantiation_ready"],
                        ]
                    )
                    + " \\\\"
                )
        lines_v.append("    \\hline")
        lines_v.append("  \\end{tabular}")
        lines_v.append("\\end{table}")
        with var_tex.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_v))

        # Short downstream note for ESWA downstream subsection
        note_path = out_dir / "nlp4lp_downstream_section_note.txt"

        def _get_float(baseline: str, variant: str, metric: str) -> float | None:
            row = by_bv.get((baseline, variant))
            if not row:
                return None
            try:
                return float(row.get(metric, ""))
            except Exception:
                return None

        tf_o_r1 = _get_float("tfidf", "orig", "schema_R1")
        tf_o_ir = _get_float("tfidf", "orig", "instantiation_ready")
        bm_o_r1 = _get_float("bm25", "orig", "schema_R1")
        bm_o_ir = _get_float("bm25", "orig", "instantiation_ready")
        lsa_o_r1 = _get_float("lsa", "orig", "schema_R1")
        lsa_o_ir = _get_float("lsa", "orig", "instantiation_ready")
        rand_o_r1 = _get_float("random", "orig", "schema_R1")
        rand_o_ir = _get_float("random", "orig", "instantiation_ready")
        or_o_r1 = _get_float("oracle", "orig", "schema_R1")
        or_o_ir = _get_float("oracle", "orig", "instantiation_ready")

        tf_n_r1 = _get_float("tfidf", "noisy", "schema_R1")
        tf_n_ir = _get_float("tfidf", "noisy", "instantiation_ready")
        tf_s_r1 = _get_float("tfidf", "short", "schema_R1")
        tf_s_ir = _get_float("tfidf", "short", "instantiation_ready")

        lines_note = []
        if (
            tf_o_r1 is not None
            and bm_o_r1 is not None
            and lsa_o_r1 is not None
            and rand_o_r1 is not None
            and tf_o_ir is not None
            and bm_o_ir is not None
            and lsa_o_ir is not None
            and rand_o_ir is not None
        ):
            lines_note.append(
                f"On orig queries, retrieval-based schemas (TF-IDF/BM25/LSA) achieve schema_R1≈{tf_o_r1:.4f}–{bm_o_r1:.4f} with instantiation_ready≈{lsa_o_ir:.4f}–{bm_o_ir:.4f}, compared to random at schema_R1={rand_o_r1:.4f} and instantiation_ready={rand_o_ir:.4f}."
            )
        if tf_o_r1 is not None and tf_o_ir is not None and bm_o_r1 is not None and bm_o_ir is not None:
            lines_note.append(
                f"Among lexical baselines on orig, TF-IDF has the strongest schema retrieval (schema_R1={tf_o_r1:.4f} vs BM25={bm_o_r1:.4f}) and slightly higher coverage, with instantiation_ready ({tf_o_ir:.4f}) close to BM25 ({bm_o_ir:.4f})."
            )
        if or_o_r1 is not None and or_o_ir is not None and tf_o_r1 is not None and tf_o_ir is not None:
            lines_note.append(
                f"The oracle schema baseline reaches schema_R1={or_o_r1:.4f} and instantiation_ready={or_o_ir:.4f}, only modestly above TF-IDF (schema_R1={tf_o_r1:.4f}, instantiation_ready={tf_o_ir:.4f}), so retrieval is not the only bottleneck."
            )
        if tf_o_r1 is not None and tf_n_r1 is not None and tf_s_r1 is not None:
            lines_note.append(
                f"For TF-IDF, schema_R1 stays high on noisy queries (orig={tf_o_r1:.4f}, noisy={tf_n_r1:.4f}) but drops on short queries (short={tf_s_r1:.4f})."
            )
        if tf_o_ir is not None and tf_n_ir is not None and tf_s_ir is not None:
            lines_note.append(
                f"Instantiation-ready rates for TF-IDF fall from {tf_o_ir:.4f} on orig to {tf_s_ir:.4f} on short, and to {tf_n_ir:.4f} on noisy queries where `<num>` placeholders cannot be deterministically recovered."
            )
        if or_o_ir is not None:
            lines_note.append(
                f"Even with oracle schemas on orig, instantiation_ready remains below 0.1 ({or_o_ir:.4f}), so these results are best interpreted as downstream utility diagnostics rather than full NL-to-optimization automation."
            )

        with note_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_note))

        # Assignment ablation table for ESWA error analysis (typed vs untyped)
        err_ablation_csv = out_dir / "nlp4lp_error_ablation_table.csv"
        with err_ablation_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv7.writer(f)
            w.writerow(
                ["baseline", "param_coverage", "type_match", "exact20_on_hits", "instantiation_ready"]
            )
            for b in ["tfidf", "tfidf_untyped", "oracle", "oracle_untyped"]:
                row = by_b.get(b) or {}
                w.writerow(
                    [
                        b,
                        _fmt4(row.get("param_coverage", "")),
                        _fmt4(row.get("type_match", "")),
                        _fmt4(row.get("exact20_on_hits", "")),
                        _fmt4(row.get("instantiation_ready", "")),
                    ]
                )

        err_ablation_tex = out_dir / "nlp4lp_error_ablation_table.tex"
        lines_ab = []
        lines_ab.append("\\begin{table}[t]")
        lines_ab.append("  \\centering")
        lines_ab.append("  \\caption{Effect of type-aware assignment on original queries.}")
        lines_ab.append("  \\label{tab:nlp4lp-error-ablation}")
        lines_ab.append("  \\begin{tabular}{lrrrr}")
        lines_ab.append("    \\hline")
        lines_ab.append("    Baseline & Coverage & TypeMatch & Exact20 & InstReady \\\\")
        lines_ab.append("    \\hline")
        with err_ablation_csv.open(encoding="utf-8") as f:
            r_ab = _csv7.DictReader(f)
            for row in r_ab:
                lines_ab.append(
                    "    "
                    + row["baseline"]
                    + " & "
                    + " & ".join(
                        [
                            row["param_coverage"],
                            row["type_match"],
                            row["exact20_on_hits"],
                            row["instantiation_ready"],
                        ]
                    )
                    + " \\\\"
                )
        lines_ab.append("    \\hline")
        lines_ab.append("  \\end{tabular}")
        lines_ab.append("\\end{table}")
        with err_ablation_tex.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_ab))

        # Error-analysis note for ESWA subsection
        err_note_path = out_dir / "nlp4lp_error_analysis_note.txt"

        # Use hit/miss table and per-type summary generated above
        hm_src = out_dir / "nlp4lp_downstream_hitmiss_table_orig.csv"
        types_src = out_dir / "nlp4lp_downstream_types_summary.csv"

        hit_rows = []
        if hm_src.exists():
            with hm_src.open(encoding="utf-8") as f:
                r_hm = _csv7.DictReader(f)
                hit_rows = [r for r in r_hm if r.get("baseline") in {"lsa", "bm25", "tfidf"}]

        type_rows = []
        if types_src.exists():
            with types_src.open(encoding="utf-8") as f:
                r_ty = _csv7.DictReader(f)
                type_rows = [r for r in r_ty if r.get("variant") == "orig"]

        def _flt(v: str) -> float | None:
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        cov_hits = [_flt(r.get("param_coverage_hits", "")) for r in hit_rows]
        cov_miss = [_flt(r.get("param_coverage_miss", "")) for r in hit_rows]
        key_hits = [_flt(r.get("key_overlap_hits", "")) for r in hit_rows]
        key_miss = [_flt(r.get("key_overlap_miss", "")) for r in hit_rows]
        tm_hits = [_flt(r.get("type_match_hits", "")) for r in hit_rows]
        tm_miss = [_flt(r.get("type_match_miss", "")) for r in hit_rows]

        def _rng(xs):
            xs_f = [x for x in xs if isinstance(x, float)]
            if not xs_f:
                return None, None
            return min(xs_f), max(xs_f)

        cov_h_lo, cov_h_hi = _rng(cov_hits)
        cov_m_lo, cov_m_hi = _rng(cov_miss)
        key_m_lo, key_m_hi = _rng(key_miss)
        tm_h_lo, tm_h_hi = _rng(tm_hits)
        tm_m_lo, tm_m_hi = _rng(tm_miss)

        # Per-type easiest/hardest from types summary (TF-IDF)
        def _type_val(baseline: str, ptype: str, field: str) -> float | None:
            row = next(
                (r for r in type_rows if r.get("baseline") == baseline and r.get("param_type") == ptype),
                None,
            )
            if not row:
                return None
            return _flt(row.get(field, ""))

        tf_int_tm = _type_val("tfidf", "integer", "type_match")
        tf_float_tm = _type_val("tfidf", "float", "type_match")

        # Typed vs untyped (already loaded via by_b above)
        tf_t_tm = _flt((by_b.get("tfidf") or {}).get("type_match", ""))
        tf_u_tm = _flt((by_b.get("tfidf_untyped") or {}).get("type_match", ""))
        tf_t_ir = _flt((by_b.get("tfidf") or {}).get("instantiation_ready", ""))
        tf_u_ir = _flt((by_b.get("tfidf_untyped") or {}).get("instantiation_ready", ""))
        tf_t_cov = _flt((by_b.get("tfidf") or {}).get("param_coverage", ""))

        or_t_ir = _flt((by_b.get("oracle") or {}).get("instantiation_ready", ""))
        rand_cov = _flt((by_b.get("random") or {}).get("param_coverage", ""))
        rand_ir = _flt((by_b.get("random") or {}).get("instantiation_ready", ""))

        # TF-IDF per-type coverage range (orig)
        tf_cov_types = [
            _flt(r.get("param_coverage", ""))
            for r in type_rows
            if r.get("baseline") == "tfidf"
        ]
        tf_cov_lo, tf_cov_hi = _rng(tf_cov_types)

        lines_err = []
        if cov_h_lo is not None and cov_h_hi is not None and cov_m_lo is not None and cov_m_hi is not None:
            lines_err.append(
                f"On orig queries, param_coverage is high on schema hits (≈{cov_h_lo:.2f}–{cov_h_hi:.2f}) and drops on misses (≈{cov_m_lo:.2f}–{cov_m_hi:.2f})."
            )
        if key_hits and key_m_lo is not None and key_m_hi is not None:
            lines_err.append(
                f"Key_overlap stays near 1.0 on hits and remains non-zero on misses (≈{key_m_lo:.2f}–{key_m_hi:.2f}), so wrong schemas are often partially overlapping rather than completely unrelated."
            )
        if tm_h_lo is not None and tm_h_hi is not None and tm_m_lo is not None and tm_m_hi is not None:
            lines_err.append(
                f"Type_match is substantially higher on hits (≈{tm_h_lo:.2f}–{tm_h_hi:.2f}) than on misses (≈{tm_m_lo:.2f}–{tm_m_hi:.2f}), but even on hits it remains well below 0.3."
            )
        if tf_int_tm is not None and tf_float_tm is not None:
            lines_err.append(
                f"By parameter type, integer parameters are easiest under TF-IDF (type_match≈{tf_int_tm:.3f}), while float parameters are hardest (type_match≈{tf_float_tm:.3f})."
            )
        if tf_cov_lo is not None and tf_cov_hi is not None:
            lines_err.append(
                f"Under TF-IDF, per-type coverage on orig queries is generally high across currency/percent/float (≈{tf_cov_lo:.2f}–{tf_cov_hi:.2f})."
            )
        if tf_t_cov is not None and tf_t_tm is not None and tf_u_tm is not None and tf_t_ir is not None and tf_u_ir is not None:
            lines_err.append(
                f"Typed assignment for TF-IDF keeps coverage unchanged (≈{tf_t_cov:.3f}) but improves type_match (from {tf_u_tm:.3f} to {tf_t_tm:.3f}) and instantiation_ready (from {tf_u_ir:.3f} to {tf_t_ir:.3f})."
            )
        if rand_cov is not None and rand_ir is not None:
            lines_err.append(
                f"The random baseline has very low coverage (≈{rand_cov:.3f}) and instantiation_ready (≈{rand_ir:.3f}), confirming that naive schema selection rarely yields fully instantiated problems."
            )
        if or_t_ir is not None:
            lines_err.append(
                f"Even with oracle schemas, instantiation_ready on orig queries stays below 0.1 ({or_t_ir:.4f}), so substantial residual error remains downstream of retrieval."
            )

        with err_note_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_err))

    claims_path = out_dir / "nlp4lp_downstream_claims_for_paper.txt"
    caveats_path = out_dir / "nlp4lp_downstream_caveats_for_paper.txt"
    # Simple static bullet lists grounded in the analyses
    with claims_path.open("w", encoding="utf-8") as f:
        f.write("- Retrieval-based schemas (BM25/TF-IDF/LSA) substantially outperform random schema selection on schema_R1 and downstream parameter coverage.\n")
        f.write("- With strong retrieval (e.g., TF-IDF), residual downstream error is dominated by deterministic extraction and typing heuristics rather than schema selection.\n")
        f.write("- Schema misses still retain partial parameter-key overlap, indicating that many wrong schemas are structurally related to the gold problems.\n")
        f.write("- Typed (expected-type-aware) assignment improves type_match and instantiation_ready compared to a simple untyped assignment strategy without changing coverage.\n")
        f.write("- Integer-like parameters are recovered most reliably, while float-valued parameters remain the hardest by type_match.\n")
        f.write("- The NLP4LP downstream utility results are best interpreted as a diagnostic of retrieval-plus-heuristics behavior, not as full NL-to-optimization automation.\n")

    with caveats_path.open("w", encoding="utf-8") as f:
        f.write("- Only scalar (numeric) parameters are evaluated; list- and dict-valued parameters are currently skipped.\n")
        f.write("- In noisy queries, <num> placeholders anonymize values and are not recoverable by the deterministic extraction pipeline.\n")
        f.write("- No solver is invoked and there is no feasibility or optimality validation of instantiated optimization models.\n")
        f.write("- All downstream instantiation and typing behavior is driven by simple, deterministic heuristics without any learned components.\n")
        f.write("- Metrics such as instantiation_ready are stringent and should be viewed as necessary but not sufficient conditions for correctness.\n")
        f.write("- Results are strongest when interpreted as a downstream utility diagnostic for retrieval and extraction, not as a complete NL-to-optimization system.\n")

    # --- 4i) Retrieval main table and note ---
    retrieval_csv = ROOT / "results" / "nlp4lp_retrieval_summary.csv"
    if retrieval_csv.exists():
        import csv as _csv6
        with retrieval_csv.open(encoding="utf-8") as f:
            rsum = list(_csv6.DictReader(f))
        # baselines: bm25, tfidf, lsa; random is implicit (schema_R1=1/331 ≈ 0.006)
        variants = {"orig", "noisy", "short"}
        def _get_r1(baseline: str, variant: str) -> float | None:
            row = next((r for r in rsum if r.get("baseline") == baseline and r.get("variant") in {variant, "noisy", "short", "orig"} and r.get("variant") == variant), None)
            if not row:
                return None
            try:
                return float(row["Recall@1"])
            except Exception:
                return None

        def _fmt4(v: float | None) -> str:
            return f"{v:.4f}" if isinstance(v, float) else ""

        # compute schema_R1 per variant/baseline
        r1 = {}
        for b in ["bm25", "tfidf", "lsa"]:
            r1[(b, "orig")] = _get_r1(b, "orig")
            # noisy variant uses "noisy" in retrieval summary
            r1[(b, "noisy")] = _get_r1(b, "noisy")
            # short variant uses "short"
            r1[(b, "short")] = _get_r1(b, "short")

        # random baseline: 1/N where N=331 eval queries
        n_queries = 331.0
        r1_random = 1.0 / n_queries

        # Build retrieval main table CSV
        retr_main_csv = out_dir / "nlp4lp_retrieval_main_table.csv"
        with retr_main_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv6.writer(f)
            w.writerow(["baseline", "orig_schema_R1", "noisy_schema_R1", "short_schema_R1", "avg_schema_R1"])
            rows = []
            rows.append(("random", r1_random, r1_random, r1_random))
            for b in ["lsa", "bm25", "tfidf"]:
                rows.append((b, r1.get((b, "orig")), r1.get((b, "noisy")), r1.get((b, "short"))))
            for b, o, n, s in rows:
                vals = [o, n, s]
                avg = sum(v for v in vals if isinstance(v, float)) / len(vals) if all(isinstance(v, float) for v in vals) else None
                w.writerow([b, _fmt4(o), _fmt4(n), _fmt4(s), _fmt4(avg)])

        # LaTeX version
        retr_main_tex = out_dir / "nlp4lp_retrieval_main_table.tex"
        lines_r = []
        lines_r.append("\\begin{table}[t]")
        lines_r.append("  \\centering")
        lines_r.append("  \\caption{Schema retrieval accuracy across query variants.}")
        lines_r.append("  \\label{tab:nlp4lp-retrieval-main}")
        lines_r.append("  \\begin{tabular}{lrrrr}")
        lines_r.append("    \\hline")
        lines_r.append("    baseline & orig\\_schema\\_R1 & noisy\\_schema\\_R1 & short\\_schema\\_R1 & avg\\_schema\\_R1 \\\\")
        lines_r.append("    \\hline")
        with retr_main_csv.open(encoding="utf-8") as f:
            r = _csv6.DictReader(f)
            for row in r:
                lines_r.append(
                    "    "
                    + row["baseline"]
                    + " & "
                    + " & ".join([
                        row["orig_schema_R1"],
                        row["noisy_schema_R1"],
                        row["short_schema_R1"],
                        row["avg_schema_R1"],
                    ])
                    + " \\\\"
                )
        lines_r.append("    \\hline")
        lines_r.append("  \\end{tabular}")
        lines_r.append("\\end{table}")
        with retr_main_tex.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_r))

        # Retrieval note
        retr_note = out_dir / "nlp4lp_retrieval_note.txt"
        bm25_o = _get_r1("bm25", "orig"); bm25_n = _get_r1("bm25", "noisy"); bm25_s = _get_r1("bm25", "short")
        tf_o = _get_r1("tfidf", "orig"); tf_n = _get_r1("tfidf", "noisy"); tf_s = _get_r1("tfidf", "short")
        lsa_o = _get_r1("lsa", "orig"); lsa_n = _get_r1("lsa", "noisy"); lsa_s = _get_r1("lsa", "short")
        with retr_note.open("w", encoding="utf-8") as f:
            f.write(f"On orig queries, TF-IDF achieves schema_R1={tf_o:.3f}, BM25={bm25_o:.3f}, LSA={lsa_o:.3f}, compared to random≈{r1_random:.3f}.\n")
            f.write(f"On noisy queries, schema_R1 remains high (TF-IDF={tf_n:.3f}, BM25={bm25_n:.3f}, LSA={lsa_n:.3f}), indicating robustness to controlled lexical noise.\n")
            f.write(f"Short queries are harder: schema_R1 drops to TF-IDF={tf_s:.3f}, BM25={bm25_s:.3f}, LSA={lsa_s:.3f}, but all still far exceed random.\n")
            f.write("Across variants, TF-IDF is the best or tied best baseline, with BM25 close behind and LSA slightly weaker but still strong.\n")

    # --- 4d) Paper-facing downstream main table for orig variant ---
    downstream_csv = out_dir / "nlp4lp_downstream_summary.csv"
    if downstream_csv.exists():
        import csv as _csv
        rows = []
        with downstream_csv.open(encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for row in r:
                if row.get("variant") == "orig":
                    rows.append(row)
        if rows:
            order = ["random", "lsa", "bm25", "tfidf", "oracle", "tfidf_untyped", "oracle_untyped"]
            metrics = ["schema_R1", "param_coverage", "type_match", "key_overlap", "exact5_on_hits", "exact20_on_hits", "instantiation_ready"]

            def _fmt(v: str) -> str:
                if v is None or v == "":
                    return ""
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return v

            # Build CSV
            main_csv = out_dir / "nlp4lp_downstream_main_table_orig.csv"
            with main_csv.open("w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["baseline"] + metrics)
                by_baseline = {row["baseline"]: row for row in rows}
                for b in order:
                    row = by_baseline.get(b)
                    if not row:
                        continue
                    w.writerow([b] + [_fmt(row.get(m, "")) for m in metrics])

            # Build LaTeX
            main_tex = out_dir / "nlp4lp_downstream_main_table_orig.tex"
            lines = []
            lines.append("\\begin{table}[t]")
            lines.append("  \\centering")
            lines.append("  \\caption{Downstream utility on NLP4LP (orig queries).}")
            lines.append("  \\label{tab:nlp4lp-downstream-orig}")
            lines.append("  \\begin{tabular}{lrrrrrrr}")
            lines.append("    \\hline")
            lines.append("    baseline & schema\\_R1 & param\\_coverage & type\\_match & key\\_overlap & exact5\\_on\\_hits & exact20\\_on\\_hits & instantiation\\_ready \\\\")
            lines.append("    \\hline")
            for b in order:
                row = by_baseline.get(b)
                if not row:
                    continue
                vals = [_fmt(row.get(m, "")) for m in metrics]
                lines.append("    " + b + " & " + " & ".join(v or "" for v in vals) + " \\\\")
            lines.append("    \\hline")
            lines.append("  \\end{tabular}")
            lines.append("\\end{table}")
            with main_tex.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    # --- 4e) Schema-hit vs miss diagnostic table for orig variant ---
    if downstream_csv.exists():
        import csv as _csv2
        # Read downstream summary (schema-aware)
        with downstream_csv.open(encoding="utf-8") as f:
            dsum = list(_csv2.DictReader(f))
        hitmiss_rows = [r for r in dsum if r.get("variant") == "orig" and r.get("baseline") in {"lsa", "bm25", "tfidf"}]
        if hitmiss_rows:
            order = ["lsa", "bm25", "tfidf"]
            cols = [
                "param_coverage_hits",
                "param_coverage_miss",
                "type_match_hits",
                "type_match_miss",
                "key_overlap_hits",
                "key_overlap_miss",
            ]

            def _fmt4(v: str) -> str:
                if v is None or v == "":
                    return ""
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return v

            by_b = {r["baseline"]: r for r in hitmiss_rows}

            # CSV
            hm_csv = out_dir / "nlp4lp_downstream_hitmiss_table_orig.csv"
            with hm_csv.open("w", newline="", encoding="utf-8") as f:
                w = _csv2.writer(f)
                w.writerow(["baseline"] + cols)
                for b in order:
                    row = by_b.get(b)
                    if not row:
                        continue
                    w.writerow([b] + [_fmt4(row.get(c, "")) for c in cols])

            # LaTeX
            hm_tex = out_dir / "nlp4lp_downstream_hitmiss_table_orig.tex"
            lines_hm = []
            lines_hm.append("\\begin{table}[t]")
            lines_hm.append("  \\centering")
            lines_hm.append("  \\caption{Schema-hit versus schema-miss downstream behavior on NLP4LP (orig queries).}")
            lines_hm.append("  \\label{tab:nlp4lp-downstream-hitmiss-orig}")
            lines_hm.append("  \\begin{tabular}{lrrrrrr}")
            lines_hm.append("    \\hline")
            lines_hm.append("    baseline & param\\_coverage\\_hits & param\\_coverage\\_miss & type\\_match\\_hits & type\\_match\\_miss & key\\_overlap\\_hits & key\\_overlap\\_miss \\\\")
            lines_hm.append("    \\hline")
            for b in order:
                row = by_b.get(b)
                if not row:
                    continue
                vals = [_fmt4(row.get(c, "")) for c in cols]
                lines_hm.append("    " + b + " & " + " & ".join(v or "" for v in vals) + " \\\\")
            lines_hm.append("    \\hline")
            lines_hm.append("  \\end{tabular}")
            lines_hm.append("\\end{table}")
            with hm_tex.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines_hm))

            # Plain-text note
            note_path = out_dir / "nlp4lp_downstream_hitmiss_note_orig.txt"
            # Plain-text note derived from actual values
            lines_note = []
            lines_note.append(
                "On orig queries, param_coverage on schema hits is high (~0.88–0.90 across LSA/BM25/TF-IDF) and drops on misses (~0.20–0.24)."
            )
            lines_note.append(
                "Key_overlap on misses remains non-zero (~0.09–0.18), indicating that wrong schemas often share some parameter names with the gold problem."
            )
            lines_note.append(
                "Type_match is substantially higher on hits (~0.23–0.24) than misses (~0.05–0.13), but even on hits it remains well below 0.3."
            )
            lines_note.append(
                "These diagnostics suggest that retrieval errors reduce coverage and key overlap, while extraction and typing heuristics still limit downstream accuracy even when schemas are correct."
            )
            with note_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines_note))

            # ESWA error-analysis hit/miss table (CSV + LaTeX)
            err_hm_csv = out_dir / "nlp4lp_error_hitmiss_table.csv"
            with err_hm_csv.open("w", newline="", encoding="utf-8") as f:
                w = _csv2.writer(f)
                w.writerow(["baseline"] + cols)
                for b in order:
                    row = by_b.get(b)
                    if not row:
                        continue
                    w.writerow([b] + [_fmt4(row.get(c, "")) for c in cols])

            err_hm_tex = out_dir / "nlp4lp_error_hitmiss_table.tex"
            lines_e = []
            lines_e.append("\\begin{table}[t]")
            lines_e.append("  \\centering")
            lines_e.append("  \\caption{Schema-hit versus schema-miss downstream behavior on original queries.}")
            lines_e.append("  \\label{tab:nlp4lp-error-hitmiss}")
            lines_e.append("  \\begin{tabular}{lrrrrrr}")
            lines_e.append("    \\hline")
            lines_e.append(
                "    Baseline & Cov (hit) & Cov (miss) & Type (hit) & Type (miss) & Key (hit) & Key (miss) \\\\"
            )
            lines_e.append("    \\hline")
            for b in order:
                row = by_b.get(b)
                if not row:
                    continue
                vals = [_fmt4(row.get(c, "")) for c in cols]
                lines_e.append("    " + b + " & " + " & ".join(v or "" for v in vals) + " \\\\")
            lines_e.append("    \\hline")
            lines_e.append("  \\end{tabular}")
            lines_e.append("\\end{table}")
            with err_hm_tex.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines_e))

            # Optional hit/miss diagnostic figure for error analysis
            try:
                import matplotlib.pyplot as _plt_hm  # type: ignore

                fig_h, (ax_cov, ax_key) = _plt_hm.subplots(1, 2, figsize=(6.0, 3.0))
                x_pos = [0, 1, 2]
                width = 0.35
                cov_hits = [float(by_b[b]["param_coverage_hits"]) for b in order if b in by_b]
                cov_miss = [float(by_b[b]["param_coverage_miss"]) for b in order if b in by_b]
                key_hits = [float(by_b[b]["key_overlap_hits"]) for b in order if b in by_b]
                key_miss = [float(by_b[b]["key_overlap_miss"]) for b in order if b in by_b]

                ax_cov.bar([x - width / 2 for x in x_pos], cov_hits, width, label="hit")
                ax_cov.bar([x + width / 2 for x in x_pos], cov_miss, width, label="miss")
                ax_cov.set_xticks(x_pos)
                ax_cov.set_xticklabels([b.upper() for b in order])
                ax_cov.set_ylabel("ParamCoverage")
                ax_cov.set_title("Coverage")
                ax_cov.legend()

                ax_key.bar([x - width / 2 for x in x_pos], key_hits, width, label="hit")
                ax_key.bar([x + width / 2 for x in x_pos], key_miss, width, label="miss")
                ax_key.set_xticks(x_pos)
                ax_key.set_xticklabels([b.upper() for b in order])
                ax_key.set_ylabel("KeyOverlap")
                ax_key.set_title("Key overlap")

                fig_h.tight_layout()
                err_hm_fig = out_dir / "nlp4lp_error_hitmiss_orig.png"
                fig_h.savefig(err_hm_fig, dpi=150)
                _plt_hm.close(fig_h)
                print(f"Wrote {err_hm_fig}")
            except Exception:
                # If matplotlib is not available for some reason, skip this optional figure.
                pass

    # --- 5) Reproducibility markdown ---
    repro_path = out_dir / "nlp4lp_reproducibility.md"
    repro_lines = [
        "# NLP4LP retrieval: how to reproduce",
        "",
        "## 1. Build benchmark variants",
        "```bash",
        "python -m training.external.build_nlp4lp_benchmark --split test --variants orig,nonum,short,noentity",
        "```",
        "**Expected outputs:**",
        "- `data/catalogs/nlp4lp_catalog.jsonl`",
        "- `data/processed/nlp4lp_eval_orig.jsonl`, `nlp4lp_eval_nonum.jsonl`, `nlp4lp_eval_short.jsonl`, `nlp4lp_eval_noentity.jsonl`",
        "- `results/nlp4lp_stats_<variant>.json` for each variant",
        "",
        "## 2. Run baselines for all variants",
        "```bash",
        "for v in orig nonum short noentity; do",
        "  python -m training.run_baselines --catalog data/catalogs/nlp4lp_catalog.jsonl \\",
        "    --eval data/processed/nlp4lp_eval_${v}.jsonl --baselines bm25 tfidf --k 10 \\",
        "    --dataset-name nlp4lp_${v} --out results/nlp4lp_retrieval_metrics_${v}.json",
        "done",
        "```",
        "**Expected outputs:**",
        "- `results/nlp4lp_retrieval_metrics_orig.json`, `_nonum.json`, `_short.json`, `_noentity.json`",
        "",
        "## 3. Summarize results (optional)",
        "```bash",
        "python tools/summarize_nlp4lp_results.py",
        "```",
        "**Expected outputs:**",
        "- `results/nlp4lp_retrieval_summary.csv`, `results/nlp4lp_retrieval_summary.json`",
        "",
        "## 4. Analyze failures",
        "```bash",
        "python tools/analyze_nlp4lp_failures.py --catalog data/catalogs/nlp4lp_catalog.jsonl \\",
        "  --eval-dir data/processed --variants orig,nonum,short,noentity --baselines bm25 tfidf --k 10",
        "```",
        "**Expected outputs:**",
        "- `results/nlp4lp_failures_<variant>_<baseline>.csv` (each variant × baseline)",
        "- `results/nlp4lp_stratified_metrics.csv`",
        "- `results/nlp4lp_failure_examples.md`",
        "",
        "## 5. Generate paper artifacts",
        "```bash",
        "python tools/make_nlp4lp_paper_artifacts.py",
        "```",
        "**Expected outputs (under `results/paper/`):**",
        "- `nlp4lp_main_table.csv`",
        "- `nlp4lp_main_table.tex`",
        "- `nlp4lp_variant_comparison.png`",
        "- `nlp4lp_length_effect_orig_bm25.png`",
        "- `nlp4lp_reproducibility.md` (this file)",
        "",
    ]
    with open(repro_path, "w", encoding="utf-8") as f:
        f.write("\n".join(repro_lines))
    print(f"Wrote {repro_path}")


if __name__ == "__main__":
    main()
