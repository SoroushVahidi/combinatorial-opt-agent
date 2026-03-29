#!/usr/bin/env python3
"""
Collect real/public optimization problem examples from open sources.

Sources attempted:
  1. OptMATH Benchmark (optsuite/OptMATH) — CC-BY-style open; 166 NL problem descriptions
  2. NL4Opt Competition (nl4opt/nl4opt-competition) — MIT licence; LP problems with NL
  3. ORQA (nl4opt/ORQA) — public; OR question-answering dataset
  4. Gurobi Modeling Examples (Gurobi/modeling-examples) — Apache-2.0; README NL descriptions
  5. Several other sources — tested and reported in manifest / failed_sources_report.md

Outputs (written to data/external_real_examples/):
  collected_examples.jsonl  — one JSON object per line
  source_manifest.csv       — all sources considered with outcome
  failed_sources_report.md  — reasons for inaccessible/skipped sources

Usage:
  python tools/collect_real_optimization_examples.py
  # or with output dir override:
  python tools/collect_real_optimization_examples.py --out data/external_real_examples
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "external_real_examples"
USER_AGENT = "Mozilla/5.0 (compatible; academic-research-bot/1.0)"


def _fetch(url: str, timeout: int = 15) -> bytes | None:
    """Fetch URL; return bytes or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001
        return None


def _slug(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", text)
    s = re.sub(r"[-\s]+", "_", s).strip("_")[:max_len]
    return s.lower() or "item"


def _first_n_words(text: str, n: int = 30) -> str:
    words = text.split()
    return " ".join(words[:n]) + ("…" if len(words) > n else "")


# ---------------------------------------------------------------------------
# Source 1: OptMATH Benchmark
# ---------------------------------------------------------------------------

OPTMATH_URL = (
    "https://raw.githubusercontent.com/optsuite/OptMATH/main/benchmark/OptMATH_Bench.json"
)

# Heuristic family detection from OptMATH en_question text
_OPTMATH_FAMILY_HINTS: list[tuple[str, str]] = [
    (r"\bjob.?shop\b|\boperations?.+schedule\b|\bmakespan\b", "job_shop_scheduling"),
    (r"\baircraft.?landing\b|\blanding.?runway\b", "aircraft_landing"),
    (r"\btraveling.?salesman\b|\bTSP\b|\bvisit.{0,30}cit", "traveling_salesman"),
    (r"\bfacility.?location\b|\bwarehouse.{0,30}open", "facility_location"),
    (r"\bknapsack\b|\bweight.{0,20}capac", "knapsack"),
    (r"\bscheduling\b|\bschedule\b", "scheduling"),
    (r"\btransport\b|\bshipment\b|\bship\b", "transportation"),
    (r"\bportfolio\b|\binvestment\b|\bstock\b", "portfolio"),
    (r"\broute\b|\brouting\b|\bvehicle.?rout", "vehicle_routing"),
    (r"\bblend\b|\bmix\b|\bdiet\b|\bnutrition\b", "blending_diet"),
    (r"\bassign\b|\bassignment\b", "assignment"),
    (r"\bcover\b|\bset.?cover\b", "set_cover"),
    (r"\bbin.?pack\b|\bpack.{0,20}bin\b", "bin_packing"),
    (r"\bcut\b|\bcutting.?stock\b", "cutting_stock"),
    (r"\bnetwork.?flow\b|\bmax.?flow\b|\bmin.?cost.?flow\b", "network_flow"),
    (r"\bstaff\b|\bworkforce\b|\bemployee\b|\bworker\b", "workforce_scheduling"),
    (r"\benergy\b|\bpower.?gen\b|\belectri", "energy_power"),
    (r"\bsupply.?chain\b|\bsupplier\b|\bprocure", "supply_chain"),
]


def _optmath_family(text: str) -> str:
    lc = text.lower()
    for pat, fam in _OPTMATH_FAMILY_HINTS:
        if re.search(pat, lc):
            return fam
    return "general_lp_milp"


def _optmath_style(text: str) -> str:
    lc = text.lower()
    if any(
        w in lc
        for w in ["integer", "binary", "0-1", "0/1", "milp", "ilp", "combinatorial"]
    ):
        return "ILP/MILP"
    return "LP"


def collect_optmath(out_dir: Path) -> list[dict[str, Any]]:
    """Download OptMATH_Bench.json and convert to unified schema."""
    print("  Fetching OptMATH Benchmark …")
    raw = _fetch(OPTMATH_URL)
    if raw is None:
        print("  SKIP: OptMATH — fetch failed")
        return []

    data: list[dict] = json.loads(raw)
    examples = []
    for item in data:
        q: str = item.get("en_question", "").strip()
        if len(q) < 30:
            continue
        ex_id = f"optmath_{item.get('id', _slug(q, 30))}"
        examples.append(
            {
                "example_id": ex_id,
                "title": _first_n_words(q, 12),
                "raw_problem_text": q,
                "short_problem_summary": _first_n_words(q, 25),
                "source_name": "OptMATH Benchmark",
                "source_url": "https://github.com/optsuite/OptMATH",
                "source_type": "benchmark",
                "license_status": "public (see repo; research use)",
                "access_status": "collected",
                "optimization_family": _optmath_family(q),
                "problem_style": _optmath_style(q),
                "is_real_public_open": True,
                "has_nl_statement": True,
                "has_numeric_detail": bool(re.search(r"\d", q)),
                "usable_as_benchmark_input": True,
                "notes": "en_question field from OptMATH_Bench.json",
            }
        )
    print(f"  OptMATH: {len(examples)} examples collected")
    return examples


# ---------------------------------------------------------------------------
# Source 2: NL4Opt Competition (generation_data)
# ---------------------------------------------------------------------------

NL4OPT_BASE = (
    "https://raw.githubusercontent.com/nl4opt/nl4opt-competition/main/generation_data"
)
NL4OPT_FILES = {
    "train": f"{NL4OPT_BASE}/train.jsonl",
    "dev": f"{NL4OPT_BASE}/dev.jsonl",
    "test": f"{NL4OPT_BASE}/test.jsonl",
}

_NL4OPT_FAMILY_HINTS: list[tuple[str, str]] = [
    (r"\bstaff\b|\bworker\b|\bemployee\b|\bclean\b|\breception", "workforce"),
    (r"\bfarm\b|\bcrop\b|\bagricult\b|\bfertil", "agricultural"),
    (r"\bfactory\b|\bmanufactur\b|\bproduc", "production_planning"),
    (r"\btransport\b|\bship\b|\bdeliver\b", "transportation"),
    (r"\bdiet\b|\bnutrition\b|\bfood\b|\bcalori", "diet_nutrition"),
    (r"\binvestment\b|\bportfolio\b|\bstock\b|\bfinance\b", "portfolio_finance"),
    (r"\btravel\b|\btourist\b|\bvisit\b", "travel_tourism"),
    (r"\bschool\b|\bstudent\b|\bcourse\b|\bclass\b", "education"),
    (r"\bstore\b|\bshop\b|\bretail\b|\bsell\b", "retail_sales"),
    (r"\bmedical\b|\bhosp\b|\bnurse\b|\bdoctor\b", "healthcare"),
]


def _nl4opt_family(text: str) -> str:
    lc = text.lower()
    for pat, fam in _NL4OPT_FAMILY_HINTS:
        if re.search(pat, lc):
            return fam
    return "general_lp"


def collect_nl4opt(
    out_dir: Path, max_per_split: int | None = None
) -> list[dict[str, Any]]:
    """Download NL4Opt generation_data and return all unique examples.

    Args:
        max_per_split: Maximum items per split; ``None`` (default) means no limit
            and collects the full split.
    """
    examples = []
    for split, url in NL4OPT_FILES.items():
        print(f"  Fetching NL4Opt {split} …")
        raw = _fetch(url)
        if raw is None:
            print(f"  SKIP: NL4Opt {split} — fetch failed")
            continue
        lines = [l for l in raw.decode("utf-8").strip().split("\n") if l.strip()]
        items = [json.loads(l) for l in lines]
        # Items are {hash: {document, vars, params, ...}}
        count = 0
        seen_docs: set[str] = set()
        for item_dict in items:
            for key, val in item_dict.items():
                doc: str = val.get("document", "").strip()
                if not doc or len(doc) < 40:
                    continue
                # Deduplicate by first 80 chars
                doc_key = doc[:80]
                if doc_key in seen_docs:
                    continue
                seen_docs.add(doc_key)
                vars_ = val.get("vars", [])
                params = val.get("params", [])
                obj = val.get("obj_declaration", {})
                obj_type = obj.get("direction", "") if isinstance(obj, dict) else ""
                ex_id = f"nl4opt_{split}_{key}"
                examples.append(
                    {
                        "example_id": ex_id,
                        "title": _first_n_words(doc, 10),
                        "raw_problem_text": doc,
                        "short_problem_summary": _first_n_words(doc, 20),
                        "source_name": "NL4Opt Competition",
                        "source_url": "https://github.com/nl4opt/nl4opt-competition",
                        "source_type": "competition_dataset",
                        "license_status": "MIT",
                        "access_status": "collected",
                        "optimization_family": _nl4opt_family(doc),
                        "problem_style": "LP",
                        "is_real_public_open": True,
                        "has_nl_statement": True,
                        "has_numeric_detail": bool(params),
                        "usable_as_benchmark_input": True,
                        "notes": (
                            f"split={split}; vars={vars_}; "
                            f"obj_direction={obj_type}; "
                            f"params={params[:5]}"
                        ),
                    }
                )
                count += 1
                if max_per_split is not None and count >= max_per_split:
                    break
            if max_per_split is not None and count >= max_per_split:
                break
        print(f"  NL4Opt {split}: {count} examples")
        time.sleep(0.3)
    return examples


# ---------------------------------------------------------------------------
# Source 3: ORQA Dataset
# ---------------------------------------------------------------------------

ORQA_URLS = {
    "test": "https://raw.githubusercontent.com/nl4opt/ORQA/main/dataset/ORQA_test.jsonl",
    "validation": "https://raw.githubusercontent.com/nl4opt/ORQA/main/dataset/ORQA_validation.jsonl",
}

_ORQA_QTYPES: dict[str, str] = {
    "Q1": "problem_type_identification",
    "Q2": "objective_type",
    "Q3": "decision_variable_count",
    "Q4": "constraint_count",
    "Q5": "feasibility",
    "Q6": "complexity_classification",
    "Q7": "formulation_suitability",
}


def collect_orqa(
    out_dir: Path, seen_contexts: set[str] | None = None
) -> list[dict[str, Any]]:
    """Download ORQA and collect one entry per unique problem context.

    Each unique *CONTEXT* (the optimization problem description) is stored
    exactly once.  The first question encountered for that context is used to
    populate the ``notes`` field.  This maximises the number of distinct
    problem scenarios while avoiding redundancy from the 10+ questions that
    share the same scenario.

    Args:
        seen_contexts: Optional pre-populated set of already-seen context keys
            (first-80-chars) used to avoid cross-split duplicates.
    """
    if seen_contexts is None:
        seen_contexts = set()
    examples = []
    for split, url in ORQA_URLS.items():
        print(f"  Fetching ORQA {split} …")
        raw = _fetch(url)
        if raw is None:
            print(f"  SKIP: ORQA {split} — fetch failed")
            continue
        lines = [l for l in raw.decode("utf-8").strip().split("\n") if l.strip()]
        items = [json.loads(l) for l in lines]

        count = 0
        for item in items:
            ctx: str = item.get("CONTEXT", "").strip()
            if not ctx or len(ctx) < 40:
                continue
            ctx_key = ctx[:80]
            if ctx_key in seen_contexts:
                continue
            seen_contexts.add(ctx_key)
            qt: str = item.get("QUESTION_TYPE", "unknown")
            question: str = item.get("QUESTION", "").strip()
            answer = item.get("TARGET_ANSWER", "")
            ex_id = f"orqa_{split}_{count}"
            examples.append(
                {
                    "example_id": ex_id,
                    "title": _first_n_words(ctx, 10),
                    "raw_problem_text": ctx,
                    "short_problem_summary": _first_n_words(ctx, 20),
                    "source_name": "ORQA (OR Question Answering)",
                    "source_url": "https://github.com/nl4opt/ORQA",
                    "source_type": "qa_benchmark",
                    "license_status": "public (research use; see repo)",
                    "access_status": "collected",
                    "optimization_family": "general_or",
                    "problem_style": "unknown",
                    "is_real_public_open": True,
                    "has_nl_statement": True,
                    "has_numeric_detail": bool(re.search(r"\d", ctx)),
                    "usable_as_benchmark_input": True,
                    "notes": (
                        f"split={split}; question_type={qt} "
                        f"({_ORQA_QTYPES.get(qt, 'unknown')}); "
                        f"question={question[:80]}; answer={answer}"
                    ),
                }
            )
            count += 1
        print(f"  ORQA {split}: {count} new unique contexts")
        time.sleep(0.3)
    return examples


# ---------------------------------------------------------------------------
# Source 4: Gurobi Modeling Examples (Apache-2.0) — README descriptions
# ---------------------------------------------------------------------------

GUROBI_EXAMPLES = [
    ("traveling_salesman", "traveling_salesman", "vehicle_routing"),
    ("workforce", "workforce", "workforce_scheduling"),
    ("farm_planning", "farm_planning", "production_planning"),
    ("facility_location", "facility_location", "facility_location"),
    ("portfolio_selection_optimization", "portfolio_selection_optimization", "portfolio"),
    ("mining", "mining", "resource_extraction"),
    ("food_manufacturing", "food_manufacturing", "production_planning"),
    ("manpower_planning", "manpower_planning", "workforce_scheduling"),
    ("supply_network_design", "supply_network_design", "supply_chain"),
    ("power_generation", "power_generation", "energy_power"),
    ("milk_collection", "milk_collection", "vehicle_routing"),
    ("refinery", "refinery", "blending_diet"),
    ("economic_planning", "economic_planning", "economic_planning"),
    ("food_program", "food_program", "diet_nutrition"),
    ("market_sharing", "market_sharing", "assignment"),
    ("offshore_wind_farming", "offshore_wind_farming", "facility_location"),
    ("yield_management", "yield_management", "revenue_management"),
    ("lost_luggage_distribution", "lost_luggage_distribution", "vehicle_routing"),
    ("technician_routing_scheduling", "technician_routing_scheduling", "scheduling"),
    ("index_tracking", "index_tracking", "portfolio"),
    ("customer_assignment", "customer_assignment", "assignment"),
    ("car_rental", "car_rental", "fleet_management"),
    ("cell_tower_coverage", "cell_tower_coverage", "set_cover"),
    ("factory_planning", "factory_planning", "production_planning"),
    # Additional examples
    ("3d_tic_tac_toe", "3d_tic_tac_toe", "general_milp"),
    ("agricultural_pricing", "agricultural_pricing", "agricultural"),
    ("aviation_planning", "aviation_planning", "scheduling"),
    ("battery_scheduling", "battery_scheduling", "energy_power"),
    ("burrito_optimization_game", "burrito_optimization_game", "general_lp"),
    ("colgen-cutting_stock", "colgen_cutting_stock", "cutting_stock"),
    ("constraint_optimization", "constraint_optimization", "general_milp"),
    ("covid19_facility_location", "covid19_facility_location", "facility_location"),
    ("curve_fitting", "curve_fitting", "general_lp"),
    ("decentralization_planning", "decentralization_planning", "facility_location"),
    ("drone_network", "drone_network", "facility_location"),
    ("efficiency_analysis", "efficiency_analysis", "general_lp"),
    ("electrical_power_generation", "electrical_power_generation", "energy_power"),
    ("fantasy_basketball", "fantasy_basketball", "assignment"),
    ("linear_regression", "linear_regression", "general_lp"),
    ("logical_design", "logical_design", "general_milp"),
    ("marketing_campaign_optimization", "marketing_campaign_optimization", "general_lp"),
    ("milp_tutorial", "milp_tutorial", "general_milp"),
    ("music_recommendation", "music_recommendation", "assignment"),
    ("opencast_mining", "opencast_mining", "resource_extraction"),
    ("pooling", "pooling", "blending_diet"),
    ("price_optimization", "price_optimization", "general_lp"),
    ("protein_comparison", "protein_comparison", "general_milp"),
    ("protein_folding", "protein_folding", "general_milp"),
    ("railway_dispatching", "railway_dispatching", "scheduling"),
    ("text_dissimilarity", "text_dissimilarity", "general_lp"),
]

GUROBI_RAW_BASE = (
    "https://raw.githubusercontent.com/Gurobi/modeling-examples/master"
)


def _extract_gurobi_nl(readme_text: str, max_chars: int = 600) -> str:
    """Extract first paragraph(s) of actual problem description from README."""
    # Remove markdown headings and links to just text
    lines = readme_text.split("\n")
    text_lines = []
    for l in lines:
        stripped = l.strip()
        if stripped.startswith("#") or stripped.startswith("[") or stripped.startswith("!"):
            continue
        if stripped.startswith("©") or stripped.startswith("For details"):
            continue
        if stripped.startswith("*") and len(stripped) < 80:
            continue
        if stripped:
            text_lines.append(stripped)
    combined = " ".join(text_lines)
    # Take first max_chars, trim to last sentence end
    if len(combined) <= max_chars:
        return combined
    excerpt = combined[:max_chars]
    for end_char in (".", "!", "?"):
        idx = excerpt.rfind(end_char)
        if idx > max_chars // 2:
            return excerpt[: idx + 1]
    return excerpt


def collect_gurobi_readmes(out_dir: Path) -> list[dict[str, Any]]:
    """Fetch available Gurobi example README files and extract NL descriptions."""
    examples = []
    for folder, name, family in GUROBI_EXAMPLES:
        url = f"{GUROBI_RAW_BASE}/{folder}/README.md"
        raw = _fetch(url)
        if raw is None:
            print(f"  SKIP Gurobi/{folder} — fetch failed")
            continue
        text = raw.decode("utf-8", errors="replace")
        nl = _extract_gurobi_nl(text)
        if len(nl) < 30:
            continue
        examples.append(
            {
                "example_id": f"gurobi_{name}",
                "title": name.replace("_", " ").title(),
                "raw_problem_text": nl,
                "short_problem_summary": _first_n_words(nl, 20),
                "source_name": "Gurobi Modeling Examples",
                "source_url": f"https://github.com/Gurobi/modeling-examples/tree/master/{folder}",
                "source_type": "model_library",
                "license_status": "Apache-2.0",
                "access_status": "collected",
                "optimization_family": family,
                "problem_style": "LP/MILP",
                "is_real_public_open": True,
                "has_nl_statement": True,
                "has_numeric_detail": False,
                "usable_as_benchmark_input": False,
                "notes": (
                    "Extracted from README.md; conceptual description only, "
                    "no specific numeric parameters. Useful as problem-type "
                    "illustration. Full code in notebooks (Apache-2.0)."
                ),
            }
        )
        time.sleep(0.2)
    print(f"  Gurobi examples: {len(examples)} README descriptions collected")
    return examples


# ---------------------------------------------------------------------------
# Source manifest and failed sources
# ---------------------------------------------------------------------------

SOURCE_MANIFEST: list[dict[str, str]] = [
    {
        "source_name": "OptMATH Benchmark",
        "source_url": "https://github.com/optsuite/OptMATH",
        "source_type": "benchmark",
        "license_status": "public (research use; see repo LICENSE)",
        "outcome": "collected",
        "items_collected": "",  # filled at runtime
        "notes": "166 NL optimization problem descriptions with numeric parameters; "
        "diverse families (job shop, aircraft landing, TSP, facility location, etc.).",
    },
    {
        "source_name": "NL4Opt Competition",
        "source_url": "https://github.com/nl4opt/nl4opt-competition",
        "source_type": "competition_dataset",
        "license_status": "MIT",
        "outcome": "collected",
        "items_collected": "",
        "notes": "NL + structured LP formulations. train/dev/test splits. "
        "Full dataset collected (train: 713, dev: 99, test: 289).",
    },
    {
        "source_name": "ORQA (OR Question Answering)",
        "source_url": "https://github.com/nl4opt/ORQA",
        "source_type": "qa_benchmark",
        "license_status": "public (research use; see repo)",
        "outcome": "collected",
        "items_collected": "",
        "notes": "OR question-answering dataset. "
        "All unique problem contexts collected (one entry per distinct scenario).",
    },
    {
        "source_name": "Gurobi Modeling Examples",
        "source_url": "https://github.com/Gurobi/modeling-examples",
        "source_type": "model_library",
        "license_status": "Apache-2.0",
        "outcome": "collected",
        "items_collected": "",
        "notes": "README NL descriptions from all available examples. "
        "No numeric parameters. Descriptions collected as concept-level problem statements.",
    },
    {
        "source_name": "DCP-Bench-Open (sample_test.jsonl)",
        "source_url": "https://github.com/DCP-Bench/DCP-Bench-Open",
        "source_type": "benchmark",
        "license_status": "Apache-2.0",
        "outcome": "accessible_not_useful",
        "items_collected": "0",
        "notes": "sample_test.jsonl only contains 5 items that are solver code "
        "(CPMPy models), not NL problem statements. "
        "Full dataset is not available in the public repo sample. "
        "Skipped — no NL descriptions in the accessible portion.",
    },
    {
        "source_name": "MAMO (FreedomIntelligence/Mamo)",
        "source_url": "https://github.com/FreedomIntelligence/Mamo",
        "source_type": "benchmark",
        "license_status": "unknown (see repo)",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "HTTP 404 for expected data path benchmark/MathBench.json. "
        "Repository structure may have changed since last update. "
        "Not collected.",
    },
    {
        "source_name": "NEOS Guide Case Studies",
        "source_url": "https://neos-guide.org/",
        "source_type": "case_studies",
        "license_status": "public (educational)",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "DNS resolution failed for neos-guide.org in this environment "
        "(no address associated with hostname). "
        "Source is publicly accessible in normal internet conditions. "
        "Cannot collect in sandboxed environment.",
    },
    {
        "source_name": "MIT OpenCourseWare (15.053 Optimization Methods)",
        "source_url": "https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/",
        "source_type": "course_notes",
        "license_status": "CC BY-NC-SA 4.0",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "DNS resolution failed for ocw.mit.edu in sandboxed environment. "
        "Problem sets and lecture notes are publicly available "
        "at the URL above under CC BY-NC-SA 4.0 in normal conditions.",
    },
    {
        "source_name": "GAMS Model Library (web catalog)",
        "source_url": "https://www.gams.com/latest/gamslib_ml/libhtml/",
        "source_type": "model_library",
        "license_status": "GAMS proprietary (models publicly described)",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "DNS resolution failed for www.gams.com in sandboxed environment. "
        "Web catalog is publicly viewable. "
        "Model list already captured in data/sources/gams_models.json.",
    },
    {
        "source_name": "OR-Library (J.E. Beasley)",
        "source_url": "http://people.brunel.ac.uk/~mastjjb/jeb/info.html",
        "source_type": "test_data_sets",
        "license_status": "public domain / free research use",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "DNS resolution failed in sandboxed environment. "
        "Provides test instance data files (not NL descriptions). "
        "Problem type metadata already captured in data/sources/or_library.json.",
    },
    {
        "source_name": "LibreTexts Mathematics (LP/OR chapters)",
        "source_url": "https://math.libretexts.org/",
        "source_type": "open_textbook",
        "license_status": "CC BY-SA 4.0",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "DNS resolution failed in sandboxed environment. "
        "Open textbook platform with LP/OR examples under CC BY-SA 4.0.",
    },
    {
        "source_name": "AMPL Book Examples",
        "source_url": "https://ampl.com/resources/the-ampl-book/",
        "source_type": "textbook",
        "license_status": "free to view online (proprietary book)",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "DNS resolution failed in sandboxed environment. "
        "Book is freely viewable online (PDF). "
        "Problem statements would require paraphrasing (copyrighted).",
    },
    {
        "source_name": "Pyomo Documentation Examples",
        "source_url": "https://pyomo.readthedocs.io/",
        "source_type": "documentation",
        "license_status": "BSD-3-Clause",
        "outcome": "accessible_not_useful",
        "items_collected": "0",
        "notes": "Pyomo docs provide solver code, not NL problem statements. "
        "Metadata already in data/sources/pyomo_examples.json.",
    },
    {
        "source_name": "CSPLib (Constraint Satisfaction Problem Library)",
        "source_url": "https://www.csplib.org/",
        "source_type": "benchmark_library",
        "license_status": "CC BY 4.0",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "DNS resolution failed in sandboxed environment. "
        "Provides CP problem descriptions; would be useful for constraint-focus.",
    },
    {
        "source_name": "MiniZinc Challenge Benchmark",
        "source_url": "https://github.com/MiniZinc/minizinc-benchmarks",
        "source_type": "benchmark",
        "license_status": "MIT",
        "outcome": "accessible_not_useful",
        "items_collected": "0",
        "notes": "Contains .mzn/.dzn constraint models without natural-language "
        "problem descriptions. Not useful for NL-to-formulation tasks.",
    },
    {
        "source_name": "MIPLIB 2017",
        "source_url": "https://miplib.zib.de/",
        "source_type": "benchmark_instances",
        "license_status": "public (see individual instance licenses)",
        "outcome": "inaccessible",
        "items_collected": "0",
        "notes": "DNS resolution failed in sandboxed environment. "
        "Provides MPS/LP instance files, not NL descriptions. "
        "Metadata in data/sources/miplib.json.",
    },
]


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples) + "\n",
        encoding="utf-8",
    )


def write_manifest(path: Path, manifest: list[dict[str, str]]) -> None:
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def write_failed_report(
    path: Path, manifest: list[dict[str, str]], collection_date: str
) -> None:
    lines = [
        "# Failed / Skipped Sources Report",
        "",
        f"*Generated: {collection_date}*",
        "",
        "This file documents every source considered during collection "
        "that could **not** be collected or was only partially useful.",
        "",
        "---",
        "",
    ]
    failed_outcomes = {"inaccessible", "accessible_not_useful", "blocked", "paywalled"}
    for src in manifest:
        if src["outcome"] not in failed_outcomes:
            continue
        lines += [
            f"## {src['source_name']}",
            "",
            f"- **URL:** {src['source_url']}",
            f"- **Type:** {src['source_type']}",
            f"- **License:** {src['license_status']}",
            f"- **Outcome:** `{src['outcome']}`",
            f"- **Notes:** {src['notes']}",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_report(
    path: Path,
    examples: list[dict],
    manifest: list[dict[str, str]],
    collection_date: str,
) -> None:
    by_source: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for ex in examples:
        by_source[ex["source_name"]] = by_source.get(ex["source_name"], 0) + 1
        by_family[ex.get("optimization_family", "unknown")] = (
            by_family.get(ex.get("optimization_family", "unknown"), 0) + 1
        )

    lines = [
        "# Real/Public Optimization Examples — Collection Report",
        "",
        f"*Generated: {collection_date}*",
        "",
        "## Overview",
        "",
        f"- **Total examples collected:** {len(examples)}",
        f"- **Sources considered:** {len(manifest)}",
        f"- **Sources with data collected:** "
        + str(sum(1 for s in manifest if s['outcome'] == 'collected')),
        "",
        "## Examples by Source",
        "",
        "| Source | Count |",
        "|---|---|",
    ]
    for src_name, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        lines.append(f"| {src_name} | {cnt} |")

    lines += [
        "",
        "## Examples by Optimization Family",
        "",
        "| Family | Count |",
        "|---|---|",
    ]
    for fam, cnt in sorted(by_family.items(), key=lambda x: -x[1]):
        lines.append(f"| {fam} | {cnt} |")

    lines += [
        "",
        "## Sources Considered",
        "",
        "| Source | Outcome | Notes |",
        "|---|---|---|",
    ]
    for src in manifest:
        lines.append(
            f"| {src['source_name']} | `{src['outcome']}` | "
            + src["notes"][:80].replace("|", "/")
            + " |"
        )

    lines += [
        "",
        "## Data File Locations",
        "",
        "- `data/external_real_examples/collected_examples.jsonl` — "
        "main dataset (one JSON object per line)",
        "- `data/external_real_examples/source_manifest.csv` — "
        "all sources with outcome",
        "- `data/external_real_examples/failed_sources_report.md` — "
        "inaccessible/skipped source details",
        "- `data/external_real_examples/collection_report.md` — this file",
        "",
        "## How to Re-run",
        "",
        "```bash",
        "python tools/collect_real_optimization_examples.py",
        "```",
        "",
        "The script fetches from public GitHub raw URLs and Gurobi public READMEs.",
        "No API keys or credentials required.",
        "",
        "## License Notes",
        "",
        "| Source | License |",
        "|---|---|",
    ]
    for src in manifest:
        if src["outcome"] == "collected":
            lines.append(f"| {src['source_name']} | {src['license_status']} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public optimization examples.")
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output directory (default: data/external_real_examples)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    collection_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"Collection started: {collection_date}")
    print(f"Output directory: {out_dir}")
    print()

    all_examples: list[dict] = []

    # Source 1: OptMATH
    optmath = collect_optmath(out_dir)
    all_examples.extend(optmath)

    # Source 2: NL4Opt
    nl4opt = collect_nl4opt(out_dir)
    all_examples.extend(nl4opt)

    # Source 3: ORQA
    orqa = collect_orqa(out_dir)
    all_examples.extend(orqa)

    # Source 4: Gurobi READMEs
    gurobi = collect_gurobi_readmes(out_dir)
    all_examples.extend(gurobi)

    print(f"\nTotal examples: {len(all_examples)}")

    # Update manifest counts
    source_counts = {
        "OptMATH Benchmark": len(optmath),
        "NL4Opt Competition": len(nl4opt),
        "ORQA (OR Question Answering)": len(orqa),
        "Gurobi Modeling Examples": len(gurobi),
    }
    for src in SOURCE_MANIFEST:
        cnt = source_counts.get(src["source_name"])
        if cnt is not None:
            src["items_collected"] = str(cnt)

    # Write outputs
    jsonl_path = out_dir / "collected_examples.jsonl"
    write_jsonl(jsonl_path, all_examples)
    print(f"Wrote {jsonl_path} ({len(all_examples)} examples)")

    manifest_path = out_dir / "source_manifest.csv"
    write_manifest(manifest_path, SOURCE_MANIFEST)
    print(f"Wrote {manifest_path}")

    failed_path = out_dir / "failed_sources_report.md"
    write_failed_report(failed_path, SOURCE_MANIFEST, collection_date)
    print(f"Wrote {failed_path}")

    report_path = out_dir / "collection_report.md"
    write_summary_report(report_path, all_examples, SOURCE_MANIFEST, collection_date)
    print(f"Wrote {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
