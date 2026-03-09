#!/usr/bin/env python3
"""Build common learning corpus JSONL from NLP4LP, NL4Opt, TAT-QA, FinQA."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Repo root: this file is src/learning/build_common_grounding_corpus.py
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.learning.common_corpus_schema import (
    DATASET_NAMES,
    SPLIT_NAMES,
    mention_to_dict,
    slot_to_dict,
    validate_record,
)


# ----- Paths (local datasets) -----
def _paths():
    return {
        "nlp4lp_eval_orig": ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl",
        "nlp4lp_catalog": ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl",
        "nl4opt_train": ROOT / "data_external" / "raw" / "nl4opt_competition" / "generation_data" / "train.jsonl",
        "nl4opt_dev": ROOT / "data_external" / "raw" / "nl4opt_competition" / "generation_data" / "dev.jsonl",
        "nl4opt_test": ROOT / "data_external" / "raw" / "nl4opt_competition" / "generation_data" / "test.jsonl",
        "tatqa_train": ROOT / "data_external" / "raw" / "tatqa" / "dataset_raw" / "tatqa_dataset_train.json",
        "tatqa_dev": ROOT / "data_external" / "raw" / "tatqa" / "dataset_raw" / "tatqa_dataset_dev.json",
        "tatqa_test": ROOT / "data_external" / "raw" / "tatqa" / "dataset_raw" / "tatqa_dataset_test.json",
        "finqa_train": ROOT / "data_external" / "raw" / "finqa" / "dataset" / "train.json",
        "finqa_dev": ROOT / "data_external" / "raw" / "finqa" / "dataset" / "dev.json",
        "finqa_test": ROOT / "data_external" / "raw" / "finqa" / "dataset" / "test.json",
    }


# ----- Number extraction (char offsets) -----
NUM_PATTERN = re.compile(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?|\d+\.\d+")


def _extract_numeric_mentions_from_text(
    text: str,
    prefix_id: str = "m",
) -> list[tuple[int, int, str, float | None, str]]:
    """Find all number-like spans; return (char_start, char_end, surface, value, type_bucket)."""
    results = []
    for m in NUM_PATTERN.finditer(text):
        start, end = m.start(), m.end()
        surface = text[start:end]
        value = None
        try:
            num_str = surface.replace("$", "").replace("%", "").replace(",", "")
            value = float(num_str)
            if "%" in surface or (surface.replace(".", "").replace(",", "").isdigit() and value > 1 and "percent" in text[max(0, start - 80) : end + 80].lower()):
                value = value / 100.0 if value > 1 else value
        except Exception:
            pass
        if "%" in surface:
            type_bucket = "percent"
            if value is not None and value > 1:
                value = value / 100.0
        elif "$" in surface or (value is not None and abs(value) >= 1000):
            type_bucket = "currency"
        elif value is not None and float(int(value)) == value:
            type_bucket = "int"
        else:
            type_bucket = "float"
        results.append((start, end, surface, value, type_bucket))
    return results


def _operator_cues_near(text: str, start: int, end: int, window: int = 60) -> list[str]:
    snippet = text[max(0, start - window) : end + window].lower()
    cues = []
    if re.search(r"\b(at least|minimum|min\.?|least)\b", snippet):
        cues.append("min")
    if re.search(r"\b(at most|maximum|max\.?|most|below|under)\b", snippet):
        cues.append("max")
    return cues


# ----- NLP4LP -----
def _expected_type_from_slot_name(name: str) -> str:
    n = (name or "").lower()
    if "percent" in n or "ratio" in n or "percentage" in n:
        return "percent"
    if any(s in n for s in ("budget", "cost", "price", "revenue", "profit", "investment", "dollar", "wage")):
        return "currency"
    return "float"


def _build_nlp4lp(
    split: str,
    output_dir: Path,
    max_examples: int | None,
    seed: int,
    verbose: bool,
) -> int:
    out_path = output_dir / f"nlp4lp_{split}.jsonl"
    out_path.unlink(missing_ok=True)
    paths = _paths()
    if split == "test":
        eval_path = paths["nlp4lp_eval_orig"]
        if not eval_path.exists():
            if verbose:
                print(f"NLP4LP test: eval not found {eval_path}", file=sys.stderr)
            return 0
        eval_items = []
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                eval_items.append(json.loads(line))
        gold_by_id = _load_nlp4lp_gold("test")
        catalog_text = {}
        if paths["nlp4lp_catalog"].exists():
            with open(paths["nlp4lp_catalog"], encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        catalog_text[obj["doc_id"]] = obj.get("text") or ""
        rng = __import__("random").Random(seed)
        if max_examples and len(eval_items) > max_examples:
            eval_items = rng.sample(eval_items, max_examples)
        count = 0
        for item in eval_items:
            query_id = item.get("query_id", "")
            query = (item.get("query") or "").strip()
            doc_id = item.get("relevant_doc_id", "")
            gold = gold_by_id.get(doc_id) or {}
            params = gold.get("parameters") or {}
            pinfo = gold.get("problem_info") or {}
            if isinstance(pinfo, str):
                try:
                    pinfo = json.loads(pinfo)
                except Exception:
                    pinfo = {}
            expected_params = list((pinfo.get("parameters") or params or {}).keys())
            scalar_params = [p for p in expected_params if _is_scalar(params.get(p))]
            if not scalar_params:
                continue
            mentions_with_pos = _extract_numeric_mentions_from_text(query)
            numeric_mentions = []
            for i, (cstart, cend, surface, value, type_bucket) in enumerate(mentions_with_pos):
                ctx_start = max(0, cstart - 50)
                ctx_end = min(len(query), cend + 50)
                local_ctx = query[ctx_start:ctx_end]
                cues = _operator_cues_near(query, cstart, cend)
                numeric_mentions.append(
                    mention_to_dict(
                        mention_id=f"m{i}",
                        surface=surface,
                        normalized_value=value,
                        type_bucket=type_bucket,
                        char_start=cstart,
                        char_end=cend,
                        local_context=local_ctx,
                        operator_cues=cues,
                    )
                )
            slots = []
            for p in scalar_params:
                et = _expected_type_from_slot_name(p)
                slot_role = "lower_bound" if "min" in p.lower() or "least" in p.lower() else ("upper_bound" if "max" in p.lower() or "most" in p.lower() else None)
                slots.append(slot_to_dict(slot_id=p, slot_name=p, slot_text=None, slot_role=slot_role, expected_type=et, variable_entity=None))
            gold_assignments = {}
            for slot_id in scalar_params:
                gval = params.get(slot_id)
                if not _is_scalar(gval):
                    gold_assignments[slot_id] = None
                    continue
                match_id = None
                for m in numeric_mentions:
                    mv = m.get("normalized_value")
                    if mv is not None and gval is not None:
                        if abs(float(mv) - float(gval)) < 1e-9:
                            match_id = m["mention_id"]
                            break
                gold_assignments[slot_id] = match_id
            bound_labels = {p: ("lower" if "min" in p.lower() or "least" in p.lower() else ("upper" if "max" in p.lower() or "most" in p.lower() else None)) for p in scalar_params}
            rec = {
                "dataset": "nlp4lp",
                "split": split,
                "instance_id": query_id,
                "source_path": str(eval_path),
                "problem_text": query,
                "schema_name": doc_id,
                "schema_description": catalog_text.get(doc_id),
                "slots": slots,
                "numeric_mentions": numeric_mentions,
                "gold_slot_assignments": gold_assignments,
                "role_labels": None,
                "entity_labels": None,
                "bound_labels": bound_labels,
                "metadata": {"gold_params": params},
            }
            errs = validate_record(rec)
            if errs:
                if verbose:
                    print(f"NLP4LP skip {query_id}: {errs}", file=sys.stderr)
                continue
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
        return count
    # train / dev: need HF
    gold_by_id = _load_nlp4lp_gold(split)
    if not gold_by_id:
        if verbose:
            print(f"NLP4LP {split}: no gold (load HF or set NLP4LP_GOLD_CACHE)", file=sys.stderr)
        return 0
    try:
        from datasets import load_dataset
    except ImportError:
        if verbose:
            print("NLP4LP train/dev requires 'datasets'; pip install datasets", file=sys.stderr)
        return 0
    raw = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
    kwargs = {"token": raw} if raw else {}
    ds = load_dataset("udell-lab/NLP4LP", split=split, **kwargs)
    items = list(ds)
    rng = __import__("random").Random(seed)
    if max_examples and len(items) > max_examples:
        indices = rng.sample(range(len(items)), max_examples)
        items = [items[i] for i in indices]
    count = 0
    for idx, ex in enumerate(items):
        doc_id = f"nlp4lp_{split}_{idx}"
        gold = gold_by_id.get(doc_id) or {}
        params = gold.get("parameters") or {}
        pinfo = gold.get("problem_info") or {}
        if isinstance(pinfo, str):
            try:
                pinfo = json.loads(pinfo)
            except Exception:
                pinfo = {}
        expected_params = list((pinfo.get("parameters") or params or {}).keys())
        scalar_params = [p for p in expected_params if _is_scalar(params.get(p))]
        description = (ex.get("description") or "").strip()
        if not description or not scalar_params:
            continue
        mentions_with_pos = _extract_numeric_mentions_from_text(description)
        numeric_mentions = []
        for i, (cstart, cend, surface, value, type_bucket) in enumerate(mentions_with_pos):
            ctx_start = max(0, cstart - 50)
            ctx_end = min(len(description), cend + 50)
            local_ctx = description[ctx_start:ctx_end]
            cues = _operator_cues_near(description, cstart, cend)
            numeric_mentions.append(
                mention_to_dict(
                    mention_id=f"m{i}",
                    surface=surface,
                    normalized_value=value,
                    type_bucket=type_bucket,
                    char_start=cstart,
                    char_end=cend,
                    local_context=local_ctx,
                    operator_cues=cues,
                )
            )
        slots = []
        for p in scalar_params:
            et = _expected_type_from_slot_name(p)
            slot_role = "lower_bound" if "min" in p.lower() or "least" in p.lower() else ("upper_bound" if "max" in p.lower() or "most" in p.lower() else None)
            slots.append(slot_to_dict(slot_id=p, slot_name=p, slot_text=None, slot_role=slot_role, expected_type=et, variable_entity=None))
        gold_assignments = {}
        for slot_id in scalar_params:
            gval = params.get(slot_id)
            if not _is_scalar(gval):
                gold_assignments[slot_id] = None
                continue
            match_id = None
            for m in numeric_mentions:
                mv = m.get("normalized_value")
                if mv is not None and gval is not None and abs(float(mv) - float(gval)) < 1e-9:
                    match_id = m["mention_id"]
                    break
            gold_assignments[slot_id] = match_id
        bound_labels = {p: ("lower" if "min" in p.lower() or "least" in p.lower() else ("upper" if "max" in p.lower() or "most" in p.lower() else None)) for p in scalar_params}
        rec = {
            "dataset": "nlp4lp",
            "split": split,
            "instance_id": doc_id,
            "source_path": f"udell-lab/NLP4LP split={split}",
            "problem_text": description,
            "schema_name": doc_id,
            "schema_description": None,
            "slots": slots,
            "numeric_mentions": numeric_mentions,
            "gold_slot_assignments": gold_assignments,
            "role_labels": None,
            "entity_labels": None,
            "bound_labels": bound_labels,
            "metadata": {"gold_params": params},
        }
        errs = validate_record(rec)
        if errs:
            if verbose:
                print(f"NLP4LP skip {doc_id}: {errs}", file=sys.stderr)
            continue
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1
    return count


def _is_scalar(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _load_nlp4lp_gold(split: str) -> dict:
    cache_path = os.environ.get("NLP4LP_GOLD_CACHE")
    if cache_path:
        p = Path(cache_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("split") == split:
                out = data.get("gold_by_id")
                if isinstance(out, dict):
                    return out
    try:
        from tools import nlp4lp_downstream_utility as _dutil
        return _dutil._load_hf_gold(split=split, use_cache=True)
    except Exception:
        return {}


# ----- NL4Opt -----
def _build_nl4opt(
    split: str,
    output_dir: Path,
    max_examples: int | None,
    seed: int,
    verbose: bool,
) -> int:
    out_path = output_dir / f"nl4opt_{split}.jsonl"
    out_path.unlink(missing_ok=True)
    paths = _paths()
    path = paths.get(f"nl4opt_{split}")
    if not path or not path.exists():
        if verbose:
            print(f"NL4Opt {split}: not found {path}", file=sys.stderr)
        return 0
    rng = __import__("random").Random(seed)
    count = 0
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
    if max_examples and len(lines) > max_examples:
        lines = rng.sample(lines, max_examples)
    for line in lines:
        obj = json.loads(line)
        # One JSON object keyed by hash
        for hash_key, rec in obj.items():
            if hash_key.startswith("_"):
                continue
            doc = (rec.get("document") or "").strip()
            if not doc:
                continue
            spans = rec.get("spans") or []
            tokens = rec.get("tokens") or []
            obj_decl = rec.get("obj_declaration") or {}
            const_decls = rec.get("const_declarations") or []
            var_mention_to_first = rec.get("var_mention_to_first_var") or {}
            slots = []
            slot_to_gold_mention: dict[str, str] = {}
            role_labels = {}
            entity_labels = {}
            bound_labels = {}
            # Slots from objective terms
            terms = obj_decl.get("terms") or {}
            for var_mention, val in terms.items():
                slot_id = f"obj_{var_mention}"
                first_var = var_mention_to_first.get(var_mention, var_mention)
                slots.append(slot_to_dict(slot_id=slot_id, slot_name=var_mention, slot_text=None, slot_role="objective_coeff", expected_type="float", variable_entity=first_var))
                slot_to_gold_mention[slot_id] = str(val)
            for c in const_decls:
                ctype = c.get("type") or ""
                limit = c.get("limit")
                var = c.get("var")
                direction = (c.get("direction") or "").lower()
                op = c.get("operator") or ""
                slot_id = f"const_{ctype}_{var}_{limit}"
                slot_role = "lower_bound" if "lower" in ctype or "at least" in direction or "GREATER" in str(op) else ("upper_bound" if "upper" in ctype or "at most" in direction or "below" in direction or "LESS" in str(op) else None)
                bound_labels[slot_id] = "lower" if slot_role == "lower_bound" else ("upper" if slot_role == "upper_bound" else None)
                first_var = var_mention_to_first.get(var, var) if var else None
                slots.append(slot_to_dict(slot_id=slot_id, slot_name=f"{var}_{limit}", slot_text=None, slot_role=slot_role, expected_type="float", variable_entity=first_var))
                slot_to_gold_mention[slot_id] = str(limit) if limit else ""
            numeric_mentions = []
            span_to_mention_id = {}
            for s in spans:
                label = s.get("label")
                if label not in ("PARAM", "LIMIT"):
                    continue
                start = s.get("start", 0)
                end = s.get("end", start)
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                try:
                    val = float(text.replace("$", "").replace("%", "").replace(",", ""))
                except Exception:
                    val = None
                type_bucket = "percent" if "%" in text else ("currency" if "$" in text else "float")
                mid = f"m{len(numeric_mentions)}"
                span_to_mention_id[(start, end, text)] = mid
                ctx_start = max(0, start - 50)
                ctx_end = min(len(doc), end + 50)
                local_ctx = doc[ctx_start:ctx_end]
                cues = _operator_cues_near(doc, start, end)
                numeric_mentions.append(
                    mention_to_dict(mention_id=mid, surface=text, normalized_value=val, type_bucket=type_bucket, char_start=start, char_end=end, local_context=local_ctx, operator_cues=cues)
                )
            gold_assignments = {}
            for slot_id, gold_val in slot_to_gold_mention.items():
                if not gold_val:
                    gold_assignments[slot_id] = None
                    continue
                try:
                    gv = float(gold_val.replace(",", ""))
                except Exception:
                    gold_assignments[slot_id] = None
                    continue
                match_id = None
                for m in numeric_mentions:
                    mv = m.get("normalized_value")
                    if mv is not None and abs(float(mv) - gv) < 1e-9:
                        match_id = m["mention_id"]
                        break
                gold_assignments[slot_id] = match_id
            for m in numeric_mentions:
                entity_labels[m["mention_id"]] = None
            rec_out = {
                "dataset": "nl4opt",
                "split": split,
                "instance_id": str(hash_key),
                "source_path": str(path),
                "problem_text": doc,
                "schema_name": str(hash_key),
                "schema_description": None,
                "slots": slots,
                "numeric_mentions": numeric_mentions,
                "gold_slot_assignments": gold_assignments,
                "role_labels": role_labels,
                "entity_labels": entity_labels,
                "bound_labels": bound_labels,
                "metadata": {"obj_declaration": obj_decl, "const_declarations": const_decls},
            }
            errs = validate_record(rec_out)
            if errs and verbose:
                print(f"NL4Opt skip {hash_key}: {errs}", file=sys.stderr)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            count += 1
    return count


# ----- TAT-QA -----
def _linearize_tatqa_table(table: list) -> str:
    if not table or not isinstance(table, list):
        return ""
    lines = []
    for row in table:
        if isinstance(row, list):
            lines.append(" | ".join(str(c) for c in row))
        else:
            lines.append(str(row))
    return "\n".join(lines)


def _extract_numbers_from_tatqa_cell(cell: str) -> list[tuple[str, float | None, str]]:
    out = []
    for m in NUM_PATTERN.finditer(cell):
        surface = m.group()
        try:
            num_str = surface.replace("$", "").replace("%", "").replace(",", "")
            val = float(num_str)
            if "%" in surface and val > 1:
                val = val / 100.0
        except Exception:
            val = None
        type_bucket = "percent" if "%" in surface else ("currency" if "$" in surface else "float")
        out.append((surface, val, type_bucket))
    return out


def _build_tatqa(
    split: str,
    output_dir: Path,
    max_examples: int | None,
    seed: int,
    verbose: bool,
) -> int:
    out_path = output_dir / f"tatqa_{split}.jsonl"
    out_path.unlink(missing_ok=True)
    paths = _paths()
    path = paths.get(f"tatqa_{split}")
    if not path or not path.exists():
        if verbose:
            print(f"TAT-QA {split}: not found {path}", file=sys.stderr)
        return 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    rng = __import__("random").Random(seed)
    if max_examples and len(data) > max_examples:
        data = rng.sample(data, max_examples)
    count = 0
    for item in data:
        table = item.get("table") or {}
        table_rows = table.get("table", table) if isinstance(table, dict) else table
        if isinstance(table_rows, list) and table_rows and isinstance(table_rows[0], dict):
            table_rows = table_rows[0].get("table", table_rows)
        paragraphs = item.get("paragraphs") or []
        para_texts = [p.get("text", "") for p in paragraphs if isinstance(p, dict)]
        context = _linearize_tatqa_table(table_rows) + "\n\n" + "\n\n".join(para_texts)
        questions = item.get("questions") or []
        for q in questions:
            if not isinstance(q, dict):
                continue
            qtext = (q.get("question") or "").strip()
            answer = q.get("answer")
            derivation = q.get("derivation") or ""
            answer_type = q.get("answer_type") or ""
            problem_text = qtext + "\n\n" + context
            mentions_with_pos = _extract_numeric_mentions_from_text(problem_text)
            numeric_mentions = []
            for i, (cstart, cend, surface, value, type_bucket) in enumerate(mentions_with_pos):
                numeric_mentions.append(
                    mention_to_dict(
                        mention_id=f"m{i}",
                        surface=surface,
                        normalized_value=value,
                        type_bucket=type_bucket,
                        char_start=cstart,
                        char_end=cend,
                        local_context=problem_text[max(0, cstart - 30) : cend + 30],
                        operator_cues=[],
                    )
                )
            uid = q.get("uid", str(count))
            rec = {
                "dataset": "tatqa",
                "split": split,
                "instance_id": uid,
                "source_path": str(path),
                "problem_text": problem_text[:8000],
                "schema_name": None,
                "schema_description": None,
                "slots": [],
                "numeric_mentions": numeric_mentions,
                "gold_slot_assignments": {"answer": answer},
                "role_labels": None,
                "entity_labels": None,
                "bound_labels": None,
                "metadata": {"derivation": derivation, "answer_type": answer_type},
            }
            errs = validate_record(rec)
            if errs and verbose:
                print(f"TAT-QA skip {uid}: {errs}", file=sys.stderr)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


# ----- FinQA -----
def _linearize_finqa_table(table: list) -> str:
    if not table or not isinstance(table, list):
        return ""
    lines = []
    for row in table:
        if isinstance(row, list):
            lines.append(" | ".join(str(c) for c in row))
        else:
            lines.append(str(row))
    return "\n".join(lines)


def _build_finqa(
    split: str,
    output_dir: Path,
    max_examples: int | None,
    seed: int,
    verbose: bool,
) -> int:
    out_path = output_dir / f"finqa_{split}.jsonl"
    out_path.unlink(missing_ok=True)
    paths = _paths()
    path = paths.get(f"finqa_{split}")
    if not path or not path.exists():
        if verbose:
            print(f"FinQA {split}: not found {path}", file=sys.stderr)
        return 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    rng = __import__("random").Random(seed)
    if max_examples and len(data) > max_examples:
        data = rng.sample(data, max_examples)
    count = 0
    for item in data:
        pre_text = item.get("pre_text") or []
        post_text = item.get("post_text") or []
        table = item.get("table") or []
        if isinstance(pre_text, list):
            pre_str = "\n".join(str(t) for t in pre_text)
        else:
            pre_str = str(pre_text)
        if isinstance(post_text, list):
            post_str = "\n".join(str(t) for t in post_text)
        else:
            post_str = str(post_text)
        table_str = _linearize_finqa_table(table)
        qa = item.get("qa") or {}
        question = (qa.get("question") or "").strip()
        answer = qa.get("answer")
        steps = qa.get("steps") or []
        problem_text = question + "\n\n" + pre_str + "\n\n" + table_str + "\n\n" + post_str
        problem_text = problem_text[:12000]
        mentions_with_pos = _extract_numeric_mentions_from_text(problem_text)
        numeric_mentions = []
        for i, (cstart, cend, surface, value, type_bucket) in enumerate(mentions_with_pos):
            numeric_mentions.append(
                mention_to_dict(
                    mention_id=f"m{i}",
                    surface=surface,
                    normalized_value=value,
                    type_bucket=type_bucket,
                    char_start=cstart,
                    char_end=cend,
                    local_context=problem_text[max(0, cstart - 30) : cend + 30],
                    operator_cues=[],
                )
            )
        instance_id = item.get("id", str(count))
        rec = {
            "dataset": "finqa",
            "split": split,
            "instance_id": instance_id,
            "source_path": str(path),
            "problem_text": problem_text,
            "schema_name": None,
            "schema_description": None,
            "slots": [],
            "numeric_mentions": numeric_mentions,
            "gold_slot_assignments": {"answer": answer},
            "role_labels": None,
            "entity_labels": None,
            "bound_labels": None,
            "metadata": {"steps": steps},
        }
        errs = validate_record(rec)
        if errs and verbose:
            print(f"FinQA skip {instance_id}: {errs}", file=sys.stderr)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Build common learning corpus JSONL")
    ap.add_argument("--dataset", choices=["nlp4lp", "nl4opt", "tatqa", "finqa", "all"], default="all")
    ap.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    ap.add_argument("--output_dir", type=Path, default=ROOT / "artifacts" / "learning_corpus")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["nlp4lp", "nl4opt", "tatqa", "finqa"] if args.dataset == "all" else [args.dataset]
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    total = 0
    for ds in datasets:
        for sp in splits:
            if ds == "nlp4lp":
                n = _build_nlp4lp(sp, output_dir, args.max_examples, args.seed, args.verbose)
            elif ds == "nl4opt":
                n = _build_nl4opt(sp, output_dir, args.max_examples, args.seed, args.verbose)
            elif ds == "tatqa":
                n = _build_tatqa(sp, output_dir, args.max_examples, args.seed, args.verbose)
            elif ds == "finqa":
                n = _build_finqa(sp, output_dir, args.max_examples, args.seed, args.verbose)
            else:
                n = 0
            total += n
            if args.verbose:
                print(f"{ds} {sp}: {n} records")
    print(f"Total: {total} records written to {output_dir}")


if __name__ == "__main__":
    main()
