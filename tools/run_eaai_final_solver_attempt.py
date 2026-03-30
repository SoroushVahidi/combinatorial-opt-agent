#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import sys
import tempfile
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.baselines import get_baseline
from tools import nlp4lp_downstream_utility as dutil


class GRB:
    CONTINUOUS = "C"
    INTEGER = "I"
    BINARY = "B"
    MAXIMIZE = -1
    MINIMIZE = 1


@dataclass
class LinExpr:
    coeffs: dict[int, float]
    const: float = 0.0

    def __add__(self, other: Any) -> "LinExpr":
        other = _to_expr(other)
        coeffs = dict(self.coeffs)
        for k, v in other.coeffs.items():
            coeffs[k] = coeffs.get(k, 0.0) + v
        return LinExpr(coeffs, self.const + other.const)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "LinExpr":
        other = _to_expr(other)
        coeffs = dict(self.coeffs)
        for k, v in other.coeffs.items():
            coeffs[k] = coeffs.get(k, 0.0) - v
        return LinExpr(coeffs, self.const - other.const)

    def __rsub__(self, other: Any) -> "LinExpr":
        return _to_expr(other) - self

    def __mul__(self, other: Any) -> "LinExpr":
        if isinstance(other, (int, float)):
            return LinExpr({k: v * float(other) for k, v in self.coeffs.items()}, self.const * float(other))
        raise TypeError("Nonlinear multiplication unsupported")

    __rmul__ = __mul__

    def __neg__(self) -> "LinExpr":
        return self * -1

    def __le__(self, other: Any) -> "ConstrExpr":
        return ConstrExpr(self - other, "<=")

    def __ge__(self, other: Any) -> "ConstrExpr":
        return ConstrExpr(self - other, ">=")

    def __eq__(self, other: Any) -> "ConstrExpr":  # type: ignore[override]
        return ConstrExpr(self - other, "==")


@dataclass
class ConstrExpr:
    expr: LinExpr
    sense: str


class Var:
    def __init__(self, model: "Model", idx: int, name: str, vtype: str, lb: float | None, ub: float | None):
        self._model = model
        self._idx = idx
        self.VarName = name
        self.vtype = vtype
        self.lb = 0.0 if lb is None else float(lb)
        self.ub = math.inf if ub is None else float(ub)
        self.x = 0.0

    @property
    def X(self) -> float:
        return self.x

    def _expr(self) -> LinExpr:
        return LinExpr({self._idx: 1.0}, 0.0)

    def __add__(self, other: Any) -> LinExpr:
        return self._expr() + other

    __radd__ = __add__

    def __sub__(self, other: Any) -> LinExpr:
        return self._expr() - other

    def __rsub__(self, other: Any) -> LinExpr:
        return _to_expr(other) - self._expr()

    def __mul__(self, other: Any) -> LinExpr:
        return self._expr() * other

    __rmul__ = __mul__

    def __le__(self, other: Any) -> ConstrExpr:
        return self._expr() <= other

    def __ge__(self, other: Any) -> ConstrExpr:
        return self._expr() >= other

    def __eq__(self, other: Any) -> ConstrExpr:  # type: ignore[override]
        return self._expr() == other


def _to_expr(v: Any) -> LinExpr:
    if isinstance(v, LinExpr):
        return v
    if isinstance(v, Var):
        return v._expr()
    if isinstance(v, (int, float, np.number)):
        return LinExpr({}, float(v))
    raise TypeError(f"Unsupported expression type: {type(v).__name__}")


class Model:
    def __init__(self):
        self.vars: list[Var] = []
        self.constrs: list[ConstrExpr] = []
        self.obj_expr: LinExpr | None = None
        self.obj_sense = GRB.MINIMIZE
        self.objVal: float | None = None
        self.Status: int = 0

    def addVar(self, vtype: str = GRB.CONTINUOUS, lb: float | None = None, ub: float | None = None, name: str = "") -> Var:
        idx = len(self.vars)
        v = Var(self, idx, name or f"x_{idx}", vtype, lb, ub)
        self.vars.append(v)
        return v

    def addVars(self, *sizes: Any, vtype: str = GRB.CONTINUOUS, lb: float | None = None, ub: float | None = None, name: str = "") -> dict[Any, Var]:
        if not sizes:
            raise ValueError("addVars requires at least one dimension")
        dims: list[list[Any]] = []
        for s in sizes:
            if isinstance(s, int):
                dims.append(list(range(s)))
            else:
                dims.append(list(s))
        out: dict[Any, Var] = {}
        for key in itertools.product(*dims):
            k = key[0] if len(key) == 1 else key
            suffix = "_".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
            out[k] = self.addVar(vtype=vtype, lb=lb, ub=ub, name=f"{name}_{suffix}")
        return out

    def addConstr(self, c: ConstrExpr, name: str | None = None) -> ConstrExpr:
        if not isinstance(c, ConstrExpr):
            raise TypeError("addConstr expects a constraint expression")
        self.constrs.append(c)
        return c

    def addConstrs(self, gen, name: str | None = None):
        out = []
        for c in gen:
            out.append(self.addConstr(c, name=name))
        return out

    def setObjective(self, expr: Any, sense: int = GRB.MINIMIZE) -> None:
        self.obj_expr = _to_expr(expr)
        self.obj_sense = sense

    def optimize(self) -> None:
        if self.obj_expr is None:
            raise ValueError("Objective not set")
        n = len(self.vars)
        c = np.zeros(n)
        for idx, coef in self.obj_expr.coeffs.items():
            c[idx] = coef
        c0 = self.obj_expr.const
        if self.obj_sense == GRB.MAXIMIZE:
            c = -c
            c0 = -c0

        lb = np.array([v.lb for v in self.vars], dtype=float)
        ub = np.array([v.ub for v in self.vars], dtype=float)
        integrality = np.array([1 if v.vtype in {GRB.INTEGER, GRB.BINARY} else 0 for v in self.vars], dtype=int)
        for i, v in enumerate(self.vars):
            if v.vtype == GRB.BINARY:
                lb[i], ub[i] = 0.0, 1.0

        constraints: list[LinearConstraint] = []
        for cexpr in self.constrs:
            row = np.zeros(n)
            for idx, coef in cexpr.expr.coeffs.items():
                row[idx] = coef
            const = cexpr.expr.const
            if cexpr.sense == "<=":
                constraints.append(LinearConstraint(row, -np.inf, -const))
            elif cexpr.sense == ">=":
                constraints.append(LinearConstraint(row, -const, np.inf))
            elif cexpr.sense == "==":
                constraints.append(LinearConstraint(row, -const, -const))
            else:
                raise ValueError(f"Unknown sense {cexpr.sense}")

        res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints or None)
        if res.success and res.x is not None:
            self.Status = 2
            for i, v in enumerate(self.vars):
                v.x = float(res.x[i])
            val = float(res.fun + c0)
            self.objVal = -val if self.obj_sense == GRB.MAXIMIZE else val
        else:
            self.Status = 3
            self.objVal = None


def quicksum(items):
    total = LinExpr({})
    for it in items:
        total = total + it
    return total


def _install_fake_gurobipy() -> None:
    mod = types.ModuleType("gurobipy")
    mod.Model = Model
    mod.GRB = GRB
    mod.quicksum = quicksum
    sys.modules["gurobipy"] = mod


def _load_eval(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _predict_schema_id(baseline_name: str, rank_fn, query: str, gold_id: str) -> str:
    if baseline_name == "oracle":
        return gold_id
    ranked = rank_fn(query, top_k=1)
    return ranked[0][0] if ranked else ""


def _ground_values(query: str, variant: str, expected_scalar: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not expected_scalar:
        return {}, {}
    vals, mentions, _ = dutil._run_optimization_role_relation_repair(query, variant, expected_scalar)
    return vals, mentions


def _build_exec_params(pred_params: dict[str, Any], filled: dict[str, Any]) -> dict[str, Any]:
    params = dict(pred_params)
    for k, v in filled.items():
        if k in pred_params and isinstance(pred_params[k], int) and not isinstance(pred_params[k], bool):
            params[k] = int(round(float(v)))
        else:
            params[k] = v
    return params


def _patch_optimus_code(code: str, params_path: str) -> str:
    pattern = r'with open\("[^"]*parameters\.json"\s*,\s*"r"\) as f:'
    repl = f'with open("{params_path}", "r") as f:'
    return re.sub(pattern, repl, code, count=1)


def _is_solver_compatible(code: str) -> bool:
    if not code:
        return False
    forbidden = ["addGenConstr", "setPWLObj", "max_(", "min_(", "abs_(", "\*\*", "gp."]
    return not any(tok in code for tok in forbidden)


def _attempt_exec(optimus_code: str, params: dict[str, Any]) -> dict[str, Any]:
    if not optimus_code:
        return {"executable_model": 0, "solver_run_success": 0, "feasible_solution": 0, "objective_produced": 0, "exec_error": "missing_optimus_code"}
    tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix="_parameters.json", delete=False, encoding="utf-8") as f:
            json.dump(params, f)
            tmp = f.name
        _install_fake_gurobipy()
        namespace: dict[str, Any] = {}
        patched = _patch_optimus_code(optimus_code, tmp)
        exec(patched, namespace)  # noqa: S102
        model = namespace.get("model")
        feasible = int(getattr(model, "Status", 0) == 2)
        objective = getattr(model, "objVal", None)
        return {
            "executable_model": int(model is not None),
            "solver_run_success": int(model is not None and getattr(model, "Status", 0) == 2),
            "feasible_solution": feasible,
            "objective_produced": int(objective is not None),
            "exec_error": "",
        }
    except Exception as e:  # noqa: BLE001
        return {"executable_model": 0, "solver_run_success": 0, "feasible_solution": 0, "objective_produced": 0, "exec_error": f"{type(e).__name__}: {e}"}
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def run(variant: str, out_dir: Path, limit: int | None) -> None:
    dutil._apply_low_resource_env()
    eval_path = ROOT / "data" / "processed" / f"nlp4lp_eval_{variant}.jsonl"
    catalog_path = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"
    gold_by_id = dutil._load_hf_gold("test")
    eval_rows = _load_eval(eval_path)

    subset = []
    for ex in eval_rows:
        gid = ex["relevant_doc_id"]
        code = (gold_by_id.get(gid) or {}).get("optimus_code") or ""
        if _is_solver_compatible(code):
            subset.append(ex)
    if limit is not None:
        subset = subset[:limit]

    catalog, _ = dutil._load_catalog_as_problems(catalog_path)
    tfidf = get_baseline("tfidf").fit(catalog)
    retrievers = {"tfidf": tfidf.rank, "oracle": None}

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "subset_instances.jsonl", "w", encoding="utf-8") as f:
        for row in subset:
            f.write(json.dumps(row) + "\n")

    instance_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for baseline_name, rank_fn in retrievers.items():
        failures = Counter()
        for ex in subset:
            query = ex["query"]
            qid = ex["query_id"]
            gold_id = ex["relevant_doc_id"]
            pred_id = _predict_schema_id(baseline_name, rank_fn, query, gold_id)
            pred = gold_by_id.get(pred_id) or {}
            pred_params = pred.get("parameters") or {}
            pred_info = pred.get("problem_info") or {}
            pred_param_keys = list((pred_info.get("parameters") or {}).keys()) if isinstance(pred_info.get("parameters"), dict) else list(pred_params.keys())
            expected_scalar = [p for p in pred_param_keys if _is_scalar(pred_params.get(p))]
            filled, _ = _ground_values(query, variant, expected_scalar)
            exec_params = _build_exec_params(pred_params, filled)

            exec_out = _attempt_exec(pred.get("optimus_code") or "", exec_params)
            if exec_out["exec_error"]:
                failures[exec_out["exec_error"].split(":", 1)[0]] += 1

            instance_rows.append(
                {
                    "baseline": baseline_name,
                    "query_id": qid,
                    "gold_doc_id": gold_id,
                    "predicted_doc_id": pred_id,
                    "schema_hit": int(pred_id == gold_id),
                    **exec_out,
                }
            )

        rows = [r for r in instance_rows if r["baseline"] == baseline_name]
        n = len(rows)
        summary_rows.append(
            {
                "baseline": baseline_name,
                "subset_size": n,
                "schema_hit_rate": round(sum(r["schema_hit"] for r in rows) / n, 4) if n else 0.0,
                "executable_model_rate": round(sum(r["executable_model"] for r in rows) / n, 4) if n else 0.0,
                "solver_run_success_rate": round(sum(r["solver_run_success"] for r in rows) / n, 4) if n else 0.0,
                "feasible_solution_rate": round(sum(r["feasible_solution"] for r in rows) / n, 4) if n else 0.0,
                "objective_produced_rate": round(sum(r["objective_produced"] for r in rows) / n, 4) if n else 0.0,
                "failure_counts": json.dumps(dict(failures), sort_keys=True),
            }
        )

    with open(out_dir / "final_solver_attempt_instances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(instance_rows[0].keys()))
        w.writeheader()
        w.writerows(instance_rows)

    with open(out_dir / "final_solver_attempt_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with open(out_dir / "final_solver_attempt_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant": variant,
                "grounding_method": "optimization_role_relation_repair",
                "subset_rule": "all eval instances whose gold optimus_code passes static compatibility filter (no addGenConstr/setPWLObj/max_/min_/abs_/gp./** tokens)",
                "subset_size": len(subset),
                "baselines": ["tfidf", "oracle"],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="orig", choices=["orig", "short", "noisy"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper" / "eaai_final_solver_attempt")
    parser.add_argument("--gold-cache", type=Path, default=ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json")
    args = parser.parse_args()

    os.environ["NLP4LP_GOLD_CACHE"] = str(args.gold_cache)
    run(args.variant, args.out_dir, args.limit)
