"""Self-augmented leaf-node modeling (`G_mod`, eq. 3) and bottom-up merge
into a complete model (eq. 4) -- PaMOP paper section 3.3, "Generation and
Improvement of Solver's Model."

Eq. 3 (leaf modeling), PAPER-SPECIFIED:
    m_{c,i} = G_mod(g, t_v, {t_{c,j} : j in cons_i})
Inputs: the global summary `g`, the FULL global variable/parameter list
`t_v` (not a node-local subset), and only this leaf's own constraint
descriptions. Output is AMPL code, not prose ("we directly generate code in
the modeling language instead of formulas"). "When modeling nodes
containing vague constraints, we can incorporate information from their
parent and sibling nodes to aid in the modeling process."

Eq. 4 (root completion), PAPER-SPECIFIED:
    M = (m_p, m_v, m_o, m_c) = G_mod(g, t_v, t_o, m_c)
One additional call, at the root, after all leaves are modeled and merged:
takes the global summary, full variable list, objective text, and the
ALREADY-MERGED constraint set `m_c`, and produces the complete model
including the objective. This is the only place the objective gets
modeled -- eq. 3's leaf calls never touch it.

Merge, PAPER-SPECIFIED mechanism (not an LLM call at internal tree layers):
"After each leaf node is modeled separately, the formulas will be merged
layer by layer from the bottom up into a complete model. Since the
constants and variables t_v have been described in advance, there is
minimal conflict between formulas modeled at different nodes. Thus, we can
directly merge the modeled formulas." I.e. internal (non-root, non-leaf)
nodes do NOT invoke the LLM at all -- they only concatenate their
children's already-modeled text, bottom-up. Only leaves (eq. 3) and the
root's final step (eq. 4) call an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import PamopConfig
from .llm.base import LLMProvider
from .llm.types import LLMResponse
from .llm.types import ModelConfig as LLMModelConfig
from .partition import PartitionNode, PartitionTree
from .prompts import PromptTemplate, load_prompt
from .representations import ConstraintInfo, StructuredProblem, VariableInfo

_IDENTIFIER_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
# AMPL keywords/builtins that would otherwise look like "unresolved
# references" in the heuristic scan below -- not exhaustive, just enough to
# keep the diagnostic usable rather than noisy.
_AMPL_KEYWORDS = {
    "subject", "to", "sum", "param", "var", "maximize", "minimize", "s",
    "t", "in", "for", "if", "then", "else", "and", "or", "not", "sqrt",
    "abs", "min", "max", "exp", "log", "integer", "binary", "le", "ge",
}
_ROOT_SECTIONS = ("PARAMETERS", "VARIABLES", "OBJECTIVE", "CONSTRAINTS")


class ModelingValidationError(ValueError):
    """Raised when a G_mod response (leaf or root) fails validation.

    Deliberately minimal, heuristic checks (this milestone has no AMPL
    parser -- see baselines/pamop/README.md "AMPL interface boundary") --
    never a substitute for real syntax checking, and never used to
    silently rewrite content, only to accept or reject it.
    """


# ---------------------------------------------------------------------
# Leaf modeling (eq. 3)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LeafModelResult:
    node_id: str
    constraint_indices: tuple[int, ...]
    ampl_fragment: str
    used_augmentation: bool
    referenced_global_symbols: tuple[str, ...]
    unresolved_references: tuple[str, ...]  # heuristic diagnostic, non-fatal
    llm_response: LLMResponse
    prompt_template: PromptTemplate
    validation_attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "constraint_indices": list(self.constraint_indices),
            "ampl_fragment": self.ampl_fragment,
            "used_augmentation": self.used_augmentation,
            "referenced_global_symbols": list(self.referenced_global_symbols),
            "unresolved_references": list(self.unresolved_references),
            "prompt_hash": self.llm_response.prompt_hash,
            "validation_attempts": self.validation_attempts,
        }


def _format_variable_list(variables: tuple[VariableInfo, ...]) -> str:
    return "\n".join(f"- {v.name} ({v.var_type or 'unspecified type'}): {v.description}" for v in variables)


def _format_constraint_list(constraints: list[ConstraintInfo]) -> str:
    return "\n".join(f"{i + 1}. {c.description}" for i, c in enumerate(constraints))


def _augmentation_block(problem: StructuredProblem, tree: PartitionTree, node: PartitionNode) -> str:
    """"incorporate information from their parent and sibling nodes" --
    REPRODUCTION CHOICE content: the parent's and siblings' constraint
    *descriptions* (not their modeled AMPL output, which may not exist yet
    for siblings under a bottom-up traversal order -- see PROVENANCE.md)."""
    if node.parent_id is None:
        return ""
    parent = tree.nodes[node.parent_id]
    sibling_ids = [c for c in parent.children if c != node.node_id]
    context_indices = set(parent.constraint_group)
    for sib_id in sibling_ids:
        context_indices.update(tree.nodes[sib_id].constraint_group)
    context_indices -= set(node.constraint_group)
    if not context_indices:
        return ""
    lines = [problem.constraints[i].description for i in sorted(context_indices)]
    return (
        "\nAdditional context from related constraints elsewhere in the problem "
        "(for understanding only -- do not model these here):\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
    )


_CONSTRAINT_LABEL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*:")
_DECLARATION_RE = re.compile(r"^\s*(param|var)\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.MULTILINE)


def _find_unresolved_references(text: str, known_names: set[str]) -> tuple[str, ...]:
    """Heuristic diagnostic, not a real AMPL parser: identifier-shaped
    tokens in ``text`` that are neither a known variable/parameter name nor
    a common AMPL keyword. False positives are expected (e.g. index
    variables like ``i``/``j``, which are dropped by the single-letter
    filter below); this is informational, never a hard validation failure.

    AMPL constraint/statement labels (``subject to c1: ...``) are excluded
    by construction -- a label is not a variable reference, and without
    this exclusion every leaf's own constraint labels would show up as
    "unresolved" noise on every single call.
    """
    labels = {m.group(0)[:-1].strip() for m in _CONSTRAINT_LABEL_RE.finditer(text)}
    tokens = set(_IDENTIFIER_TOKEN_RE.findall(text))
    unresolved = tokens - known_names - _AMPL_KEYWORDS - labels
    unresolved = {t for t in unresolved if len(t) > 1}  # drop single-letter index variables
    return tuple(sorted(unresolved))


def _find_referenced_global_symbols(text: str, known_names: set[str]) -> tuple[str, ...]:
    """Known t_v names that appear in a modeled leaf fragment.

    This is serializable provenance for downstream merge/correction, not
    an AMPL parse and not a validation gate.
    """
    tokens = set(_IDENTIFIER_TOKEN_RE.findall(text))
    return tuple(sorted(tokens & known_names))


def validate_leaf_output(text: str) -> str:
    """Minimal structural sanity check -- see ModelingValidationError."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:ampl)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    if not stripped:
        raise ModelingValidationError("empty response")
    if ";" not in stripped:
        raise ModelingValidationError("no ';' found -- does not look like AMPL statements")
    return stripped


def model_leaf(
    node: PartitionNode,
    tree: PartitionTree,
    problem: StructuredProblem,
    provider: LLMProvider,
    config: PamopConfig,
) -> LeafModelResult:
    max_retries = config.require("llm", "modeling_max_retries")
    vague_threshold = config.require("llm", "vague_threshold")
    model_config = LLMModelConfig(
        provider=config.require("llm", "provider"),
        model=config.require("llm", "model"),
        temperature=config.require("llm", "temperature"),
        max_tokens=config.llm.max_tokens,
        top_p=config.llm.top_p,
    )

    leaf_constraints = [problem.constraints[i] for i in node.constraint_group]
    is_vague = any((c.vagueness_score or 0.0) >= vague_threshold for c in leaf_constraints)
    augmentation = _augmentation_block(problem, tree, node) if is_vague else ""

    template = load_prompt("modeling_leaf_v1.txt")
    prompt = template.render(
        global_summary=problem.global_summary,
        variable_list=_format_variable_list(problem.variables),
        constraint_list=_format_constraint_list(leaf_constraints),
        augmentation_block=augmentation,
    )

    known_names = {v.name for v in problem.variables}
    attempt = 0
    last_error: Exception | None = None
    last_response: LLMResponse | None = None
    while attempt <= max_retries:
        attempt += 1
        response = provider.generate(prompt, model_config)
        last_response = response
        try:
            fragment = validate_leaf_output(response.text)
        except ModelingValidationError as exc:
            last_error = exc
            continue

        return LeafModelResult(
            node_id=node.node_id,
            constraint_indices=node.constraint_group,
            ampl_fragment=fragment,
            used_augmentation=bool(augmentation),
            referenced_global_symbols=_find_referenced_global_symbols(fragment, known_names),
            unresolved_references=_find_unresolved_references(fragment, known_names),
            llm_response=response,
            prompt_template=template,
            validation_attempts=attempt,
        )

    raise ModelingValidationError(
        f"leaf {node.node_id!r} modeling failed validation after {attempt} attempt(s); "
        f"last response prompt_hash={last_response.prompt_hash if last_response else 'n/a'}; "
        f"last error: {last_error}"
    ) from last_error


def model_all_leaves(
    tree: PartitionTree,
    problem: StructuredProblem,
    provider: LLMProvider,
    config: PamopConfig,
) -> dict[str, LeafModelResult]:
    """Model every leaf, in a deterministic order (ascending node_id).

    Order doesn't affect correctness here (leaves don't depend on each
    other's *output*, only on shared, already-known context) but a fixed
    order keeps run logs and any future cost/latency accounting
    reproducible.
    """
    results: dict[str, LeafModelResult] = {}
    for leaf in sorted(tree.leaves(), key=lambda n: n.node_id):
        results[leaf.node_id] = model_leaf(leaf, tree, problem, provider, config)
    return results


# ---------------------------------------------------------------------
# Bottom-up merge (paper: "directly merge the modeled formulas" -- no LLM
# call at internal tree layers)
# ---------------------------------------------------------------------


def merge_bottom_up(tree: PartitionTree, leaf_results: dict[str, LeafModelResult]) -> str:
    """Recursively concatenate modeled constraint text from the leaves up
    through every internal node to the root, in ``children`` order at each
    level -- literal text concatenation, no LLM call, per the paper's own
    "directly merge" wording."""

    def _merge_node(node_id: str) -> str:
        node = tree.nodes[node_id]
        if node.node_type == "leaf" or not node.children:
            return leaf_results[node_id].ampl_fragment
        return "\n".join(_merge_node(child_id) for child_id in node.children)

    return _merge_node(tree.root_id)


# ---------------------------------------------------------------------
# Root completion (eq. 4)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MergedModel:
    problem_id: str
    parameters_text: str
    variables_text: str
    objective_text: str
    constraints_text: str
    leaf_results: tuple[LeafModelResult, ...]
    root_llm_response: LLMResponse
    root_prompt_template: PromptTemplate
    symbol_conflicts: tuple[str, ...]
    config_hash: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "parameters_text": self.parameters_text,
            "variables_text": self.variables_text,
            "objective_text": self.objective_text,
            "constraints_text": self.constraints_text,
            "leaf_results": [r.to_dict() for r in self.leaf_results],
            "root_prompt_hash": self.root_llm_response.prompt_hash,
            "symbol_conflicts": list(self.symbol_conflicts),
            "config_hash": self.config_hash,
            "provenance": self.provenance,
        }


def _split_root_sections(text: str) -> dict[str, str]:
    """Parse the four ``### SECTION`` blocks eq. 4's prompt asks for.

    Raises ModelingValidationError if the headers are missing or
    out of order -- this is our own output-structuring choice
    (PROVENANCE.md), not something the paper mandates, so the parser is
    correspondingly explicit about what it requires.
    """
    pattern = re.compile(
        r"###\s*(" + "|".join(_ROOT_SECTIONS) + r")\s*\n(.*?)(?=###\s*(?:"
        + "|".join(_ROOT_SECTIONS) + r")|\Z)",
        re.DOTALL,
    )
    matches = pattern.findall(text)
    found_order = [m[0] for m in matches]
    if found_order != list(_ROOT_SECTIONS):
        raise ModelingValidationError(
            f"expected sections {_ROOT_SECTIONS} in order, found {found_order or 'none'}"
        )
    sections = {name: body.strip() for name, body in matches}
    if not sections["OBJECTIVE"]:
        raise ModelingValidationError("OBJECTIVE section is empty")
    if not sections["CONSTRAINTS"]:
        raise ModelingValidationError("CONSTRAINTS section is empty")
    _validate_root_declarations(sections)
    return sections


def _validate_root_declarations(sections: dict[str, str]) -> None:
    """Reject duplicate parameter/variable declarations in the root model.

    This is the minimum detectable structural conflict the reconstructed
    sectioned Eq. 4 output can catch without a real AMPL parser.
    """
    declarations: dict[str, list[str]] = {}
    for section_name in ("PARAMETERS", "VARIABLES"):
        for kind, name in _DECLARATION_RE.findall(sections[section_name]):
            declarations.setdefault(name, []).append(kind)

    duplicates = {name: kinds for name, kinds in declarations.items() if len(kinds) > 1}
    if duplicates:
        details = ", ".join(f"{name} as {kinds}" for name, kinds in sorted(duplicates.items()))
        raise ModelingValidationError(f"duplicate parameter/variable declarations: {details}")


def model_root_objective(
    problem: StructuredProblem,
    merged_constraints: str,
    provider: LLMProvider,
    config: PamopConfig,
) -> tuple[dict[str, str], LLMResponse, PromptTemplate, int]:
    max_retries = config.require("llm", "modeling_max_retries")
    model_config = LLMModelConfig(
        provider=config.require("llm", "provider"),
        model=config.require("llm", "model"),
        temperature=config.require("llm", "temperature"),
        max_tokens=config.llm.max_tokens,
        top_p=config.llm.top_p,
    )
    template = load_prompt("modeling_root_v1.txt")
    prompt = template.render(
        global_summary=problem.global_summary,
        objective_text=problem.objective_text,
        variable_list=_format_variable_list(problem.variables),
        merged_constraints=merged_constraints,
    )

    attempt = 0
    last_error: Exception | None = None
    last_response: LLMResponse | None = None
    while attempt <= max_retries:
        attempt += 1
        response = provider.generate(prompt, model_config)
        last_response = response
        try:
            sections = _split_root_sections(response.text)
        except ModelingValidationError as exc:
            last_error = exc
            continue
        return sections, response, template, attempt

    raise ModelingValidationError(
        f"root modeling failed validation after {attempt} attempt(s); "
        f"last response prompt_hash={last_response.prompt_hash if last_response else 'n/a'}; "
        f"last error: {last_error}"
    ) from last_error


def _detect_symbol_conflicts(problem: StructuredProblem, leaf_results: dict[str, LeafModelResult]) -> tuple[str, ...]:
    """Diagnostic only, never a hard failure -- the paper's own design
    assumes "minimal conflict... since t_v have been described in advance"
    (§3.3). This surfaces cases worth a human glance, e.g. a name multiple
    leaves flag as unresolved (possibly a real gap in t_v) -- it does not
    attempt automatic reconciliation, which would be scope beyond what the
    paper describes."""
    seen: dict[str, list[str]] = {}
    labels: dict[str, list[str]] = {}
    leaf_declarations: dict[str, list[str]] = {}
    for node_id, result in leaf_results.items():
        for name in result.unresolved_references:
            seen.setdefault(name, []).append(node_id)
        for label in _CONSTRAINT_LABEL_RE.findall(result.ampl_fragment):
            labels.setdefault(label[:-1].strip(), []).append(node_id)
        for _kind, name in _DECLARATION_RE.findall(result.ampl_fragment):
            leaf_declarations.setdefault(name, []).append(node_id)

    conflicts = [
        f"{name!r} referenced but not in the declared variable list, in leaf(s): {', '.join(nodes)}"
        for name, nodes in sorted(seen.items())
    ]
    conflicts.extend(
        f"constraint label {label!r} appears in multiple leaf fragments: {', '.join(nodes)}"
        for label, nodes in sorted(labels.items())
        if len(nodes) > 1
    )
    conflicts.extend(
        f"leaf fragment declares {name!r}; leaf G_mod output should contain constraints only, in leaf(s): {', '.join(nodes)}"
        for name, nodes in sorted(leaf_declarations.items())
    )
    return tuple(conflicts)


def build_merged_model(
    tree: PartitionTree,
    problem: StructuredProblem,
    provider: LLMProvider,
    config: PamopConfig,
) -> MergedModel:
    """Full eq. 3 + merge + eq. 4 pipeline for one problem."""
    from .partition import _config_hash  # reuse the same hashing convention as partition.py

    leaf_results = model_all_leaves(tree, problem, provider, config)
    merged_constraints = merge_bottom_up(tree, leaf_results)
    sections, root_response, root_template, root_attempts = model_root_objective(
        problem, merged_constraints, provider, config
    )
    conflicts = _detect_symbol_conflicts(problem, leaf_results)

    return MergedModel(
        problem_id=problem.problem_id,
        parameters_text=sections["PARAMETERS"],
        variables_text=sections["VARIABLES"],
        objective_text=sections["OBJECTIVE"],
        constraints_text=sections["CONSTRAINTS"],
        leaf_results=tuple(leaf_results[nid] for nid in sorted(leaf_results)),
        root_llm_response=root_response,
        root_prompt_template=root_template,
        symbol_conflicts=conflicts,
        config_hash=_config_hash(config),
        provenance={
            "config_kind": config.config_kind,
            "structured_problem_source": problem.source,
            "root_validation_attempts": root_attempts,
        },
    )
