"""Pure objective-text parsing and explicit comparison predicates.

No solver/client imports. Numeric lexemes remain Decimal values until the caller
chooses a predicate. A parse is evidence of a printed number, not semantic truth.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
_TERMINAL = re.compile(rf"^\s*(?:Best objective|Optimal objective)\s+({NUMBER})(?=\s|,|$)", re.I | re.M)
_PLAIN = re.compile(rf"^\s*({NUMBER})\s*$")
_STOP = re.compile(r"Model is infeasible|infeasible or unbounded|Unbounded model|Time limit reached|Interrupted", re.I)


def decimal_token(token):
    value = Decimal(token.replace(',', '').replace('d', 'e').replace('D', 'E'))
    if not value.is_finite():
        raise ValueError('Nonfinite objective')
    return value


def resolution(token):
    """One displayed last-place unit; an explicit conservative text bound."""
    return Decimal(1).scaleb(decimal_token(token).as_tuple().exponent)


@dataclass(frozen=True)
class ObjectiveEvidence:
    value: Decimal | None
    token: str | None
    source: str
    status: str
    corroborating_token: str | None = None
    uncertainty: Decimal | None = None


def extract_gurobi_objective(stdout, return_code=0):
    """Require a terminal optimum, then cross-check a final standalone print.

    First heuristic solutions, bounds, progress rows and coefficient counts do
    not match. A failed process or later unsuccessful solver termination cannot
    supply a successful objective comparison. Multiple terminal blocks use last.
    """
    if return_code != 0:
        return ObjectiveEvidence(None, None, '', 'PROCESS_FAILED')
    found = list(_TERMINAL.finditer(stdout))
    if not found:
        return ObjectiveEvidence(None, None, '', 'NO_TERMINAL_OBJECTIVE')
    terminal = found[-1]
    if _STOP.search(stdout[terminal.end():]):
        return ObjectiveEvidence(None, None, '', 'LATER_UNSUCCESSFUL_TERMINATION')
    prefix = stdout[:terminal.end()]
    if not (re.search(r'Optimal objective', terminal.group(), re.I) or
            re.search(r'Optimal solution found', prefix, re.I)):
        return ObjectiveEvidence(None, None, '', 'OPTIMALITY_NOT_ESTABLISHED')
    token = terminal.group(1)
    value = decimal_token(token)
    prints = [(line, _PLAIN.fullmatch(line)) for line in stdout[terminal.end():].splitlines()]
    prints = [match.group(1) for _, match in prints if match]
    if prints:
        printed = prints[-1]
        pv = decimal_token(printed)
        if abs(pv-value) > max(resolution(token), resolution(printed)):
            return ObjectiveEvidence(None, printed, 'last_numeric_print', 'OBJECTIVE_CONFLICT', token)
        return ObjectiveEvidence(pv, printed, 'terminal_optimum+last_numeric_print',
                                 'CORROBORATED', token, resolution(printed))
    return ObjectiveEvidence(value, token, 'terminal_optimum', 'TERMINAL_ONLY',
                             uncertainty=resolution(token))


def extract_ampl_objective(stdout, objective):
    """Choose the most precise labeled objective in the final solved block.

    Archived AMPL display output can be less precise than its solver summary.
    Neither becomes a full-precision historical return through this choice.
    """
    states = list(re.finditer(r'\bsolve_result\s*=\s*(\w+)', stdout))
    if states and states[-1].group(1).lower() not in ('solved', 'optimal'):
        return ObjectiveEvidence(None, None, '', 'SOLVER_NOT_SOLVED')
    if not states and not re.search(r'optimal solution|Solution determined by presolve', stdout, re.I):
        return ObjectiveEvidence(None, None, '', 'SOLVER_STATUS_UNAVAILABLE')
    candidates = []
    if objective:
        pattern = re.compile(rf'^\s*{re.escape(objective)}\s*=\s*({NUMBER})\s*$', re.M)
        for match in pattern.finditer(stdout):
            candidates.append((match.group(1), 'ampl_display'))
    for match in re.finditer(rf'optimal solution;\s*objective\s+({NUMBER})(?=\s|$)', stdout, re.I):
        candidates.append((match.group(1), 'ampl_solver_summary'))
    if objective:
        for match in re.finditer(rf'objective\s+{re.escape(objective)}\s*=\s*({NUMBER})', stdout, re.I):
            candidates.append((match.group(1), 'ampl_presolve'))
    if not candidates:
        return ObjectiveEvidence(None, None, '', 'NO_LABELED_OBJECTIVE')
    token, source = min(candidates, key=lambda c: resolution(c[0]))
    value = decimal_token(token)
    for other, _ in candidates:
        if abs(value-decimal_token(other)) > max(resolution(token), resolution(other)):
            return ObjectiveEvidence(None, token, source, 'OBJECTIVE_CONFLICT')
    return ObjectiveEvidence(value, token, source, 'PRINTED_PRECISION_ONLY',
                             uncertainty=resolution(token))


def legacy_rounded_agreement(prediction, reference, tolerance=0.05):
    """Original Python float/round ties-to-even predicate; explicitly versioned."""
    if prediction is None or reference is None or isinstance(reference, str):
        return None
    p, g = round(float(prediction)), round(float(reference))
    return abs(p-g) <= tolerance if g == 0 else abs((p-g)/g) <= tolerance


def unrounded_agreement(prediction, reference, tolerance=Decimal('.05')):
    if prediction is None or reference is None:
        return None
    p, g = Decimal(str(prediction)), Decimal(str(reference))
    bound = tolerance if g == 0 else tolerance*abs(g)
    return abs(p-g) <= bound


def precision_bounded_absolute_agreement(evidence, reference, tolerance=Decimal('1e-6')):
    """Do not force PASS/FAIL when the retained decimal resolution straddles it.

    Uses +/- one displayed unit, conservatively, without assuming an unrecorded
    full-precision solver value or recovering extra digits from a formulation.
    """
    if evidence.value is None or reference is None:
        return None
    gap = abs(evidence.value-Decimal(str(reference)))
    error = evidence.uncertainty
    if error is None:
        return None
    if gap+error <= tolerance:
        return True
    if gap-error > tolerance:
        return False
    return None
