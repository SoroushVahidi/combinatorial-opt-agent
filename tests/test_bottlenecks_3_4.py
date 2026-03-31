"""
Tests for Bottleneck 3 and Bottleneck 4 fixes.

Bottleneck 3 — NLP4LP downstream instantiation gap:
  - _word_to_number: word ↔ integer value lookup
  - _int_to_word:    inverse mapping (for training-pair augmentation)
  - _extract_num_tokens: now recognises written-word numbers
  - _extract_num_mentions: same, used by constrained assignment
  - _expected_type:  extended slot-type pattern coverage

Bottleneck 4 — Catalog formulation coverage:
  - format_problem_and_ip: graceful notice when formulation is missing
  - custom_problems.json: new formulation entries override no-formulation stubs
  - generate_mention_slot_pairs: written-word paraphrase augmentation helpers
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck 3: _word_to_number
# ─────────────────────────────────────────────────────────────────────────────

class TestWordToNumber:
    """_word_to_number must map English number words to their float value."""

    def test_single_digit_words(self):
        from tools.nlp4lp_downstream_utility import _word_to_number
        assert _word_to_number("one") == 1.0
        assert _word_to_number("five") == 5.0
        assert _word_to_number("nine") == 9.0

    def test_teens(self):
        from tools.nlp4lp_downstream_utility import _word_to_number
        assert _word_to_number("ten") == 10.0
        assert _word_to_number("twelve") == 12.0
        assert _word_to_number("nineteen") == 19.0

    def test_round_tens(self):
        from tools.nlp4lp_downstream_utility import _word_to_number
        assert _word_to_number("twenty") == 20.0
        assert _word_to_number("fifty") == 50.0
        assert _word_to_number("ninety") == 90.0

    def test_hundred_thousand(self):
        from tools.nlp4lp_downstream_utility import _word_to_number
        assert _word_to_number("hundred") == 100.0
        assert _word_to_number("thousand") == 1000.0

    def test_hyphenated_compounds(self):
        from tools.nlp4lp_downstream_utility import _word_to_number
        assert _word_to_number("twenty-three") == 23.0
        assert _word_to_number("forty-five") == 45.0
        assert _word_to_number("ninety-nine") == 99.0

    def test_case_insensitive(self):
        from tools.nlp4lp_downstream_utility import _word_to_number
        assert _word_to_number("Two") == 2.0
        assert _word_to_number("TWENTY") == 20.0
        assert _word_to_number("Forty-Five") == 45.0

    def test_unknown_word_returns_none(self):
        from tools.nlp4lp_downstream_utility import _word_to_number
        assert _word_to_number("warehouse") is None
        assert _word_to_number("") is None
        assert _word_to_number("million-billion") is None

    def test_zero(self):
        from tools.nlp4lp_downstream_utility import _word_to_number
        assert _word_to_number("zero") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck 3: _int_to_word (training augmentation)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntToWord:
    """_int_to_word must return the English word for small integers."""

    def test_single_digits(self):
        from training.generate_mention_slot_pairs import _int_to_word
        assert _int_to_word(1) == "one"
        assert _int_to_word(5) == "five"
        assert _int_to_word(9) == "nine"

    def test_teens(self):
        from training.generate_mention_slot_pairs import _int_to_word
        assert _int_to_word(10) == "ten"
        assert _int_to_word(15) == "fifteen"
        assert _int_to_word(19) == "nineteen"

    def test_round_tens(self):
        from training.generate_mention_slot_pairs import _int_to_word
        assert _int_to_word(20) == "twenty"
        assert _int_to_word(30) == "thirty"
        assert _int_to_word(90) == "ninety"

    def test_compound_twenty_one_to_ninety_nine(self):
        from training.generate_mention_slot_pairs import _int_to_word
        assert _int_to_word(21) == "twenty-one"
        assert _int_to_word(45) == "forty-five"
        assert _int_to_word(99) == "ninety-nine"

    def test_hundred_thousand(self):
        from training.generate_mention_slot_pairs import _int_to_word
        assert _int_to_word(100) == "hundred"
        assert _int_to_word(1000) == "thousand"

    def test_out_of_range_returns_none(self):
        from training.generate_mention_slot_pairs import _int_to_word
        assert _int_to_word(101) is None   # not in the lookup
        assert _int_to_word(10000) is None

    def test_zero(self):
        from training.generate_mention_slot_pairs import _int_to_word
        assert _int_to_word(0) == "zero"


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck 3: _extract_num_tokens — written-word recognition
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractNumTokensWordRecognition:
    """_extract_num_tokens must extract written-number words as NumToks."""

    def test_digit_token_still_extracted(self):
        from tools.nlp4lp_downstream_utility import _extract_num_tokens
        toks = _extract_num_tokens("Budget is 500 dollars", "orig")
        assert len(toks) == 1
        assert toks[0].value == 500.0

    def test_written_word_extracted(self):
        from tools.nlp4lp_downstream_utility import _extract_num_tokens
        toks = _extract_num_tokens("There are two warehouses and five customers", "orig")
        values = {t.value for t in toks}
        assert 2.0 in values
        assert 5.0 in values

    def test_written_word_kind_is_int(self):
        from tools.nlp4lp_downstream_utility import _extract_num_tokens
        toks = _extract_num_tokens("three trucks available", "orig")
        assert any(t.value == 3.0 and t.kind == "int" for t in toks)

    def test_hyphenated_written_word(self):
        from tools.nlp4lp_downstream_utility import _extract_num_tokens
        # "twenty-five percent" → the hyphenated compound is parsed as 25 and then
        # normalized to 0.25 (kind="percent") because "percent" is the next token.
        # This mirrors digit-based behaviour: "25 percent" → value=0.25, kind="percent".
        toks = _extract_num_tokens("twenty-five percent discount applies", "orig")
        assert any(abs(t.value - 0.25) < 1e-9 and t.kind == "percent" for t in toks)

    def test_mixed_digit_and_word(self):
        from tools.nlp4lp_downstream_utility import _extract_num_tokens
        toks = _extract_num_tokens("100 items across three categories", "orig")
        values = {t.value for t in toks}
        assert 100.0 in values
        assert 3.0 in values

    def test_non_number_words_not_extracted(self):
        from tools.nlp4lp_downstream_utility import _extract_num_tokens
        toks = _extract_num_tokens("warehouse facility route", "orig")
        assert toks == []


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck 3: _extract_num_mentions — written-word recognition
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractNumMentionsWordRecognition:
    """_extract_num_mentions (for constrained assignment) must recognise written numbers."""

    def test_digit_mention_extracted(self):
        from tools.nlp4lp_downstream_utility import _extract_num_mentions
        mentions = _extract_num_mentions("Budget is $500", "orig")
        assert len(mentions) == 1
        assert mentions[0].tok.value == 500.0

    def test_written_word_mention_extracted(self):
        from tools.nlp4lp_downstream_utility import _extract_num_mentions
        mentions = _extract_num_mentions("four factories and six depots must be planned", "orig")
        values = {m.tok.value for m in mentions}
        assert 4.0 in values
        assert 6.0 in values

    def test_written_word_mention_has_context_tokens(self):
        from tools.nlp4lp_downstream_utility import _extract_num_mentions
        mentions = _extract_num_mentions("two facilities with shared capacity limit", "orig")
        two_mention = next((m for m in mentions if m.tok.value == 2.0), None)
        assert two_mention is not None
        assert len(two_mention.context_tokens) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck 3: _expected_type — extended slot patterns
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectedTypeExtended:
    """_expected_type must cover all common slot name patterns correctly."""

    def test_percent_slots(self):
        from tools.nlp4lp_downstream_utility import _expected_type
        assert _expected_type("interestRate") == "percent"
        assert _expected_type("fractionProduced") == "percent"
        assert _expected_type("shareOfMarket") == "percent"
        assert _expected_type("proportionAllocated") == "percent"

    def test_int_slots_extended(self):
        from tools.nlp4lp_downstream_utility import _expected_type
        assert _expected_type("numMachines") == "int"
        assert _expected_type("numWorkers") == "int"
        assert _expected_type("numFacilities") == "int"
        assert _expected_type("numWarehouses") == "int"
        assert _expected_type("numVehicles") == "int"
        assert _expected_type("numPeriods") == "int"

    def test_currency_slots_extended(self):
        from tools.nlp4lp_downstream_utility import _expected_type
        # "amount" is ambiguous (AmountPerPill is float) so totalAmount → float
        assert _expected_type("totalAmount") == "float"
        # supplyLimit is a quantity constraint bound (not monetary) → float
        assert _expected_type("supplyLimit") == "float"
        assert _expected_type("allocationBudget") == "currency"
        assert _expected_type("incomeThreshold") == "currency"
        assert _expected_type("salaryLimit") == "currency"

    def test_original_slots_unchanged(self):
        """Existing slot patterns must not regress."""
        from tools.nlp4lp_downstream_utility import _expected_type
        assert _expected_type("totalBudget") == "currency"
        assert _expected_type("unitCost") == "currency"
        assert _expected_type("percentageUsed") == "percent"
        assert _expected_type("numItems") == "int"

    def test_float_fallback(self):
        from tools.nlp4lp_downstream_utility import _expected_type
        assert _expected_type("someUnknownParam") == "float"


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck 4: format_problem_and_ip — graceful missing-formulation notice
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatProblemNoFormulation:
    """format_problem_and_ip must show a clear notice when formulation is missing."""

    def _make_problem_no_form(self):
        return {
            "id": "stub_problem",
            "name": "Stub Problem",
            "description": "A problem without a formulation.",
        }

    def _make_problem_empty_form(self):
        return {
            "id": "empty_form",
            "name": "Empty Form Problem",
            "description": "A problem with an empty formulation dict.",
            "formulation": {"variables": [], "constraints": [], "objective": {}},
        }

    def _make_problem_with_form(self):
        return {
            "id": "full_problem",
            "name": "Full Problem",
            "description": "A problem with a full formulation.",
            "formulation": {
                "variables": [{"symbol": "x", "description": "var", "domain": "x ≥ 0"}],
                "objective": {"sense": "minimize", "expression": "x"},
                "constraints": [{"expression": "x ≥ 1", "description": "lower bound"}],
            },
        }

    def test_no_formulation_shows_notice(self):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(self._make_problem_no_form())
        assert "Formulation not yet available" in output
        assert "Stub Problem" in output

    def test_empty_formulation_shows_notice(self):
        """A formulation dict with all-empty lists is also considered missing."""
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(self._make_problem_empty_form())
        assert "Formulation not yet available" in output

    def test_full_formulation_does_not_show_notice(self):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(self._make_problem_with_form())
        assert "Formulation not yet available" not in output
        assert "Variables" in output
        assert "Objective" in output
        assert "Constraints" in output

    def test_no_formulation_still_shows_description(self):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(self._make_problem_no_form())
        assert "A problem without a formulation." in output

    def test_no_formulation_with_score_prefix(self):
        from retrieval.search import format_problem_and_ip
        output = format_problem_and_ip(self._make_problem_no_form(), score=0.95)
        assert "0.950" in output
        assert "Formulation not yet available" in output


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck 4: custom_problems.json new formulations
# ─────────────────────────────────────────────────────────────────────────────

class TestNewCustomFormulations:
    """The 16 new ILP formulations added to custom_problems.json must be structurally valid."""

    NEW_IDS = [
        "gurobi_optimods_bipartite_matching",
        "gurobi_optimods_max_flow",
        "gurobi_optimods_min_cost_flow",
        "gurobi_optimods_min_cut",
        "gurobi_optimods_portfolio",
        "gurobi_optimods_workforce",
        "gurobi_optimods_mwis",
        "or_lib_assignment",
        "or_lib_bin_packing_1d",
        "or_lib_generalised_assignment",
        "or_lib_graph_colouring",
        "or_lib_crew_scheduling",
        "gurobi_ex_cell_tower_coverage",
        "gurobi_ex_customer_assignment",
        "pyomo_diet",
        "pyomo_facility_location",
    ]

    def _load_custom(self):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        path = root / "data" / "processed" / "custom_problems.json"
        return {p["id"]: p for p in json.loads(path.read_text())}

    def test_all_new_ids_present(self):
        by_id = self._load_custom()
        for pid in self.NEW_IDS:
            assert pid in by_id, f"Expected {pid!r} in custom_problems.json"

    def test_all_new_formulations_have_variables(self):
        by_id = self._load_custom()
        for pid in self.NEW_IDS:
            p = by_id[pid]
            vars_ = p.get("formulation", {}).get("variables", [])
            assert len(vars_) > 0, f"{pid}: variables list is empty"

    def test_all_new_formulations_have_objective(self):
        by_id = self._load_custom()
        for pid in self.NEW_IDS:
            p = by_id[pid]
            obj = p.get("formulation", {}).get("objective", {})
            assert obj.get("sense") in ("minimize", "maximize"), (
                f"{pid}: objective.sense missing or invalid"
            )
            assert obj.get("expression"), f"{pid}: objective.expression is empty"

    def test_all_new_formulations_have_constraints(self):
        by_id = self._load_custom()
        for pid in self.NEW_IDS:
            p = by_id[pid]
            constr = p.get("formulation", {}).get("constraints", [])
            assert len(constr) > 0, f"{pid}: constraints list is empty"

    def test_all_new_formulations_pass_schema_check(self):
        from formulation.verify import verify_problem_schema, verify_formulation_structure
        by_id = self._load_custom()
        for pid in self.NEW_IDS:
            p = by_id[pid]
            schema_errs = verify_problem_schema(p)
            form_errs = verify_formulation_structure(p)
            assert schema_errs == [], f"{pid} schema errors: {schema_errs}"
            assert form_errs == [], f"{pid} formulation errors: {form_errs}"


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck 4: build_extended_catalog merges correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildExtendedCatalog:
    """build_extended_catalog.py must produce a catalog where custom entries override stubs."""

    def test_extended_catalog_larger_than_base(self):
        """The extended catalog must contain at least as many entries as the base catalog."""
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        base = json.loads((root / "data" / "processed" / "all_problems.json").read_text())
        ext = json.loads((root / "data" / "processed" / "all_problems_extended.json").read_text())
        assert len(ext) >= len(base), (
            f"Extended catalog ({len(ext)}) should be ≥ base catalog ({len(base)})"
        )

    def test_new_formulations_override_stubs_in_extended_catalog(self):
        """Problems with new formulations in custom_problems.json must have them in extended catalog."""
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        ext = json.loads((root / "data" / "processed" / "all_problems_extended.json").read_text())
        by_id = {p["id"]: p for p in ext}
        for pid in TestNewCustomFormulations.NEW_IDS:
            assert pid in by_id, f"{pid} missing from extended catalog"
            p = by_id[pid]
            vars_ = (p.get("formulation") or {}).get("variables") or []
            assert len(vars_) > 0, f"{pid} in extended catalog still has no variables"
