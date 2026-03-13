"""Stage 8 regression tests for the harder-grounding distractor-suppression layer.

These tests validate the improvements introduced by four targeted changes:

1. _ENUM_STOP_WORDS extended with modifier adjectives ("total", "overall", etc.)
   that are never standalone enumeration nouns in optimization contexts — prevents
   spurious derived-count mentions like "derived:2 (hours, and total)".

2. Comma + coordinating-conjunction clause-boundary stop in the right narrow
   context window — prevents cross-clause is_total_like contamination (e.g. "4
   labor hours, and total budget is 5000" no longer makes 4 appear total-like).

3. narrow_measure_overlap feature in relation_aware_linking — tight clause-local
   lexical overlap between a mention's narrow context and the slot name tokens,
   providing a strong uncontaminated measure/attribute matching signal.

4. Stage 4 distractor suppression:
   a. "demand" / "supply" removed from _PER_UNIT_LEFT_VERBS — these are noun-form
      words that should not trigger is_per_unit (e.g. "total demand is 300").
   b. Sentence-boundary stop on the operator left window — prevents "at least N"
      from the previous sentence contaminating the operator_tags of a number in the
      next sentence (e.g. "at least 50 units. The profit is 20 per unit." no longer
      tags 20 as a lower-bound value).
   c. bound_to_objective_mismatch penalty — a mention that has a min/max operator
      tag (lower/upper polarity) should not fill a purely objective/coefficient slot
      (e.g. ProfitPerUnit) that carries no bound character.

Test families (matching problem-statement Stage 8 A–F + new G):
  A. Distractor-role mismatch
  B. Measure-family mismatch (protein vs fat)
  C. Cost vs profit discrimination
  D. Heating vs cooling discrimination
  E. Early entity-anchor linking (chair vs dresser)
  F. Preserve easy-family behaviour (percent bounds, derived counts, min/max)
  G. Stage 4 new cases (demand noun fix, operator contamination fix, bound mismatch)
"""
import pytest

from tools.relation_aware_linking import run_relation_aware_grounding


def _ground(query: str, schema: list[str]) -> dict[str, float | None]:
    """Convenience wrapper: run full grounding and return slot→value mapping."""
    _, mentions, _ = run_relation_aware_grounding(query, "orig", schema)
    return {slot: m.value for slot, m in mentions.items()}


# ---------------------------------------------------------------------------
# A. Distractor-role mismatch
# ---------------------------------------------------------------------------

class TestDistractorRoleMismatch:
    """Stage 8 A — wrong assignment / distractor number.

    A total-budget number must not fill a per-unit slot; a worker-count must
    not fill a budget slot; and a per-unit coefficient must prefer the matching
    coefficient slot over the global-capacity slot.
    """

    def test_labor_hours_not_total_budget(self):
        """4 labor hours must go to the per-unit slot, not TotalBudget."""
        q = "There are 100 workers, each product needs 4 labor hours, and total budget is 5000."
        result = _ground(q, ["NumWorkers", "LaborHoursPerProduct", "TotalBudget"])
        assert result["LaborHoursPerProduct"] == pytest.approx(4.0), (
            "4 (per-unit labor hours) should fill LaborHoursPerProduct, not TotalBudget"
        )

    def test_total_budget_not_workers(self):
        """5000 (budget) must not fill NumWorkers."""
        q = "There are 100 workers, each product needs 4 labor hours, and total budget is 5000."
        result = _ground(q, ["NumWorkers", "LaborHoursPerProduct", "TotalBudget"])
        assert result["TotalBudget"] == pytest.approx(5000.0), (
            "5000 (total budget) should fill TotalBudget"
        )
        assert result["NumWorkers"] != pytest.approx(5000.0), (
            "5000 (budget) should not fill the worker-count slot"
        )

    def test_workers_not_budget(self):
        """100 workers must not fill TotalBudget."""
        q = "There are 100 workers, each product needs 4 labor hours, and total budget is 5000."
        result = _ground(q, ["NumWorkers", "LaborHoursPerProduct", "TotalBudget"])
        assert result["TotalBudget"] != pytest.approx(100.0), (
            "100 (worker count) should not fill TotalBudget"
        )

    def test_worker_count_slot(self):
        """100 must eventually fill NumWorkers (the only count slot remaining)."""
        q = "There are 100 workers, each product needs 4 labor hours, and total budget is 5000."
        result = _ground(q, ["NumWorkers", "LaborHoursPerProduct", "TotalBudget"])
        assert result["NumWorkers"] == pytest.approx(100.0), (
            "100 should fill NumWorkers after better mentions take their slots"
        )

    def test_no_spurious_derived_count_from_modifier_adjective(self):
        """'labor hours, and total budget' must NOT produce a derived count of 2.

        Before the _ENUM_STOP_WORDS fix, 'hours' + 'total' were both counted as
        valid enumeration items (total was not in stop words), yielding a spurious
        derived:2 mention that competed with 100 for NumWorkers.
        """
        from tools.nlp4lp_downstream_utility import _extract_enum_derived_counts
        q = "There are 100 workers, each product needs 4 labor hours, and total budget is 5000."
        derived = _extract_enum_derived_counts(q)
        assert derived == [], (
            "No spurious derived count should be produced from 'hours, and total budget'"
        )

    def test_is_total_like_not_contaminated_by_cross_clause(self):
        """4 (per-unit) must NOT be flagged as is_total_like.

        Before the clause-boundary stop, 'total' in the right context window of
        '4' made is_total_like=True, inflating 4's score for TotalBudget.
        """
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        q = "There are 100 workers, each product needs 4 labor hours, and total budget is 5000."
        mentions = _extract_opt_role_mentions(q, "orig")
        mention_4 = next((m for m in mentions if m.value == 4.0 and "derived" not in m.raw_surface), None)
        assert mention_4 is not None, "Mention with value 4.0 must be extracted"
        assert not mention_4.is_total_like, (
            "4 (per-unit labor hours) must not be flagged is_total_like; "
            "'total budget' is in a different clause after the comma-and boundary"
        )

    def test_narrow_context_does_not_include_cross_clause_tokens(self):
        """The narrow_context_tokens of '4' must not include 'total' or 'budget'."""
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        q = "There are 100 workers, each product needs 4 labor hours, and total budget is 5000."
        mentions = _extract_opt_role_mentions(q, "orig")
        mention_4 = next((m for m in mentions if m.value == 4.0 and "derived" not in m.raw_surface), None)
        assert mention_4 is not None
        assert "total" not in mention_4.narrow_context_tokens, (
            "'total' from the next clause must not appear in the narrow context of '4'"
        )
        assert "budget" not in mention_4.narrow_context_tokens, (
            "'budget' from the next clause must not appear in the narrow context of '4'"
        )


# ---------------------------------------------------------------------------
# B. Measure-family mismatch — protein vs fat
# ---------------------------------------------------------------------------

class TestMeasureFamilyMismatch:
    """Stage 8 B — avoid swapping protein and fat values for feed slots."""

    def test_protein_to_protein_slot(self):
        q = "Feed A contains 10 protein and 8 fat."
        result = _ground(q, ["ProteinFeedA", "FatFeedA"])
        assert result["ProteinFeedA"] == pytest.approx(10.0), (
            "10 (protein) should fill ProteinFeedA, not FatFeedA"
        )

    def test_fat_to_fat_slot(self):
        q = "Feed A contains 10 protein and 8 fat."
        result = _ground(q, ["ProteinFeedA", "FatFeedA"])
        assert result["FatFeedA"] == pytest.approx(8.0), (
            "8 (fat) should fill FatFeedA, not ProteinFeedA"
        )

    def test_no_protein_fat_swap(self):
        """Values must not be swapped: protein ≠ 8, fat ≠ 10."""
        q = "Feed A contains 10 protein and 8 fat."
        result = _ground(q, ["ProteinFeedA", "FatFeedA"])
        assert result["ProteinFeedA"] != pytest.approx(8.0), "Protein slot must not get fat value"
        assert result["FatFeedA"] != pytest.approx(10.0), "Fat slot must not get protein value"


# ---------------------------------------------------------------------------
# C. Cost vs profit discrimination
# ---------------------------------------------------------------------------

class TestCostVsProfit:
    """Stage 8 C — profit and cost values must not be swapped."""

    def test_profit_slot(self):
        q = "Product X yields 12 dollars profit and costs 5 dollars to produce."
        result = _ground(q, ["ProfitPerUnit", "CostPerUnit"])
        assert result["ProfitPerUnit"] == pytest.approx(12.0), (
            "12 (profit) should fill ProfitPerUnit"
        )

    def test_cost_slot(self):
        q = "Product X yields 12 dollars profit and costs 5 dollars to produce."
        result = _ground(q, ["ProfitPerUnit", "CostPerUnit"])
        assert result["CostPerUnit"] == pytest.approx(5.0), (
            "5 (cost) should fill CostPerUnit"
        )

    def test_no_profit_cost_swap(self):
        q = "Product X yields 12 dollars profit and costs 5 dollars to produce."
        result = _ground(q, ["ProfitPerUnit", "CostPerUnit"])
        assert result["ProfitPerUnit"] != pytest.approx(5.0), "Profit slot must not get cost value"
        assert result["CostPerUnit"] != pytest.approx(12.0), "Cost slot must not get profit value"


# ---------------------------------------------------------------------------
# D. Heating vs cooling discrimination
# ---------------------------------------------------------------------------

class TestHeatingVsCooling:
    """Stage 8 D — heating and cooling hours must not be swapped."""

    def test_heating_slot(self):
        q = "Regular glass requires 3 heating hours and 5 cooling hours."
        result = _ground(q, ["HeatingHours", "CoolingHours"])
        assert result["HeatingHours"] == pytest.approx(3.0), (
            "3 (heating) should fill HeatingHours"
        )

    def test_cooling_slot(self):
        q = "Regular glass requires 3 heating hours and 5 cooling hours."
        result = _ground(q, ["HeatingHours", "CoolingHours"])
        assert result["CoolingHours"] == pytest.approx(5.0), (
            "5 (cooling) should fill CoolingHours"
        )

    def test_no_heating_cooling_swap(self):
        q = "Regular glass requires 3 heating hours and 5 cooling hours."
        result = _ground(q, ["HeatingHours", "CoolingHours"])
        assert result["HeatingHours"] != pytest.approx(5.0), "Heating slot must not get cooling value"
        assert result["CoolingHours"] != pytest.approx(3.0), "Cooling slot must not get heating value"


# ---------------------------------------------------------------------------
# E. Early entity-anchor linking — chair vs dresser
# ---------------------------------------------------------------------------

class TestEntityAnchorLinking:
    """Stage 8 E — entity-local context should guide chair vs dresser assignment."""

    def test_chair_wood(self):
        q = "A chair uses 2 units of wood, while a dresser uses 5."
        result = _ground(q, ["WoodPerChair", "WoodPerDresser"])
        assert result["WoodPerChair"] == pytest.approx(2.0), (
            "2 (chair context) should fill WoodPerChair"
        )

    def test_dresser_wood(self):
        q = "A chair uses 2 units of wood, while a dresser uses 5."
        result = _ground(q, ["WoodPerChair", "WoodPerDresser"])
        assert result["WoodPerDresser"] == pytest.approx(5.0), (
            "5 (dresser context) should fill WoodPerDresser"
        )

    def test_no_chair_dresser_swap(self):
        q = "A chair uses 2 units of wood, while a dresser uses 5."
        result = _ground(q, ["WoodPerChair", "WoodPerDresser"])
        assert result["WoodPerChair"] != pytest.approx(5.0), "Chair slot must not get dresser value"
        assert result["WoodPerDresser"] != pytest.approx(2.0), "Dresser slot must not get chair value"


# ---------------------------------------------------------------------------
# F. Preserve easy-family behaviour
# ---------------------------------------------------------------------------

class TestPreserveEasyFamilyBehavior:
    """Stage 8 F — new hard-family layer must not break easy-family gains."""

    def test_percent_upper_bound_preserved(self):
        """'at most 40%' → percent upper-bound slot must still work."""
        q = "At most 40% of workers can be assigned to night shift."
        result = _ground(q, ["MaxNightShiftPercent"])
        assert result["MaxNightShiftPercent"] == pytest.approx(0.4), (
            "40% should be stored as 0.4 in a percent slot"
        )

    def test_derived_count_two_types_preserved(self):
        """'two types of jars' → NumJarTypes = 2 (word-number extraction)."""
        q = "There are two types of jars."
        result = _ground(q, ["NumJarTypes"])
        assert result["NumJarTypes"] == pytest.approx(2.0), (
            "Word number 'two' should still fill the count slot"
        )

    def test_enumeration_count_preserved(self):
        """'phones and laptops' → NumProducts = 2 (enum-derived count)."""
        q = "A company produces phones and laptops."
        result = _ground(q, ["NumProducts"])
        assert result["NumProducts"] == pytest.approx(2.0), (
            "Enum-derived count (phones + laptops = 2) should fill NumProducts"
        )

    def test_bound_directions_preserved(self):
        """'at least 10 and at most 20' → min=10, max=20 (bound disambiguation)."""
        q = "At least 10 and at most 20 units must be produced."
        result = _ground(q, ["MinUnits", "MaxUnits"])
        assert result["MinUnits"] == pytest.approx(10.0), "10 should fill the lower-bound slot"
        assert result["MaxUnits"] == pytest.approx(20.0), "20 should fill the upper-bound slot"

    def test_no_bound_direction_swap(self):
        q = "At least 10 and at most 20 units must be produced."
        result = _ground(q, ["MinUnits", "MaxUnits"])
        assert result["MinUnits"] != pytest.approx(20.0), "Min slot must not get max value"
        assert result["MaxUnits"] != pytest.approx(10.0), "Max slot must not get min value"

    def test_total_vs_per_unit_preserved(self):
        """'cost 15 each, budget 3000' → coefficient and budget correctly assigned."""
        q = "Each item costs 15 dollars. The company can spend at most 3000 dollars."
        result = _ground(q, ["Budget", "CostPerItem"])
        assert result["CostPerItem"] == pytest.approx(15.0), (
            "15 (per-unit cost) should fill the coefficient slot"
        )
        assert result["Budget"] == pytest.approx(3000.0), (
            "3000 (total budget) should fill the capacity slot"
        )


# ---------------------------------------------------------------------------
# G. Stage 4 new distractor-suppression cases
# ---------------------------------------------------------------------------

class TestStage4DistractorSuppression:
    """Stage 4 G — demand-noun fix, operator-contamination fix, bound-to-objective mismatch.

    Three deterministic fixes that prevent cross-sentence and cross-role
    distractor assignments discovered during Stage 4 auditing.
    """

    def test_total_demand_not_profit_slot(self):
        """300 (total demand) must not fill ProfitPerUnit; 15 (profit) must."""
        q = "Total demand is 300. Profit per unit is 15."
        result = _ground(q, ["TotalDemand", "ProfitPerUnit"])
        assert result["TotalDemand"] == pytest.approx(300.0), (
            "300 is the total demand value and should fill TotalDemand"
        )
        assert result["ProfitPerUnit"] == pytest.approx(15.0), (
            "15 is the per-unit profit and should fill ProfitPerUnit"
        )

    def test_lower_bound_not_profit_slot(self):
        """50 (min-production bound) must not fill ProfitPerUnit; 20 (profit) must."""
        q = "We must produce at least 50 units. The profit is 20 per unit."
        result = _ground(q, ["MinProduction", "ProfitPerUnit"])
        assert result["MinProduction"] == pytest.approx(50.0), (
            "50 has 'at least' → lower-bound polarity; should fill MinProduction"
        )
        assert result["ProfitPerUnit"] == pytest.approx(20.0), (
            "20 has 'profit per unit' context; should fill ProfitPerUnit"
        )

    def test_demand_noun_not_per_unit_flag(self):
        """'total demand is 300' must not set is_per_unit=True on 300.

        Previously 'demand' was in _PER_UNIT_LEFT_VERBS, causing is_per_unit=True
        for the noun usage in 'total demand is N'.  After the fix, 300 should be
        total-like, not per-unit.
        """
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions = _extract_opt_role_mentions("Total demand is 300.", "orig")
        demand_mention = next((m for m in mentions if m.value == 300.0), None)
        assert demand_mention is not None
        assert not demand_mention.is_per_unit, (
            "'demand' is a noun here; the mention should NOT be flagged is_per_unit"
        )

    def test_operator_not_contaminated_across_sentences(self):
        """'at least 50 units. The profit is 20 per unit.' — 20 must not get min operator.

        Previously the operator left window crossed the sentence boundary, making
        'at least' contaminate the operator_tags of 20 in the next sentence.
        After the fix, 20 should have neutral polarity (no min operator).
        """
        from tools.nlp4lp_downstream_utility import _extract_opt_role_mentions
        mentions = _extract_opt_role_mentions(
            "We must produce at least 50 units. The profit is 20 per unit.", "orig"
        )
        profit_mention = next((m for m in mentions if m.value == 20.0), None)
        assert profit_mention is not None
        assert "min" not in profit_mention.operator_tags, (
            "20 in the second sentence should not inherit 'at least' from the first"
        )

    def test_upper_bound_not_cost_coefficient_slot(self):
        """45 (at-most bound) must not fill CostPerUnit; 12 (cost) must."""
        q = "Production must not exceed 45 units. Each unit costs 12 dollars."
        result = _ground(q, ["MaxProduction", "CostPerUnit"])
        assert result["MaxProduction"] == pytest.approx(45.0), (
            "45 has 'not exceed' → upper-bound polarity; should fill MaxProduction"
        )
        assert result["CostPerUnit"] == pytest.approx(12.0), (
            "12 has 'costs' context and is per-unit; should fill CostPerUnit"
        )
