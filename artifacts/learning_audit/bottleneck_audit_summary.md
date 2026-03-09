# NLP4LP Bottleneck Audit Summary

## Overview

- **Total eval examples**: 331
- **Pairwise ranker data available**: False
- **Total ranker rows**: 0
- **Flagged in any slice**: 320
- **Flagged in ≥2 slices**: 213

## Slice Counts

| Slice | Count | Fraction |
|-------|-------|----------|
| entity_association_risk | 14 | 4.2% |
| lower_upper_risk | 116 | 35.0% |
| multi_numeric_confusion | 288 | 87.0% |
| total_vs_per_unit_risk | 191 | 57.7% |
| percent_vs_absolute_risk | 43 | 13.0% |

## Heuristic Definitions

### entity_association_risk

Query mentions entity/person names alongside numbers, raising risk of wrong variable-entity association.

### lower_upper_risk

Query contains both lower-bound and upper-bound cues (e.g., 'at least' and 'at most'), risking lower/upper confusion.

### multi_numeric_confusion

Query contains >=3 distinct numeric values, raising multi-float confusable grounding risk.

### total_vs_per_unit_risk

Query uses both 'total/aggregate' and 'per/each/unit' language alongside numbers, risking total-vs-per-unit confusion.

### percent_vs_absolute_risk

Query mixes percentage mentions and large absolute numeric values, risking percent vs absolute value confusion.

## Example Flagged Cases

### entity_association_risk (first 3 examples)

- **ID**: `nlp4lp_test_0`
  - **Reason**: entity cues (['Mrs. Watson', 'Mrs. Watson']) co-occur with 5 numeric mentions
  - **Query snippet**: *Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested *

- **ID**: `nlp4lp_test_6`
  - **Reason**: entity cues (['Elm Furniture']) co-occur with 8 numeric mentions
  - **Query snippet**: *A chair produced by Elm Furniture yields a profit of $43, while every dresser yields a $52 profit. Each week, 17 gallons of stain and 11 lengths of oak wood are available. Each cha*

- **ID**: `nlp4lp_test_25`
  - **Reason**: entity cues (['Oil Max', 'Oil Max Pro', 'Oil Max']) co-occur with 11 numeric mentions
  - **Query snippet**: *A car manufacturer makes two types of car oils: Oil Max and Oil Max Pro. A container of Oil Max contains 46 grams of substance A, 43 grams of substance B and 56 grams of substance *

### lower_upper_risk (first 3 examples)

- **ID**: `nlp4lp_test_0`
  - **Reason**: lower cues ['minimum', 'at least'] AND upper cues ['at most'] both present
  - **Query snippet**: *Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested *

- **ID**: `nlp4lp_test_2`
  - **Reason**: lower cues ['at least', 'at least'] AND upper cues ['no more than', 'maximum'] both present
  - **Query snippet**: *A cleaning company located in Edmonton wants to get the best exposure possible for promoting their new dishwashing detergent without exceeding their $250,000 advertising budget. To*

- **ID**: `nlp4lp_test_5`
  - **Reason**: lower cues ['at least'] AND upper cues ['at most'] both present
  - **Query snippet**: *A company is deciding where to promote their product. Some options include z-tube, soorchle engine, and wassa advertisements. The cost for each option and the number of viewers the*

### multi_numeric_confusion (first 3 examples)

- **ID**: `nlp4lp_test_0`
  - **Reason**: 5 distinct numeric values: ['0.50', '1', '20', '20000', '760000']
  - **Query snippet**: *Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested *

- **ID**: `nlp4lp_test_1`
  - **Reason**: 6 distinct numeric values: ['2', '3', '4', '40', '5', '70']
  - **Query snippet**: *A breakfast joint makes two different sandwiches: a regular and a special. Both need eggs and bacon. Each regular sandwich requires 2 eggs and 3 slices of bacon. Each special sandw*

- **ID**: `nlp4lp_test_2`
  - **Reason**: 10 distinct numeric values: ['1', '15', '2', '250000', '35', '40']
  - **Query snippet**: *A cleaning company located in Edmonton wants to get the best exposure possible for promoting their new dishwashing detergent without exceeding their $250,000 advertising budget. To*

### total_vs_per_unit_risk (first 3 examples)

- **ID**: `nlp4lp_test_0`
  - **Reason**: total cues ['total', 'total'] AND per-unit cues ['Each', 'each'] both present with 5 numbers
  - **Query snippet**: *Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested *

- **ID**: `nlp4lp_test_1`
  - **Reason**: total cues ['total'] AND per-unit cues ['Each', 'Each'] both present with 8 numbers
  - **Query snippet**: *A breakfast joint makes two different sandwiches: a regular and a special. Both need eggs and bacon. Each regular sandwich requires 2 eggs and 3 slices of bacon. Each special sandw*

- **ID**: `nlp4lp_test_5`
  - **Reason**: total cues ['total', 'total'] AND per-unit cues ['each', 'each'] both present with 8 numbers
  - **Query snippet**: *A company is deciding where to promote their product. Some options include z-tube, soorchle engine, and wassa advertisements. The cost for each option and the number of viewers the*

### percent_vs_absolute_risk (first 3 examples)

- **ID**: `nlp4lp_test_0`
  - **Reason**: percent cues ['20%'] AND absolute value cues ['$760000', '$0.50'] co-present
  - **Query snippet**: *Mrs. Watson wants to invest in the real-estate market and has a total budget of at most $760000. She has two choices which include condos and detached houses. Each dollar invested *

- **ID**: `nlp4lp_test_5`
  - **Reason**: percent cues ['5%'] AND absolute value cues ['$1000', '$200'] co-present
  - **Query snippet**: *A company is deciding where to promote their product. Some options include z-tube, soorchle engine, and wassa advertisements. The cost for each option and the number of viewers the*

- **ID**: `nlp4lp_test_14`
  - **Reason**: percent cues ['10%', '15%'] AND absolute value cues ['$600,000', '$200,000'] co-present
  - **Query snippet**: *My family has decided to invest in real state for the first time. Currently, they have $600,000 to invest, some in apartments and the rest in townhouses. The money invested in apar*

