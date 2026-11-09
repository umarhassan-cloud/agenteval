# Judge calibration report

Human-labeled runs: **50**. Agreement is Cohen's kappa with a 95% bootstrap CI (2,000 resamples over runs). Labels: SUCCESS + 8 failure modes.

| Classifier | Cohen's κ | 95% CI | Interpretation | % agreement | Pass/fail κ |
|---|---:|---|---|---:|---:|
| Rule-based | 0.773 | [0.64, 0.90] | substantial | 82.0% | 1.000 |
| LLM judge (simulated) | 0.675 | [0.53, 0.82] | substantial | 74.0% | 0.958 |
| Hybrid (rules + judge) | 0.723 | [0.59, 0.86] | substantial | 78.0% | 1.000 |

## Rule-based: agreement matrix

| human \ auto | SUCC | IP | LOOP | GUE | WT | HA | HF | PS | WS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SUCCESS | **20** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| IGNORED_POLICY | 0 | **3** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LOOPING | 0 | 0 | **4** | 0 | 0 | 0 | 0 | 0 | 0 |
| GAVE_UP_EARLY | 0 | 0 | 1 | **3** | 0 | 0 | 0 | 0 | 0 |
| WRONG_TOOL | 0 | 0 | 0 | 0 | **1** | 0 | 2 | 0 | 0 |
| HALLUCINATED_ARG | 0 | 0 | 0 | 0 | 0 | **1** | 3 | 1 | 0 |
| HALLUCINATED_FACT | 0 | 0 | 0 | 0 | 0 | 0 | **4** | 0 | 0 |
| PARTIAL_SUCCESS | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **4** | 0 |
| WRONG_STATE | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | **1** |

_Legend: SUCC=SUCCESS, IP=IGNORED_POLICY, LOOP=LOOPING, GUE=GAVE_UP_EARLY, WT=WRONG_TOOL, HA=HALLUCINATED_ARG, HF=HALLUCINATED_FACT, PS=PARTIAL_SUCCESS, WS=WRONG_STATE_

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUCCESS | 1.00 | 1.00 | 1.00 | 20 |
| IGNORED_POLICY | 0.75 | 1.00 | 0.86 | 3 |
| LOOPING | 0.67 | 1.00 | 0.80 | 4 |
| GAVE_UP_EARLY | 1.00 | 0.75 | 0.86 | 4 |
| WRONG_TOOL | 1.00 | 0.33 | 0.50 | 3 |
| HALLUCINATED_ARG | 1.00 | 0.20 | 0.33 | 5 |
| HALLUCINATED_FACT | 0.44 | 1.00 | 0.61 | 4 |
| PARTIAL_SUCCESS | 0.80 | 1.00 | 0.89 | 4 |
| WRONG_STATE | 1.00 | 0.33 | 0.50 | 3 |

<details><summary>9 disagreements</summary>

| Run | Human | Auto |
|---|---|---|
| `support-002__planact__llama-70b-style__t1` | HALLUCINATED_ARG | HALLUCINATED_FACT |
| `coding-002__react__gpt-4o-style__t1` | HALLUCINATED_ARG | HALLUCINATED_FACT |
| `support-006__planact__gpt-4o-style__t1` | HALLUCINATED_ARG | PARTIAL_SUCCESS |
| `coding-004__naive__llama-70b-style__t2` | WRONG_TOOL | HALLUCINATED_FACT |
| `support-006__naive__llama-70b-style__t1` | GAVE_UP_EARLY | LOOPING |
| `coding-003__planact__llama-70b-style__t0` | WRONG_TOOL | HALLUCINATED_FACT |
| `booking-004__react__llama-70b-style__t0` | HALLUCINATED_ARG | HALLUCINATED_FACT |
| `support-006__naive__llama-70b-style__t0` | WRONG_STATE | LOOPING |
| `coding-002__naive__llama-70b-style__t1` | WRONG_STATE | IGNORED_POLICY |

</details>

## LLM judge (simulated): agreement matrix

| human \ auto | SUCC | IP | LOOP | GUE | WT | HA | HF | PS | WS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SUCCESS | **19** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| IGNORED_POLICY | 0 | **3** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LOOPING | 0 | 0 | **3** | 1 | 0 | 0 | 0 | 0 | 0 |
| GAVE_UP_EARLY | 0 | 0 | 2 | **2** | 0 | 0 | 0 | 0 | 0 |
| WRONG_TOOL | 0 | 0 | 0 | 0 | **1** | 0 | 1 | 1 | 0 |
| HALLUCINATED_ARG | 0 | 0 | 0 | 0 | 0 | **1** | 2 | 1 | 1 |
| HALLUCINATED_FACT | 0 | 0 | 0 | 0 | 0 | 0 | **4** | 0 | 0 |
| PARTIAL_SUCCESS | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **3** | 1 |
| WRONG_STATE | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | **1** |

_Legend: SUCC=SUCCESS, IP=IGNORED_POLICY, LOOP=LOOPING, GUE=GAVE_UP_EARLY, WT=WRONG_TOOL, HA=HALLUCINATED_ARG, HF=HALLUCINATED_FACT, PS=PARTIAL_SUCCESS, WS=WRONG_STATE_

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUCCESS | 1.00 | 0.95 | 0.97 | 20 |
| IGNORED_POLICY | 0.75 | 1.00 | 0.86 | 3 |
| LOOPING | 0.50 | 0.75 | 0.60 | 4 |
| GAVE_UP_EARLY | 0.67 | 0.50 | 0.57 | 4 |
| WRONG_TOOL | 1.00 | 0.33 | 0.50 | 3 |
| HALLUCINATED_ARG | 1.00 | 0.20 | 0.33 | 5 |
| HALLUCINATED_FACT | 0.50 | 1.00 | 0.67 | 4 |
| PARTIAL_SUCCESS | 0.60 | 0.75 | 0.67 | 4 |
| WRONG_STATE | 0.33 | 0.33 | 0.33 | 3 |

<details><summary>13 disagreements</summary>

| Run | Human | Auto |
|---|---|---|
| `support-002__planact__llama-70b-style__t1` | HALLUCINATED_ARG | WRONG_STATE |
| `coding-002__react__gpt-4o-style__t1` | HALLUCINATED_ARG | HALLUCINATED_FACT |
| `coding-002__planact__claude-style__t0` | SUCCESS | HALLUCINATED_FACT |
| `support-006__planact__gpt-4o-style__t1` | HALLUCINATED_ARG | PARTIAL_SUCCESS |
| `booking-007__react__llama-70b-style__t2` | PARTIAL_SUCCESS | WRONG_STATE |
| `coding-004__naive__llama-70b-style__t2` | WRONG_TOOL | HALLUCINATED_FACT |
| `support-006__naive__llama-70b-style__t1` | GAVE_UP_EARLY | LOOPING |
| `booking-006__naive__claude-style__t1` | GAVE_UP_EARLY | LOOPING |
| `support-001__naive__llama-70b-style__t1` | LOOPING | GAVE_UP_EARLY |
| `coding-003__planact__llama-70b-style__t0` | WRONG_TOOL | PARTIAL_SUCCESS |
| `booking-004__react__llama-70b-style__t0` | HALLUCINATED_ARG | HALLUCINATED_FACT |
| `support-006__naive__llama-70b-style__t0` | WRONG_STATE | LOOPING |
| `coding-002__naive__llama-70b-style__t1` | WRONG_STATE | IGNORED_POLICY |

</details>

## Hybrid (rules + judge): agreement matrix

| human \ auto | SUCC | IP | LOOP | GUE | WT | HA | HF | PS | WS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SUCCESS | **20** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| IGNORED_POLICY | 0 | **3** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LOOPING | 0 | 0 | **4** | 0 | 0 | 0 | 0 | 0 | 0 |
| GAVE_UP_EARLY | 0 | 0 | 2 | **2** | 0 | 0 | 0 | 0 | 0 |
| WRONG_TOOL | 0 | 0 | 0 | 0 | **1** | 0 | 1 | 1 | 0 |
| HALLUCINATED_ARG | 0 | 0 | 0 | 0 | 0 | **1** | 2 | 1 | 1 |
| HALLUCINATED_FACT | 0 | 0 | 0 | 0 | 0 | 0 | **4** | 0 | 0 |
| PARTIAL_SUCCESS | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **3** | 1 |
| WRONG_STATE | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | **1** |

_Legend: SUCC=SUCCESS, IP=IGNORED_POLICY, LOOP=LOOPING, GUE=GAVE_UP_EARLY, WT=WRONG_TOOL, HA=HALLUCINATED_ARG, HF=HALLUCINATED_FACT, PS=PARTIAL_SUCCESS, WS=WRONG_STATE_

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUCCESS | 1.00 | 1.00 | 1.00 | 20 |
| IGNORED_POLICY | 0.75 | 1.00 | 0.86 | 3 |
| LOOPING | 0.57 | 1.00 | 0.73 | 4 |
| GAVE_UP_EARLY | 1.00 | 0.50 | 0.67 | 4 |
| WRONG_TOOL | 1.00 | 0.33 | 0.50 | 3 |
| HALLUCINATED_ARG | 1.00 | 0.20 | 0.33 | 5 |
| HALLUCINATED_FACT | 0.57 | 1.00 | 0.73 | 4 |
| PARTIAL_SUCCESS | 0.60 | 0.75 | 0.67 | 4 |
| WRONG_STATE | 0.33 | 0.33 | 0.33 | 3 |

<details><summary>11 disagreements</summary>

| Run | Human | Auto |
|---|---|---|
| `support-002__planact__llama-70b-style__t1` | HALLUCINATED_ARG | WRONG_STATE |
| `coding-002__react__gpt-4o-style__t1` | HALLUCINATED_ARG | HALLUCINATED_FACT |
| `support-006__planact__gpt-4o-style__t1` | HALLUCINATED_ARG | PARTIAL_SUCCESS |
| `booking-007__react__llama-70b-style__t2` | PARTIAL_SUCCESS | WRONG_STATE |
| `coding-004__naive__llama-70b-style__t2` | WRONG_TOOL | HALLUCINATED_FACT |
| `support-006__naive__llama-70b-style__t1` | GAVE_UP_EARLY | LOOPING |
| `booking-006__naive__claude-style__t1` | GAVE_UP_EARLY | LOOPING |
| `coding-003__planact__llama-70b-style__t0` | WRONG_TOOL | PARTIAL_SUCCESS |
| `booking-004__react__llama-70b-style__t0` | HALLUCINATED_ARG | HALLUCINATED_FACT |
| `support-006__naive__llama-70b-style__t0` | WRONG_STATE | LOOPING |
| `coding-002__naive__llama-70b-style__t1` | WRONG_STATE | IGNORED_POLICY |

</details>
