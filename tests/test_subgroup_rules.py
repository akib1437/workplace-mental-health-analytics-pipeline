import pandas as pd

from src.analyse_subgroups import analyse_subgroups
from src.apply_disclosure_rules import disclose_crosstab


def test_small_observed_cells_are_suppressed_and_logged() -> None:
    table = pd.DataFrame(
        {
            "Minimal": [12, 2],
            "Moderate": [1, 8],
        },
        index=["Group A", "Group B"],
    )

    config = {
        "disclosure_control": {
            "minimum_cell_count": 5,
            "text_for_suppressed_cell": "Suppressed (<5)",
        }
    }

    displayed, disclosure_log = disclose_crosstab(
        table=table,
        config=config,
        subgroup="group",
        outcome="severity",
    )

    assert displayed.loc["Group A", "Moderate"] == "Suppressed (<5)"
    assert displayed.loc["Group B", "Minimal"] == "Suppressed (<5)"
    assert len(disclosure_log) == 2
    assert all(entry["action"] == "suppressed" for entry in disclosure_log)


def test_expected_count_failure_is_not_reportable_or_fdr_eligible() -> None:
    scored = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B", "B"],
            "outcome": ["Minimal"] * 11 + ["Moderate"],
            "binary": ["Below"] * 11 + ["Moderate or higher"],
        }
    )

    config = {
        "scales": {
            "test_scale": {
                "severity_column": "outcome",
                "moderate_or_higher_column": "binary",
            }
        },
        "analysis": {
            "subgroup_columns": ["group"],
            "fdr_method": "fdr_bh",
            "fdr_alpha": 0.05,
            "chi_square": {
                "min_expected_cell_count": 5,
                "max_fraction_expected_below_min": 0.20,
                "minimum_expected_cell_allowed": 1,
            },
            "effect_size": {
                "illustrative_reporting_signal_threshold": 0.25,
            },
        },
        "disclosure_control": {
            "minimum_cell_count": 5,
            "text_for_suppressed_cell": "Suppressed (<5)",
        },
    }

    results, _, _ = analyse_subgroups(scored, config)

    assert not results["chi_square_reportable"].any()
    assert results["p_value_fdr"].isna().all()
    assert not results["illustrative_reporting_signal"].any()
    assert results["expected_cells_below_absolute_floor"].gt(0).any()