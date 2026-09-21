"""End-to-end and unit tests on synthetic data."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from subpopfig import (FigureError, WorkbookError, fit_mixtures, from_dataframe,
                       plot_subpopulations)
from subpopfig.fitting import _compute_component_stats

SCHEMA = {
    "EntryID": "entry",
    "Nominal": {"column": "group", "label": "Group"},
    "Ordinal": {"column": "level", "display_column": "level_display", "label": "Level"},
    "Measurement": {"column": "size", "label": "Size"},
    "Selected": "selected",
}


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for group in ("A", "B"):
        for level in (1, 2, 3):
            for rep in range(3):
                n = 400
                if group == "B" and level > 1:
                    x = np.concatenate([rng.lognormal(np.log(50), 0.1, n // 2),
                                        rng.lognormal(np.log(500), 0.1, n // 2)])
                else:
                    x = rng.lognormal(np.log(200), 0.1, n)
                selected = rep == 0 and not (group == "A" and level == 2)
                for v in x:
                    rows.append({"entry": f"{group}{level}_{rep}", "group": group,
                                 "level": level, "level_display": f"L{level}",
                                 "size": v, "selected": selected})
    return pd.DataFrame(rows)


def test_pipeline_runs_and_finds_two_components():
    wb = from_dataframe(make_data(), SCHEMA, verbose=False)
    assert len(wb.entries) == 18
    assert wb.levels.nominal == ["A", "B"]
    fits = fit_mixtures(wb, verbose=False)
    ks = {e.entry_id: e.k for e in fits.entries}
    assert ks["A1_0"] == 1
    assert ks["B2_0"] == 2
    b2 = next(e for e in fits.entries if e.entry_id == "B2_0")
    assert b2.component_medians[0] < b2.component_medians[1]
    np.testing.assert_allclose(b2.component_weights, [0.5, 0.5])

    result = plot_subpopulations(fits, verbose=False)
    assert [(b.nominal_label, b.ordinal_label) for b in result.bridged] == [("A", "L2")]
    assert result.figure is not None


def test_fit_is_reproducible():
    wb = from_dataframe(make_data(), SCHEMA, verbose=False)
    a = fit_mixtures(wb, verbose=False)
    b = fit_mixtures(wb, verbose=False)
    for ea, eb in zip(a.entries, b.entries):
        np.testing.assert_array_equal(ea.component_medians, eb.component_medians)


def test_percentiles_use_hazen_definition():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    stats = _compute_component_stats(x, np.zeros(4, dtype=int))
    # Hazen: sorted values sit at the 12.5, 37.5, 62.5, 87.5 percentiles.
    assert stats.p25[0] == pytest.approx(1.5)
    assert stats.p75[0] == pytest.approx(3.5)


def test_inconsistent_entry_is_rejected():
    data = make_data()
    data.loc[0, "group"] = "B"
    with pytest.raises(WorkbookError) as err:
        from_dataframe(data, SCHEMA, verbose=False)
    assert err.value.code == "inconsistentEntry"


def test_nonpositive_measurement_is_rejected():
    data = make_data()
    data.loc[5, "size"] = 0
    with pytest.raises(WorkbookError) as err:
        from_dataframe(data, SCHEMA, verbose=False)
    assert err.value.code == "invalidValues"


def test_bad_format_string_is_rejected():
    fits = fit_mixtures(from_dataframe(make_data(), SCHEMA, verbose=False), verbose=False)
    with pytest.raises(FigureError) as err:
        plot_subpopulations(fits, text={"weight_format": "Weight"}, verbose=False)
    assert err.value.code == "textFormat"
