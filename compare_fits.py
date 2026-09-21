"""Compare the Python fits with fits exported from the MATLAB pipeline.

Run exportFitsForComparison.m in MATLAB first, then:

    python validation/compare_fits.py Input.xlsx matlab_fits.csv

The two implementations use different random number generators and EM
implementations, so an entry whose BIC values at two k lie close together can
settle on a different k. The report lists those entries, and for entries where
k agrees it reports how far the component statistics differ.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from subpopfig import fit_mixtures, read_workbook

STAT_COLUMNS = ("median", "weight", "p25", "p75")


def main(argv=None) -> None:
    """CONTROLLER: fit in Python, load the MATLAB export, compare, report."""
    args = _parse_args(argv)
    matlab = _load_matlab_fits(args.matlab_csv)
    python = _python_fit_table(args.workbook)
    comparison = _compare_tables(matlab, python)
    _print_comparison(comparison, args.rtol, args.show)


def _python_fit_table(workbook_path: str) -> pd.DataFrame:
    """SUB-CONTROLLER (model): read and fit the workbook, then tabulate."""
    fits = fit_mixtures(read_workbook(workbook_path, verbose=False), verbose=False)
    return _fits_to_table(fits)


def _parse_args(argv):
    """MODEL: command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workbook", help="the input workbook both pipelines read")
    parser.add_argument("matlab_csv", help="output of exportFitsForComparison.m")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="relative difference counted as a mismatch (default 1e-3)")
    parser.add_argument("--show", type=int, default=10,
                        help="entries listed per section (default 10)")
    return parser.parse_args(argv)


def _load_matlab_fits(path: str) -> pd.DataFrame:
    """MODEL: the MATLAB export with entry IDs as text."""
    table = pd.read_csv(path, dtype={"entry_id": str})
    table["entry_id"] = table["entry_id"].str.strip()
    return table


def _fits_to_table(fits) -> pd.DataFrame:
    """MODEL: one row per component, components numbered from 1 as in MATLAB."""
    rows = []
    for e in fits.entries:
        for c in range(e.n_components):
            rows.append({"entry_id": e.entry_id, "k": e.k, "component": c + 1,
                         "median": e.component_medians[c], "weight": e.component_weights[c],
                         "p25": e.component_p25[c], "p75": e.component_p75[c]})
    return pd.DataFrame(rows)


def _compare_tables(matlab: pd.DataFrame, python: pd.DataFrame) -> dict:
    """MODEL: entries present in only one table, entries whose k or component
    count differs, and the largest relative difference of each statistic for
    entries whose component count agrees."""
    m_ids, p_ids = set(matlab["entry_id"]), set(python["entry_id"])
    m_k = matlab.groupby("entry_id").agg(k=("k", "first"), n=("component", "size"))
    p_k = python.groupby("entry_id").agg(k=("k", "first"), n=("component", "size"))
    shared = sorted(m_ids & p_ids)
    k_table = m_k.loc[shared].join(p_k.loc[shared], lsuffix="_matlab", rsuffix="_python")
    k_mismatch = k_table[(k_table["k_matlab"] != k_table["k_python"])
                         | (k_table["n_matlab"] != k_table["n_python"])]

    agree = k_table.index[(k_table["n_matlab"] == k_table["n_python"])]
    merged = matlab[matlab["entry_id"].isin(agree)].merge(
        python[python["entry_id"].isin(agree)], on=["entry_id", "component"],
        suffixes=("_matlab", "_python"))
    for stat in STAT_COLUMNS:
        a, b = merged[f"{stat}_matlab"], merged[f"{stat}_python"]
        merged[f"{stat}_rel"] = (a - b).abs() / np.maximum(a.abs(), np.finfo(float).tiny)
    per_entry = merged.groupby("entry_id")[[f"{s}_rel" for s in STAT_COLUMNS]].max()
    per_entry["worst"] = per_entry.max(axis=1)

    return {
        "n_matlab": len(m_ids), "n_python": len(p_ids), "n_shared": len(shared),
        "only_matlab": sorted(m_ids - p_ids), "only_python": sorted(p_ids - m_ids),
        "k_mismatch": k_mismatch,
        "per_entry": per_entry.sort_values("worst", ascending=False),
    }


def _print_comparison(comparison: dict, rtol: float, show: int) -> None:
    """VIEW: entry coverage, k agreement, and statistic differences."""
    c = comparison
    print(f"Entries: MATLAB {c['n_matlab']}, Python {c['n_python']}, shared {c['n_shared']}")
    for label, ids in (("only in MATLAB", c["only_matlab"]), ("only in Python", c["only_python"])):
        if ids:
            print(f"  {label}: {len(ids)} ({', '.join(ids[:show])})")

    km = c["k_mismatch"]
    print(f"\nk agreement: {c['n_shared'] - len(km)} of {c['n_shared']}")
    if len(km):
        print(f"  {'entry':<30} {'k MATLAB':>9} {'k Python':>9}")
        for entry_id, row in km.head(show).iterrows():
            print(f"  {entry_id:<30} {row['k_matlab']:>9d} {row['k_python']:>9d}")

    pe = c["per_entry"]
    if pe.empty:
        return
    n_over = int((pe["worst"] > rtol).sum())
    print(f"\nEntries with matching component count: {len(pe)}")
    for stat in STAT_COLUMNS:
        print(f"  max relative difference in {stat}: {pe[f'{stat}_rel'].max():.3g}")
    print(f"  entries exceeding rtol {rtol:g}: {n_over}")
    if n_over:
        print(f"  {'entry':<30} {'worst relative difference':>26}")
        for entry_id, row in pe[pe["worst"] > rtol].head(show).iterrows():
            print(f"  {entry_id:<30} {row['worst']:>26.3g}")


if __name__ == "__main__":
    main()
