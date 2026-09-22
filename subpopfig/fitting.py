"""Fit a LogNormal mixture to each entry and compute subpopulation statistics.

For each entry, Gaussian mixtures with 1 to ``max_components`` components are
fit to the log measurements, and the component count with the lowest BIC is
kept along with the model that produced it. Each measurement is assigned to
the component with the highest posterior. Component median, weight, 25th
percentile, and 75th percentile are computed on the raw measurements assigned
to each component, and components are numbered in ascending order of median.
A component that receives no measurements is dropped, so ``n_components`` can
fall below ``k``.

One random generator seeded with ``seed`` is created per entry and passed
through the k = 1 to max_components fits in order, so an entry's fit does not
depend on the other entries. An entry with fewer than
``min_measurements_for_mixture`` measurements, or with a single distinct
value, takes one component without fitting. An entry on which every k fails
takes one component and is listed in the summary.

Percentiles use the midpoint (Hazen) definition, which is the definition
MATLAB's ``prctile`` uses.

Structure: public functions are controllers. Private functions marked MODEL
compute and return values; private functions marked VIEW print or warn.
"""
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from .errors import FitError
from .workbook import Labels, Levels, WorkbookData

__all__ = ["FitConfig", "KSummary", "FitIssues", "FitResult", "fit_mixtures"]


@dataclass(frozen=True)
class FitConfig:
    """Resolved fit settings."""

    max_components: int = 4
    replicates: int = 5
    max_iter: int = 1000
    tol: float = 1e-6
    regularization: float = 1e-3
    min_measurements_for_mixture: int = 10
    seed: int = 0


@dataclass
class ComponentSearch:
    """Outcome of the BIC search on one entry."""

    k: int
    bic: np.ndarray
    model: Optional[Any]
    status: str
    converged: Optional[bool]
    n_failed: int
    warned_k: np.ndarray


@dataclass
class ComponentStats:
    """Per-component statistics sorted by median, with renumbered assignments."""

    median: np.ndarray
    weight: np.ndarray
    p25: np.ndarray
    p75: np.ndarray
    n: np.ndarray
    hard: np.ndarray


@dataclass
class KSummary:
    """Entries at each k (rows: nominal levels; columns: k = 1..K) and mean k."""

    nominal_display: list
    counts: np.ndarray
    mean_k: np.ndarray


@dataclass
class FitIssues:
    """Entries the fit handled in a nonstandard way."""

    n_too_few: int
    n_no_spread: int
    failed_ids: list
    non_converged_ids: list
    partial_failure_ids: list
    warned_ids: list
    n_warned_fits: int
    empty_component_ids: list


@dataclass
class FitResult:
    """Workbook context plus the fitted entries."""

    source: str
    labels: Labels
    levels: Levels
    selection_rule: str
    config: FitConfig
    k_summary: KSummary
    issues: FitIssues
    entries: list


# ---------------------------------------------------------------------------
# Public controller
# ---------------------------------------------------------------------------

def fit_mixtures(
    workbook: WorkbookData,
    *,
    max_components: int = 4,
    replicates: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-6,
    regularization: float = 1e-3,
    min_measurements_for_mixture: int = 10,
    seed: int = 0,
    verbose: bool = True,
) -> FitResult:
    """Fit every entry of ``workbook`` and summarize the fits.

    Parameters
    ----------
    max_components : highest number of subpopulations tried per entry.
    replicates : random starting points per fit (``n_init``).
    max_iter : EM iteration limit per starting point.
    tol : convergence tolerance on the per-measurement log-likelihood.
    regularization : variance added to each component (``reg_covar``), which
        keeps a component from collapsing onto repeated identical values.
    min_measurements_for_mixture : entries with fewer measurements receive
        one subpopulation without fitting.
    seed : random seed, reset for each entry.
    verbose : print the fit settings, k summary, and fit issues.
    """
    config = _normalize_fit_config(
        max_components=max_components, replicates=replicates, max_iter=max_iter,
        tol=tol, regularization=regularization,
        min_measurements_for_mixture=min_measurements_for_mixture, seed=seed)
    result = _resolve_fits(workbook, config)
    if verbose:
        _report_fits(result)
    _warn_fit_failures(result)
    return result


# ---------------------------------------------------------------------------
# Model sub-controllers
# ---------------------------------------------------------------------------

def _resolve_fits(workbook: WorkbookData, config: FitConfig) -> FitResult:
    """SUB-CONTROLLER (model): fit every entry, then summarize k per nominal
    level and collect fit issues. Calls model functions only."""
    entries = [_fit_single_entry(e, config) for e in workbook.entries]
    return FitResult(
        source=workbook.source,
        labels=workbook.labels,
        levels=workbook.levels,
        selection_rule=workbook.selection_rule,
        config=config,
        k_summary=_summarize_k_by_nominal(entries, workbook.levels, config),
        issues=_collect_fit_issues(entries),
        entries=entries,
    )


def _fit_single_entry(entry, config: FitConfig):
    """SUB-CONTROLLER (model): select the component count, assign
    measurements, compute component statistics, and return a fitted copy of
    the entry. Calls model functions only."""
    search = _select_component_count(entry.measurements, config)
    hard = _assign_to_components(entry.measurements, search)
    stats = _compute_component_stats(entry.measurements, hard)
    return _attach_fit_results(entry, search, stats)


# ---------------------------------------------------------------------------
# Model terminals
# ---------------------------------------------------------------------------

def _normalize_fit_config(**settings) -> FitConfig:
    """MODEL: validate the fit settings. Count settings must be positive
    integers; tol and regularization must be positive."""
    for name in ("max_components", "replicates", "max_iter", "min_measurements_for_mixture"):
        v = settings[name]
        if isinstance(v, bool) or not float(v).is_integer() or v < 1:
            raise FitError(name, f"{name} must be a positive integer; got {v!r}.")
        settings[name] = int(v)
    for name in ("tol", "regularization"):
        v = float(settings[name])
        if not np.isfinite(v) or v <= 0:
            raise FitError(name, f"{name} must be a positive number; got {settings[name]!r}.")
        settings[name] = v
    settings["seed"] = int(settings["seed"])
    return FitConfig(**settings)


def _select_component_count(measurements: np.ndarray, config: FitConfig) -> ComponentSearch:
    """MODEL: BIC search over k = 1 to max_components on log measurements.
    The winning model is kept, so the assignments come from the same fit that
    won BIC. Status is 'fit', 'tooFewMeasurements', 'noSpread', or
    'fitFailed'; every status other than 'fit' leaves k = 1 with no model.
    warned_k marks each k whose fit raised a ConvergenceWarning or a
    RuntimeWarning; other warnings are not counted."""
    K = config.max_components
    bic = np.full(K, np.nan)
    warned = np.zeros(K, dtype=bool)
    log_data = np.log(np.asarray(measurements, dtype=float)).reshape(-1, 1)
    n = log_data.shape[0]

    if n < config.min_measurements_for_mixture:
        return ComponentSearch(1, bic, None, "tooFewMeasurements", None, 0, warned)
    if np.unique(log_data).size < 2:
        return ComponentSearch(1, bic, None, "noSpread", None, 0, warned)

    rng = np.random.RandomState(config.seed)
    models = [None] * K
    n_failed = 0
    for k in range(1, K + 1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                gm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    tol=config.tol,
                    reg_covar=config.regularization,
                    max_iter=config.max_iter,
                    n_init=config.replicates,
                    init_params="k-means++",
                    random_state=rng,
                ).fit(log_data)
                bic[k - 1] = gm.bic(log_data)
                models[k - 1] = gm
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                bic[k - 1] = np.inf
                n_failed += 1
        warned[k - 1] = any(issubclass(w.category, (ConvergenceWarning, RuntimeWarning))
                            for w in caught)

    if not np.any(np.isfinite(bic)):
        return ComponentSearch(1, bic, None, "fitFailed", None, n_failed, warned)
    best = int(np.argmin(bic))
    return ComponentSearch(best + 1, bic, models[best], "fit",
                           bool(models[best].converged_), n_failed, warned)


def _assign_to_components(measurements: np.ndarray, search: ComponentSearch) -> np.ndarray:
    """MODEL: 0-based component index per measurement by maximum posterior
    under the winning model. Without a model every measurement belongs to
    component 0."""
    if search.model is None:
        return np.zeros(len(measurements), dtype=int)
    log_data = np.log(np.asarray(measurements, dtype=float)).reshape(-1, 1)
    return np.argmax(search.model.predict_proba(log_data), axis=1)


def _compute_component_stats(measurements: np.ndarray, hard: np.ndarray) -> ComponentStats:
    """MODEL: median, weight, P25, P75, and count per component that received
    measurements, computed on raw measurements and sorted by median.
    Assignments are renumbered to match the sorted order. Percentiles use the
    Hazen definition."""
    x_all = np.asarray(measurements, dtype=float).ravel()
    labels = np.unique(hard)
    m = labels.size
    med, w, p25, p75, cnt = (np.zeros(m) for _ in range(5))
    for c, label in enumerate(labels):
        x = x_all[hard == label]
        med[c] = np.median(x)
        w[c] = x.size / x_all.size
        p25[c], p75[c] = np.percentile(x, [25, 75], method="hazen")
        cnt[c] = x.size
    order = np.argsort(med, kind="stable")
    rank_of = np.zeros(labels.max() + 1, dtype=int)
    rank_of[labels[order]] = np.arange(m)
    return ComponentStats(median=med[order], weight=w[order], p25=p25[order],
                          p75=p75[order], n=cnt[order].astype(int), hard=rank_of[hard])


def _attach_fit_results(entry, search: ComponentSearch, stats: ComponentStats):
    """MODEL: a copy of the entry with the fit fields filled."""
    return dataclasses.replace(
        entry,
        k=search.k,
        bic=search.bic,
        fit_status=search.status,
        converged=search.converged,
        n_failed_k=search.n_failed,
        warned_k=search.warned_k,
        n_components=int(stats.median.size),
        hard_assignments=stats.hard,
        component_medians=stats.median,
        component_weights=stats.weight,
        component_p25=stats.p25,
        component_p75=stats.p75,
        component_n=stats.n,
    )


def _summarize_k_by_nominal(entries: list, levels: Levels, config: FitConfig) -> KSummary:
    """MODEL: entry counts at each k and mean k per nominal level."""
    K = config.max_components
    n_levels = len(levels.nominal)
    counts = np.zeros((n_levels, K), dtype=int)
    mean_k = np.full(n_levels, np.nan)
    nom_idx = np.array([e.nominal_idx for e in entries])
    ks = np.array([e.k for e in entries])
    for li in range(n_levels):
        mask = nom_idx == li
        for k in range(1, K + 1):
            counts[li, k - 1] = int(np.sum(ks[mask] == k))
        if mask.any():
            mean_k[li] = ks[mask].mean()
    return KSummary(nominal_display=list(levels.nominal_display), counts=counts, mean_k=mean_k)


def _collect_fit_issues(entries: list) -> FitIssues:
    """MODEL: entries that skipped or failed fitting, winning models that did
    not converge, entries where some k failed, entries where some k fit raised
    a warning, the total count of warned k fits, and entries with components
    that received no measurements."""
    return FitIssues(
        n_too_few=sum(e.fit_status == "tooFewMeasurements" for e in entries),
        n_no_spread=sum(e.fit_status == "noSpread" for e in entries),
        failed_ids=[e.entry_id for e in entries if e.fit_status == "fitFailed"],
        non_converged_ids=[e.entry_id for e in entries if e.converged is False],
        partial_failure_ids=[e.entry_id for e in entries if e.n_failed_k > 0],
        warned_ids=[e.entry_id for e in entries if e.warned_k.any()],
        n_warned_fits=int(sum(e.warned_k.sum() for e in entries)),
        empty_component_ids=[e.entry_id for e in entries if e.n_components < e.k],
    )


# ---------------------------------------------------------------------------
# View sub-controller and terminals
# ---------------------------------------------------------------------------

def _report_fits(result: FitResult) -> None:
    """SUB-CONTROLLER (view): print the fit settings, the k summary per
    nominal level, and the fit issues. Calls view functions only."""
    _print_fit_header(result)
    _print_k_summary(result)
    _print_fit_issues(result)


def _print_fit_header(result: FitResult) -> None:
    """VIEW: entry count and fit settings."""
    c = result.config
    print("\n=== fit_mixtures ===")
    print(f"Entries: {len(result.entries)}")
    print(f"LogNormal mixture, BIC over k = 1 to {c.max_components}; "
          f"replicates {c.replicates}; max_iter {c.max_iter}; tol {c.tol:g}; "
          f"regularization {c.regularization:g}; seed {c.seed}; "
          f"minimum {c.min_measurements_for_mixture} measurements for k > 1")


def _print_k_summary(result: FitResult) -> None:
    """VIEW: entries at each k and mean k per nominal level, in figure order."""
    s = result.k_summary
    K = s.counts.shape[1]
    print("\nComponent count by BIC:")
    header = f"  {result.labels.nominal:<14}" + "".join(f" {'k=%d' % k:>5}" for k in range(1, K + 1))
    print(header + f" {'mean k':>7}")
    for li, name in enumerate(s.nominal_display):
        row = f"  {name:<14}" + "".join(f" {v:>5d}" for v in s.counts[li])
        print(row + f" {s.mean_k[li]:>7.2f}")


def _print_fit_issues(result: FitResult) -> None:
    """VIEW: counts of entries that skipped fitting and of warned k fits, then
    the lists of entries handled in a nonstandard way."""
    issues = result.issues
    print("\nFit issues:")
    print(f"  below {result.config.min_measurements_for_mixture} measurements "
          f"(k = 1, not fit): {issues.n_too_few}")
    print(f"  single distinct value (k = 1, not fit): {issues.n_no_spread}")
    print(f"  k fits that raised a convergence or numerical warning: {issues.n_warned_fits}")
    lists = (
        ("entries where at least one k failed", issues.partial_failure_ids),
        ("entries where a k fit raised a warning", issues.warned_ids),
        ("every k failed (k = 1)", issues.failed_ids),
        ("winning model did not converge", issues.non_converged_ids),
        ("components with no measurements dropped", issues.empty_component_ids),
    )
    for label, ids in lists:
        line = f"  {label}: {len(ids)}"
        if ids:
            line += f" ({', '.join(ids)})"
        print(line)
    print()


def _warn_fit_failures(result: FitResult) -> None:
    """VIEW: one warning when any entry failed every k."""
    if result.issues.failed_ids:
        warnings.warn(
            f"{len(result.issues.failed_ids)} entries failed every k and were given "
            "one component.", stacklevel=3)
