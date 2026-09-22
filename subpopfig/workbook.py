"""Read a figure workbook, or a DataFrame and schema, into one record per entry.

The schema maps each role to a data column. EntryID, Nominal, Ordinal, and
Measurement are required; SortKey and Selected are optional. Data columns the
schema does not name are ignored. The role name PlateID is accepted as an
alias for EntryID.

Every entry-level value must agree on every row of an entry, the ordinal and
sort key must be finite numbers, and every measurement must be positive and
finite. A violation raises WorkbookError naming the entries involved.

Without a SortKey column, the sort key is the number of rows per entry.
Without a Selected column, each nominal and ordinal cluster selects the entry
whose sort key lies closest to the median sort key across all entries; a tie
resolves to the first entry in row order. Nominal levels follow first
appearance in the data, and ordinal levels are sorted numerically.

Structure: public functions are controllers. Private functions marked MODEL
compute and return values; private functions marked VIEW print or warn. Model
and view functions do not call other functions of this package.
"""
from __future__ import annotations

import math
import numbers
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .errors import WorkbookError

__all__ = [
    "Role",
    "Entry",
    "Levels",
    "Labels",
    "ClusterSummary",
    "WorkbookData",
    "read_workbook",
    "from_dataframe",
]

REQUIRED_ROLES = ("EntryID", "Nominal", "Ordinal", "Measurement")
OPTIONAL_ROLES = ("SortKey", "Selected")
ALL_ROLES = REQUIRED_ROLES + OPTIONAL_ROLES
ROLE_ALIASES = {"EntryID": ("EntryID", "PlateID")}
SCHEMA_HEADERS = ("Role", "Column", "DisplayColumn", "Label")
DEFAULT_SORT_KEY_LABEL = "Measurements per entry"
SELECTED_TEXT = frozenset({"true", "yes", "y", "1"})
MAX_LISTED = 10


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _empty_float() -> np.ndarray:
    return np.empty(0, dtype=float)


def _empty_int() -> np.ndarray:
    return np.empty(0, dtype=int)


def _empty_bool() -> np.ndarray:
    return np.empty(0, dtype=bool)


@dataclass
class Role:
    """How one schema role maps onto the data."""

    present: bool = False
    column: str = ""
    display_column: str = ""
    label: str = ""
    schema_label: str = ""


@dataclass
class Entry:
    """One entry (a plate, well, or other sample unit).

    The fields after ``nominal_idx`` stay empty until ``fit_mixtures`` fills
    them. Component arrays are ordered by ascending median.
    """

    entry_id: str
    nominal: str
    nominal_display: str
    ordinal: float
    ordinal_display: str
    sort_key: float
    selected: bool
    measurements: np.ndarray
    n_rows: int
    nominal_idx: int = -1
    k: Optional[int] = None
    bic: np.ndarray = field(default_factory=_empty_float)
    fit_status: str = ""
    converged: Optional[bool] = None
    n_failed_k: int = 0
    warned_k: np.ndarray = field(default_factory=_empty_bool)
    n_components: int = 0
    hard_assignments: np.ndarray = field(default_factory=_empty_int)
    component_medians: np.ndarray = field(default_factory=_empty_float)
    component_weights: np.ndarray = field(default_factory=_empty_float)
    component_p25: np.ndarray = field(default_factory=_empty_float)
    component_p75: np.ndarray = field(default_factory=_empty_float)
    component_n: np.ndarray = field(default_factory=_empty_int)


@dataclass
class Levels:
    """Nominal levels in figure order and ordinal levels in numeric order."""

    nominal: list
    nominal_display: list
    ordinal: np.ndarray
    ordinal_display: list


@dataclass
class Labels:
    """Role labels as written in the schema (``unit`` may be blank)."""

    unit: str
    nominal: str
    ordinal: str
    measurement: str
    sort_key: str


@dataclass
class ClusterSummary:
    """Entry, row, and selection counts for one nominal and ordinal cluster."""

    nominal_display: str
    ordinal_display: str
    n_entries: int
    n_rows: int
    n_selected: int


@dataclass
class WorkbookData:
    """Everything the fit and the figure need from the input."""

    source: str
    sheets: Optional[tuple]
    roles: dict
    labels: Labels
    levels: Levels
    selection_rule: str
    clusters: list
    n_rows: int
    entries: list


# ---------------------------------------------------------------------------
# Public controllers
# ---------------------------------------------------------------------------

def read_workbook(
    path: Union[str, Path],
    data_sheet: str = "Data",
    schema_sheet: str = "Schema",
    *,
    verbose: bool = True,
) -> WorkbookData:
    """Read an .xlsx workbook with a data sheet and a schema sheet.

    Parameters
    ----------
    path : path to the workbook.
    data_sheet, schema_sheet : sheet names, ``'Data'`` and ``'Schema'`` by
        default.
    verbose : print the role mapping, level order, and per-cluster selection.
    """
    data, schema = _read_workbook_sheets(Path(path), data_sheet, schema_sheet)
    workbook = _resolve_workbook(data, schema, str(path), (data_sheet, schema_sheet))
    if verbose:
        _report_workbook(workbook)
    _warn_workbook(workbook)
    return workbook


def from_dataframe(
    data: pd.DataFrame,
    schema: Union[pd.DataFrame, Mapping[str, Any]],
    *,
    verbose: bool = True,
    source: str = "DataFrame",
) -> WorkbookData:
    """Build workbook data from a DataFrame and a schema.

    ``schema`` is either a DataFrame with the headers Role, Column,
    DisplayColumn, and Label, or a mapping from role name to a column name or
    to a mapping with the keys ``column``, ``display_column``, and ``label``::

        schema = {
            "EntryID": "plate_id",
            "Nominal": {"column": "genotype", "display_column": "genotype_display",
                        "label": "Genotype"},
            "Ordinal": {"column": "passage", "display_column": "passage_display"},
            "Measurement": {"column": "radius_um", "label": "Radius (µm)"},
        }
    """
    workbook = _resolve_workbook(data, schema, source, None)
    if verbose:
        _report_workbook(workbook)
    _warn_workbook(workbook)
    return workbook


# ---------------------------------------------------------------------------
# Model sub-controller
# ---------------------------------------------------------------------------

def _resolve_workbook(data, schema, source, sheets) -> WorkbookData:
    """SUB-CONTROLLER (model): map roles to columns, group rows into entries,
    validate entry values, order the levels, resolve the selection, and
    summarize each cluster. Calls model functions only."""
    data = _prepare_data(data)
    schema_frame = _schema_to_frame(schema)
    schema_text = {h: _coerce_text(schema_frame[h]) for h in SCHEMA_HEADERS}
    roles = _resolve_roles(schema_text, list(data.columns))

    text_columns, numeric_columns = _role_column_names(roles)
    cols = {name: _coerce_text(data[col]) for name, col in text_columns.items()}
    cols.update({name: _coerce_numeric(data[col], col)
                 for name, col in numeric_columns.items()})
    cols["selected"] = (_coerce_selected(data[roles["Selected"].column])
                        if roles["Selected"].present else None)

    entries = _group_rows_into_entries(cols, roles)
    _validate_entry_values(entries)
    entries, levels = _order_levels(entries)
    entries, selection_rule = _resolve_selection(entries, roles)

    return WorkbookData(
        source=source,
        sheets=sheets,
        roles=roles,
        labels=_collect_labels(roles),
        levels=levels,
        selection_rule=selection_rule,
        clusters=_summarize_clusters(entries, levels),
        n_rows=len(data),
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Model terminals
# ---------------------------------------------------------------------------

def _read_workbook_sheets(path: Path, data_sheet: str, schema_sheet: str):
    """MODEL: read the data and schema sheets with headers kept as written."""
    if not path.is_file():
        raise WorkbookError("noFile", f"No workbook at {path}.")
    with pd.ExcelFile(path) as xls:
        present = list(xls.sheet_names)
        for sheet in (data_sheet, schema_sheet):
            if sheet not in present:
                raise WorkbookError(
                    "noSheet",
                    f"Workbook has no '{sheet}' sheet. Sheets present: {', '.join(present)}")
        schema = xls.parse(schema_sheet)
        data = xls.parse(data_sheet)
    return data, schema


def _prepare_data(data) -> pd.DataFrame:
    """MODEL: drop fully blank rows and make every header a string."""
    if not isinstance(data, pd.DataFrame):
        raise WorkbookError("dataType", "Data must be a pandas DataFrame.")
    data = data.dropna(how="all").reset_index(drop=True)
    data = data.rename(columns=str)
    if len(data) == 0:
        raise WorkbookError("emptyData", "The data have no rows.")
    return data


def _schema_to_frame(schema) -> pd.DataFrame:
    """MODEL: the schema as a DataFrame with the four schema headers. A
    mapping is converted row by row. Role and Column are required headers;
    a missing DisplayColumn or Label column is treated as blank."""
    if isinstance(schema, pd.DataFrame):
        frame = schema.dropna(how="all").reset_index(drop=True).rename(columns=str)
    elif isinstance(schema, Mapping):
        rows = []
        for role, spec in schema.items():
            if isinstance(spec, str):
                spec = {"column": spec}
            if not isinstance(spec, Mapping):
                raise WorkbookError(
                    "schemaType",
                    f"Schema entry for role '{role}' must be a column name or a mapping.")
            keys = {str(k).lower().replace("_", ""): v for k, v in spec.items()}
            rows.append({
                "Role": role,
                "Column": keys.get("column", ""),
                "DisplayColumn": keys.get("displaycolumn", ""),
                "Label": keys.get("label", ""),
            })
        frame = pd.DataFrame(rows, columns=list(SCHEMA_HEADERS))
    else:
        raise WorkbookError("schemaType", "Schema must be a DataFrame or a mapping.")

    for header in SCHEMA_HEADERS[:2]:
        if header not in frame.columns:
            raise WorkbookError(
                "schemaHeader",
                f"Schema has no '{header}' column. Expected headers: {', '.join(SCHEMA_HEADERS)}")
    frame = frame.copy()
    for header in SCHEMA_HEADERS[2:]:
        if header not in frame.columns:
            frame[header] = ""
    return frame


def _coerce_text(values) -> list:
    """MODEL: each value as stripped text. Integer-valued numbers are written
    without a decimal point, other numbers with up to 15 significant digits,
    and blanks become empty strings."""
    out = []
    for x in values:
        if isinstance(x, str):
            out.append(x.strip())
        elif isinstance(x, (bool, np.bool_)):
            out.append("1" if x else "0")
        elif isinstance(x, numbers.Real) and not math.isnan(float(x)):
            xf = float(x)
            out.append(str(int(xf)) if xf.is_integer() else "%.15g" % xf)
        else:
            out.append("")
    return out


def _coerce_numeric(series: pd.Series, column: str) -> np.ndarray:
    """MODEL: the column as float. Text converts where it parses as a number
    and becomes NaN otherwise, which fails validation downstream."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float, na_value=np.nan)
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        stripped = series.map(lambda v: v.strip() if isinstance(v, str) else v)
        return pd.to_numeric(stripped, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    raise WorkbookError("notNumeric", f"Column '{column}' must hold numbers.")


def _coerce_selected(series: pd.Series) -> np.ndarray:
    """MODEL: the Selected column as bool. Logical values stand; numbers are
    selected when nonzero (blank is not); text is selected when it reads
    true, yes, y, or 1."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(series):
        v = series.to_numpy(dtype=float, na_value=np.nan)
        return ~np.isnan(v) & (v != 0)
    out = np.zeros(len(series), dtype=bool)
    for i, x in enumerate(series):
        if isinstance(x, (bool, np.bool_)):
            out[i] = bool(x)
        elif isinstance(x, numbers.Real):
            out[i] = not math.isnan(float(x)) and float(x) != 0
        elif isinstance(x, str):
            out[i] = x.strip().lower() in SELECTED_TEXT
    return out


def _resolve_roles(schema_text: dict, data_columns: list) -> dict:
    """MODEL: one Role per role name. Role matching ignores case. A missing
    required role, a duplicated role, or a column absent from the data raises
    WorkbookError. ``schema_label`` holds the Label cell as written; ``label``
    falls back to the column name when that cell is blank."""
    role_names = [r.lower() for r in schema_text["Role"]]
    roles = {}
    for role in ALL_ROLES:
        names = {a.lower() for a in ROLE_ALIASES.get(role, (role,))}
        idx = [i for i, r in enumerate(role_names) if r in names]
        if len(idx) > 1:
            raise WorkbookError("duplicateRole", f"Schema lists role '{role}' {len(idx)} times.")
        if not idx or not schema_text["Column"][idx[0]]:
            if role in REQUIRED_ROLES:
                raise WorkbookError("missingRole",
                                    f"Schema has no column for required role '{role}'.")
            roles[role] = Role()
            continue
        i = idx[0]
        column = schema_text["Column"][i]
        if column not in data_columns:
            raise WorkbookError(
                "missingColumn",
                f"Role '{role}' maps to column '{column}', which the data lack. "
                f"Data columns: {', '.join(data_columns)}")
        display = schema_text["DisplayColumn"][i]
        if display and display not in data_columns:
            raise WorkbookError(
                "missingColumn",
                f"Role '{role}' maps its display to column '{display}', which the data lack.")
        schema_label = schema_text["Label"][i]
        roles[role] = Role(present=True, column=column, display_column=display,
                           label=schema_label or column, schema_label=schema_label)
    if not roles["SortKey"].present:
        roles["SortKey"].label = DEFAULT_SORT_KEY_LABEL
    return roles


def _role_column_names(roles: dict):
    """MODEL: the data column behind each text field and each numeric field.
    A role with no display column uses its value column as the display."""
    nominal, ordinal = roles["Nominal"], roles["Ordinal"]
    text = {
        "entry_id": roles["EntryID"].column,
        "nominal": nominal.column,
        "nominal_display": nominal.display_column or nominal.column,
        "ordinal_display": ordinal.display_column or ordinal.column,
    }
    numeric = {"ordinal": ordinal.column, "measurement": roles["Measurement"].column}
    if roles["SortKey"].present:
        numeric["sort_key"] = roles["SortKey"].column
    return text, numeric


def _group_rows_into_entries(cols: dict, roles: dict) -> list:
    """MODEL: one Entry per entry ID in order of first appearance. Every
    entry-level value must agree across the entry's rows; entries that
    disagree are listed in one error. Without a SortKey column the sort key
    is the entry's row count."""
    ids = cols["entry_id"]
    n_blank = sum(1 for v in ids if not v)
    if n_blank:
        raise WorkbookError("blankEntryID", f"{n_blank} data rows have no entry ID.")

    rows_by_id = {}
    for row, entry_id in enumerate(ids):
        rows_by_id.setdefault(entry_id, []).append(row)

    has_sort = roles["SortKey"].present
    has_sel = roles["Selected"].present
    entries, issues = [], []
    for entry_id, rows in rows_by_id.items():
        idx = np.asarray(rows)
        first = rows[0]
        bad = []
        for name, label in (("nominal", "nominal"),
                            ("nominal_display", "nominal display"),
                            ("ordinal_display", "ordinal display")):
            if any(cols[name][r] != cols[name][first] for r in rows):
                bad.append(label)
        for name, label, present in (("ordinal", "ordinal", True),
                                     ("sort_key", "sort key", has_sort)):
            if present:
                v = cols[name][idx]
                if not (np.all(v == v[0]) or np.all(np.isnan(v))):
                    bad.append(label)
        if has_sel and np.any(cols["selected"][idx] != cols["selected"][first]):
            bad.append("selected")
        if bad:
            issues.append(f"{entry_id} ({', '.join(bad)})")

        entries.append(Entry(
            entry_id=entry_id,
            nominal=cols["nominal"][first],
            nominal_display=cols["nominal_display"][first],
            ordinal=float(cols["ordinal"][first]),
            ordinal_display=cols["ordinal_display"][first],
            sort_key=float(cols["sort_key"][first]) if has_sort else float(len(rows)),
            selected=bool(cols["selected"][first]) if has_sel else False,
            measurements=cols["measurement"][idx].copy(),
            n_rows=len(rows),
        ))

    if issues:
        shown = issues[:MAX_LISTED]
        raise WorkbookError(
            "inconsistentEntry",
            f"{len(issues)} entries carry entry-level values that differ between rows. "
            f"First {len(shown)}: {'; '.join(shown)}")
    return entries


def _validate_entry_values(entries: list) -> None:
    """MODEL: every entry needs a finite ordinal, a finite sort key, and
    measurements that are all positive and finite, since the LogNormal fit
    and the log axis require positive values. Violations are raised together,
    listing up to ten entries per check."""
    msgs = []
    bad_ord = [e.entry_id for e in entries if not np.isfinite(e.ordinal)]
    if bad_ord:
        msgs.append(f"ordinal not a finite number on {len(bad_ord)} entries: "
                    f"{', '.join(bad_ord[:MAX_LISTED])}")
    bad_key = [e.entry_id for e in entries if not np.isfinite(e.sort_key)]
    if bad_key:
        msgs.append(f"sort key not a finite number on {len(bad_key)} entries: "
                    f"{', '.join(bad_key[:MAX_LISTED])}")
    bad_meas = []
    for e in entries:
        m = e.measurements
        n_bad = int(np.sum(~np.isfinite(m) | (m <= 0)))
        if n_bad:
            bad_meas.append(f"{e.entry_id} ({n_bad})")
    if bad_meas:
        msgs.append(f"measurements blank, non-numeric, or not positive on {len(bad_meas)} "
                    f"entries: {', '.join(bad_meas[:MAX_LISTED])}")
    if msgs:
        raise WorkbookError("invalidValues", "\n".join(msgs))


def _order_levels(entries: list):
    """MODEL: nominal levels in order of first appearance and ordinal levels
    sorted numerically, each carrying one display value. A value with two
    display values raises WorkbookError. Each entry receives the 0-based
    index of its nominal level."""
    nominal = list(dict.fromkeys(e.nominal for e in entries))
    nominal_display = []
    for li, value in enumerate(nominal):
        members = [e for e in entries if e.nominal == value]
        displays = list(dict.fromkeys(e.nominal_display for e in members))
        if len(displays) > 1:
            raise WorkbookError(
                "nominalDisplay",
                f"Nominal value '{value}' carries display values: {', '.join(displays)}")
        nominal_display.append(displays[0])
        for e in members:
            e.nominal_idx = li

    ordinal = np.unique(np.array([e.ordinal for e in entries], dtype=float))
    ordinal_display = []
    for value in ordinal:
        displays = list(dict.fromkeys(e.ordinal_display for e in entries if e.ordinal == value))
        if len(displays) > 1:
            raise WorkbookError(
                "ordinalDisplay",
                f"Ordinal value {value:g} carries display values: {', '.join(displays)}")
        ordinal_display.append(displays[0])

    return entries, Levels(nominal=nominal, nominal_display=nominal_display,
                           ordinal=ordinal, ordinal_display=ordinal_display)


def _resolve_selection(entries: list, roles: dict):
    """MODEL: with a Selected column, entry flags stand as read. Without one,
    each nominal and ordinal cluster selects the entry whose sort key lies
    closest to the median sort key across all entries; a tie resolves to the
    first entry in row order."""
    if roles["Selected"].present:
        return entries, f"Selected column '{roles['Selected'].column}'"
    target = float(np.median([e.sort_key for e in entries]))
    clusters = {}
    for i, e in enumerate(entries):
        clusters.setdefault((e.nominal_idx, e.ordinal), []).append(i)
    for members in clusters.values():
        keys = np.array([entries[i].sort_key for i in members])
        entries[members[int(np.argmin(np.abs(keys - target)))]].selected = True
    return entries, f"sort key closest to the median across entries ({target:g})"


def _collect_labels(roles: dict) -> Labels:
    """MODEL: role labels the fit and the figure read."""
    return Labels(unit=roles["EntryID"].schema_label,
                  nominal=roles["Nominal"].label,
                  ordinal=roles["Ordinal"].label,
                  measurement=roles["Measurement"].label,
                  sort_key=roles["SortKey"].label)


def _summarize_clusters(entries: list, levels: Levels) -> list:
    """MODEL: entries, rows, and selected entries per nominal and ordinal
    cluster, in level order."""
    clusters = []
    for li, nom_display in enumerate(levels.nominal_display):
        for oi, value in enumerate(levels.ordinal):
            members = [e for e in entries if e.nominal_idx == li and e.ordinal == value]
            if not members:
                continue
            clusters.append(ClusterSummary(
                nominal_display=nom_display,
                ordinal_display=levels.ordinal_display[oi],
                n_entries=len(members),
                n_rows=sum(e.n_rows for e in members),
                n_selected=sum(e.selected for e in members),
            ))
    return clusters


# ---------------------------------------------------------------------------
# View sub-controller and terminals
# ---------------------------------------------------------------------------

def _report_workbook(wb: WorkbookData) -> None:
    """SUB-CONTROLLER (view): print totals, the role mapping, the level
    order, and the per-cluster selection. Calls view functions only."""
    _print_workbook_header(wb)
    _print_role_map(wb)
    _print_level_order(wb)
    _print_cluster_selection(wb)


def _print_workbook_header(wb: WorkbookData) -> None:
    """VIEW: source, sheets, and row and entry totals."""
    print("\n=== read_workbook ===")
    if wb.sheets is None:
        print(f"Source: {wb.source}")
    else:
        print(f"Workbook: {wb.source} (sheets {wb.sheets[0]}, {wb.sheets[1]})")
    print(f"Rows: {wb.n_rows}; entries: {len(wb.entries)}")


def _print_role_map(wb: WorkbookData) -> None:
    """VIEW: each role with its column, display column, and label. An absent
    optional role shows its default."""
    print("\nRole mapping:")
    print(f"  {'role':<12} {'column':<20} {'display':<20} label")
    for role in ALL_ROLES:
        r = wb.roles[role]
        if not r.present:
            if role == "SortKey":
                print(f"  {role:<12} {'(absent: rows/entry)':<20} {'':<20} {r.label}")
            elif role == "Selected":
                print(f"  {role:<12} (absent: rule)")
            continue
        print(f"  {role:<12} {r.column:<20} {r.display_column:<20} {r.label}")


def _print_level_order(wb: WorkbookData) -> None:
    """VIEW: nominal levels in figure order, ordinal levels in numeric order,
    and the selection rule."""
    print(f"\nNominal order: {', '.join(wb.levels.nominal_display)}")
    print(f"Ordinal order: {', '.join(wb.levels.ordinal_display)}")
    print(f"Selection: {wb.selection_rule}")


def _print_cluster_selection(wb: WorkbookData) -> None:
    """VIEW: entries, rows, and selected entries per cluster, with a note on
    clusters holding no selection or more than one."""
    print("\nEntries per cluster:")
    print(f"  {'nominal':<14} {'ordinal':<8} {'entries':>7} {'rows':>8} {'selected':>9}")
    for c in wb.clusters:
        note = ""
        if c.n_selected == 0:
            note = "  no selection; trajectory bridges this level"
        elif c.n_selected > 1:
            note = "  more than one selection"
        print(f"  {c.nominal_display:<14} {c.ordinal_display:<8} {c.n_entries:>7d} "
              f"{c.n_rows:>8d} {c.n_selected:>9d}{note}")
    print()


def _warn_workbook(wb: WorkbookData) -> None:
    """VIEW: one warning per cluster holding more than one selected entry."""
    for c in wb.clusters:
        if c.n_selected > 1:
            warnings.warn(
                f"{c.nominal_display} {c.ordinal_display}: {c.n_selected} selected entries; "
                "the trajectory uses the one with the largest sort key.",
                stacklevel=3)
