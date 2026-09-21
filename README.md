# subpopfig


subpopfig fits a LogNormal mixture to the measurements of each entry (a plate, well, or any other sample unit) and plots the resulting subpopulations across groups and ordered levels. A trajectory line through one selected entry per group and level follows how the subpopulations shift from level to level.

Each entry is drawn as a column of circles, one per subpopulation. Circle height is the subpopulation median, circle size is its share of the entry, and a vertical whisker spans its 25th to 75th percentile. Entries are clustered by a nominal variable (such as genotype) and an ordinal variable (such as passage or time point).

## Quick start

Requires Python 3.9 or later.

1. Download the repository (Code → Download ZIP on GitHub) and unzip it. Keep the `subpopfig` folder next to `Run Full Analysis (Start Here).py`.
2. Open `Run Full Analysis (Start Here).py`, set `DATA_FILE` to your workbook, and adjust any other settings at the top of the file.
3. Run the file.

On the first run, the script installs any missing libraries (numpy, pandas, openpyxl, scikit-learn, matplotlib) into the Python that runs it.

## Use from your own code

Install the package from the repository folder:

```bash
pip install -e .
```

Then:

```python
from subpopfig import read_workbook, fit_mixtures, plot_subpopulations

wb = read_workbook("Input.xlsx")
fits = fit_mixtures(wb)
result = plot_subpopulations(fits)
result.figure.savefig("subpopulations.pdf")
```

Each function prints a summary. Read those summaries before trusting the figure, since they list any entries the fit or the figure handled in a nonstandard way. Pass `verbose=False` to silence them.

Data already in a DataFrame skip the workbook:

```python
from subpopfig import from_dataframe

wb = from_dataframe(df, {
    "EntryID": "plate_id",
    "Nominal": {"column": "genotype", "display_column": "genotype_display", "label": "Genotype"},
    "Ordinal": {"column": "passage", "display_column": "passage_display", "label": "Passage"},
    "Measurement": {"column": "radius_um", "label": "Radius (µm)"},
    "SortKey": "colony_count",
    "Selected": "selected",
})
```

## Input workbook

The workbook holds two sheets. **Data** has one row per measurement. **Schema** tells the reader which Data column plays which role, so columns can carry any names.

### Data sheet

One row per measured object (a colony, cell, or particle). Values that describe the whole entry repeat on every row of that entry.

| plate_id | genotype | genotype_display | passage | passage_display | radius_um | colony_count | selected |
|---|---|---|---|---|---|---|---|
| wild_type_P1_312 | wild_type | Wild Type | 1 | P1 | 412.7 | 312 | 1 |
| wild_type_P1_312 | wild_type | Wild Type | 1 | P1 | 388.1 | 312 | 1 |
| dbf4_1_P1_160 | dbf4_1 | dbf4-1 | 1 | P1 | 251.4 | 160 | 0 |

Columns the Schema sheet does not name are ignored, so the Data sheet can keep tracing columns (source filenames, dates).

### Schema sheet

One row per role, with the headers `Role`, `Column`, `DisplayColumn`, and `Label`.

| Role | Column | DisplayColumn | Label |
|---|---|---|---|
| EntryID | plate_id | | Plate |
| Nominal | genotype | genotype_display | Genotype |
| Ordinal | passage | passage_display | Passage |
| Measurement | radius_um | | Radius (µm) |
| SortKey | colony_count | | Colony count |
| Selected | selected | | |

- **Column** names the Data column that holds the role.
- **DisplayColumn** names an optional column of display text for the Nominal and Ordinal roles. Without one, the value column is also the display text.
- **Label** is the text the figure shows for the role, used exactly as written.

### Roles

| Role | Required | Level | Meaning |
|---|---|---|---|
| EntryID | yes | entry | Groups rows into entries; one mixture is fit per entry. Its label names the entry unit in the legend (default "Entry"). `PlateID` is accepted as the same role |
| Nominal | yes | entry | Grouping variable. Sets cluster order and color |
| Ordinal | yes | entry | Ordered variable. Must be numeric; levels are sorted by value |
| Measurement | yes | row | The measured quantity. Every value must be positive and finite, since the fit runs on its logarithm |
| SortKey | no | entry | Orders entries within a cluster, largest first. Without it, the number of rows per entry is used |
| Selected | no | entry | Marks the entry the trajectory passes through in each cluster (1, true, or yes). Without it, each cluster selects the entry whose sort key lies closest to the median sort key across all entries |

Role names are matched without regard to case.

### Rules the reader enforces

- Every entry-level value (nominal, ordinal, their display text, sort key, selected) must be identical on every row of an entry.
- Each nominal or ordinal value must map to one display value.
- Nominal levels appear in the figure in order of their first appearance in the data, so row order sets group order.

A violation raises `WorkbookError` naming the entries involved. Filter the data before building the workbook; the package applies no exclusion rules of its own.

## Function reference

### `read_workbook(path, data_sheet="Data", schema_sheet="Schema", *, verbose=True)`

Reads an .xlsx workbook and returns a `WorkbookData` record.

### `from_dataframe(data, schema, *, verbose=True, source="DataFrame")`

Builds the same record from a DataFrame. `schema` is either a DataFrame with the four schema headers or a mapping from role name to a column name or to a mapping with the keys `column`, `display_column`, and `label`.

### `fit_mixtures(workbook, **settings)`

For each entry, Gaussian mixtures with 1 to `max_components` components are fit to the log measurements. The component count with the lowest BIC is kept. Each measurement is assigned to the component with the highest posterior probability. Each subpopulation's median, weight, 25th percentile, and 75th percentile are then computed on the original (unlogged) measurements, with percentiles following the midpoint (Hazen) definition. Subpopulations are numbered in ascending order of median.

| Setting | Default | Effect |
|---|---|---|
| `max_components` | 4 | Highest number of subpopulations tried per entry |
| `regularization` | 0.001 | Variance added to each component, which keeps a component from collapsing onto repeated identical values. Raise it if the summary lists many warned fits |
| `replicates` | 5 | Random starting points per fit. More starts make the best solution more likely to be found, at the cost of run time |
| `min_measurements_for_mixture` | 10 | Entries with fewer measurements receive one subpopulation without fitting |
| `seed` | 0 | Random seed, reset for each entry, so an entry's fit does not depend on the other entries |
| `max_iter` | 1000 | EM iteration limit per starting point |
| `tol` | 1e-6 | Convergence tolerance on the per-measurement log-likelihood |
| `verbose` | True | Print the fit summary |

```python
fits = fit_mixtures(wb, max_components=3, replicates=20)
```

### `plot_subpopulations(fit_result, **options)`

| Option | Default | Effect |
|---|---|---|
| `trajectory` | True | Draw the trajectory overlay |
| `selected_only` | False | Draw only the selected entries; a cluster with no selection drops out |
| `y_limits` | None | `(low, high)` of the data region. None fits the range of every drawn median and whisker |
| `colors` | None | n-by-3 RGB values in [0, 1], one row per nominal level in figure order. None uses a built-in palette |
| `trajectory_weight_min` | 0.01 | Minimum subpopulation weight for a trajectory node |
| `figsize` | (15, 7.5) | Figure size in inches |
| `dpi` | 100 | Figure resolution |
| `text` | None | Figure wording, described below |
| `style` | None | A `ComponentStyle` holding glyph sizes, alphas, line widths, and panel geometry |
| `verbose` | True | Print the figure summary |

**Figure wording.** `text` is a mapping of the fields below, or a `FigureText`. Every string is used exactly as written. A field given in `text` wins. Otherwise, `unit_header` takes the EntryID label and `y_label` is built from the Measurement label. Otherwise, the default applies. An unknown field name raises `FigureError`.

| Field | Default |
|---|---|
| `title` | Subpopulation Analysis Across Selected Entries |
| `y_label` | "Component median " followed by the Measurement label |
| `unit_header` | EntryID label, else "Entry" |
| `selected` | Selected |
| `other` | Other |
| `whisker` | Component IQR |
| `weight_header` | Component weight |
| `weight_format` | Weight {:.1f} |
| `trajectory_header` | Trajectory |
| `consecutive` | Consecutive |
| `gap` | Gap |
| `mean_weight_format` | Mean weight {:.1f} |

The two format fields each hold exactly one `str.format` replacement field, such as `{:.1f}`.

```python
result = plot_subpopulations(fits, y_limits=(50, 5000),
                             text={"title": "Colony size subpopulations", "gap": "Missing passage"})
```

**Styling.** Glyph and layout constants live in `ComponentStyle`:

```python
from dataclasses import replace
from subpopfig import ComponentStyle

style = replace(ComponentStyle(), size_scale=320, other_face_alpha=0.1)
result = plot_subpopulations(fits, style=style)
```

`result.figure` and `result.axes` are ordinary matplotlib objects, so any further edits go through the matplotlib API before saving. The figure uses Times New Roman when it is installed and a serif fallback otherwise.

## Reading the trajectory

Within each nominal group, the selected entry at each ordinal level contributes one or two nodes: its subpopulations at or above `trajectory_weight_min`, with weights rescaled to sum to 1. When an entry has more than two such subpopulations, the two heaviest are used, and the entry is listed in the summary.

- Between two levels that each hold two nodes, the line connects small to small and large to large.
- When either level holds one node, every node connects to every node, which draws a split or a merge.
- A solid line joins consecutive levels. A dotted line crosses an ordinal level with no selected entry.
- Line width scales with the mean weight of the two nodes it connects.
- When a cluster holds more than one selected entry, the trajectory passes through the one with the largest sort key.

## Outputs

Each function returns a dataclass.

- **`WorkbookData`** holds the role mapping, labels, levels, the selection rule, a per-cluster summary, and one `Entry` per entry with its measurements.
- **`FitResult`** holds the resolved `FitConfig`, the k summary per nominal level, the fit issues, and fitted copies of the entries. Each fitted entry carries the chosen component count, the BIC at each k, fit status, the component index of each measurement, and the subpopulation statistics (`component_medians`, `component_weights`, `component_p25`, `component_p75`, `component_n`).
- **`FigureResult`** holds the resolved options and wording, the layout, axis bounds, trajectory segments, the entries reduced to two trajectory nodes, the bridged levels, and the matplotlib figure and axes.

## Reproducibility

`fit_mixtures` returns identical fits for identical input and settings. Fits depend on the random starting points, so a different `seed` or `replicates` can change the chosen k for an entry whose BIC values at two k lie close together. Raising `replicates` makes that less likely.

This package is a port of a MATLAB implementation built on `fitgmdist`. The two implementations use different random number generators and EM code, so their fits agree on most entries but not necessarily all. `validation/` holds a MATLAB export script and a Python comparison script that list the entries where they differ.

## Common errors

Each error carries a `code` attribute.

| Code | Meaning |
|---|---|
| `inconsistentEntry` | An entry-level column differs between rows of one entry |
| `invalidValues` | A measurement is blank, non-numeric, zero, or negative, or an ordinal or sort key is not numeric |
| `missingRole` / `missingColumn` | The schema lacks a required role, or names a column the data lack |
| `nominalDisplay` / `ordinalDisplay` | One nominal or ordinal value carries two display values |
| `textFormat` | A format field in `text` does not hold exactly one replacement field |

## Summary messages

| Message | Meaning |
|---|---|
| "entries where a k fit raised a warning" | Some fits did not converge cleanly. A few are expected; many suggest raising `regularization` |
| "no selection; trajectory bridges this level" | A cluster has no selected entry, so the trajectory crosses it with a dotted line |
| "more than one selection" | A cluster has several selected entries; the trajectory uses the one with the largest sort key |

## Tests

```bash
pip install -e ".[test]"
pytest
```