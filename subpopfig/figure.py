"""Plot the subpopulations of every entry, grouped by nominal and ordinal level,
with a trajectory through the selected entries.

Entries run left to right along the x axis, grouped into one cluster per
nominal and ordinal level with labels in a bottom gutter. Component median
runs up a log-scaled y axis. Each component is a circle sized by its weight
with a vertical whisker spanning its P25 to P75. Within a cluster, entries are
ordered by sort key, largest first. Selected entries are drawn opaque and the
rest are faded back. A legend panel in the left margin decodes circle size,
selected and faded entries, the whisker, and the trajectory lines.

Trajectory overlay: per nominal level, each selected entry contributes its
components at or above ``trajectory_weight_min``, renormalized to sum to 1.
When more than two components pass, the two heaviest are kept and the entry is
listed in the summary. With two nodes, the larger median is the large class.
Consecutive levels connect large to large and small to small; a single node
connects to every node of its neighbor. An ordinal level with no selected
entry is bridged with dotted lines. Lines take the nominal color, and line
width scales with the mean weight of the two connected nodes. When a cluster
holds more than one selected entry, the one with the largest sort key is used.

Structure: public functions are controllers. Private functions marked MODEL
compute and return values; private functions marked VIEW draw, print, or warn.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from .errors import FigureError
from .fitting import FitResult
from .workbook import Labels, Levels

__all__ = [
    "FigureText",
    "ComponentStyle",
    "FigureOptions",
    "FigureResult",
    "plot_subpopulations",
]

BASE_COLORS = np.array([[0.20, 0.40, 0.85],
                        [0.95, 0.55, 0.10],
                        [0.20, 0.65, 0.30],
                        [0.85, 0.40, 0.65]])
# MATLAB's default "lines" palette; fills nominal levels beyond the fourth.
LINES_PALETTE = np.array([[0.000, 0.447, 0.741],
                          [0.850, 0.325, 0.098],
                          [0.929, 0.694, 0.125],
                          [0.494, 0.184, 0.556],
                          [0.466, 0.674, 0.188],
                          [0.301, 0.745, 0.933],
                          [0.635, 0.078, 0.184]])
FONT_RC = {"font.family": "serif",
           "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]}
FORMAT_FIELDS = ("weight_format", "mean_weight_format")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FigureText:
    """Figure wording. Every string is used as written. A blank ``y_label``
    is built from the Measurement label, and a blank ``unit_header`` takes the
    EntryID label, else 'Entry'. The two format fields each hold exactly one
    ``str.format`` replacement field, such as ``{:.1f}``."""

    title: str = "Subpopulation Analysis Across Selected Entries"
    y_label: str = ""
    unit_header: str = ""
    selected: str = "Selected"
    other: str = "Other"
    whisker: str = "Component IQR"
    weight_header: str = "Component weight"
    weight_format: str = "Weight {:.1f}"
    trajectory_header: str = "Trajectory"
    consecutive: str = "Consecutive"
    gap: str = "Gap"
    mean_weight_format: str = "Mean weight {:.1f}"


@dataclass(frozen=True)
class ComponentStyle:
    """Glyph styling and panel geometry, read by the plot and the legend.
    Replace fields with ``dataclasses.replace(ComponentStyle(), ...)``."""

    # Selected vs faded entries
    sel_face_alpha: float = 0.90
    sel_edge_alpha: float = 1.00
    sel_whisker_width: float = 2.50
    other_face_alpha: float = 0.20
    other_edge_alpha: float = 0.20
    other_whisker_width: float = 1.50
    other_whisker_fade: float = 0.65
    # Marker area (points^2) = size_base + size_scale * weight
    size_base: float = 30.0
    size_scale: float = 250.0
    # Trajectory line width at mean weight 0 and 1
    traj_min_width: float = 1.0
    traj_max_width: float = 4.0
    # Layout
    cluster_gap: float = 1.5
    gutter_frac: float = 0.18
    # Panel geometry (figure fractions: left, bottom, width, height)
    axes_position: tuple = (0.21, 0.08, 0.77, 0.84)
    legend_position: tuple = (0.005, 0.08, 0.145, 0.84)
    # Legend appearance
    legend_glyph_color: tuple = (0.45, 0.45, 0.45)
    legend_font_size: float = 12.0
    legend_header_size: float = 13.0
    legend_row_step: float = 0.052
    legend_header_gap: float = 0.025


@dataclass
class FigureOptions:
    """Resolved figure options."""

    trajectory: bool
    selected_only: bool
    y_limits: Optional[tuple]
    colors: Optional[np.ndarray]
    trajectory_weight_min: float
    figsize: tuple
    dpi: float
    text: FigureText


@dataclass
class Cluster:
    """One nominal and ordinal cluster with its entries in drawing order."""

    nominal_idx: int
    ordinal: float
    ordinal_idx: int
    color: np.ndarray
    ordinal_label: str
    nominal_label: str
    row_start: float
    row_end: float
    records: list
    row_positions: list


@dataclass
class Layout:
    clusters: list
    max_row: float


@dataclass
class Bounds:
    radius_min: float
    radius_max: float
    axis_min: float
    label_edge: float
    ticks: np.ndarray


@dataclass
class TrajectorySegment:
    nominal_idx: int
    from_ordinal: float
    to_ordinal: float
    x: tuple
    y: tuple
    mean_weight: float
    dotted: bool
    color: np.ndarray


@dataclass
class Reduction:
    entry_id: str
    nominal_label: str
    ordinal_label: str
    n_above: int


@dataclass
class BridgedLevel:
    nominal_label: str
    ordinal_label: str


@dataclass
class LegendEntry:
    kind: str
    label: str
    y: float = float("nan")
    size: float = float("nan")
    face_alpha: float = float("nan")
    edge_alpha: float = float("nan")
    line_style: str = ""
    line_width: float = float("nan")


@dataclass
class _Generation:
    ordinal_idx: int
    row: float
    radius: np.ndarray
    w: np.ndarray
    color: np.ndarray


@dataclass
class FigureResult:
    """The resolved figure model and the drawn figure."""

    options: FigureOptions
    style: ComponentStyle
    labels: Labels
    text: FigureText
    layout: Layout
    bounds: Bounds
    trajectory: list
    reductions: list
    bridged: list
    legend_entries: list
    figure: Any = None
    axes: Any = None
    legend_axes: Any = None


# ---------------------------------------------------------------------------
# Public controller
# ---------------------------------------------------------------------------

def plot_subpopulations(
    fit_result: FitResult,
    *,
    trajectory: bool = True,
    selected_only: bool = False,
    y_limits: Optional[Sequence[float]] = None,
    colors: Optional[Any] = None,
    trajectory_weight_min: float = 0.01,
    figsize: Sequence[float] = (15.0, 7.5),
    dpi: float = 100,
    text: Union[None, FigureText, Mapping[str, str]] = None,
    style: Optional[ComponentStyle] = None,
    verbose: bool = True,
) -> FigureResult:
    """Draw the subpopulation figure.

    Parameters
    ----------
    trajectory : draw the trajectory overlay.
    selected_only : draw only the selected entries; a cluster with no
        selection drops out.
    y_limits : (low, high) of the data region. None fits the range of every
        drawn median and whisker with a small margin.
    colors : n-by-3 RGB values in [0, 1], one row per nominal level in figure
        order. None uses the built-in palette.
    trajectory_weight_min : minimum component weight for a trajectory node.
    figsize, dpi : figure size in inches and resolution.
    text : a FigureText or a mapping of FigureText field names to strings.
    style : a ComponentStyle; None uses the defaults.
    verbose : print what was drawn, bridged levels, and trajectory reductions.

    Returns a FigureResult whose ``figure`` and ``axes`` fields hold the
    matplotlib objects, so the figure can be edited or saved with
    ``result.figure.savefig(...)``.
    """
    model = _build_figure_model(
        fit_result, style,
        trajectory=trajectory, selected_only=selected_only, y_limits=y_limits,
        colors=colors, trajectory_weight_min=trajectory_weight_min,
        figsize=figsize, dpi=dpi, text=text)
    fig, ax, legend_ax = _draw_component_figure(model)
    result = replace(model, figure=fig, axes=ax, legend_axes=legend_ax)
    if verbose:
        _report_figure_model(result)
    return result


# ---------------------------------------------------------------------------
# Model sub-controller
# ---------------------------------------------------------------------------

def _build_figure_model(fit_result: FitResult, style, **raw_options) -> FigureResult:
    """SUB-CONTROLLER (model): resolve options and style, assign colors, lay
    out clusters and rows, set the axis bounds, build the trajectory
    segments, resolve the wording, and build the legend entries. Calls model
    functions only."""
    opts = _normalize_figure_options(**raw_options)
    style = _resolve_component_style(style)
    colors = _assign_nominal_colors(len(fit_result.levels.nominal), opts)
    layout = _build_clustered_layout(fit_result, colors, opts, style)
    bounds = _compute_bounds(layout, opts, style)
    segments, reductions, bridged = _build_trajectory_segments(layout, fit_result.levels, opts)
    fig_text = _resolve_figure_text(fit_result.labels, opts)
    legend_entries = _build_legend_entries(opts, style, fig_text)
    return FigureResult(
        options=opts, style=style, labels=fit_result.labels, text=fig_text,
        layout=layout, bounds=bounds, trajectory=segments, reductions=reductions,
        bridged=bridged, legend_entries=legend_entries)


# ---------------------------------------------------------------------------
# Model terminals
# ---------------------------------------------------------------------------

def _normalize_figure_options(*, trajectory, selected_only, y_limits, colors,
                              trajectory_weight_min, figsize, dpi, text) -> FigureOptions:
    """MODEL: validate the options and merge the wording with its defaults.
    An unknown text field raises FigureError. Blank text fields keep their
    defaults. y_limits must be two positive increasing values, and each
    format field must hold exactly one replacement field that formats a
    number."""
    if text is None:
        fig_text = FigureText()
    elif isinstance(text, FigureText):
        fig_text = text
    elif isinstance(text, Mapping):
        known = {f.name for f in fields(FigureText)}
        unknown = sorted(set(text) - known)
        if unknown:
            raise FigureError("textField",
                              f"Unknown text fields: {', '.join(unknown)}. "
                              f"Known fields: {', '.join(sorted(known))}")
        fig_text = replace(FigureText(), **{k: str(v) for k, v in text.items()
                                            if v is not None and str(v) != ""})
    else:
        raise FigureError("textField", "text must be a FigureText or a mapping.")

    for name in FORMAT_FIELDS:
        fmt = getattr(fig_text, name)
        try:
            n_fields = sum(1 for _, f, _, _ in string.Formatter().parse(fmt) if f is not None)
            fmt.format(0.5)
        except (ValueError, IndexError, KeyError):
            n_fields = -1
        if n_fields != 1:
            raise FigureError("textFormat",
                              f"text {name} must hold exactly one replacement field such as "
                              f"{{:.1f}}; got {fmt!r}.")

    limits = None
    if y_limits is not None:
        y = np.asarray(y_limits, dtype=float).ravel()
        if y.size != 2 or not np.all(np.isfinite(y)) or np.any(y <= 0) or y[0] >= y[1]:
            raise FigureError("yLimits", "y_limits must be two positive values, low then high.")
        limits = (float(y[0]), float(y[1]))

    color_array = None if colors is None else np.asarray(colors, dtype=float)

    return FigureOptions(
        trajectory=bool(trajectory),
        selected_only=bool(selected_only),
        y_limits=limits,
        colors=color_array,
        trajectory_weight_min=float(trajectory_weight_min),
        figsize=tuple(float(v) for v in figsize),
        dpi=float(dpi),
        text=fig_text,
    )


def _resolve_component_style(style) -> ComponentStyle:
    """MODEL: the given style, or the defaults when None."""
    if style is None:
        return ComponentStyle()
    if not isinstance(style, ComponentStyle):
        raise FigureError("style", "style must be a ComponentStyle.")
    return style


def _assign_nominal_colors(n_levels: int, opts: FigureOptions) -> np.ndarray:
    """MODEL: one RGB row per nominal level. A colors option supplies them
    directly and must hold at least one row per level. Otherwise the four
    base colors come first and MATLAB's lines palette fills levels beyond
    four."""
    if opts.colors is not None:
        c = opts.colors
        if c.ndim != 2 or c.shape[1] != 3 or c.shape[0] < n_levels:
            raise FigureError("colors",
                              f"colors must be an n-by-3 RGB array with at least {n_levels} rows.")
        return c[:n_levels].copy()
    if n_levels <= len(BASE_COLORS):
        return BASE_COLORS[:n_levels].copy()
    extra = np.array([LINES_PALETTE[i % len(LINES_PALETTE)]
                      for i in range(len(BASE_COLORS), n_levels)])
    return np.vstack([BASE_COLORS, extra])


def _build_clustered_layout(fit_result: FitResult, colors: np.ndarray,
                            opts: FigureOptions, style: ComponentStyle) -> Layout:
    """MODEL: one cluster per nominal and ordinal level holding entries, in
    nominal then ordinal order. Entries within a cluster are ordered by sort
    key, largest first (ties keep row order), and each receives a row
    position along x. With selected_only, only selected entries are kept and
    a cluster with none is skipped. An empty layout raises FigureError."""
    levels = fit_result.levels
    clusters = []
    row = 0.0
    for li in range(len(levels.nominal)):
        for oi, value in enumerate(levels.ordinal):
            recs = [e for e in fit_result.entries
                    if e.nominal_idx == li and e.ordinal == value
                    and e.component_medians.size > 0
                    and (e.selected or not opts.selected_only)]
            if not recs:
                continue
            order = np.argsort(-np.array([e.sort_key for e in recs]), kind="stable")
            recs = [recs[i] for i in order]

            row += 1
            start = row
            positions = []
            for ri in range(len(recs)):
                positions.append(row)
                if ri < len(recs) - 1:
                    row += 1
            clusters.append(Cluster(
                nominal_idx=li, ordinal=float(value), ordinal_idx=oi,
                color=colors[li], ordinal_label=levels.ordinal_display[oi],
                nominal_label=levels.nominal_display[li],
                row_start=start, row_end=row, records=recs, row_positions=positions))
            row += style.cluster_gap
    if not clusters:
        raise FigureError("nothingToDraw",
                          "No entry has components to draw under the current options.")
    return Layout(clusters=clusters, max_row=row)


def _compute_bounds(layout: Layout, opts: FigureOptions, style: ComponentStyle) -> Bounds:
    """MODEL: the data region, the axis extension that holds the label
    gutter, and the y ticks. The data region is y_limits when given;
    otherwise it spans every drawn median, P25, and P75, widened by 5% of that
    span in log space on each side. The axis extends downward in log space so
    the gutter takes gutter_frac of the axis height. Ticks run 1, 2, 5 per
    decade inside the data region; when fewer than two fall inside, the
    region's ends are the ticks."""
    if opts.y_limits is not None:
        lo, hi = opts.y_limits
    else:
        vals = np.concatenate([
            np.concatenate([e.component_medians, e.component_p25, e.component_p75])
            for c in layout.clusters for e in c.records])
        vals = vals[np.isfinite(vals) & (vals > 0)]
        log_lo, log_hi = np.log10(vals.min()), np.log10(vals.max())
        pad = 0.05 * (log_hi - log_lo)
        if pad == 0:
            pad = 0.1
        lo, hi = 10 ** (log_lo - pad), 10 ** (log_hi + pad)

    log_min, log_max = np.log10(lo), np.log10(hi)
    frac = style.gutter_frac
    log_axis = log_min - frac / (1 - frac) * (log_max - log_min)

    decades = np.arange(np.floor(log_min), np.ceil(log_max) + 1)
    cand = (10.0 ** decades[:, None] * np.array([1.0, 2.0, 5.0])[None, :]).ravel()
    ticks = cand[(cand >= lo) & (cand <= hi)]
    if ticks.size < 2:
        ticks = np.array([lo, hi])

    return Bounds(radius_min=float(lo), radius_max=float(hi),
                  axis_min=float(10 ** log_axis),
                  label_edge=float(10 ** (log_axis + frac * (log_max - log_axis))),
                  ticks=ticks)


def _build_trajectory_segments(layout: Layout, levels: Levels, opts: FigureOptions):
    """MODEL: trajectory segments from the selected entries. Per nominal
    level, each ordinal level with a selected entry becomes a generation of
    one or two nodes: the entry's components at or above
    trajectory_weight_min, reduced to the two heaviest when more pass,
    weights renormalized to sum to 1, sorted by median so node 0 is the small
    class. Consecutive generations connect small to small and large to large
    when both hold two nodes; otherwise every node of one connects to every
    node of the other. A pair separated by a level with no selected entry is
    dotted, and each skipped level is recorded in bridged. Segment x holds
    row positions; y holds medians."""
    segments, reductions, bridged = [], [], []
    cl_nom = np.array([c.nominal_idx for c in layout.clusters])
    cl_ord = np.array([c.ordinal_idx for c in layout.clusters])

    for li in np.unique(cl_nom):
        gens = []
        nom_label = ""
        for oi in range(len(levels.ordinal)):
            match = np.flatnonzero((cl_nom == li) & (cl_ord == oi))
            if match.size == 0:
                continue
            cl = layout.clusters[match[0]]
            nom_label = cl.nominal_label
            sel = [i for i, e in enumerate(cl.records) if e.selected]
            if not sel:
                continue
            r = cl.records[sel[0]]

            meds, wts = r.component_medians, r.component_weights
            keep = np.isfinite(meds) & np.isfinite(wts) & (wts >= opts.trajectory_weight_min)
            if not keep.any():
                continue
            meds, wts = meds[keep], wts[keep]
            if meds.size > 2:
                reductions.append(Reduction(r.entry_id, cl.nominal_label,
                                            cl.ordinal_label, int(meds.size)))
                heavy = np.argsort(-wts, kind="stable")[:2]
                meds, wts = meds[heavy], wts[heavy]
            wts = wts / wts.sum()
            o = np.argsort(meds, kind="stable")
            gens.append(_Generation(oi, cl.row_positions[sel[0]], meds[o], wts[o], cl.color))

        for a, b in zip(gens[:-1], gens[1:]):
            is_bridge = (b.ordinal_idx - a.ordinal_idx) > 1
            if is_bridge:
                for skip in range(a.ordinal_idx + 1, b.ordinal_idx):
                    bridged.append(BridgedLevel(nom_label, levels.ordinal_display[skip]))
            if a.radius.size == 2 and b.radius.size == 2:
                pairs = [(0, 0), (1, 1)]
            else:
                pairs = [(i, j) for j in range(b.radius.size) for i in range(a.radius.size)]
            for i, j in pairs:
                segments.append(TrajectorySegment(
                    nominal_idx=int(li),
                    from_ordinal=float(levels.ordinal[a.ordinal_idx]),
                    to_ordinal=float(levels.ordinal[b.ordinal_idx]),
                    x=(a.row, b.row),
                    y=(float(a.radius[i]), float(b.radius[j])),
                    mean_weight=float((a.w[i] + b.w[j]) / 2),
                    dotted=is_bridge,
                    color=a.color))
    return segments, reductions, bridged


def _resolve_figure_text(labels: Labels, opts: FigureOptions) -> FigureText:
    """MODEL: final wording. Fields given in the options stand as written. A
    blank unit_header takes the EntryID label, else 'Entry'. A blank y_label
    becomes 'Component median ' followed by the Measurement label."""
    t = opts.text
    unit = t.unit_header or labels.unit or "Entry"
    y_label = t.y_label or f"Component median {labels.measurement}"
    return replace(t, unit_header=unit, y_label=y_label)


def _build_legend_entries(opts: FigureOptions, style: ComponentStyle,
                          fig_text: FigureText) -> list:
    """MODEL: ordered legend entries with their vertical positions in legend
    axes units. The faded circle drops out under selected_only, and the
    trajectory section drops out when the overlay is off."""
    def area_at(w):
        return style.size_base + style.size_scale * w

    def width_at(w):
        return style.traj_min_width + (style.traj_max_width - style.traj_min_width) * w

    entries = [LegendEntry("header", fig_text.weight_header)]
    for w in (0.1, 0.5, 1.0):
        entries.append(LegendEntry("circle", fig_text.weight_format.format(w), size=area_at(w),
                                   face_alpha=style.sel_face_alpha,
                                   edge_alpha=style.sel_edge_alpha, line_width=0.5))
    entries.append(LegendEntry("header", fig_text.unit_header))
    entries.append(LegendEntry("circle", fig_text.selected, size=area_at(0.5),
                               face_alpha=style.sel_face_alpha,
                               edge_alpha=style.sel_edge_alpha, line_width=0.5))
    if not opts.selected_only:
        entries.append(LegendEntry("circle", fig_text.other, size=area_at(0.5),
                                   face_alpha=style.other_face_alpha,
                                   edge_alpha=style.other_edge_alpha, line_width=0.5))
    entries.append(LegendEntry("whisker", fig_text.whisker, line_style="-",
                               line_width=style.sel_whisker_width))
    if opts.trajectory:
        entries.append(LegendEntry("header", fig_text.trajectory_header))
        entries.append(LegendEntry("line", fig_text.consecutive, line_style="-",
                                   line_width=width_at(0.5)))
        entries.append(LegendEntry("line", fig_text.gap, line_style=":",
                                   line_width=width_at(0.5)))
        entries.append(LegendEntry("line", fig_text.mean_weight_format.format(0.1),
                                   line_style="-", line_width=width_at(0.1)))
        entries.append(LegendEntry("line", fig_text.mean_weight_format.format(0.9),
                                   line_style="-", line_width=width_at(0.9)))

    y = 0.97
    for i, e in enumerate(entries):
        if e.kind == "header" and i > 0:
            y -= style.legend_header_gap
        e.y = y
        y -= style.legend_row_step
    return entries


# ---------------------------------------------------------------------------
# View sub-controller and terminals: drawing
# ---------------------------------------------------------------------------

def _draw_component_figure(model: FigureResult):
    """SUB-CONTROLLER (view): create the figure and axes, draw the trajectory,
    the markers, the gutter brackets and labels, the axis finish, and the
    legend. Explicit z-orders keep trajectory lines under whiskers and
    whiskers under circles. Calls view functions only."""
    with plt.rc_context(FONT_RC):
        fig = _create_figure(model.options, model.text)
        ax = _setup_axes(fig, model)
        if model.options.trajectory:
            _draw_trajectory_segments(ax, model.trajectory, model.style)
        _draw_component_markers(ax, model.layout, model.style)
        _draw_cluster_brackets(ax, model.layout, model.bounds)
        _finalize_component_plot(ax, model)
        legend_ax = _render_legend_panel(fig, ax, model.legend_entries, model.style)
    return fig, ax, legend_ax


def _create_figure(opts: FigureOptions, fig_text: FigureText):
    """VIEW: white figure window titled with the figure title."""
    fig = plt.figure(figsize=opts.figsize, dpi=opts.dpi, facecolor="white")
    manager = fig.canvas.manager
    if manager is not None:
        manager.set_window_title(fig_text.title)
    return fig


def _setup_axes(fig, model: FigureResult):
    """VIEW: log-scaled axes right of the legend margin, with no x ticks and
    outward y ticks. Grid lines sit beneath the data."""
    ax = fig.add_axes(model.style.axes_position)
    ax.set_yscale("log")
    ax.set_xlim(0, model.layout.max_row + 1)
    ax.set_ylim(model.bounds.axis_min, model.bounds.radius_max)
    ax.set_xticks([])
    ax.tick_params(axis="y", which="both", direction="out", labelsize=12)
    ax.set_axisbelow(True)
    return ax


def _draw_trajectory_segments(ax, segments: list, style: ComponentStyle) -> None:
    """VIEW: trajectory segments in the nominal color, solid between
    consecutive levels and dotted across a skipped level. Line width is
    linear in the mean weight of the two connected nodes."""
    for s in segments:
        width = style.traj_min_width + (style.traj_max_width - style.traj_min_width) * s.mean_weight
        ax.plot(s.x, s.y, ":" if s.dotted else "-", color=s.color, linewidth=width, zorder=2)


def _draw_component_markers(ax, layout: Layout, style: ComponentStyle) -> None:
    """VIEW: for every entry, a vertical whisker from P25 to P75 and a circle
    at the median of each component, with circle area set by weight.
    Selected entries are drawn opaque; the rest are faded toward white."""
    for cluster in layout.clusters:
        color = np.asarray(cluster.color, dtype=float)
        base_whisker = color * 0.5 + 0.5
        for entry, row in zip(cluster.records, cluster.row_positions):
            valid = np.isfinite(entry.component_medians)
            meds = entry.component_medians[valid]
            wts = entry.component_weights[valid]
            p25s = entry.component_p25[valid]
            p75s = entry.component_p75[valid]

            if entry.selected:
                face_alpha, edge_alpha = style.sel_face_alpha, style.sel_edge_alpha
                whisker_color, whisker_width = base_whisker, style.sel_whisker_width
            else:
                face_alpha, edge_alpha = style.other_face_alpha, style.other_edge_alpha
                whisker_color = (base_whisker * (1 - style.other_whisker_fade)
                                 + style.other_whisker_fade)
                whisker_width = style.other_whisker_width

            for lo, hi in zip(p25s, p75s):
                ax.plot([row, row], [lo, hi], "-", color=whisker_color,
                        linewidth=whisker_width, solid_capstyle="butt", zorder=3)
            n = meds.size
            ax.scatter(np.full(n, row), meds,
                       s=style.size_base + style.size_scale * wts,
                       facecolors=np.tile(np.append(color, face_alpha), (n, 1)),
                       edgecolors=np.tile([0.0, 0.0, 0.0, edge_alpha], (n, 1)),
                       linewidths=0.5, zorder=4)


def _draw_cluster_brackets(ax, layout: Layout, bounds: Bounds) -> None:
    """VIEW: in the bottom gutter, a bracket with upward end ticks under each
    cluster with its ordinal label beneath, and each nominal label centered
    beneath all of that level's clusters."""
    log_gutter = np.log10(bounds.axis_min)
    log_edge = np.log10(bounds.label_edge)
    span = log_edge - log_gutter
    bracket_y = 10 ** (log_edge - 0.15 * span)
    bracket_tick_y = 10 ** (log_edge - 0.05 * span)
    ordinal_label_y = 10 ** (log_edge - 0.40 * span)
    nominal_label_y = 10 ** (log_edge - 0.75 * span)

    for c in layout.clusters:
        x1, x2 = c.row_start - 0.4, c.row_end + 0.4
        ax.plot([x1, x2], [bracket_y, bracket_y], "-", color=c.color, linewidth=2, zorder=5)
        ax.plot([x1, x1], [bracket_y, bracket_tick_y], "-", color=c.color, linewidth=2, zorder=5)
        ax.plot([x2, x2], [bracket_y, bracket_tick_y], "-", color=c.color, linewidth=2, zorder=5)
        ax.text((x1 + x2) / 2, ordinal_label_y, c.ordinal_label,
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=np.asarray(c.color) * 0.6, parse_math=False, zorder=5)

    for li in sorted({c.nominal_idx for c in layout.clusters}):
        members = [c for c in layout.clusters if c.nominal_idx == li]
        x_center = (min(c.row_start for c in members) + max(c.row_end for c in members)) / 2
        ax.text(x_center, nominal_label_y, members[0].nominal_label,
                ha="center", va="center", fontsize=14, fontweight="bold",
                color=np.asarray(members[0].color) * 0.6, parse_math=False, zorder=5)


def _finalize_component_plot(ax, model: FigureResult) -> None:
    """VIEW: gutter separator at the bottom of the data region, y ticks with
    plain numeric labels and unlabeled minor ticks, axis label, title, and
    horizontal grid."""
    ax.axhline(model.bounds.radius_min, color="k", linewidth=0.5, alpha=0.3, zorder=1)
    ticks = model.bounds.ticks
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%g" % v for v in ticks], fontsize=12)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(model.bounds.axis_min, model.bounds.radius_max)
    ax.set_ylabel(model.text.y_label, fontsize=16, parse_math=False)
    ax.set_title(model.text.title, fontsize=20, parse_math=False)
    ax.grid(True, axis="y", which="major", color="0.15", alpha=0.2, linewidth=0.5)


def _render_legend_panel(fig, ax, entries: list, style: ComponentStyle):
    """VIEW: a hidden axes in the left margin with each entry's glyph and
    label. Marker areas and line widths are absolute (points^2 and points),
    so the glyphs match the plot one to one. The main axes is made current
    afterward."""
    glyph_x, line_x, text_x, half_h = 0.14, (0.03, 0.25), 0.30, 0.018
    gray = np.asarray(style.legend_glyph_color, dtype=float)
    whisker_gray = gray * 0.5 + 0.5

    lax = fig.add_axes(style.legend_position)
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")

    for e in entries:
        if e.kind == "header":
            lax.text(0.02, e.y, e.label, fontsize=style.legend_header_size,
                     fontweight="bold", va="center", parse_math=False)
            continue
        if e.kind == "circle":
            lax.scatter([glyph_x], [e.y], s=e.size,
                        facecolors=[np.append(gray, e.face_alpha)],
                        edgecolors=[[0.0, 0.0, 0.0, e.edge_alpha]],
                        linewidths=e.line_width, clip_on=False)
        elif e.kind == "whisker":
            lax.plot([glyph_x, glyph_x], [e.y - half_h, e.y + half_h], e.line_style,
                     color=whisker_gray, linewidth=e.line_width, solid_capstyle="butt")
        elif e.kind == "line":
            lax.plot(line_x, [e.y, e.y], e.line_style, color=gray, linewidth=e.line_width)
        lax.text(text_x, e.y, e.label, fontsize=style.legend_font_size,
                 va="center", parse_math=False)

    fig.sca(ax)
    return lax


# ---------------------------------------------------------------------------
# View sub-controller and terminals: report
# ---------------------------------------------------------------------------

def _report_figure_model(result: FigureResult) -> None:
    """SUB-CONTROLLER (view): print what the figure drew, the levels the
    trajectory bridges, and the entries reduced to two trajectory nodes.
    Calls view functions only."""
    _print_figure_header(result)
    _print_bridged_levels(result)
    _print_trajectory_reductions(result)


def _print_figure_header(result: FigureResult) -> None:
    """VIEW: clusters and entries drawn, the data region, and the overlay
    settings."""
    n_entries = sum(len(c.records) for c in result.layout.clusters)
    o = result.options
    print("\n=== plot_subpopulations ===")
    print(f"Clusters drawn: {len(result.layout.clusters)}; entries drawn: {n_entries}")
    print(f"Data region: {result.bounds.radius_min:g} to {result.bounds.radius_max:g}")
    print(f"Trajectory: {'on' if o.trajectory else 'off'}; "
          f"selected only: {'on' if o.selected_only else 'off'}; "
          f"node weight minimum: {o.trajectory_weight_min:g}")


def _print_bridged_levels(result: FigureResult) -> None:
    """VIEW: each level the trajectory crosses with a dotted line because no
    selected entry was drawn there."""
    if not result.options.trajectory:
        return
    if not result.bridged:
        print("Bridged levels: none.")
        return
    print("Bridged levels (no selected entry, dotted line):")
    for b in result.bridged:
        print(f"  {b.nominal_label} {b.ordinal_label}")


def _print_trajectory_reductions(result: FigureResult) -> None:
    """VIEW: each selected entry with more than two components at or above
    the node weight minimum, which the trajectory drew through its two
    heaviest."""
    if not result.options.trajectory:
        print()
        return
    if not result.reductions:
        print("Trajectory reductions: none.\n")
        return
    print("Trajectory drawn through the two heaviest components:")
    for r in result.reductions:
        print(f"  {r.entry_id:<26} {r.nominal_label} {r.ordinal_label}  "
              f"({r.n_above} components above minimum)")
    print()
