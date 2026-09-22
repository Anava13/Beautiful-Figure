"""Fit LogNormal mixtures per entry and plot subpopulations across groups and
ordered levels.

    from subpopfig import read_workbook, fit_mixtures, plot_subpopulations

    wb = read_workbook("Input.xlsx")
    fits = fit_mixtures(wb)
    result = plot_subpopulations(fits)
    result.figure.savefig("subpopulations.pdf")
"""
from .errors import FigureError, FitError, SubpopFigError, WorkbookError
from .figure import ComponentStyle, FigureResult, FigureText, plot_subpopulations
from .fitting import FitConfig, FitResult, fit_mixtures
from .workbook import Entry, WorkbookData, from_dataframe, read_workbook

__version__ = "0.1.0"

__all__ = [
    "read_workbook",
    "from_dataframe",
    "fit_mixtures",
    "plot_subpopulations",
    "WorkbookData",
    "Entry",
    "FitResult",
    "FitConfig",
    "FigureResult",
    "FigureText",
    "ComponentStyle",
    "SubpopFigError",
    "WorkbookError",
    "FitError",
    "FigureError",
]
