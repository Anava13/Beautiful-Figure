"""Exceptions raised by subpopfig.

Each exception carries a short code naming the check that failed, so callers
can branch on ``err.code`` instead of parsing the message.
"""
from __future__ import annotations


class SubpopFigError(Exception):
    """Base class for every error this package raises."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code


class WorkbookError(SubpopFigError):
    """The workbook, DataFrame, or schema violates a reader rule."""


class FitError(SubpopFigError):
    """The fit configuration is invalid."""


class FigureError(SubpopFigError):
    """The figure options are invalid or leave nothing to draw."""
