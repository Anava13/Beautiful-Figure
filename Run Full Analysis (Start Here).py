"""Start here.

1. Edit the SETTINGS section below. At minimum, set DATA_FILE.
2. Run this file (the play button in VS Code, or `python "Run Full Analysis (Start Here).py"`).

On the first run, any missing libraries are installed automatically into the
Python that runs this file. The `subpopfig` folder must sit next to this file.
"""

# =============================================================================
# SETTINGS: edit this section
# =============================================================================

# Input workbook. A bare filename is looked up in the folder holding this file.
# Keep the r before the quotes so Windows backslashes are read correctly.
DATA_FILE = r"ExampleInput.xlsx"
DATA_SHEET = "Data"
SCHEMA_SHEET = "Schema"

# Where to save the figure. The extension sets the format (.pdf, .png, .svg).
# Leave as "" to skip saving. A bare filename saves next to this file.
OUTPUT_FILE = r"subpopulations.pdf"

# Open the figure in a window after drawing it.
SHOW_FIGURE = True

# Print the summaries from each step.
VERBOSE = True

# Mixture fit (see README for what each setting does).
FIT_SETTINGS = {
    "max_components": 4,                # highest number of subpopulations tried per entry
    "replicates": 5,                    # random starting points per fit
    "regularization": 1e-3,             # raise if many fits raise warnings
    "min_measurements_for_mixture": 10, # entries with fewer get one subpopulation
    "seed": 0,                          # random seed, reset for each entry
    "max_iter": 1000,                   # iteration limit per starting point
    "tol": 1e-6,                        # convergence tolerance
}

# Figure layout.
FIGURE_SETTINGS = {
    "trajectory": True,                 # draw the trajectory line
    "selected_only": False,             # draw only the selected entries
    "y_limits": None,                   # e.g. (50, 5000); None fits the data
    "colors": None,                     # e.g. [(0.2, 0.4, 0.85), (0.95, 0.55, 0.1)]; None uses the built-in palette
    "trajectory_weight_min": 0.01,      # minimum subpopulation weight for a trajectory point
    "figsize": (15, 7.5),               # width and height in inches
    "dpi": 100,                         # resolution
}

# Figure wording. Every string is used exactly as written.
# Leave y_label or unit_header as "" to build them from the Schema sheet labels.
FIGURE_TEXT = {
    "title": "Subpopulation Analysis Across Selected Entries",
    "y_label": "",
    "unit_header": "",
    "selected": "Selected",
    "other": "Other",
    "whisker": "Component IQR",
    "weight_header": "Component weight",
    "weight_format": "Weight {:.1f}",           # must keep one {:...} slot for the number
    "trajectory_header": "Trajectory",
    "consecutive": "Consecutive",
    "gap": "Gap",
    "mean_weight_format": "Mean weight {:.1f}", # must keep one {:...} slot for the number
}

# Glyph and layout adjustments. Leave empty for the defaults, or add any
# ComponentStyle field from subpopfig/figure.py, for example:
#   STYLE_CHANGES = {"size_scale": 320, "other_face_alpha": 0.1}
STYLE_CHANGES = {}

# =============================================================================
# END OF SETTINGS: nothing below needs editing
# =============================================================================

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# import name -> (pip name, minimum version)
REQUIRED_LIBRARIES = {
    "numpy": ("numpy", (1, 22)),
    "pandas": ("pandas", (1, 4)),
    "openpyxl": ("openpyxl", (3, 0)),
    "sklearn": ("scikit-learn", (1, 1)),
    "matplotlib": ("matplotlib", (3, 5)),
}


def main() -> None:
    """CONTROLLER: install what is missing, resolve the paths, then run the
    three pipeline steps and save or show the figure."""
    to_install = _libraries_to_install(REQUIRED_LIBRARIES)
    if to_install:
        _install_libraries(to_install)
    data_path, output_path = _resolve_paths(DATA_FILE, OUTPUT_FILE, HERE)
    _run_pipeline(data_path, output_path)


def _run_pipeline(data_path, output_path) -> None:
    """SUB-CONTROLLER: read, fit, draw, then save and show."""
    sys.path.insert(0, str(HERE))
    from dataclasses import replace

    import matplotlib.pyplot as plt
    from subpopfig import ComponentStyle, fit_mixtures, plot_subpopulations, read_workbook

    wb = read_workbook(data_path, DATA_SHEET, SCHEMA_SHEET, verbose=VERBOSE)
    fits = fit_mixtures(wb, verbose=VERBOSE, **FIT_SETTINGS)
    result = plot_subpopulations(fits, text=FIGURE_TEXT,
                                 style=replace(ComponentStyle(), **STYLE_CHANGES),
                                 verbose=VERBOSE, **FIGURE_SETTINGS)
    if output_path is not None:
        result.figure.savefig(output_path)
        print(f"Figure saved to {output_path}")
    if SHOW_FIGURE:
        plt.show()


def _libraries_to_install(required: dict) -> list:
    """MODEL: pip requirement strings for libraries that are missing or older
    than their minimum version."""
    from importlib.metadata import PackageNotFoundError, version

    needed = []
    for module, (pip_name, minimum) in required.items():
        spec = f"{pip_name}>={'.'.join(map(str, minimum))}"
        if importlib.util.find_spec(module) is None:
            needed.append(spec)
            continue
        try:
            found = re.match(r"(\d+)\.(\d+)", version(pip_name))
        except PackageNotFoundError:
            found = None
        if found is None or (int(found.group(1)), int(found.group(2))) < minimum:
            needed.append(spec)
    return needed


def _install_libraries(specs: list) -> None:
    """VIEW: install the given libraries into the Python running this file."""
    print(f"Installing: {', '.join(specs)} (first run only; this can take a few minutes)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *specs])
    importlib.invalidate_caches()
    print("Installation finished.\n")


def _resolve_paths(data_file: str, output_file: str, here: Path):
    """MODEL: absolute input and output paths. Bare filenames resolve against
    the folder holding this file. A missing workbook stops the run with a
    plain message."""
    data_path = Path(data_file)
    if not data_path.is_absolute():
        data_path = here / data_path
    if not data_path.is_file():
        sys.exit(f"\nNo data file found at:\n  {data_path}\n"
                 "Check DATA_FILE in the SETTINGS section at the top of this file.")
    output_path = None
    if output_file:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = here / output_path
    return data_path, output_path


if __name__ == "__main__":
    main()