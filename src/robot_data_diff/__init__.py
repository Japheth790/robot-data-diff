"""robot-data-diff: episode-aware diffs for LeRobot robot-training datasets.

Answers: "we retrained the policy and it got worse — what changed in the data?"
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("robot-data-diff")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"

from .differ import SCHEMA_VERSION, DiffResult, EpisodeDelta, diff_datasets  # noqa: E402
from .loader import DatasetError, DatasetView, EpisodeInfo, load_dataset  # noqa: E402
from .report import render_html, render_text  # noqa: E402

__all__ = [
    "SCHEMA_VERSION",
    "DatasetError",
    "DatasetView",
    "DiffResult",
    "EpisodeDelta",
    "EpisodeInfo",
    "__version__",
    "diff_datasets",
    "load_dataset",
    "render_html",
    "render_text",
]
