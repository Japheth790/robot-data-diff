"""robot-data-diff: episode-aware diffs for LeRobot robot-training datasets.

Answers: "we retrained the policy and it got worse — what changed in the data?"
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("robot-data-diff")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"
