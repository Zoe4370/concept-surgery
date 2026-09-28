"""concept-surgery: surgical concept erasure for transformer language models.

Locate a concept's linear representation inside a transformer's residual
stream and remove it at inference time with orthogonal projection -- no
weight edits, no retraining.
"""

__version__ = "0.1.0"
__author__ = "Zoe Faith Gumise"

from .core_math import (
    fit_direction,
    fit_directions_inlp,
    fit_leace_eraser,
    project_out,
    oblique_project,
)
from .config import ErasureConfig

__all__ = [
    "fit_direction",
    "fit_directions_inlp",
    "fit_leace_eraser",
    "project_out",
    "oblique_project",
    "ErasureConfig",
    "__version__",
]
