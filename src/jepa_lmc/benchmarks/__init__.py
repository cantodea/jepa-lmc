"""Reusable benchmark definitions for JEPA-LMC experiments."""

from jepa_lmc.benchmarks.ctl_suite import CTLProperty, default_ctl_suite
from jepa_lmc.benchmarks.perturbations import block_entries_to_proposition

__all__ = [
    "CTLProperty",
    "block_entries_to_proposition",
    "default_ctl_suite",
]
