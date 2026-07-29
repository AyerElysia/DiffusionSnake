"""Explicit adapters to frozen or inherited single-slice components."""

from .legacy_dataset import (
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)
from .v4_6c import V46cContourAdapter

__all__ = [
    "V46cContourAdapter",
    "configure_single_slice_compatibility",
    "make_single_slice_dataset_class",
]
