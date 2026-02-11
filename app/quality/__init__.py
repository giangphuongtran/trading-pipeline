"""Data quality checks (schema + invariants).

This package is intentionally lightweight and can be disabled via env:
`ENABLE_DATA_QUALITY_CHECKS=0`.
"""

from .freshness import check_data_freshness

__all__ = ["check_data_freshness"]

