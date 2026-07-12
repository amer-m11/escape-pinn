"""Reusable building blocks for escape-equation networks (pluggable trunks)."""

from .trunks import TRUNK_KINDS, build_trunk

__all__ = ["TRUNK_KINDS", "build_trunk"]
