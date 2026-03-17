"""Adapter registry: discover PoseAdapter subclasses."""

from label3d.adapters.base import PoseAdapter
from label3d.adapters.sleap_adapter import SleapAdapter


_ADAPTERS: dict[str, type[PoseAdapter]] = {}


def _register(cls: type[PoseAdapter]) -> None:
    _ADAPTERS[cls.name()] = cls


_register(SleapAdapter)


def get_adapters() -> dict[str, type[PoseAdapter]]:
    """Return registered adapter classes keyed by name."""
    return dict(_ADAPTERS)
