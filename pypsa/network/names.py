# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Names are unique across entity types.

Entity types are every component type except Bus, Carrier, SubNetwork, LineType and
TransformerType. A name may be used by one entity type only.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pypsa.network.abstract import _NetworkABC as Network

logger = logging.getLogger(__name__)

F = TypeVar("F", bound="Callable[..., Any]")

EXEMPT_TYPES: frozenset[str] = frozenset(
    {"Bus", "Carrier", "SubNetwork", "LineType", "TransformerType"}
)
NAMESPACE_ORDER: tuple[str, ...] = (
    "Line",
    "Transformer",
    "Link",
    "Process",
    "Generator",
    "Load",
    "StorageUnit",
    "Store",
    "ShuntImpedance",
    "GlobalConstraint",
    "Shape",
)


# Up to this many names, hash lookups beat building an indexer per entity type.
SMALL_LOOKUP = 16


def _ordered_types(n: Network) -> list[str]:
    """Entity types of `n`, in namespace order (unknown types last, sorted)."""
    present = set(n.all_components) - EXEMPT_TYPES
    known = [t for t in NAMESPACE_ORDER if t in present]
    return known + sorted(present - set(NAMESPACE_ORDER))


def _names_of(n: Network, c_name: str) -> pd.Index:
    idx = n.components[c_name].static.index
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values(-1).drop_duplicates()
    return idx


def name_owners(
    n: Network, names: pd.Index, *, exclude: str | None = None
) -> dict[str, list[str]]:
    """Find entity types holding the given names.

    Parameters
    ----------
    n : pypsa.Network
        Network to inspect.
    names : pandas.Index
        Names to look up.
    exclude : str, optional
        Entity type to ignore.

    Returns
    -------
    dict[str, list[str]]
        Each name held by an entity type other than `exclude`, mapped to the
        holding types.

    """
    names = pd.Index(names).astype(str)
    owners: dict[str, list[str]] = {}
    for c_name in _ordered_types(n):
        if c_name == exclude:
            continue
        held = _names_of(n, c_name)
        if held.empty:
            continue
        if len(names) <= SMALL_LOOKUP:
            found = [name for name in dict.fromkeys(names) if name in held]
        else:
            found = names[held.get_indexer(names) >= 0].unique()
        for name in found:
            owners.setdefault(name, []).append(c_name)
    return owners


def check_names_free(n: Network, cls_name: str, names: pd.Index) -> None:
    """Raise if `names` are held by another entity type.

    No-op for exempt types or while [unchecked_names][pypsa.network.names.unchecked_names]
    is active.

    Parameters
    ----------
    n : pypsa.Network
        Network to check against.
    cls_name : str
        Component type the names are meant for.
    names : pandas.Index
        Names to check.

    """
    if cls_name in EXEMPT_TYPES or getattr(n, "_names_unchecked", False):
        return
    owners = name_owners(n, names, exclude=cls_name)
    if not owners:
        return
    shown = ", ".join(
        f"{name} (held by {', '.join(types)})"
        for name, types in list(owners.items())[:10]
    )
    msg = (
        f"Names must be unique across entity types; {len(owners)} name(s) of "
        f"{cls_name} are already in use: {shown}. Fix: rename, or build "
        "under pypsa.network.names.unchecked_names(n) and then call "
        "pypsa.network.names.deduplicate_names(n)."
    )
    raise ValueError(msg)


@contextmanager
def unchecked_names(n: Network) -> Iterator[None]:
    """Suspend the cross-type name check on `n`.

    Parameters
    ----------
    n : pypsa.Network
        Network to suspend the check on.

    """
    previous = getattr(n, "_names_unchecked", False)
    n._names_unchecked = True  # noqa: SLF001
    try:
        yield
    finally:
        n._names_unchecked = previous  # noqa: SLF001


def deduplicate_names(n: Network) -> dict[str, dict[str, str]]:
    """Rename names shared by several entity types.

    The type earlier in `NAMESPACE_ORDER` keeps its name; later types are renamed to
    `"{name}-{Type}"`, or `"{name}-{Type}-{k}"` with `k` from 2 while taken.
    Cross-references follow the renaming.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify in place.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping `{Type: {old: new}}` of the renamed names.

    """
    types = _ordered_types(n)
    taken: set[str] = set()
    for c_name in types:
        taken.update(_names_of(n, c_name).astype(str))

    seen: set[str] = set()
    result: dict[str, dict[str, str]] = {}
    for c_name in types:
        names = _names_of(n, c_name).astype(str)
        mapping: dict[str, str] = {}
        for name in names:
            if name not in seen:
                continue
            new = f"{name}-{c_name}"
            k = 2
            while new in taken:
                new = f"{name}-{c_name}-{k}"
                k += 1
            taken.add(new)
            mapping[name] = new
        if mapping:
            n.components[c_name].rename_component_names(**mapping)
            result[c_name] = mapping
        seen.update(names.difference(mapping))
        seen.update(mapping.values())

    if result:
        counts = ", ".join(f"{len(m)} {t}" for t, m in result.items())
        logger.warning(
            "Names must be unique across entity types; renamed %s name(s) to "
            "'<name>-<Type>': %s. See n.deduplicated_names for the mapping.",
            sum(len(m) for m in result.values()),
            counts,
        )
    return result


def deduplicating_import(func: F) -> F:
    """Run an import method without name checks, then deduplicate names.

    The result is stored on `n.deduplicated_names`.
    """

    @functools.wraps(func)
    def wrapper(self: Network, *args: Any, **kwargs: Any) -> Any:
        with unchecked_names(self):
            out = func(self, *args, **kwargs)
        self.deduplicated_names = deduplicate_names(self)
        return out

    return wrapper  # type: ignore[return-value]
