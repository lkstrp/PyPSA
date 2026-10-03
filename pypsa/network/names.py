# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Cross-type component name namespace.

PyPSA v2 treats component names as unique across a subset of component
types, the namespace types. Shape, SubNetwork and the standard types sit
outside that namespace and may reuse a namespace type's name: a shape is
conventionally named after the bus it describes.

This module answers "who already holds this name" for the namespace types,
and formats the answer as the export-check error wording. `check_names_free`
enforces it on import and `n.add`; `rename_component_names` uses it to
refuse a taken target name.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Generator

    from pypsa import Network

logger = logging.getLogger(__name__)

NAMESPACE_ORDER: tuple[str, ...] = (
    "Bus",
    "Carrier",
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
)
"""The namespace types, in rename order (buses, carriers, branches, one-ports, global constraints).

Carrier sits right after Bus so that on a clash it yields only to a bus and
every other type yields to it: downstream code filters by carrier name far
more often than by a generator's.
"""


def _name_level(index: pd.Index) -> pd.Index:
    """Return the unique names held by a static index, dropping any scenario level.

    Uses the last level by position, not by the label "name", since
    `check_names_free` sees incoming indices before `_import_components_from_df`
    has normalized the MultiIndex level names.
    """
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(-1).unique()
    return index


def name_owners(
    n: Network, names: pd.Index, exclude: str | None = None
) -> dict[str, list[str]]:
    """Map each of `names` already held by a namespace type to its owning types.

    Checks every namespace type except `exclude`. A name absent from all of
    them is left out of the result, so an empty dict means no clash.

    A unique flat index is probed with `get_indexer`, which reuses the
    index's cached hash table across calls. A scenario MultiIndex yields a
    fresh name level on every call, so it is probed with `isin`, which only
    hashes the (usually short) `names`.
    """
    owners: dict[str, list[str]] = {}
    for type_name in NAMESPACE_ORDER:
        if type_name == exclude:
            continue
        index = n.components[type_name].static.index
        if isinstance(index, pd.MultiIndex) or not index.is_unique:
            level = (
                index.get_level_values(-1)
                if isinstance(index, pd.MultiIndex)
                else index
            )
            found = level[level.isin(names)].unique()
        else:
            found = names[index.get_indexer(names) >= 0].unique()
        for name in found:
            owners.setdefault(str(name), []).append(type_name)
    for types in owners.values():
        types.sort()
    return owners


def format_clashes(owners: dict[str, list[str]]) -> str:
    """Render a name-owners clash dict in the export-check wording."""
    detail = "; ".join(
        f"{name}: {', '.join(types)}" for name, types in sorted(owners.items())
    )
    return f"names claimed by more than one component type: {detail}"


def check_names_free(n: Network, cls_name: str, names: pd.Index) -> None:
    """Raise if any of `names` is already held by another namespace type.

    No-op for exempt types (Shape, SubNetwork, the standard types) and while
    the check is suspended with `unchecked_names`. Only the `name`
    level is checked for a MultiIndex, never scenario labels.
    """
    if cls_name not in NAMESPACE_ORDER:
        return
    if n._names_unchecked:
        return
    owners = name_owners(n, _name_level(names), exclude=cls_name)
    if owners:
        owners = {name: sorted({cls_name, *types}) for name, types in owners.items()}
        msg = format_clashes(owners)
        raise ValueError(msg)


@contextmanager
def unchecked_names(n: Network) -> Generator[None, None, None]:
    """Suspend `check_names_free` for `n` for the duration of the context.

    Reentrant, counted with a private flag on the network. Importers and
    clustering use this while they build a network that may clash, then call
    `deduplicate_names` after leaving the context.
    """
    n._names_unchecked += 1
    try:
        yield
    finally:
        n._names_unchecked -= 1


def deduplicate_names(n: Network) -> dict[str, dict[str, str]]:
    """Rename namespace-type component names that clash across types.

    Walks `NAMESPACE_ORDER` with a running set of names already claimed. The
    first type to claim a name keeps it. A later type's clashing name is
    renamed to `"<name>-<Type>"`, with `-2`, `-3`, ... appended while that
    candidate is still taken, either by another type, by the same type's own
    names, or by a candidate already assigned to another name of the same
    type in this call.

    Renames are applied one type at a time, in reverse `NAMESPACE_ORDER`, so
    that by the time a type's renames are applied, their target names are
    free. A later type's target can still be held by an earlier type at
    planning time (e.g. Line renamed to "x-Line" while Load still holds
    "x-Line"), and reverse order renames Load away first.

    Returns a map `{type: {old: new}}` for every type it renamed, empty if
    nothing clashed. Logs one warning with the total and per-type counts
    when anything was renamed.
    """
    taken: set[str] = set()
    renames: dict[str, dict[str, str]] = {}

    for type_name in NAMESPACE_ORDER:
        names = _name_level(n.components[type_name].static.index)
        same_type = set(names)
        type_map: dict[str, str] = {}
        assigned: set[str] = set()

        candidates = [name for name in map(str, names) if name in taken]
        for name in candidates:
            counter = 1
            candidate = f"{name}-{type_name}"
            while candidate in taken or candidate in same_type or candidate in assigned:
                counter += 1
                candidate = f"{name}-{type_name}-{counter}"
            type_map[name] = candidate
            assigned.add(candidate)

        if type_map:
            renames[type_name] = type_map

        taken.update(same_type - type_map.keys())
        taken.update(type_map.values())

    if renames:
        for type_name, type_map in reversed(renames.items()):
            n.rename_component_names(type_name, **type_map)

        total = sum(len(type_map) for type_map in renames.values())
        detail = ", ".join(
            f"{type_name} {len(type_map)}" for type_name, type_map in renames.items()
        )
        logger.warning(
            "Renamed %s component names that clash across component types: %s.",
            total,
            detail,
        )

    return renames
