# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Cross-type component name namespace.

PyPSA v2 treats component names as unique across a subset of component
types, the namespace types. Carrier, Shape, SubNetwork and the standard
types sit outside that namespace and may reuse a namespace type's name.

This module answers "who already holds this name" for the namespace types,
and formats the answer as the export-check error wording. It does not
enforce anything itself. `rename_component_names` uses it to refuse a
taken target name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pypsa import Network

logger = logging.getLogger(__name__)

NAMESPACE_ORDER: tuple[str, ...] = (
    "Bus",
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
"""The namespace types, in rename order (buses, branches, one-ports, global constraints)."""


def _name_level(index: pd.Index) -> pd.Index:
    """Return the unique names held by a static index, dropping any scenario level."""
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values("name").unique()
    return index


def name_owners(
    n: Network, names: pd.Index, exclude: str | None = None
) -> dict[str, list[str]]:
    """Map each of `names` already held by a namespace type to its owning types.

    Checks every namespace type except `exclude`. A name absent from all of
    them is left out of the result, so an empty dict means no clash.

    Each type's own `static.index` engine answers the membership check
    (`idx.get_indexer(names) >= 0`), so repeated calls reuse pandas' cached
    hash table instead of rebuilding a set of all names.
    """
    owners: dict[str, list[str]] = {}
    for type_name in NAMESPACE_ORDER:
        if type_name == exclude:
            continue
        index = _name_level(n.components[type_name].static.index)
        hits = index.get_indexer(names) >= 0
        for name in names[hits]:
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


def deduplicate_names(n: Network) -> dict[str, dict[str, str]]:
    """Rename namespace-type component names that clash across types.

    Walks `NAMESPACE_ORDER` with a running set of names already claimed. The
    first type to claim a name keeps it. A later type's clashing name is
    renamed to `"<name>-<Type>"`, with `-2`, `-3`, ... appended while that
    candidate is still taken, either by another type, by the same type's own
    names, or by a candidate already assigned to another name of the same
    type in this call.

    Renames are applied one type at a time, in `NAMESPACE_ORDER`, so that by
    the time a type's renames are applied, their target names are free.

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

        for name in names.unique():
            name = str(name)
            if name not in taken:
                continue
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
        for type_name, type_map in renames.items():
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
