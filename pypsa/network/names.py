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

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pypsa import Network

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
