# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Per-type port table, the bus ports of a component type and their column names."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pypsa.components import types

if TYPE_CHECKING:
    from pypsa.components.components import Components

# Stems a type leaves bare at port 1, the one fact the registry cannot express.
_UNSUFFIXED = {"Link": frozenset({"efficiency", "delay", "cyclic_delay"})}
_SUFFIXED = re.compile(r"(.+?)(\d+)")


@dataclass(frozen=True)
class PortTable:
    """Bus ports of a component type and the column names of its per-port attributes.

    Attributes
    ----------
    name : str
        Canonical component type name, e.g. `"Link"`.
    ports : tuple[int, ...]
        Declared ports, sorted. `(0,)` for one-port types, empty without a bus.
    one_port : bool
        Whether the type attaches to a single bus in the bare `bus` column.
    extends : bool
        Whether instances can grow ports at runtime (`bus2`, `bus3`, ...).
    unsuffixed : frozenset[str]
        Stems whose port-1 column has no suffix (`efficiency` on a Link).
    stems : frozenset[str]
        Every per-port attribute stem, `bus` included. Empty for one-port types.

    Examples
    --------
    >>> from pypsa.components import port_table
    >>> t = port_table("Link")
    >>> t.ports, t.extends
    ((0, 1), True)
    >>> t.column("efficiency", 1), t.column("efficiency", 2)
    ('efficiency', 'efficiency2')
    >>> port_table("Generator").column("bus", 0)
    'bus'

    """

    name: str
    ports: tuple[int, ...]
    one_port: bool
    extends: bool
    unsuffixed: frozenset[str]
    stems: frozenset[str]
    _attributes: frozenset[str] = field(repr=False, compare=False)

    def column(self, stem: str, port: int) -> str:
        """Column name of `stem` at `port`."""
        if self.one_port or (port == 1 and stem in self.unsuffixed):
            return stem
        return f"{stem}{port}"

    def carries(self, stem: str, port: int) -> bool:
        """Whether the type has `stem` at `port`.

        Ports beyond the declared ones carry every stem on extending types.

        Examples
        --------
        >>> from pypsa.components import port_table
        >>> port_table("Link").carries("efficiency", 0)
        False
        >>> port_table("Link").carries("efficiency", 2)
        True
        >>> port_table("Line").carries("bus", 3)
        False

        """
        if port not in self.ports:
            return self.extends
        return self.column(stem, port) in self._attributes

    def ports_of(self, c: Components) -> list[int]:
        """Sorted ports a live component instance has."""
        found = {int(p or 0) for p in c.ports}
        return sorted(p for p in found if self.extends or p in self.ports)

    def buses(self, c: Components) -> dict[int, str]:
        """Bus column name of each port a live component instance has."""
        return {p: self.column("bus", p) for p in self.ports_of(c)}


@functools.cache
def _build(list_name: str) -> PortTable:
    # Keyed by list name, which the registry resolves for custom types too.
    ctype = types.get(list_name)
    one_port = ctype.category in ("controllable_one_port", "passive_one_port")
    unsuffixed = _UNSUFFIXED.get(ctype.name, frozenset())
    defaults = ctype.defaults
    # A custom type's defaults may keep the names in a column, not the index.
    names = defaults["attribute"] if "attribute" in defaults.columns else defaults.index
    attrs = frozenset(names.astype(str))
    ports: tuple[int, ...] = (0,)
    stems: frozenset[str] = frozenset()
    if not one_port:
        suffixed = [(m[1], int(m[2])) for a in attrs if (m := _SUFFIXED.fullmatch(a))]
        ports = tuple(sorted(p for s, p in suffixed if s == "bus"))
        stems = frozenset(s for s, p in suffixed if p in ports) | unsuffixed
    return PortTable(
        name=ctype.name,
        ports=ports,
        one_port=one_port,
        extends=ctype.category == "controllable_branch",
        unsuffixed=unsuffixed,
        stems=stems,
        _attributes=attrs,
    )


def port_table(name: str) -> PortTable:
    """Return the cached port table of a type, by name or list name.

    Runtime port extension does not change it, see `extends` and `ports_of`.

    Examples
    --------
    >>> from pypsa.components import port_table
    >>> port_table("links") is port_table("Link")
    True

    """
    return _build(types.get(name).list_name)


__all__ = ["PortTable", "port_table"]
