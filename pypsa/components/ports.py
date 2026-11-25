# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Per-type port table, the bus ports of a component type and how its columns are named.

An exporter that turns `bus`, `bus0`, `bus1`, ... columns into port-keyed
records reads these facts from [`port_table`][pypsa.components.ports.port_table]
instead of parsing column names itself.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pypsa.components import types

if TYPE_CHECKING:
    from pypsa.components.components import Components

_ONE_PORT_CATEGORIES = frozenset({"controllable_one_port", "passive_one_port"})
_EXTENDING_CATEGORIES = frozenset({"controllable_branch"})

# Stems a type leaves bare at port 1, the one port fact the registry cannot express.
_UNSUFFIXED: dict[str, frozenset[str]] = {
    "Link": frozenset({"efficiency", "delay", "cyclic_delay"}),
}

_BUS_PORT = re.compile(r"bus(\d+)")
_SUFFIXED = re.compile(r"(.+?)(\d+)")


@dataclass(frozen=True)
class PortTable:
    """Bus ports of a component type and the column names of its per-port attributes.

    Obtain one through [`port_table`][pypsa.components.ports.port_table] or
    `c.port_table` on a component instance.

    Attributes
    ----------
    name : str
        Canonical component type name, e.g. `"Link"`.
    ports : tuple[int, ...]
        Ports the registry declares, sorted. `(0,)` for one-port types, empty
        for types without a bus port (e.g. Bus, Carrier, GlobalConstraint).
    one_port : bool
        Whether the type attaches to a single bus, held in the bare `bus` column.
    extends : bool
        Whether instances can grow ports at runtime (`bus2`, `bus3`, ...).
    unsuffixed : frozenset[str]
        Stems whose port-1 column carries no suffix (`efficiency` on a Link).
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
    _attributes: frozenset[str] = field(default=frozenset(), repr=False, compare=False)

    def _takes(self, port: int) -> bool:
        return self.extends or port in self.ports

    def column(self, stem: str, port: int) -> str:
        """Column name of `stem` at `port`.

        Parameters
        ----------
        stem : str
            Attribute stem, e.g. `"bus"`, `"p"` or `"efficiency"`.
        port : int
            Port number.

        Returns
        -------
        str
            The bare stem for one-port types and for an unsuffixed stem at
            port 1, otherwise `f"{stem}{port}"`.

        Examples
        --------
        >>> from pypsa.components import port_table
        >>> port_table("Process").column("rate", 1)
        'rate1'

        """
        if self.one_port or (port == 1 and stem in self.unsuffixed):
            return stem
        return f"{stem}{port}"

    def carries(self, stem: str, port: int) -> bool:
        """Whether the type has an attribute `stem` at `port`.

        A port beyond the declared ones carries every stem, since the runtime
        port extension adds all of them.

        Parameters
        ----------
        stem : str
            Attribute stem.
        port : int
            Port number.

        Returns
        -------
        bool

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
        if not self._takes(port):
            return False
        if port not in self.ports:
            return True
        return self.column(stem, port) in self._attributes

    def ports_of(self, c: Components) -> list[int]:
        """Ports a live component instance has, as sorted integers.

        Parameters
        ----------
        c : pypsa.Components
            Component instance of this type.

        Returns
        -------
        list[int]

        Examples
        --------
        >>> n.c.links.port_table.ports_of(n.c.links)
        [0, 1]

        """
        found = {0 if p == "" else int(p) for p in c.ports}
        return sorted(p for p in found if self._takes(p))

    def buses(self, c: Components) -> dict[int, str]:
        """Bus column name of each port a live component instance has.

        Parameters
        ----------
        c : pypsa.Components
            Component instance of this type.

        Returns
        -------
        dict[int, str]

        Examples
        --------
        >>> n.c.generators.port_table.buses(n.c.generators)
        {0: 'bus'}

        """
        return {p: self.column("bus", p) for p in self.ports_of(c)}


@functools.cache
def _build(list_name: str) -> PortTable:
    # Keyed by list name: the registry resolves it for custom types too, the
    # canonical name only for the defaults.
    ctype = types.get(list_name)
    name = ctype.name
    category = ctype.category if isinstance(ctype.category, str) else ""
    one_port = category in _ONE_PORT_CATEGORIES
    extends = category in _EXTENDING_CATEGORIES
    unsuffixed = _UNSUFFIXED.get(name, frozenset())
    defaults = ctype.defaults
    # A custom type's defaults may keep the names in a column, not the index.
    index = (
        defaults["attribute"] if "attribute" in defaults.columns else defaults.index
    ).astype(str)
    if one_port:
        ports: tuple[int, ...] = (0,)
        stems: frozenset[str] = frozenset()
    else:
        ports = tuple(
            sorted(int(m.group(1)) for row in index if (m := _BUS_PORT.fullmatch(row)))
        )
        stems = (
            frozenset(
                m.group(1)
                for row in index
                if (m := _SUFFIXED.fullmatch(row)) and int(m.group(2)) in ports
            )
            | unsuffixed
        )
    return PortTable(
        name=name,
        ports=ports,
        one_port=one_port,
        extends=extends,
        unsuffixed=unsuffixed,
        stems=stems,
        _attributes=frozenset(index),
    )


def port_table(name: str) -> PortTable:
    """Port table of a component type.

    Built from the component type registry and cached per type, so
    repeated calls are cheap. Runtime port extension on a network does not
    change it: see `extends` and `ports_of`.

    Parameters
    ----------
    name : str
        Component type name or list name, e.g. `"Link"` or `"links"`.

    Returns
    -------
    PortTable

    Examples
    --------
    >>> from pypsa.components import port_table
    >>> port_table("links") is port_table("Link")
    True
    >>> port_table("Line").stems == {"bus", "p", "q"}
    True

    """
    return _build(types.get(name).list_name)


__all__ = ["PortTable", "port_table"]
