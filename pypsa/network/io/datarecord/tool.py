# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The PyPSA tool: record -> `pypsa.Network` -> record."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import pandas as pd
from datarecord.protocols import Flags, RecordLike
from datarecord.schema.categories import with_defaults
from datarecord.tools.base import (
    Requirements,
    Schema,
    Tool,
    UnsupportedRecordError,
)
from duckdb import (
    DuckDBPyConnection,
    DuckDBPyRelation,
)
from duckdb import StarExpression as star

if TYPE_CHECKING:
    import pypsa
from pypsa.network.io._common import (
    _broadcast_standard_types,
)
from pypsa.network.io.datarecord.build import (
    _add_component_type,
    _collapse_connections,
    _components,
    _connection_frame,
    _new_network,
    _port_columns,
    _required_attributes,
    _tuples_of,
    _TypeMembers,
    _unpack_connection,
)
from pypsa.network.io.datarecord.schema import (
    _AXIS_TYPES,
    CONNECTION,
    CTYPE,
    ENTITY,
    REQUIRED_DIMS,
    SCENARIO,
    SNAPSHOT,
    _taken,
    pypsa_attribute,
)
from pypsa.network.io.datarecord.shape import (
    NetworkShape,
    _connection,
)
from pypsa.network.io.datarecord.source import (
    _clashing_columns,
    _colliding_names,
    _empty_for_none,
    _keyed_as_entity,
    _NetworkSource,
    _repeated_buses,
    _ResultsSource,
)
from pypsa.network.names import name_owners


def _refusal(
    tool: str, message: str, missing: Requirements | None = None
) -> UnsupportedRecordError:
    """An `UnsupportedRecordError` carrying `message` rather than a requirement list."""
    error = UnsupportedRecordError(tool, missing or Requirements())
    error.args = (message,)
    return error


class PyPSATool(Tool):
    """Build a `pypsa.Network` from a record and read its results back."""

    name = "pypsa"

    # PyPSA defines the record vocabulary, so every attribute maps to
    # itself and there is nothing to declare. The seam is here for the first
    # tool whose names differ, or whose value is computed from several of the
    # record's (`Attr.compute`).
    schema = Schema()

    def component_types(self) -> frozenset[str]:
        """The component types PyPSA knows, from its type registry.

        Taken from the component *type registry*, not a network instance:
        `pypsa.Network().components` iterates only the components attached to
        that network, which for an empty one is just the standard types.
        """
        from pypsa.components.types import all_components

        return frozenset(c.name for c in all_components.values())

    def requires(self, record: RecordLike) -> Requirements:
        """PyPSA's axes, the record's own component types, and their required attributes.

        The attribute half is record-dependent: only types the record
        actually defines members for are required, and which of their
        attributes are mandatory comes from PyPSA's registry
        (`defaults["status"] == "Input (required)"`), not a list kept here.
        Reported by source attribute (`Schema.sources`), so a renamed or
        computed one names what the record must actually supply.
        """
        known = self.component_types()
        ctypes = {ct for ct in _components(record) if ct in known}
        return Requirements(
            dims=REQUIRED_DIMS,
            categories=frozenset(ctypes),
            attributes=frozenset(
                (ct, src)
                for ct in ctypes
                for attr in _required_attributes(ct)
                for src in self.schema.sources(ct, attr)
            ),
        )

    def verify(self, record: RecordLike) -> Requirements:
        """What this record fails to supply for a PyPSA build; falsy if it is usable.

        Checks what a build needs: the schema declares PyPSA's axes, its key dims are ones this tool can honour, every component
        type it names is one PyPSA knows, and each type's required attributes
        are resolvable - either owned by some layer (the owner map), a
        property of the entity axis, or carrying a declared default for that type. A
        renamed or computed attribute is checked against its sources
        (`Schema.sources`), which is also how it is reported. A port's bus is
        supplied only where every member has a connection there, so an entity
        attached to no bus reports its type's `bus` (`_TypeMembers.port_attributes`).
        A stored attribute a type does not take is left out by `build` and
        reported in `ignored`, which leaves the result falsy.

        Row ownership and curves are read per type, from `flags(ctype=...)`.

        The key checks come first and return early: the owner-map fold keys
        its relations by those dims, so reading the map at all could raise a
        binder error before anything below could report it.

        Notes
        -----
        - [AttributeSpec](https://energy-models.github.io/datarecord/design/model/attributes/#attributespec)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

        """
        return self._verify(record)[0]

    def _verify(
        self, record: RecordLike
    ) -> tuple[Requirements, dict[str, _TypeMembers]]:
        """`verify`, with the members it read for each known type.

        `build` builds from the same reads rather than repeating them: each is
        a fold over the record's layers. Empty where a key check returned early.
        """
        dims = REQUIRED_DIMS - set(record.schema.dims)
        unsupported_keys = self._unsupported_keys(record)
        if unsupported_keys:
            return Requirements(
                dims=frozenset(dims), unsupported_keys=frozenset(unsupported_keys)
            ), {}

        categories: set[str] = set()
        attributes: set[tuple[str, str]] = set()
        unsupported_values: set[tuple[str, str]] = set()
        ignored: set[tuple[str, str]] = set()

        known = self.component_types()
        declared = record.schema
        read: dict[str, _TypeMembers] = {}
        ctypes = _components(record)
        connection_frame = _connection_frame(record) if known & set(ctypes) else None
        for ctype in ctypes:
            # A type PyPSA has no registry entry for, though the record's own
            # schema declares it. Reported rather than raised: the record layer
            # upholds its schema's vocabulary and knows no framework's, so this
            # reads back fine and it is this tool's business that it cannot be
            # built.
            if ctype not in known:
                categories.add(ctype)
                continue
            resolved = record.flags(**{CTYPE: ctype})
            taken = _taken(declared, ctype, resolved)
            ignored |= {(ctype, attr) for attr in resolved if attr not in taken}
            # A curve, not a scalar (https://energy-models.github.io/datarecord/design/model/attributes/#wide-and-long-rows). PyPSA takes a scalar for every
            # attribute this build assigns, so the record is storing something
            # correct that this translation cannot express - reported here
            # rather than silently pivoting one arbitrary breakpoint.
            unsupported_values |= {
                (ctype, attr) for attr in taken if resolved[attr].breakpoints
            }
            read[ctype] = _TypeMembers.of(record, ctype, connection_frame)
            static_cols = read[ctype].columns
            # A port attribute the record supplies as connection rows rather
            # than as a column: `bus0`/`bus1` are satisfied by a connection
            # per port, so the collapse in `build` can name them (https://energy-models.github.io/datarecord/design/tools/pypsa/#ports).
            from_connections = read[ctype].port_attributes(ctype)
            for attr in _required_attributes(ctype):
                for src in self.schema.sources(ctype, attr):
                    if src in resolved or src in static_cols or src in from_connections:
                        continue
                    # A declared default makes the attribute resolvable even
                    # with no row anywhere (the broadcast rule).
                    if (
                        src in declared.attributes
                        and declared.default_for(src, **{CTYPE: ctype}) is not None
                    ):
                        continue
                    attributes.add((ctype, src))
        return Requirements(
            dims=frozenset(dims),
            categories=frozenset(categories),
            attributes=frozenset(attributes),
            unsupported_values=frozenset(unsupported_values),
            ignored=frozenset(ignored),
        ), read

    def _unsupported_keys(self, record: RecordLike) -> set[tuple[str, str]]:
        """`(key, dim)` pairs this tool cannot honour: `snapshot`, as any key.

        PyPSA's static/series split needs a component's whole series to come
        from one layer, so a snapshot-keyed overlay could leave a broadcast row
        and a descendant's per-snapshot row with no single container to land in. A limit of the *representation*, not the format - which is why
        the record layer permits the declaration and this reports it.

        Notes
        -----
        - [partial](https://energy-models.github.io/datarecord/design/layers/tree/#partial-the-granularity-of-an-override)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

        """
        defs = record.schema
        return {("owner_map_key", SNAPSHOT)} if SNAPSHOT in defs.partial_dims else set()

    def build(self, record: RecordLike) -> pypsa.Network:
        """The resolved network: its carriers and buses, then one component type at a time.

        Raises
        ------
        UnsupportedRecordError
            If `verify` reports anything missing - a partial build would
            surface as a confusing PyPSA error several frames later.

        Notes
        -----
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        - [the broadcast rule](https://energy-models.github.io/datarecord/design/model/attributes/#the-broadcast-rule)

        """
        missing, read = self._verify(record)
        if missing:
            raise UnsupportedRecordError(self.name, missing)

        con = _connection(record)
        schema = record.schema
        shape = NetworkShape(record.dims, record.properties)
        n = _new_network(record, shape, con)
        flagged = record.flags()

        over_entity = [a for a in schema.attributes if schema.addresses(a, ENTITY)]
        scenarios = {SCENARIO: record.dims[SCENARIO]} if shape.stochastic else {}
        tuples = record.dims.get(CONNECTION)
        for ctype in _AXIS_TYPES:
            self._add_labels(n, record, ctype, flagged, shape, scenarios, con)
        for ctype, members in read.items():
            static = members.members
            if static is None:
                continue
            labels = {**scenarios, ENTITY: static}
            if tuples is not None:
                labels[CONNECTION] = _tuples_of(tuples, static)
            if not shape.stochastic and SCENARIO in static.columns:
                static = static.project(star(exclude=[SCENARIO]))
            # `_pos` in member order before any join scrambles it: the member
            # frame arrives in member order (https://energy-models.github.io/datarecord/design/layers/resolution/#one-fold-for-every-axis),
            # and `_assign_static` sorts the wide frame back by it at the end.
            static = static.project("*, row_number() OVER () AS _pos")
            # Connections back to the positional columns PyPSA expects (https://energy-models.github.io/datarecord/design/tools/).
            connections = members.connections
            static = _collapse_connections(static, connections, ctype, con)

            # Frames are built and released per type, so peak memory is one
            # type's wide frames rather than the whole network (https://energy-models.github.io/datarecord/design/tools/).
            attributes = {}
            carried = _taken(schema, ctype, over_entity)
            for attr, flags in flagged.items():
                if attr not in carried:
                    continue
                # A named row of another type's entity is left to the joins on
                # `entity` (https://energy-models.github.io/datarecord/design/storage/format/#a-label-is-unique-within-its-dim).
                long = _unpack_connection(
                    self.schema.resolve(
                        record, ctype, attr, labels, flagged, filter=False
                    )
                )
                if CONNECTION in schema.attributes[attr].dims:
                    for column, rows in _port_columns(
                        long, attr, connections, ctype, con
                    ).items():
                        attributes[column] = (rows, flags)
                else:
                    attributes[pypsa_attribute(ctype, attr)] = (long, flags)

            _add_component_type(n, ctype, static, attributes, shape, con)

        _broadcast_standard_types(n)
        return n

    def _add_labels(
        self,
        n: pypsa.Network,
        record: RecordLike,
        ctype: str,
        flagged: dict[str, Flags],
        shape: NetworkShape,
        scenarios: Mapping[str, DuckDBPyRelation],
        con: DuckDBPyConnection,
    ) -> None:
        """Add an `_AXIS_TYPES` type's members: its dim's labels, as `n.buses`, `n.carriers` or `n.sub_networks`.

        Each label with its properties, defaults applied, then the attributes
        over the dim, a series by its flags, so a solved bus keeps its
        `marginal_price`. Keyed as
        `entity`, the static frame and the series pivot are a component's.

        Notes
        -----
        - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)
        - [default_by](https://energy-models.github.io/datarecord/design/model/properties/#default_by-a-default-per-category)
        - [flags](https://energy-models.github.io/datarecord/design/use/reading/#flags)

        """
        dim = _AXIS_TYPES[ctype]
        frame = record.properties.get(dim)
        if frame is None:
            return
        schema = record.schema
        labels = with_defaults(schema, dim, frame, list(schema.properties[dim]), con)
        static = _keyed_as_entity(_empty_for_none(labels), dim).project(
            "*, row_number() OVER () AS _pos"
        )
        over = {**scenarios, dim: labels}
        attributes = {
            pypsa_attribute(ctype, attr): (
                _keyed_as_entity(
                    self.schema.resolve(
                        record, ctype, attr, over, flagged, filter=False
                    ),
                    dim,
                ),
                flags,
            )
            for attr, flags in flagged.items()
            if attr in schema.attributes and dim in schema.attributes[attr].dims
        }
        _add_component_type(n, ctype, static, attributes, shape, con)

    def to_datarecord(
        self,
        model: pypsa.Network,
        con: DuckDBPyConnection,
        *,
        results_only: bool = False,
        base: RecordLike | None = None,
    ) -> RecordLike:
        """Present a `Network` as the `Record` `write_layer` persists.

        The inverse of `build`, and the only place PyPSA's shape is undone:
        `c.static`/`c.dynamic` become long rows on `con`, and
        `bus0`/`bus1`/`efficiency2` become connection rows carrying a `role`,
        and `n.buses` and `n.carriers` the `bus` and `carrier` axes, in PyPSA's
        order, with every carrier named but not listed added and `""` stored as
        NULL. Its outputs, by the registry's status "Output", are attributes
        like any other, those still at their default left out. Key sets are
        read off the network, so listing unpivots no input.

        With `results_only` the record is a thin layer over `base`: the
        outputs, at their default too, `base`'s output rows it does not write
        again restated as NULL, and this solve's SubNetworks as the
        `sub_network` axis, which drops `base`'s this solve did not number
        again.

        Raises
        ------
        ValueError
            If `base` is given without `results_only`, or missing with it.
        UnsupportedRecordError
            If two entity types share a name: PyPSA scopes names per type, a
            record record-wide, and this reports rather than renames. Likewise
            for a static column PyPSA's registry has no entry for on its type
            whose attribute is declared over another dim (`Carrier.efficiency`,
            `Bus.country` beside `Generator.country`).

        Notes
        -----
        - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
        - [a label is unique within its dim](https://energy-models.github.io/datarecord/design/storage/format/#a-label-is-unique-within-its-dim)
        - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)
        - [writing a whole record](https://energy-models.github.io/datarecord/design/use/writing/)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)

        """
        if results_only != (base is not None):
            msg = (
                "`to_datarecord` takes `base` exactly when `results_only=True`: "
                "a results layer is written over the record the network was "
                "built from, and a full export over nothing"
            )
            raise ValueError(msg)
        if len(model.snapshots) == 1 and model.snapshots[0] == "now":
            msg = (
                "the network has the default snapshots `['now']`, which a record "
                "cannot tell from a real time axis: call `n.set_snapshots(...)` first"
            )
            raise _refusal(self.name, msg)
        repeated = _repeated_buses(model)
        if repeated:
            msg = (
                "a record keys a connection by (entity, bus), so a component on "
                f"one bus through two ports loses a port: {'; '.join(repeated)}"
            )
            raise _refusal(self.name, msg)
        clashing = _colliding_names(model)
        columns = _clashing_columns(model)
        if clashing:
            shown = sorted(clashing)[:10]
            more = f" and {len(clashing) - 10} more" if len(clashing) > 10 else ""
            held = name_owners(model, pd.Index(shown))
            owners = sorted({c for cs in held.values() for c in cs})
            msg = (
                f"entity names are shared by more than one component type "
                f"({', '.join(owners)}): {', '.join(shown)}{more}. A record has "
                "one entity per name; rename them with "
                "`pypsa.network.names.deduplicate_names(n)`"
            )
            raise _refusal(self.name, msg, Requirements(names=clashing))
        if columns:
            raise UnsupportedRecordError(self.name, Requirements(columns=columns))
        if base is not None:
            return _ResultsSource(model, con, base)
        return _NetworkSource(model, con)


# The tool is a module-level singleton, imported rather than looked up by
# name: `from pypsa.network.io.datarecord import PyPSA`, then `PyPSA.build(record)`.
PyPSA = PyPSATool()
