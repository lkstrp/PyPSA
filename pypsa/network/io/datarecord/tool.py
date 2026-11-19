# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The PyPSA tool: record -> `pypsa.Network` -> record."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from datarecord.protocols import RecordLike
from datarecord.select import Selection
from datarecord.tools.base import (
    Requirements,
    Schema,
    Tool,
    UnsupportedRecordError,
)
from duckdb import ColumnExpression as col
from duckdb import (
    DuckDBPyConnection,
)

if TYPE_CHECKING:
    import pypsa
from pypsa.descriptors import _update_ports_component_attrs
from pypsa.network.io._common import (
    _broadcast_standard_types,
)
from pypsa.network.io.datarecord.build import (
    _apply_axes,
    _components,
    _connection_frame,
    _narrowed,
    _own_dims,
    _required_attributes,
    _squeezed,
    _TypeMembers,
    _unheld,
    _untimed,
)
from pypsa.network.io.datarecord.lazy import RecordSource
from pypsa.network.io.datarecord.schema import (
    _AXIS_TYPES,
    _SHAPE_PREFIX,
    BUS,
    CTYPE,
    ENTITY,
    PERIOD,
    REQUIRED_DIMS,
    SCENARIO,
    SNAPSHOT,
    _port_attribute,
    _registered,
    _shape_properties,
    _taken,
)
from pypsa.network.io.datarecord.shape import NetworkShape, _multiperiod
from pypsa.network.io.datarecord.source import (
    _axis_labels,
    _colliding_names,
    _exported,
    _from_wkt,
    _NetworkSource,
    _OpenedSource,
    _repeated_buses,
    _ResultsSource,
    _root,
    _StackedSource,
    untouched_types,
)
from pypsa.network.names import name_owners

#: The axis type each of PyPSA's axis dims (`bus`, `carrier`, `sub_network`) holds.
_TYPE_OF = {dim: ctype for ctype, dim in _AXIS_TYPES.items()}


def _require_base(results_only: bool, base: RecordLike | None) -> None:
    """Refuse results without the `base` they are written over.

    Raises
    ------
    ValueError
        If `base` is missing with `results_only`.

    """
    if results_only and base is None:
        msg = (
            "`to_datarecord` takes `base` with `results_only=True`: a results "
            "layer is written over the record the network was built from"
        )
        raise ValueError(msg)


def _refusal(
    tool: str, message: str, missing: Requirements | None = None
) -> UnsupportedRecordError:
    """An `UnsupportedRecordError` carrying `message` rather than a requirement list."""
    error = UnsupportedRecordError(tool, missing or Requirements())
    error.args = (message,)
    return error


def _earlier_pin(record: RecordLike) -> bool:
    """Whether `record` has the layout of an earlier datarecord pin (`365f7a50`).

    That layout kept a multi-period network's `snapshot` as a flat dim with
    properties, and held shapes as `Shape` entities; this one holds neither.
    """
    return bool(record.schema.properties.get(SNAPSHOT)) or "Shape" in _components(
        record
    )


def _refuse_earlier_pin(tool: str, record: RecordLike) -> None:
    """Raise where `record` was written by an earlier datarecord pin, which is not migrated."""
    if _earlier_pin(record):
        msg = (
            "the record was written by an earlier datarecord pin (365f7a50) and "
            "is not migrated: its multi-period `snapshot` dim or its `Shape` "
            "entities are laid out differently; export it again from its network"
        )
        raise _refusal(tool, msg)


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

        A selection (`Record.select`) is narrowed as `build` narrows it
        (`_narrowed`). Dims of the record's own are reported in
        `unsupported_dims`, and an attribute whose rows differ along a dim
        PyPSA holds it without in `unsupported_values` (`_unheld`).

        Row ownership and curves are read per type, from `flags(ctype=...)`.

        The key checks come first and return early: the owner-map fold keys
        its relations by those dims, so reading the map at all could raise a
        binder error before anything below could report it.

        Notes
        -----
        - [AttributeSpec](https://energy-models.github.io/datarecord/design/model/attributes/#attributespec)
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        - [selecting before a build](https://energy-models.github.io/datarecord/reference/tools/#selecting-before-a-build)

        """
        return self._verify(_narrowed(record))[0]

    def _verify(
        self, record: RecordLike, unheld: dict[str, list[str]] | None = None
    ) -> tuple[Requirements, dict[str, _TypeMembers]]:
        """`verify`, with the members it read for each known type.

        `build` builds from the same reads rather than repeating them: each is
        a fold over the record's layers. Empty where a key check returned early.
        `unheld` is `_unheld` of the record where the caller has it already.
        """
        _refuse_earlier_pin(self.name, record)
        dims = REQUIRED_DIMS - set(record.schema.dimensions) - _squeezed(record)
        unsupported_keys = self._unsupported_keys(record)
        unsupported_dims = self._unsupported_dims(record)
        if unsupported_keys or unsupported_dims:
            return Requirements(
                dims=frozenset(dims),
                unsupported_keys=frozenset(unsupported_keys),
                unsupported_dims=frozenset(unsupported_dims),
            ), {}

        categories: set[str] = set()
        attributes: set[tuple[str, str]] = set()
        unsupported_values: set[tuple[str, str]] = set()
        ignored: set[tuple[str, str]] = set()

        known = self.component_types()
        declared = record.schema
        if unheld is None:
            unheld = _unheld(record, record.flags())
        unsupported_values |= {
            (_TYPE_OF[dim], attr)
            for attr in unheld
            for dim in declared.attributes[attr].dims & set(_TYPE_OF)
        }
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
                (ctype, attr)
                for attr in taken
                if resolved[attr].breakpoints
                or resolved[attr].varies.intersection(unheld.get(attr, ()))
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

    def _unsupported_dims(self, record: RecordLike) -> set[str]:
        """The record's dims of its own (`_own_dims`), which a network has none of and a selection keeping one is no network's."""
        return set(_own_dims(record.schema))

    def open(
        self, record: RecordLike, into: pypsa.Network | None = None
    ) -> pypsa.Network:
        """The network over `record`: axes now, each frame on first access.

        Snapshots, weightings, periods, scenarios, meta and network attributes
        are applied at once. Every component type the record has members of
        reads its static frame and each of its dynamic attributes from the
        record when first asked for; the others keep their in-memory containers.

        A selection (`Record.select`) opens the network of its labels, selected
        once more as PyPSA holds it (`_narrowed`), before `into` is looked at:
        a squeezed `period` is a single-period network without
        `investment_period`, a squeezed `scenario` a deterministic one, and kept
        scenarios' weights are renormalised.

        Parameters
        ----------
        record
            The record to open, or a selection of one; the network keeps it
            (narrowed) as `n.record`.
        into
            A network to open the record into, by default a fresh one. It must
            hold no components outside the standard types.

        Raises
        ------
        UnsupportedRecordError
            If `verify` reports anything missing - a partial build would
            surface as a confusing PyPSA error several frames later.
        TypeError
            If `record` is this tool's own network source, whose `flags` the
            build reads and which answers none.
        ValueError
            If `into` already holds components outside the standard types, or
            `record` is a selection squeezing a dim but `period`, `scenario`
            or a dim of its own, holding no snapshot, or keeping scenarios
            weighing nothing in all, or if an attribute's rows name several
            labels of a dim PyPSA holds it without (`_unheld`).

        Notes
        -----
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)
        - [the broadcast rule](https://energy-models.github.io/datarecord/design/model/attributes/#the-broadcast-rule)

        """
        import pypsa

        if isinstance(record, _NetworkSource):
            msg = (
                "the PyPSA tool builds from a Record, not a network source: "
                "a network source answers no flags"
            )
            raise TypeError(msg)
        record = _narrowed(record)
        _refuse_earlier_pin(self.name, record)
        if unheld := _unheld(record, record.flags()):
            over = ", ".join(f"`{attr}` over {dims}" for attr, dims in unheld.items())
            msg = (
                f"A network holds one value of {over}, where the record holds "
                "several labels its rows name: select one label as a list of one"
            )
            raise ValueError(msg)
        missing, read = self._verify(record, unheld)
        if missing:
            raise UnsupportedRecordError(self.name, missing)
        if into is not None:
            held = sorted(c.name for c in into.components if _exported(c))
            if held:
                msg = f"`into` already holds components of {held}; open into an empty network"
                raise ValueError(msg)

        con = record.con
        shape = NetworkShape(record.schema, record.dims)
        n = pypsa.Network() if into is None else into
        _apply_axes(n, record, shape, con)

        # Ports beyond the registry's two exist before any frame is read.
        for ctype in ("Link", "Process"):
            connections = read[ctype].connections if ctype in read else None
            if connections is None:
                continue
            registered = _registered(ctype)
            ports = {
                _port_attribute(BUS, port, registered)
                for port in connections["port"].unique()
            }
            _update_ports_component_attrs(
                n, where=pd.DataFrame(columns=sorted(ports)), c_name=ctype
            )

        source = RecordSource(record, n, shape, con, read, record.flags(), self.schema)
        for ctype in source.types():
            n.c[ctype]._set_source(source)
        _add_shapes(n, record)
        _broadcast_standard_types(n)
        n._record = record
        return n

    def build(self, record: RecordLike) -> pypsa.Network:
        """The resolved network, every frame read from the record.

        `open` with each container then materialised, so nothing depends on
        the record afterwards.

        Raises
        ------
        UnsupportedRecordError
            If `verify` reports anything missing, a dim of the record's own included.
        ValueError
            As `open` does.

        """
        n = self.open(record)
        for c in n.components:
            c._materialize()
        return n

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

        Over a `base`, a static value or series repeating what the record
        resolves to without it is left out: `base`'s value, else the default.
        With `base=n.record`, the record the network was opened from,
        attributes the network never read are not compared and fall through to
        `base`.

        With `results_only` the record is a thin layer over `base`: the
        outputs, at their default too, `base`'s output rows it does not write
        again restated as NULL, and this solve's SubNetworks as the
        `sub_network` axis, which drops `base`'s this solve did not number
        again.

        Raises
        ------
        ValueError
            If `base` is missing with `results_only`. If `base` is a selection
            or a record with a dim of its own and the export is not
            `results_only`; or a multi-period `base` whose investment periods
            the network does not carry is written without it.
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
        _require_base(results_only, base)
        opened = not results_only and base is not None and base is model.record
        loaded = (
            {c.name: c.loaded_attributes for c in model.components} if opened else {}
        )
        if not results_only and isinstance(base, Selection):
            msg = (
                "A full export is written over a record, not over a selection "
                "of one: its rows would miss the labels the selection squeezes. "
                "Write a solved network's results over it with `results_only=True`"
            )
            raise ValueError(msg)
        if len(model.snapshots) == 1 and model.snapshots[0] == "now":
            msg = (
                "the network has the default snapshots `['now']`, which a record "
                "cannot tell from a real time axis: call `n.set_snapshots(...)` first"
            )
            raise _refusal(self.name, msg)
        untouched = untouched_types(loaded)
        repeated = _repeated_buses(model, untouched)
        if repeated:
            msg = (
                "a record keys a connection by (entity, bus), so a component on "
                f"one bus through two ports loses a port: {'; '.join(repeated)}"
            )
            raise _refusal(self.name, msg)
        clashing = _colliding_names(model, untouched)
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
        own: list[str] = []
        if base is not None:
            own = _own_dims(_root(base).schema)
            squeezed = _squeezed(base)
            if own and (not results_only or len(own) > 1 or own[0] not in squeezed):
                msg = (
                    f"A network is not written over a record with a dim of its "
                    f"own, {own}, where its rows would reach every label: write "
                    "its results over the selection it was built from, "
                    f"`record.select({own[0]}=label)`"
                )
                raise ValueError(msg)
            if not results_only and _multiperiod(base):
                periods = _axis_labels(base, PERIOD)
                if list(model.investment_periods) != periods:
                    msg = (
                        f"A network of investment periods "
                        f"{list(model.investment_periods)} is written over a "
                        f"`base` of {periods} with `results_only=True` only"
                    )
                    raise ValueError(msg)
            axis = base.dims.get(SCENARIO)
            if (
                results_only
                and axis is not None
                and axis.limit(1).fetchall()
                and not model.has_scenarios
            ):
                msg = (
                    "A deterministic network's results are not written over a "
                    "`base` holding scenarios: NULL on `scenario`, they would "
                    "reach every scenario. Write them over the selection it was "
                    "built from, `record.select(scenario=label)`"
                )
                raise ValueError(msg)
            period = None
            if isinstance(base, Selection) and PERIOD in squeezed:
                period = base.labels[PERIOD]
        if results_only and base is not None:
            source: _NetworkSource = _ResultsSource(model, con, _untimed(base), period)
        elif opened:
            source = _OpenedSource(model, con, base, loaded=loaded)
        else:
            source = _NetworkSource(model, con, base)
        source.check()
        if own and isinstance(base, Selection):
            (dim,) = own
            return _StackedSource.of({base.labels[dim]: source}, dim, con, base)
        return source


def _add_shapes(n: pypsa.Network, record: RecordLike) -> None:
    """Import the shape properties of `record` as `n.shapes`, in `n.crs`."""
    rows = []
    for dim in record.schema.properties:
        outlines = _shape_properties(record.schema, dim)
        frame = record.properties.get(dim)
        if not outlines or frame is None:
            continue
        keys = [ENTITY, CTYPE] if dim == ENTITY else [dim]
        held = frame.project(*map(col, [*keys, *outlines])).df()
        long = held.melt(id_vars=keys, var_name="type", value_name="geometry")
        long = long.dropna(subset=["geometry"])
        component = long[CTYPE] if dim == ENTITY else _TYPE_OF.get(dim, dim)
        rows.append(
            pd.DataFrame(
                {
                    "component": component,
                    "idx": long[dim].astype(str),
                    "type": long["type"].str.removeprefix(_SHAPE_PREFIX),
                    "geometry": long["geometry"],
                }
            )
        )
    if not rows:
        return
    static = pd.concat(rows, ignore_index=True)
    if static.empty:
        return
    static.index = pd.Index(
        static["component"] + " " + static["idx"] + " " + static["type"], name="name"
    )
    static["geometry"] = _from_wkt(static["geometry"], n.crs)
    if n.has_scenarios:
        static = pd.concat(dict.fromkeys(n.scenarios, static), names=[SCENARIO])
    n._import_components_from_df(static, "Shape")


# The tool is a module-level singleton, imported rather than looked up by
# name: `from pypsa.network.io.datarecord import PyPSA`, then `PyPSA.build(record)`.
PyPSA = PyPSATool()
