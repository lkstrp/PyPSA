# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Record-to-network construction for the PyPSA tool."""

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from datarecord.duck import ex_all, fn, scratch_table
from datarecord.protocols import Flags, RecordLike
from datarecord.schema.categories import matching, with_defaults
from datarecord.schema.schema import Schema as RecordSchema
from datarecord.schema.schema import unpacked
from datarecord.select import Selection, select
from datarecord.settle import settle
from duckdb import CoalesceOperator as coalesce
from duckdb import ColumnExpression as col
from duckdb import ConstantExpression as lit
from duckdb import (
    DuckDBPyConnection,
    DuckDBPyRelation,
    sqltypes,
)
from duckdb import StarExpression as star

if TYPE_CHECKING:
    import pypsa
from pypsa.network.io._common import (
    _apply_network_attributes,
    _apply_snapshots_import,
)
from pypsa.network.io.datarecord.schema import (
    _AXIS_TYPES,
    _INPUT_PORT,
    _SINGLE_PORT,
    BUS,
    CONNECTION,
    CTYPE,
    ENTITY,
    GLOBAL_CONSTRAINT,
    PERIOD,
    SCENARIO,
    SNAPSHOT,
    TIMESTEP,
    _carries,
    _per_port_stems,
    _port_attribute,
    _registered,
    _registry_schema,
    _shape_properties,
    _taken,
)
from pypsa.network.io.datarecord.shape import (
    Index,
    NetworkShape,
    _multiperiod,
)
from pypsa.network.io.datarecord.source import (
    _active,
    _applying,
    _axis_labels,
    _empty_for_none,
    _root,
    _weightings,
)


def _assign_static(
    static: DuckDBPyRelation,
    attributes: dict[str, tuple[DuckDBPyRelation, Flags]],
    shape: NetworkShape,
    scenarios: Index,
    con: DuckDBPyConnection,
) -> pd.DataFrame:
    """Set each varying attribute's static-valued entries, resolved to a wide frame."""
    keys = shape.index_names

    # `static` already carries a `_pos` in member order, assigned before the
    # connection collapse could scramble it (`build`); the wide frame is ordered
    # by it below (https://energy-models.github.io/datarecord/design/layers/resolution/#one-fold-for-every-axis).

    # An entity exists once, whatever the scenario, so the record's member
    # frame carries one row per entity - where PyPSA's static container is
    # `(scenario, name)`-indexed. Repeated here, and any attribute that really
    # differs between scenarios arrived as `attributes/` rows and is assigned over
    # the repeats below.
    if shape.stochastic and SCENARIO not in static.columns:
        static = _at_each_scenario(static, scenarios.rel)

    picked: dict[str, DuckDBPyRelation] = {}
    over_snapshots: set[str] = set()
    for attr, (long, flags) in attributes.items():
        # PyPSA's `static` container takes the rows that do not vary over
        # `snapshot`. Two ways an attribute has them, and the flags distinguish
        # the two because they are scoped to what it is addressed by (https://energy-models.github.io/datarecord/design/use/reading/#flags):
        # a NULL-snapshot row on an attribute that *may* vary over it puts
        # `snapshot` in `broadcast`, while one not addressed by it at all has
        # `snapshot` in neither set and every row static.
        addressed = SNAPSHOT in (flags.varies | flags.broadcast)
        if addressed and SNAPSHOT not in flags.broadcast:
            continue
        rows = long
        if shape.stochastic and SCENARIO not in rows.columns:
            rows = _at_each_scenario(rows, scenarios.rel)
        picked[attr] = rows
        if addressed:
            over_snapshots.add(attr)

    frame = _left_join_values(static, picked, over_snapshots, keys, con)
    # Scenario-major: PyPSA's `(scenario, name)` index groups a scenario's
    # components together, and `_pos` gives the order within one - the
    # entity axis carrying one row per entity rather than one per scenario, so
    # it cannot order the outer level itself. A stable sort keeps that order.
    if shape.stochastic:
        frame = frame.sort_values(
            SCENARIO,
            kind="stable",
            key=lambda s: s.map({v: i for i, v in enumerate(scenarios.index)}),
        )
    return frame.set_index(keys)


def _at_each_scenario(
    rel: DuckDBPyRelation, scenarios: DuckDBPyRelation
) -> DuckDBPyRelation:
    """`rel`, over no `scenario`, at each of `scenarios`, as the broadcast rule reads it."""
    over = frozenset({SCENARIO})
    unset = rel.project(star(), lit(None).cast(sqltypes.VARCHAR).alias(SCENARIO))
    return settle(unset, {SCENARIO: scenarios}, Flags(frozenset(), over))


def _left_join_values(
    static: DuckDBPyRelation,
    values: Mapping[str, DuckDBPyRelation],
    over_snapshots: Container[str],
    keys: Sequence[str],
    con: DuckDBPyConnection,
) -> pd.DataFrame:
    """`static` with each of `values` left-joined on `keys` as its named column, in `_pos` order.

    A value fills its column where `static` has none (`COALESCE`); `_pos` is
    dropped. A left join from `static` drops a value with no matching static
    row rather than adding one, which also scopes the rows to `static`'s
    members.

    Each relation is held in a table of its own before the joins, dropped
    after: the relation API binds the whole tree again at every step, and a
    resolved attribute's tree is deep, so chaining the joins over the lazy
    relations made one type's static frame quadratic in its attribute count.

    Parameters
    ----------
    values
        Long rows carrying `keys` and `value`, by the column they fill, one
        per key (`settle`).
    over_snapshots
        Names in `values` addressed by `snapshot`, of whose rows only the
        NULL-snapshot ones are static.

    """
    with ExitStack() as tables:
        joined = tables.enter_context(scratch_table(con, static)).set_alias("s")
        exprs = {}
        for i, (name, long) in enumerate(values.items()):
            rows = (
                long.filter(col(SNAPSHOT).isnull()) if name in over_snapshots else long
            )
            alias = f"a{i}"
            held = rows.project(*(col(k) for k in keys), col("value"))
            picked = tables.enter_context(scratch_table(con, held)).set_alias(alias)
            joined = joined.join(
                picked, ex_all(col("s", k) == col(alias, k) for k in keys), how="left"
            )
            value = col(alias, "value")
            exprs[name] = (
                coalesce(value, col("s", name)) if name in static.columns else value
            ).alias(name)
        return (
            joined.project(
                *(col("s", c) for c in static.columns if c not in exprs),
                *exprs.values(),
            )
            .order(col("_pos"))
            .project(star(exclude=["_pos"]))
            .df()
        )


def _series_frame(
    rows: DuckDBPyRelation,
    shape: NetworkShape,
    snapshots: Index,
    static_index: pd.Index,
    con: DuckDBPyConnection,
) -> pd.DataFrame:
    """Scatter long series rows into the float `n.snapshots x static_index` frame, NaN where no row is.

    Only the columns the rows fill, since `_import_series_from_df` takes
    `df.columns` as-is and an added column would force that component into
    `dynamic`.

    A row is placed by the positions of its snapshot and entity, so a name
    never becomes an SQL identifier.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

    """
    group_by = list(shape.snapshots.columns)
    keys = shape.index_names
    columns = Index.of(static_index.set_names(keys), con).rel.set_alias("k")
    rows = unpacked(shape.schema, SNAPSHOT, rows).set_alias("r")
    placed = (
        rows.join(columns, ex_all(col("r", k) == col("k", k) for k in keys))
        .project(
            *(col("r", g) for g in group_by),
            col("k", "_pos").alias("_col"),
            col("r", "value"),
        )
        .fetchnumpy()
    )
    # Replaced upstream's second join on the snapshot key by an index lookup: the join cost GBs of peak memory.
    at = [placed[g] for g in group_by]
    pos = snapshots.index.get_indexer(
        at[0] if len(at) == 1 else pd.MultiIndex.from_arrays(at)
    )
    held = pos >= 0
    pos, cols, values = pos[held], placed["_col"][held], placed["value"][held]
    taken = np.zeros(len(static_index) + 1, dtype=bool)
    taken[cols] = True
    present = np.flatnonzero(taken)
    wide = np.full((len(snapshots.index), len(present)), np.nan)
    wide[pos, np.cumsum(taken)[cols] - 1] = values
    return pd.DataFrame(wide, index=snapshots.index, columns=static_index[present - 1])


def _apply_axes(
    n: pypsa.Network,
    record: RecordLike,
    shape: NetworkShape,
    con: DuckDBPyConnection,
) -> None:
    """Establish the record's axes on `n`: attributes, meta, crs, snapshots, periods, scenarios.

    Meant for a network whose components are still in memory, so that
    `set_snapshots` and `set_scenarios` reshape only empty frames.
    """
    schema = record.schema
    _apply_network_attributes(n, schema.meta.get("attributes") or {})
    n.meta = schema.meta.get("meta") or {}
    crs = schema.meta.get("crs")
    if crs is not None:
        n.to_crs(crs)

    _apply_snapshots_import(n, shape.snapshots)
    if shape.multiperiod:
        n.periods = shape.periods[PERIOD]
        n.investment_period_weightings = _weightings(
            record, PERIOD, n.investment_period_weightings
        )
    n.snapshot_weightings = _weightings(record, SNAPSHOT, n.snapshot_weightings)
    if shape.stochastic:
        scenarios = shape.scenarios[SCENARIO].astype(str)
        # `set_scenarios` rather than assigning `_scenarios_data`: it rebuilds
        # each component's empty `static`/`dynamic` frames with the `scenario`
        # level, so later imports union against a correctly named 2-level index
        # instead of the single-level one a fresh component carries.
        n.set_scenarios(_scenario_weights(record, scenarios))


def _required_attributes(ctype: str) -> frozenset[str]:
    """`ctype`'s mandatory input attributes, from PyPSA's registry.

    `name` is the component's identity, always supplied by the record's member
    list rather than an attribute row, so it never counts as missing.
    """
    from pypsa.components.types import get as get_component_type

    defaults = get_component_type(ctype).defaults
    required = defaults.index[defaults["status"].astype(str) == "Input (required)"]
    return frozenset(required) - {"name"}


def _connection_frame(record: RecordLike) -> pd.DataFrame | None:
    """Every type's connections as flat `(entity, bus, ...)` rows, read once.

    `_ordered_connections` scopes it per type, so the multi-dim is not folded
    again for each.
    """
    frame = record.properties.get(CONNECTION)
    if frame is None:
        return None
    return _unpack_connection(frame).df()


def _tuples_of(tuples: DuckDBPyRelation, members: DuckDBPyRelation) -> DuckDBPyRelation:
    """The `connection` tuples of `tuples` whose entity is one of `members`."""
    return tuples.set_alias("t").join(
        members.project(col(ENTITY)).set_alias("m"),
        fn.struct_extract(col("t", CONNECTION), lit(ENTITY)) == col("m", ENTITY),
        how="semi",
    )


def _ordered_connections(
    frame: pd.DataFrame | None, members: DuckDBPyRelation | None
) -> pd.DataFrame | None:
    """One type's connections with a port index assigned per component.

    The positional collapse: connections come in member order and are
    numbered within each component, so a port index follows the multi-dim's
    row order, and a restatement that appends a connection renumbers none. Inputs are placed before outputs, so `bus0` is the input end
    PyPSA's sign convention expects.

    Scoped to one type by its `members`, `frame` (`_connection_frame`) holding
    every type's connections.

    Notes
    -----
    - [the owner map](https://energy-models.github.io/datarecord/design/layers/resolution/#owner-map)
    - [one record over one fold](https://energy-models.github.io/datarecord/design/layers/resolution/#one-record-over-one-fold)
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

    """
    if frame is None or members is None:
        return None
    df = frame
    mine = {entity for (entity,) in members.project(col(ENTITY)).fetchall()}
    df = df[df["entity"].isin(mine)]
    if df.empty:
        return None
    # Per component: ports are numbered within the entity they belong to.
    within = ["entity"]
    if "role" in df.columns:
        df = df.assign(_role_rank=(df["role"] != _INPUT_PORT).astype(int)).sort_values(
            [*within, "_role_rank"], kind="stable"
        )
    # A single-attachment component keeps PyPSA's unsuffixed `bus`, so its port
    # is the empty index rather than `0` (`_port_attribute`).
    # `dropna=False`: a dim left NULL means "every value of it" (https://energy-models.github.io/datarecord/design/model/attributes/#the-broadcast-rule), which
    # is an ordinary group here, not a row to drop.
    grouped = df.groupby(within, dropna=False)
    single = grouped["bus"].transform("size").eq(1) & df["role"].eq(_SINGLE_PORT)
    df["port"] = grouped.cumcount().astype(str).where(~single, "")
    return df


@dataclass(frozen=True)
class _TypeMembers:
    """One type's members and numbered connections, as `verify` and `build` read them.

    Read once per type and shared by the two: each read is a fold over the
    record's layers.
    """

    #: `_members`: the static frame PyPSA's `n.<type>` starts from.
    members: DuckDBPyRelation | None
    #: `_ordered_connections`: the type's connections with their port index.
    connections: pd.DataFrame | None

    @classmethod
    def of(
        cls, record: RecordLike, ctype: str, connections: pd.DataFrame | None
    ) -> _TypeMembers:
        """`ctype`'s reads, scoping `connections` (`_connection_frame`) to it."""
        members = _members(record, ctype)
        return cls(members, _ordered_connections(connections, members))

    @property
    def columns(self) -> frozenset[str]:
        """The properties the members carry, defaults applied.

        A required input among them needs no `attributes/` row to be resolvable.

        Notes
        -----
        - [where a value lives](https://energy-models.github.io/datarecord/design/storage/format/#where-a-value-lives)

        """
        return (
            frozenset(self.members.columns) if self.members is not None else frozenset()
        )

    def port_attributes(self, ctype: str) -> frozenset[str]:
        """Port attribute names the connection rows supply for every member.

        `bus0`/`bus1`/... for the ports every member has a connection at, so
        `verify` knows a required `bus0` is satisfied by a connection rather
        than by a column, and reports one an entity left attached to no bus
        lacks, as `remove("bus", ...)` may leave it.

        Notes
        -----
        - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
        - [deletion](https://energy-models.github.io/datarecord/design/layers/removal/)

        """
        if self.connections is None or self.members is None:
            return frozenset()
        members = {entity for (entity,) in self.members.project(col(ENTITY)).fetchall()}
        registered = _registered(ctype)
        return frozenset(
            _port_attribute(BUS, port, registered)
            for port, rows in self.connections.groupby("port")
            if members <= set(rows[ENTITY])
        )


def _collapse_connections(
    static: DuckDBPyRelation,
    connections: pd.DataFrame | None,
    ctype: str,
    con: DuckDBPyConnection,
) -> DuckDBPyRelation:
    """Add `bus0`/`bus1`/... columns to a static frame from its connection rows.

    PyPSA wants a column per port; the record stores a row per connection. This is the seam between them, and the only direction that needs
    positions at all.

    Parameters
    ----------
    connections
        `ctype`'s connections as `_ordered_connections` numbers them, which
        `_port_columns` places per-connection values by.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

    """
    if connections is None:
        return static
    # Pivoted on the same key the ports were numbered within, so each
    # (component, scenario, ...) keeps its own port columns.
    index = ["entity"]
    # A dim left NULL means "every value of it" (https://energy-models.github.io/datarecord/design/model/attributes/#the-broadcast-rule), so it is a real key
    # here - but `pivot` drops NaN index levels, so such a dim is folded away
    # instead: every row shares its value, and the join below matches on the
    # dims that actually distinguish rows.
    varying = [d for d in index[1:] if connections[d].notna().any()]
    registered = _registered(ctype)
    stems = [BUS, *(s for s in sorted(_per_port_stems() - {BUS}) if s in connections)]
    pieces = []
    for stem in stems:
        piece = connections.pivot(
            index=["entity", *varying], columns="port", values=stem
        )
        piece = piece[[p for p in piece if _carries(stem, p, registered)]]
        piece.columns = [_port_attribute(stem, port, registered) for port in piece]
        pieces.append(piece if stem == BUS else piece.dropna(axis=1, how="all"))
    wide = pd.concat(pieces, axis=1).reset_index()  # noqa: F841 - queried by name below
    ports = con.sql("FROM wide").set_alias("ports")
    on = ["entity", *(d for d in varying if d in static.columns)]
    joined = static.set_alias("s").join(
        ports, ex_all(col("s", c) == col("ports", c) for c in on), how="left"
    )
    return joined.project(
        *(col("s", c) for c in static.columns),
        *(col("ports", c) for c in ports.columns if c not in ("entity", *varying)),
    )


def _unpack_connection(long: DuckDBPyRelation) -> DuckDBPyRelation:
    """`long` with its `connection` struct spread to `entity` and `bus`, if it has one.

    The flat pair is what PyPSA's per-type, per-port frames are keyed by, so the
    struct is unpacked once, where an attribute is read for the build.

    Notes
    -----
    - [multi-dims](https://energy-models.github.io/datarecord/design/model/multi-dims/)

    """
    if CONNECTION not in long.columns:
        return long
    return long.project(
        *(col(c) for c in long.columns if c != CONNECTION),
        *(
            fn.struct_extract(col(CONNECTION), lit(field)).alias(field)
            for field in (ENTITY, BUS)
        ),
    )


def _port_columns(
    long: DuckDBPyRelation,
    stem: str,
    connections: pd.DataFrame | None,
    ctype: str,
    con: DuckDBPyConnection,
) -> dict[str, DuckDBPyRelation]:
    """A per-connection attribute's rows split into PyPSA's port columns.

    The inverse of `_NetworkSource._per_port_long_rels`: `efficiency` on a
    Link's bus1 connection is `efficiency`, on its bus2 connection
    `efficiency2`. A row takes the port `_ordered_connections` numbered its
    `(entity, bus)` with, the numbering `_collapse_connections` names
    `bus0`/`bus1`/... by, so a value lands beside the bus it is keyed by. A
    row naming no connection of the entity is dropped, as a value for no
    member is.

    Only the ports that carry `stem` (`_carries`).

    Parameters
    ----------
    long
        The attribute's rows, carrying `entity` and `bus`.
    connections
        `ctype`'s connections as `_ordered_connections` numbers them.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

    """
    if connections is None:
        return {}
    keyed = connections[[ENTITY, BUS, "port"]]  # noqa: F841 - queried by name below
    ports = con.sql("FROM keyed").set_alias("ports")
    placed = (
        long.set_alias("v")
        .join(ports, ex_all(col("v", c) == col("ports", c) for c in (ENTITY, BUS)))
        .project(*(col("v", c) for c in long.columns if c != BUS), col("ports", "port"))
    )
    registered = _registered(ctype)
    columns = {
        _port_attribute(stem, port, registered): port
        for port in connections["port"].unique()
        if _carries(stem, port, registered)
    }
    return {
        column: placed.filter(col("port") == lit(port))
        .project(star(exclude=["port"]))
        .set_alias(column)
        for column, port in columns.items()
    }


def _components(record: RecordLike) -> list[str]:
    """The `ctype` labels `record`'s entity axis has members of, sorted."""
    frame = record.properties.get(ENTITY)
    if frame is None or CTYPE not in frame.columns:
        return []
    labels = frame.project(col(CTYPE)).distinct().fetchall()
    return sorted(str(label) for (label,) in labels if label is not None)


def _members(record: RecordLike, ctype: str) -> DuckDBPyRelation | None:
    """`ctype`'s members in axis order, with the properties it takes (`_taken`), defaults applied.

    The static frame PyPSA's `n.<type>` starts from, `ctype` left out;
    `build` joins the entity-only attribute rows to it. `None` where the
    record has no `ctype` property on its entity axis.

    Notes
    -----
    - [where a value lives](https://energy-models.github.io/datarecord/design/storage/format/#where-a-value-lives)
    - [axis order](https://energy-models.github.io/datarecord/design/model/dims/#axis-order)

    """
    schema = record.schema
    if CTYPE not in schema.categorizing(ENTITY):
        return None
    rel = matching(record.properties[ENTITY], {CTYPE: (ctype,)})
    outlines = _shape_properties(schema, ENTITY)
    held = [name for name in schema.properties[ENTITY] if name not in outlines]
    names = _taken(schema, ctype, held)
    defaulted = with_defaults(schema, ENTITY, rel, names, record.con)
    return _empty_for_none(defaulted.project(star(exclude=[CTYPE])), ENTITY)


# -- selections --------------------------------------------------------------

#: The dims a network is built from a selection squeezing, beside a record's own.
_SQUEEZABLE = frozenset({PERIOD, SCENARIO})
#: The keywords selecting snapshots, at all of which a results layer restates a series.
_TIME = frozenset({SNAPSHOT, TIMESTEP, PERIOD})
#: The dims PyPSA holds a value over, which an attribute its registry declares no spec of may be over.
_HELD = frozenset({ENTITY, CONNECTION, SCENARIO, SNAPSHOT, *_AXIS_TYPES.values()})


def _own_dims(schema: RecordSchema) -> list[str]:
    """The dims `schema` declares that PyPSA's does not and no shape made, a collection's, in declared order.

    A dim outside PyPSA's schema that holds a shape's geometry
    (`_shape_properties`) is one shapes made, not a collection's.
    """
    declared = _registry_schema().dimensions
    return [
        d
        for d in schema.dimensions
        if d not in declared and not _shape_properties(schema, d)
    ]


def _squeezed(record: RecordLike) -> frozenset[str]:
    """The dims a selection squeezes out of its base, none for any other record."""
    dims = record.schema.dimensions
    return frozenset(d for d in _root(record).schema.dimensions if d not in dims)


def _narrow(record: RecordLike) -> tuple[RecordLike, bool]:
    """`record`, a selection building a network, selected once more as PyPSA holds it, any other record as it is, and whether it keeps no global constraint.

    A period holding no kept snapshot goes, and an entity attached to a bus
    the selection drops. Where the selection holds fewer periods than its
    base, an entity active in none of them goes (`_active`), and a global
    constraint whose `investment_period` names none of them (`_applying`).
    A selection cannot keep zero labels of `global_constraint`, so the flag
    says all of the record's constraints are dropped. A selection keeping a
    dim of the record's own is left as it is for `verify` to report.

    Raises
    ------
    ValueError
        If the selection squeezes a dim but `period`, `scenario` or a dim of
        the record's own, holds no snapshot, or as `_active` does.

    """
    if not isinstance(record, Selection):
        return record, False
    base = record.base
    refused = sorted(_squeezed(record) - _SQUEEZABLE - set(_own_dims(base.schema)))
    if refused:
        msg = (
            f"A network is built from a selection squeezing `period`, `scenario` "
            f"or a dim of the record's own, not {refused}: select one label as a "
            "list of one to keep the dim"
        )
        raise ValueError(msg)
    snapshots = _axis_labels(record, SNAPSHOT)
    if not snapshots:
        msg = (
            f"A network is built from a selection holding no snapshot: {record.labels}"
        )
        raise ValueError(msg)
    if _own_dims(record.schema):
        return record, False
    labels: dict[str, Any] = {}
    selected: list[Any] | None = None
    if PERIOD in _squeezed(record):
        selected = [record.labels[PERIOD]]
    elif _multiperiod(record):
        axis = _axis_labels(record, PERIOD)
        held = {s[PERIOD] for s in snapshots}
        periods = [p for p in axis if p in held]
        if len(periods) < len(axis):
            labels[PERIOD] = periods
        if len(periods) < len(_axis_labels(base, PERIOD)):
            selected = periods
    entities = _axis_labels(record, ENTITY)
    kept = set(entities) - _detached(record)
    if selected is not None:
        kept &= _active(record, selected)
    if len(kept) < len(entities):
        labels[ENTITY] = [e for e in entities if e in kept]
    constraintless = False
    if selected is not None and GLOBAL_CONSTRAINT in record.dims:
        axis = _axis_labels(record, GLOBAL_CONSTRAINT)
        applying = _applying(record, selected)
        if len(applying) < len(axis):
            constraintless = not applying
            if applying:
                labels[GLOBAL_CONSTRAINT] = [g for g in axis if g in applying]
    return (record.select(**labels) if labels else record), constraintless


def _detached(record: Selection) -> set[Any]:
    """The entities with a connection to a bus `record` drops, which PyPSA would hold dangling."""
    connections = _connection_frame(record.base)
    if connections is None or BUS not in record.schema.dimensions:
        return set()
    buses = _axis_labels(record, BUS)
    return set(connections.loc[~connections[BUS].isin(buses), ENTITY])


def _untimed(record: RecordLike) -> RecordLike:
    """`record` at every snapshot of its base, a selection without its keywords selecting snapshots (`_TIME`)."""
    if not isinstance(record, Selection):
        return record
    rest = {k: v for k, v in record.labels.items() if k not in _TIME}
    return select(record.base, **rest) if rest else record.base


def _equal_weighting(labels: Iterable[Any]) -> pd.DataFrame:
    """PyPSA's scenario weighting frame of `labels`, each weighing equally."""
    index = pd.Index(labels, name=SCENARIO)
    return pd.DataFrame({"weight": 1 / len(index)}, index=index)


def _scenario_weights(record: RecordLike, scenarios: pd.Series) -> pd.Series:
    """The weights of `scenarios`, `record`'s scenario axis, renormalised where it holds fewer than its base.

    A scenario no row reaches takes its equal share of the base's whole axis
    first, so the weights PyPSA requires sum to 1.

    Raises
    ------
    ValueError
        If the kept scenarios weigh nothing in all.

    """
    whole = _equal_weighting(_axis_labels(_root(record), SCENARIO))
    weights = _weightings(record, SCENARIO, whole.loc[scenarios])["weight"]
    if len(weights) == len(whole):
        return weights
    if not weights.sum() > 0:
        msg = f"The selection keeps scenarios {list(weights.index)} weighing nothing in all"
        raise ValueError(msg)
    return weights / weights.sum()


def _unheld(record: RecordLike, flags: Mapping[str, Flags]) -> dict[str, list[str]]:
    """Each attribute of `flags`, with the dims its registry spec lacks along which its rows differ (`_apart`).

    A network holds one value of the attribute across such a dim, so it
    would keep one row's and drop the others. An attribute the registry
    declares no spec of is held over `_HELD`. A dim of the record's own is
    left to `unsupported_dims`.
    """
    specs = _registry_schema().attributes
    own = set(_own_dims(record.schema))
    unheld: dict[str, list[str]] = {}
    for attr, f in flags.items():
        spec = specs.get(attr)
        beyond = f.varies - (_HELD if spec is None else spec.dims) - own
        if dims := [d for d in sorted(beyond) if _apart(record, attr, d)]:
            unheld[attr] = dims
    return unheld


def _apart(record: RecordLike, attr: str, dim: str) -> bool:
    """Whether `record` holds more than one label of `dim` and a row of `attr` names one.

    Counted from the rows, not `Flags.varies` which a selection overstates, and
    from the axis, not the labels the rows name, since a row naming one label
    leaves the others at the default.
    """
    if len(record.dims[dim]) < 2:
        return False
    return bool(
        record.attributes[attr].filter(col(dim).isnotnull()).limit(1).fetchall()
    )


def _one_labelled(record: RecordLike) -> dict[str, DuckDBPyRelation]:
    """Each dim but `snapshot` that `record` holds exactly one label of, with its axis, for `settle`.

    Over one label, the row naming it is kept over one leaving it NULL. Over
    several, a row naming one is refused before (`_unheld`) unless PyPSA holds
    the dim, so settling would only copy a NULL row to every label. A NULL
    `snapshot` marks a static row (`_assign_static`), which settling would
    give the one snapshot's label.
    """
    return {
        dim: rel
        for dim, rel in record.dims.items()
        if dim != SNAPSHOT and len(rel) == 1
    }
