# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Record-to-network construction for the PyPSA tool."""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import pandas as pd
from datarecord.duck import ex_all, fn
from datarecord.protocols import Flags, RecordLike
from datarecord.schema.categories import matching, with_defaults
from duckdb import CoalesceOperator as coalesce
from duckdb import ColumnExpression as col
from duckdb import ConstantExpression as lit
from duckdb import (
    DuckDBPyConnection,
    DuckDBPyRelation,
)
from duckdb import StarExpression as star

if TYPE_CHECKING:
    import pypsa
from pypsa.network.io._common import (
    _apply_network_attributes,
    _apply_snapshots_import,
)
from pypsa.network.io.datarecord.schema import (
    _INPUT_PORT,
    _SINGLE_PORT,
    BUS,
    CONNECTION,
    CTYPE,
    ENTITY,
    PERIOD,
    SCENARIO,
    SNAPSHOT,
    _carries,
    _geometries,
    _per_port_stems,
    _port_attribute,
    _registered,
    _taken,
)
from pypsa.network.io.datarecord.shape import (
    Index,
    NetworkShape,
    _connection,
)
from pypsa.network.io.datarecord.source import (
    _empty_for_none,
    _from_wkt,
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
        static = scenarios.expand(
            static.project(f"*, NULL AS {SCENARIO}").set_alias("m"), SCENARIO
        )

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
            rows = rows.project(f"*, CAST(NULL AS VARCHAR) AS {SCENARIO}")
            rows = scenarios.expand(rows, SCENARIO)
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
    tag = uuid4().hex
    tables = [f"_pypsa_static_{tag}"]
    static.create(tables[0])
    try:
        joined = con.table(tables[0]).set_alias("s")
        exprs = {}
        for i, (name, long) in enumerate(values.items()):
            rows = (
                long.filter(col(SNAPSHOT).isnull()) if name in over_snapshots else long
            )
            tables.append(f"_pypsa_value_{tag}_{i}")
            rows.project(*(col(k) for k in keys), col("value")).create(tables[-1])
            alias = f"a{i}"
            picked = con.table(tables[-1]).set_alias(alias)
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
            .order("_pos")
            .project(star(exclude=["_pos"]))
            .df()
        )
    finally:
        for table in tables:
            con.execute(f"DROP TABLE IF EXISTS {table}")


def _series_frame(
    rows: DuckDBPyRelation,
    shape: NetworkShape,
    scenarios: Index,
    snapshots: Index,
    static_index: pd.Index,
    con: DuckDBPyConnection,
) -> pd.DataFrame:
    """Pivot long series rows to the `n.snapshots x static_index` frame.

    `snapshots` (the row axis) is known before the pivot, so missing rows are
    filled by joining against it directly rather than reindexing pandas-side
    after. Columns stay restricted to what the data actually pivoted, only
    ordered to match `static_index`: `_import_series_from_df` takes
    `df.columns` as-is without merging against `static`, so a column added
    just to complete `static_index` would wrongly force that component into
    `dynamic` instead of leaving it static.

    The pivot is on each column's position in `static_index` rather than its
    name, so a name never becomes an SQL identifier: one equal to a group-by
    column would collide with it, and DuckDB folds identifiers' case.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

    """
    group_by = [SNAPSHOT]
    keys = shape.index_names
    columns = Index.of(static_index.set_names(keys), con).rel.set_alias("k")
    rows = rows.set_alias("r")
    positioned = rows.join(  # noqa: F841 - bound by name for the query below
        columns, ex_all(col("r", k) == col("k", k) for k in keys)
    ).project(
        *(col("r", g) for g in group_by),
        col("k", "_pos").alias("_col"),
        col("r", "value"),
    )
    piv = con.sql(
        f"PIVOT positioned ON _col USING first(value) GROUP BY {', '.join(group_by)}"
    )

    present = sorted(int(c) for c in set(piv.columns) - set(group_by))
    if not present:
        return pd.DataFrame(index=snapshots.index, columns=static_index[:0])

    wide = (
        snapshots.left_join(piv.set_alias("piv"), group_by)
        .project(*(col(str(p)) for p in present))
        .df()
    )
    wide.index = snapshots.index
    wide.columns = static_index[[p - 1 for p in present]]
    return wide


def _new_network(
    record: RecordLike, shape: NetworkShape, con: DuckDBPyConnection
) -> pypsa.Network:
    """An empty network with the record's axes already established."""
    import pypsa

    schema = record.schema
    n = pypsa.Network()
    _apply_network_attributes(n, schema.meta.get("attributes") or {})
    n.meta = schema.meta.get("meta") or {}
    crs = schema.meta.get("crs")
    if crs is not None:
        n.to_crs(crs)

    _apply_snapshots_import(n, shape.snapshot_levels)
    if shape.multiperiod:
        n.periods = shape.periods[PERIOD]
        n.investment_period_weightings = _weightings(
            record, PERIOD, n.investment_period_weightings, con
        )
    labels = pd.Index(shape.snapshots[SNAPSHOT], name=SNAPSHOT)
    n.snapshot_weightings = _weightings(
        record, SNAPSHOT, n.snapshot_weightings.set_axis(labels), con
    ).set_axis(n.snapshots)
    if shape.stochastic:
        index = pd.Index(shape.scenarios[SCENARIO].astype(str), name=SCENARIO)
        equal = pd.DataFrame({"weight": 1 / len(index)}, index=index)
        # `set_scenarios` rather than assigning `_scenarios_data`: it rebuilds
        # each component's empty `static`/`dynamic` frames with the `scenario`
        # level, so later imports union against a correctly named 2-level index
        # instead of the single-level one a fresh component carries.
        n.set_scenarios(_weightings(record, SCENARIO, equal, con)["weight"])
    return n


def _add_component_type(
    n: pypsa.Network,
    ctype: str,
    static: DuckDBPyRelation,
    attributes: dict[str, tuple[DuckDBPyRelation, Flags]],
    shape: NetworkShape,
    con: DuckDBPyConnection,
) -> None:
    """Assign one type's static frame, then its series."""
    scenarios = Index.of(n.scenarios, con)
    static_df = _assign_static(static, attributes, shape, scenarios, con)
    for column in _geometries(ctype).intersection(static_df.columns):
        static_df[column] = _from_wkt(static_df[column], n.crs)
    n._import_components_from_df(static_df, ctype)

    snapshots = Index.of(n.snapshots, con, shape.snapshots[[SNAPSHOT]])
    static_index = n.c[ctype].static.index
    for attr, (long, flags) in attributes.items():
        # ... and `dynamic` the per-snapshot ones.
        if SNAPSHOT not in flags.varies:
            continue
        ts_rows = long.filter("snapshot IS NOT NULL AND value IS NOT NULL")
        # Partial columns keep non-series components static (https://energy-models.github.io/datarecord/design/tools/).
        wide = _series_frame(ts_rows, shape, scenarios, snapshots, static_index, con)
        n._import_series_from_df(wide, ctype, attr)


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
    names = _taken(schema, ctype, schema.properties[ENTITY])
    defaulted = with_defaults(schema, ENTITY, rel, names, _connection(record))
    return _empty_for_none(defaulted.project(star(exclude=[CTYPE])))
