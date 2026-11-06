# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Hand-built datarecord layers and network comparison for the datarecord tests."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from datarecord.schema.schema import Schema
from datarecord.schema.specs import AttributeSpec, Dimension, PropertySpec
from datarecord.store.store import Revision, Store

TYPES = (
    "Bus",
    "Carrier",
    "Generator",
    "Line",
    "Link",
    "Load",
    "Process",
    "StorageUnit",
    "Store",
    "Transformer",
    "Widget",
)


# No `ctype`, as an attribute row is keyed by `name`, unique across every type.
LONG_COLUMNS = [
    "entity",
    "bus",
    "snapshot",
    "scenario",
    "period",
    "attribute",
    "breakpoint",
    "value",
]


def write_attribute(
    layer: str,
    attribute: str,
    rows: list[dict],
    *,
    snapshot_dtype="datetime64[ns]",
    connection: bool | None = None,
) -> None:
    """Write `attributes/<attribute>.parquet` in the long schema.

    Each row needs at least `name` and `value`; missing dimension columns
    default to NULL, i.e. "applies to the whole axis".
    A `bus` on any row marks a per-connection attribute, its `entity` and `bus`
    packed into the one `connection` struct column - unless `connection` says
    otherwise, for an attribute over the `bus` dim itself; `breakpoint` marks a
    piecewise-linear one; neither is the ordinary component-level scalar.

    Write every file of a layer before its first read: a read freezes the
    layer for its workspace, as a layer is write-once.

    Notes
    -----
    - [wide and long rows](https://energy-models.github.io/datarecord/design/model/attributes/#wide-and-long-rows)
    - [attributes over a multi-dim](https://energy-models.github.io/datarecord/design/model/multi-dims/#attributes-over-a-multi-dim)
    """
    df = pd.DataFrame(rows)
    if connection is None:
        connection = "bus" in df
    df["attribute"] = attribute
    for col in LONG_COLUMNS:
        if col not in df:
            df[col] = None
    df["snapshot"] = pd.Series(df["snapshot"]).astype(snapshot_dtype)
    df["scenario"] = df["scenario"].astype("string")
    df["entity"] = df["entity"].astype("string")
    df["bus"] = df["bus"].astype("string")
    df["period"] = df["period"].astype("Int64")
    df["breakpoint"] = df["breakpoint"].astype("float64")
    df["value"] = df["value"].astype("float64")

    target = Path(layer, "attributes")
    target.mkdir(parents=True, exist_ok=True)
    rel = duckdb.connect().from_df(df[LONG_COLUMNS])
    if connection:
        rel = rel.project(
            "{'entity': entity, 'bus': bus} AS connection, * EXCLUDE (entity, bus)"
        )
    rel.to_parquet(str(target / f"{attribute}.parquet"))


def _hashable(value: Any) -> Any:
    """A key value as a dict key: a struct's fields as a tuple."""
    return tuple(value.values()) if isinstance(value, dict) else value


def _as_written(revision: Revision, kind: str, name: str) -> pd.DataFrame:
    """The layer's file of `name` so far, else its parent's resolved one, else nothing.

    A layer restates an axis whole, so a first write in a child starts
    from what its parent resolves, properties and order included.
    """
    path = Path(revision.store.layer_dir(revision.id), kind, f"{name}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    if revision.parent is None:
        return pd.DataFrame()
    record = revision.store.revision(revision.parent).record
    if name in record.schema.properties:
        rel = record.properties.get(name)
    else:
        rel = record.dims.get(name)
    if rel is None:
        return pd.DataFrame()
    frame = rel.df()
    enums = [c for c, t in frame.dtypes.items() if isinstance(t, pd.CategoricalDtype)]
    return frame.astype(dict.fromkeys(enums, object))


def restate(
    revision: Revision,
    kind: str,
    name: str,
    rows: list[dict],
    *,
    drop: Sequence[Any] = (),
) -> None:
    """Write `{kind}/{name}.parquet` of `revision`'s layer whole, as a layer restates it.

    It starts from `_as_written`. A row of `rows` replaces the row of its key
    in place, a new key is appended, and a key in `drop` is left out.

    Notes
    -----
    - [deletion](https://energy-models.github.io/datarecord/design/layers/removal/)
    """
    old = _as_written(revision, kind, name)
    records = old.to_dict("records")
    at = {_hashable(r[name]): i for i, r in enumerate(records)}
    for row in rows:
        key = _hashable(row[name])
        if key in at:
            records[at[key]] = row
        else:
            at[key] = len(records)
            records.append(row)
    gone = {_hashable(k) for k in drop}
    kept = [r for r in records if _hashable(r[name]) not in gone]
    columns = list(dict.fromkeys([*old.columns, *(c for r in rows for c in r)]))
    frame = pd.DataFrame(kept, columns=columns)
    target = Path(revision.store.layer_dir(revision.id), kind)
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / f"{name}.parquet", index=False)


def unpacked(rel: Any, multi: str = "connection") -> pd.DataFrame:
    """A relation keyed by a multi-dim as pandas, its struct column spread flat.

    A numpy-backed pandas frame has no struct dtype, so the unnest happens on
    the DuckDB side.
    """
    return rel.project(f"unnest({multi}), * EXCLUDE ({multi})").df()


def write_component(revision: Revision, ctype: str, rows: list[dict]) -> None:
    """Restate `dims/entity.parquet` with `rows`, each an entity of type `ctype`."""
    restate(revision, "dims", "entity", [{**r, "ctype": ctype} for r in rows])


def empty_child(parent: Revision) -> Revision:
    """A new child of `parent` with no layer.

    Write every file of a layer before its first read: a read freezes the
    layer for its workspace, as a layer is write-once.
    """
    return parent.store._create(parent.id)


def export_network(n, store: Store) -> Revision:
    """Write `n` as a new root of `store`, through the record layer's own writer.

    Not `n.export_to_parquet`: that emits PyPSA's upstream manifest format,
    which is a different vocabulary from the schema a record declares.
    Going through `store.root` means a test record is written exactly as
    `blocks` writes one.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/model/schema/#one-schema-per-record)
    """
    from pypsa.network.io.datarecord import PyPSA

    return store.root(PyPSA.to_datarecord(n, store.workspace.con))


def write_schema(schema: Schema, store: Store) -> None:
    """Declare the record's one schema, beside the layers.

    Not per layer: a layer holds only data, so this writes the record-level
    `manifest.json` that every layer in the tree is read under.

    Notes
    -----
    - [one schema per record](https://energy-models.github.io/datarecord/design/model/schema/#one-schema-per-record)
    """
    tree = schema.model_copy(update={"id": None})
    store.files.write_text(store.manifest_uri, tree.model_dump_json(), expect=None)


def _default_attributes(dims: dict[str, Any], multi_dims: dict[str, dict[str, str]]):
    """The attributes tests write, declared over whichever dims are in play.

    Writing an attribute the schema does not declare is rejected, since its
    `dims` are what say which columns its file carries - so every attribute a
    test writes has to be declared, and these are the ones they write.

    Addressed over every declared dim rather than a narrower set, which is the
    widest shape and so the one that accepts any row a test writes.
    `efficiency` is the exception, being over the `connection` multi-dim where one
    is declared: that is what puts a `connection` struct column on its file.

    `weight` is the other, addressed by `scenario` alone - long rows of
    `attributes/weight.parquet` like any attribute, whatever its `dims`.
    """
    varying = frozenset({"entity", *dims})
    connection = "connection" if "connection" in multi_dims else "entity"
    # A dim named like a coordinate of the connection would clash with its keyword.
    per_connection = set(dims) - set(multi_dims.get(connection, ()))
    declared = {
        "p_nom": AttributeSpec(dtype="float", dims=varying),
        "e_nom": AttributeSpec(dtype="float", dims=varying),
        "p_max_pu": AttributeSpec(dtype="float", dims=varying),
        "p_min_pu": AttributeSpec(dtype="float", dims=varying),
        "marginal_cost": AttributeSpec(dtype="float", dims=varying, breakpoints=True),
        "efficiency": AttributeSpec(
            dtype="float", dims=frozenset({connection, *per_connection})
        ),
    }
    if "scenario" in dims:
        declared["weight"] = AttributeSpec(
            dtype="float",
            dims=frozenset({"scenario"}),
            description="How much this scenario counts in the expectation.",
        )
    return declared


def schema(
    *,
    partial: set[str] = {"scenario", "connection"},
    attributes: dict[str, dict[str, AttributeSpec]] | None = None,
    dims: dict[str, Any] = {
        "snapshot": "datetime",
        "period": "int",
        "scenario": "str",
    },
    multi_dims: dict[str, dict[str, str]] = {
        "connection": {"entity": "entity", "bus": "bus"}
    },
) -> Schema:
    """A schema shaped like the PyPSA records most tests build on.

    Defaults match `PyPSA.to_datarecord`: the `entity` axis carrying the
    `ctype` property, which refers to the dim `ctype` listing `TYPES`, a
    `connection` multi-dim over
    `(entity, bus)`, and three declared dims. Override `partial` to pin a
    different layering granularity, `dims` to declare another axis by its
    dtype or as a `Dimension`, `multi_dims` to
    declare a different sparse relation.

    A multi-dim's coordinates are declared dims. `entity` is always `partial`, so
    a layer patches one component's value without restating the rest, and
    `connection` is by default, as the PyPSA tool declares it, so the same
    holds per connection.

    Notes
    -----
    - [the schema](https://energy-models.github.io/datarecord/design/model/schema/)
    - [multi-dims](https://energy-models.github.io/datarecord/design/model/multi-dims/)
    """
    # Callers declare per type, the schema stores one spec per attribute.
    flat: dict[str, AttributeSpec] = {}
    for attrs in (attributes or {}).values():
        for attr, spec in attrs.items():
            flat.setdefault(attr, spec)
    # Defaults are declared even when a caller did not name them.
    for attr, spec in _default_attributes(dims, multi_dims).items():
        flat.setdefault(attr, spec)
    # Declared rather than assumed, so a multi-dim may use coordinates not called `bus`.
    coordinates = {c for over in multi_dims.values() for c in over}
    declared = {
        "entity": "str",
        **dict.fromkeys(coordinates, "str"),
        **dims,
    }
    return Schema(
        dimensions={
            **{
                d: t if isinstance(t, Dimension) else Dimension(dtype=t)
                for d, t in declared.items()
            },
            "ctype": Dimension(one_of=TYPES),
            **{m: Dimension(over=over) for m, over in multi_dims.items()},
        },
        properties={"entity": {"ctype": PropertySpec(one_of="ctype")}},
        attributes=flat,
        # The default names `connection`, which a caller's own `multi_dims` may drop.
        partial=frozenset(
            p for p in {"entity", *partial} if p in declared or p in multi_dims
        ),
    )


def relation(revision, attribute: str):
    """The resolved long relation for one attribute, as a DuckDB relation.

    A test helper rather than a `Revision` method: `Revision` presents its data
    through `.record` (a `Record`), and a DuckDB-shaped accessor beside it would
    duplicate `record.attributes[attr]`. Tests want relations because they
    assert on `.df()`, so the affordance lives here.
    """
    return revision.record.attributes[attribute]


def assert_networks_equal(got, expected):
    """Compare component frames, which is what the record round-trips."""
    for exp_c in expected.components:
        ctype = exp_c.name
        if exp_c.static.empty:
            continue
        got_c = got.c[ctype]
        pd.testing.assert_index_equal(
            got_c.static.index, exp_c.static.index, check_names=False
        )
        # No junk columns (join artifacts, `deleted`, ...) may leak through.
        assert not set(got_c.static.columns) - set(exp_c.static.columns)
        for col in exp_c.static.columns:
            if col == "obj":
                continue
            pd.testing.assert_series_equal(
                got_c.static[col],
                exp_c.static[col],
                check_dtype=False,
                check_names=False,
            )
        assert set(got_c.dynamic) >= {
            a for a, f in exp_c.dynamic.items() if not f.empty
        }
        for attr, frame in exp_c.dynamic.items():
            if frame.empty:
                continue
            pd.testing.assert_frame_equal(
                got_c.dynamic[attr][frame.columns],
                frame,
                check_dtype=False,
                check_freq=False,
                check_column_type=False,
                check_index_type=False,
            )
