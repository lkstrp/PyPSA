# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Hand-built datarecord layers and network comparison for the datarecord tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd
from datarecord.schema.schema import Schema
from datarecord.schema.specs import AttributeSpec, Dimension, PropertySpec
from datarecord.store.manifest import LayerFiles, Manifest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from datarecord.store.store import Claim, Revision, Store

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
    "port",
    "snapshot",
    "scenario",
    "period",
    "attribute",
    "breakpoint",
    "value",
]


def write_attribute(
    layer: HandBuilt,
    attribute: str,
    rows: list[dict],
    *,
    snapshot_dtype="datetime64[ns]",
    connection: bool | None = None,
) -> None:
    """Write `attributes/<attribute>.parquet` in the long schema.

    Each row needs at least `name` and `value`; missing dimension columns
    default to NULL, i.e. "applies to the whole axis".
    A `port` on any row marks a per-connection attribute, its `entity` and `port`
    packed into the one `connection` struct column - unless `connection` says
    otherwise, for an attribute over the `bus` dim itself; `breakpoint` marks a
    piecewise-linear one; neither is the ordinary component-level scalar.

    Notes
    -----
    - [wide and long rows](https://energy-models.github.io/datarecord/design/model/attributes/#wide-and-long-rows)
    - [attributes over a multi-dim](https://energy-models.github.io/datarecord/design/model/multi-dims/#attributes-over-a-multi-dim)
    """
    df = pd.DataFrame(rows)
    if connection is None:
        connection = "port" in df
    df["attribute"] = attribute
    for col in LONG_COLUMNS:
        if col not in df:
            df[col] = None
    df["snapshot"] = pd.Series(df["snapshot"]).astype(snapshot_dtype)
    df["scenario"] = df["scenario"].astype("string")
    df["entity"] = df["entity"].astype("string")
    df["port"] = df["port"].astype("Int64")
    df["period"] = df["period"].astype("Int64")
    df["breakpoint"] = df["breakpoint"].astype("float64")
    df["value"] = df["value"].astype("float64")

    target = Path(layer.dir, "attributes")
    target.mkdir(parents=True, exist_ok=True)
    rel = duckdb.connect().from_df(df[LONG_COLUMNS])
    if connection:
        rel = rel.project(
            "{'entity': entity, 'port': port} AS connection, * EXCLUDE (entity, port)"
        )
    rel.to_parquet(str(target / f"{attribute}.parquet"))


def _hashable(value: Any) -> Any:
    """A key value as a dict key: a struct's fields as a tuple."""
    return tuple(value.values()) if isinstance(value, dict) else value


def _as_written(layer: HandBuilt, kind: str, name: str) -> pd.DataFrame:
    """The layer's file of `name` so far, else its parent's resolved one, else nothing.

    A layer restates an axis whole, so a first write in a child starts
    from what its parent resolves, properties and order included.
    """
    path = Path(layer.dir, kind, f"{name}.parquet")
    if path.exists():
        return pd.read_parquet(path)
    if layer.parent is None:
        return pd.DataFrame()
    record = layer.parent.record
    if name in record.schema.properties:
        rel = record.properties.get(name)
    else:
        rel = record.dimensions.get(name)
    if rel is None:
        return pd.DataFrame()
    frame = rel.df()
    enums = [c for c, t in frame.dtypes.items() if isinstance(t, pd.CategoricalDtype)]
    return frame.astype(dict.fromkeys(enums, object))


def restate(
    layer: HandBuilt,
    kind: str,
    name: str,
    rows: list[dict],
    *,
    drop: Sequence[Any] = (),
) -> None:
    """Write `{kind}/{name}.parquet` of `layer` whole, as a layer restates it.

    It starts from `_as_written`. A row of `rows` replaces the row of its key
    in place, a new key is appended, and a key in `drop` is left out.

    Notes
    -----
    - [deletion](https://energy-models.github.io/datarecord/design/layers/removal/)
    """
    old = _as_written(layer, kind, name)
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
    target = Path(layer.dir, kind)
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / f"{name}.parquet", index=False)


def unpacked(rel: Any, multi: str = "connection") -> pd.DataFrame:
    """A relation keyed by a multi-dim as pandas, its struct column spread flat.

    A numpy-backed pandas frame has no struct dtype, so the unnest happens on
    the DuckDB side.
    """
    return rel.project(f"unnest({multi}), * EXCLUDE ({multi})").df()


def write_component(layer: HandBuilt, ctype: str, rows: list[dict]) -> None:
    """Restate `dimensions/entity.parquet` with `rows`, each an entity of type `ctype`."""
    restate(layer, "dimensions", "entity", [{**r, "ctype": ctype} for r in rows])


@dataclass(frozen=True)
class HandBuilt:
    """A claimed layer a test writes the files of directly, a revision once committed.

    Attributes
    ----------
    schema
        What `commit` writes into the manifest.
    """

    claim: Claim
    schema: Schema

    @property
    def id(self) -> UUID:
        return self.claim.id

    @property
    def store(self) -> Store:
        return self.claim.store

    @property
    def dir(self) -> Path:
        return Path(self.store.layer_dir(self.id))

    @property
    def parent(self) -> Revision | None:
        """The revision the layer folds over, which a first restatement starts from."""
        return (
            None
            if self.claim.parent is None
            else self.store.revision(self.claim.parent)
        )

    def commit(self) -> Revision:
        """Write the manifest listing the layer's files, which makes it a revision."""
        files = LayerFiles(
            dimensions=tuple(
                sorted(p.stem for p in self.dir.glob("dimensions/*.parquet"))
            ),
            attributes=tuple(
                sorted(p.stem for p in self.dir.glob("attributes/*.parquet"))
            ),
        )
        manifest = Manifest(
            id=self.id,
            parent=self.claim.parent,
            schema=self.schema,
            files=files,
        )
        manifest.write(self.store.files, self.store.manifest_uri(self.id))
        return self.store.revision(self.id)


def empty_root(store: Store, schema: Schema) -> HandBuilt:
    """A new root layer of `store` under `schema`, for a test to hand-write.

    No public API writes files into a layer, so this claims one beside the
    write path.
    """
    return HandBuilt(store.claim(parent=None), schema)


def empty_child(parent: Revision, schema: Schema | None = None) -> HandBuilt:
    """A new child layer of `parent`, under `schema`, else the parent's.

    A hand-built layer is checked against no parent schema, so `schema` may be
    one `Claim.commit` refuses.
    """
    declared = parent.manifest.schema if schema is None else schema
    return HandBuilt(parent.store.claim(parent=parent), declared)


def export_network(n, store: Store) -> Revision:
    """Write `n` as a new root of `store`, through the record layer's own writer.

    Not `n.export_to_parquet`: that emits PyPSA's upstream manifest format,
    which is a different vocabulary from the schema a record declares.
    Going through `store.commit` means a test record is written exactly as
    `blocks` writes one.
    """
    from pypsa.network.io.datarecord import PyPSA

    return store.commit(PyPSA.to_datarecord(n, store.workspace.con), parent=None)


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
    partial: set[str] = {"scenario", "connection", "port"},
    attributes: dict[str, dict[str, AttributeSpec]] | None = None,
    dims: dict[str, Any] = {
        "snapshot": "datetime",
        "period": "int",
        "scenario": "str",
    },
    multi_dims: dict[str, dict[str, str]] = {
        "connection": {"entity": "entity", "port": "port"}
    },
) -> Schema:
    """A schema shaped like the PyPSA records most tests build on.

    Defaults match `PyPSA.to_datarecord`: the `entity` axis carrying the
    `ctype` property, which refers to the dim `ctype` listing `TYPES`, a
    `connection` multi-dim over
    `(entity, port)`, and three declared dims. Override `partial` to pin a
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
    # Declared rather than assumed, so a multi-dim may use coordinates not called `port`.
    coordinates = {c for over in multi_dims.values() for c in over}
    declared = {
        "entity": "str",
        **dict.fromkeys(coordinates, "str"),
        **({"port": "int"} if "port" in coordinates else {}),
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


def as_written_at_8889d623(path) -> None:
    """Rewrite the fresh export at `path` to the layout of datarecord pin 8889d623.

    `dimensions/` becomes `dims/`, and so does `files.dimensions` in the manifest.
    """
    import json  # noqa: PLC0415

    root = Path(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {
        ("dims" if key == "dimensions" else key): value
        for key, value in manifest["files"].items()
    }
    manifest_path.write_text(json.dumps(manifest))
    (root / "dimensions").rename(root / "dims")
