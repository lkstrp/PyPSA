# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The canonical datarecord schema built from PyPSA's component registry.

Built from `pypsa.components.types.all_components`, never from a network's
contents, so the schema is the same for every network of a given shape
(multi-period or not, integer or datetime snapshots).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import narwhals as nw
from datarecord.schema import (
    AttributeSpec,
    Dimension,
    Group,
    Schema,
    TypeAttribute,
    TypeSpec,
)

from pypsa.components.types import all_components
from pypsa.constants import piecewise_attrs

if TYPE_CHECKING:
    import pandas as pd

    from pypsa.components.components import Components

TIMESTEP, PERIOD, SCENARIO, ENTITY_TYPE, PORT, CARRIER, SHAPE = (
    "timestep",
    "period",
    "scenario",
    "entity_type",
    "port",
    "carrier",
    "shape",
)
# The two narwhals dtype names `build_schema` accepts for `timestep_dtype`.
TIMESTEP_DTYPES = ("Int64", "Datetime")

SNAPSHOT_WEIGHTINGS = ("objective", "generators", "stores")
PERIOD_WEIGHTINGS = {"objective": "period_objective", "years": "years"}
SCENARIO_WEIGHTINGS = {"weight": "scenario_weight"}

# The entity axis and its bus attachment aren't part of the public interface,
# addressed record-wide only through the `connection` group.
_ENTITY, _BUS, _CONNECTION = "entity", "bus", "connection"

# Component types the schema does not export: templates/library rows, derived,
# non-schema types, and Carrier/Shape, which the record addresses as dims
# (`CARRIER`/`SHAPE`) rather than as entity types.
_EXCLUDED_TYPES = {
    "LineType",
    "TransformerType",
    "SubNetwork",
    "Network",
    "Carrier",
    "Shape",
}

# Carrier/Shape attributes declared on the `carrier`/`shape` dims rather than
# granted to an entity type. Shape's own `type` shares its PyPSA name with
# Bus/Generator/Line's entity-addressed `type`, so it is registered under
# `_RECORD_NAME_OVERRIDES` instead of the bare name.
_DIM_ATTRS: dict[str, tuple[str, ...]] = {
    CARRIER: (
        "co2_emissions",
        "color",
        "nice_name",
        "max_growth",
        "max_relative_growth",
    ),
    SHAPE: ("geometry", "component", "idx", "type"),
}
# Dim name -> the entity type it rebuilds on import.
DIM_TYPES: dict[str, str] = {CARRIER: "Carrier", SHAPE: "Shape"}
# Entity type -> its dim, the inverse of `DIM_TYPES`.
_TYPE_DIMS: dict[str, str] = {v: k for k, v in DIM_TYPES.items()}

# `schema.meta` key for `declare_custom`'s merge bookkeeping, popped before
# the schema is used (`NetworkRecord.schema`), never written to disk.
_CUSTOM_ATTRS_META_KEY = "_datarecord_custom_attrs"

# Derived topology outputs dropped from the schema and never written: Bus,
# Line and Transformer's `sub_network`, and Bus's `generator`. Both are set by
# `determine_network_topology`, which the datarecord import never calls.
_TOPOLOGY_OUTPUTS = {
    ("Bus", "sub_network"),
    ("Bus", "generator"),
    ("Line", "sub_network"),
    ("Transformer", "sub_network"),
}

# PyPSA's `defaults["typ"]` mapped to the narwhals type the record stores.
# `String` for anything unlisted, which covers `geometry` (WKT text) too.
_DTYPES: dict[Any, nw.dtypes.DType] = {
    bool: nw.Boolean(),
    int: nw.Int64(),
    float: nw.Float64(),
    str: nw.String(),
}

# Which entity types have a per-port coefficient attribute, and its name.
_COEFFICIENT_ATTR = {"Link": "efficiency", "Process": "rate"}

_BUS_RE = re.compile(r"^bus(\d*)$")

# One name has one record-wide address, so Bus's own p/q and Link/Process's
# aggregate p (component-addressed) are renamed away from the name every
# per-port flow uses (connection-addressed).
_RECORD_NAME_OVERRIDES = {
    ("Bus", "p"): "p_balance",
    ("Bus", "q"): "q_balance",
    ("Link", "p"): "p_activity",
    ("Process", "p"): "p_activity",
    ("Shape", "type"): "shape_type",
}
_PYPSA_NAME_OVERRIDES = {
    (ctype, record_attr): attr
    for (ctype, attr), record_attr in _RECORD_NAME_OVERRIDES.items()
}


def record_name(ctype: str, attr: str) -> str:
    """PyPSA attribute -> record-wide attribute name."""
    return _RECORD_NAME_OVERRIDES.get((ctype, attr), attr)


def custom_dim_attr_name(dim: str, attr: str) -> str:
    """Record-wide name for a custom Carrier/Shape attribute, namespaced by `dim`.

    A custom name could otherwise collide with an unrelated per-component
    registry attribute, the way `marginal_cost` does with Generator's own.
    The same pattern already stores `Shape.type` as `shape_type`.
    """
    return f"{dim}_{attr}"


def pypsa_name(ctype: str, record_attr: str) -> str:
    """Record-wide attribute name -> PyPSA attribute, inverse of `record_name`."""
    return _PYPSA_NAME_OVERRIDES.get((ctype, record_attr), record_attr)


def _ports(defaults: pd.DataFrame) -> list[str]:
    """Port labels from a type's `bus`/`bus0`/`bus1`/... columns."""
    return [m.group(1) for col in defaults.index if (m := _BUS_RE.match(col))]


def _port_stems(ctype: str, defaults: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """PyPSA column -> (record stem, port label) for one type's port columns.

    Shared by `port_columns` (a live `Components`) and `build_schema` (the
    registry's `ComponentType.defaults`), since both only need column names.

    A single-port type (one bus, labelled `""`) also maps its own
    `efficiency` column to that port, the same connection-addressed quantity
    as Link's per-port `efficiency`, not a per-component one.
    """
    ports = _ports(defaults)
    if not ports:
        return {}
    result: dict[str, tuple[str, str]] = {}
    coefficient_attr = _COEFFICIENT_ATTR.get(ctype)
    single_port = ports == [""]
    if single_port and coefficient_attr is None and "efficiency" in defaults.index:
        result["efficiency"] = ("efficiency", ports[0])
    for port in ports:
        for stem in ("bus", "p", "q"):
            col = f"{stem}{port}"
            if col in defaults.index:
                result[col] = (stem, port)
        if coefficient_attr is None:
            continue
        # Only Link leaves port "1" unsuffixed (`efficiency`, not `efficiency1`).
        # Every other port, and Process's own port "1", suffixes with the label.
        suffix = "" if ctype == "Link" and port == "1" else port
        col = f"{coefficient_attr}{suffix}"
        if col in defaults.index:
            result[col] = (coefficient_attr, port)
        for stem in ("delay", "cyclic_delay"):
            col = f"{stem}{suffix}"
            if col in defaults.index:
                result[col] = (stem, port)
    return result


def port_columns(c: Components) -> dict[str, tuple[str, str]]:
    """PyPSA column -> (record stem, port label) for one component type's ports.

    E.g. `{"bus0": ("bus", "0"), "bus1": ("bus", "1"), "efficiency": ("efficiency", "1"),
    "p0": ("p", "0"), "p1": ("p", "1")}` for a Link, whose entity-addressed `p`
    is not a port column, or `{"bus": ("bus", ""), "p": ("p", ""), "q": ("q", "")}`
    for a single-port type such as Generator.
    """
    return _port_stems(c.name, c.defaults)


def _default(value: Any) -> Any:
    """One `defaults["default"]` cell as JSON-storable, NaN as absent."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _text(value: Any) -> str | None:
    """One `unit`/`description` cell as text, or None where PyPSA has none."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _register(
    attributes: dict[str, AttributeSpec], name: str, spec: AttributeSpec
) -> None:
    """Add `name` -> `spec`, or raise if a prior type already declared it differently.

    Record-wide attributes are declared once and shared by every type that
    carries them (`AttributeSpec` is flat, per the datarecord schema), so two
    types disagreeing on one name's shape is a schema bug, not a shadowing to
    resolve silently.
    """
    existing = attributes.get(name)
    if existing is not None and existing != spec:
        msg = f"{name!r} already declared as {existing!r}, conflicting with {spec!r}"
        raise ValueError(msg)
    attributes[name] = spec


def _timestep_dtype(name: str) -> nw.dtypes.DType:
    if name == "Int64":
        return nw.Int64()
    if name == "Datetime":
        return nw.Datetime()
    msg = f"unknown timestep dtype {name!r}; expected one of {TIMESTEP_DTYPES}"
    raise ValueError(msg)


def build_schema(*, multiperiod: bool, timestep_dtype: str, stochastic: bool) -> Schema:
    """Build the canonical datarecord schema for a PyPSA network of this shape.

    Parameters
    ----------
    multiperiod
        Whether `timestep` nests within `period`.
    timestep_dtype
        One of `TIMESTEP_DTYPES`: the narwhals dtype name for the snapshot
        axis, integer or datetime.
    stochastic
        Whether Carrier/Shape attributes vary by scenario. `False` puts them
        on the `carrier`/`shape` axis files as columns; `True` puts them in
        long input rows addressed by `(scenario, carrier)`/`(scenario, shape)`.

    """
    dimensions = {
        PERIOD: Dimension(
            dtype=nw.Int64(),
            unit="years",
            description="An investment period, labelled by its year.",
        ),
        TIMESTEP: Dimension(
            dtype=_timestep_dtype(timestep_dtype),
            within=frozenset({PERIOD}) if multiperiod else frozenset(),
            description="A point in the operational time series.",
        ),
        SCENARIO: Dimension(
            dtype=nw.String(), description="One realisation of a stochastic problem."
        ),
        CARRIER: Dimension(dtype=nw.String(), description="An energy carrier."),
        SHAPE: Dimension(dtype=nw.String(), description="A named geographic shape."),
        _ENTITY: Dimension(dtype=nw.String(), description="A component."),
        _BUS: Dimension(dtype=nw.String(), description="A node of the network."),
    }
    all_by_name = {ct.name: ct for ct in all_components.values()}
    types_by_name = {
        name: ct for name, ct in all_by_name.items() if name not in _EXCLUDED_TYPES
    }
    type_names = sorted(types_by_name)
    dimensions[ENTITY_TYPE] = Dimension(
        dtype=nw.Enum(type_names),
        description="What kind of component an entity is.",
    )
    groups = {
        _CONNECTION: Group(
            over={_ENTITY: _ENTITY, _BUS: _BUS},
            description="A component's attachment to one bus.",
        ),
        ENTITY_TYPE: Group(
            over={_ENTITY: _ENTITY},
            into=ENTITY_TYPE,
            description="What kind of component each entity is.",
        ),
    }

    breakpoint_stems = {y for ctype in type_names for y in piecewise_attrs(ctype)["y"]}

    attributes: dict[str, AttributeSpec] = {}
    results: dict[str, AttributeSpec] = {}
    types: dict[str, TypeSpec] = {}

    for ctype, ct in types_by_name.items():
        defaults = ct.defaults
        stems = _port_stems(ctype, defaults)
        grants: dict[str, TypeAttribute] = {}
        for attr, row in defaults.iterrows():
            if attr == "name" or (ctype, attr) in _TOPOLOGY_OUTPUTS:
                continue
            stem, port = stems.get(attr, (attr, None))
            # A `_RECORD_NAME_OVERRIDES` entry disambiguates a type's own,
            # component-addressed column from the connection-addressed one
            # every per-port flow shares, so it is scoped to `port is None`.
            name = record_name(ctype, stem) if port is None else stem
            dims = {_CONNECTION if port is not None else _ENTITY, SCENARIO}
            if row["varying"]:
                dims.add(TIMESTEP)
                if multiperiod:
                    dims.add(PERIOD)
            spec = AttributeSpec(
                dtype=_DTYPES.get(row["typ"], nw.String()),
                dims=frozenset(dims),
                breakpoints=stem in breakpoint_stems,
            )
            is_output = row["status"].startswith("Output")
            _register(results if is_output else attributes, name, spec)
            if is_output:
                continue
            grants.setdefault(
                name,
                TypeAttribute(
                    default=_default(row["default"]),
                    unit=_text(row["unit"]),
                    description=_text(row["description"]),
                ),
            )
        if stems:
            attributes.setdefault(
                PORT,
                AttributeSpec(
                    dtype=nw.String(), dims=frozenset({_CONNECTION, SCENARIO})
                ),
            )
            grants[PORT] = TypeAttribute()
        types[ctype] = TypeSpec(attributes=grants, description=_text(ct.description))

    for dim, ctype in DIM_TYPES.items():
        defaults = all_by_name[ctype].defaults
        dim_dims = frozenset({dim, SCENARIO}) if stochastic else frozenset({dim})
        for attr in _DIM_ATTRS[dim]:
            row = defaults.loc[attr]
            _register(
                attributes,
                record_name(ctype, attr),
                AttributeSpec(
                    dtype=_DTYPES.get(row["typ"], nw.String()),
                    dims=dim_dims,
                    default=_default(row["default"]),
                    unit=_text(row["unit"]),
                    description=_text(row["description"]),
                ),
            )

    _weighting_descriptions = {
        "objective": "Weight of this snapshot in the objective function.",
        "generators": "Weight of this snapshot for generator energy sums.",
        "stores": "Weight of this snapshot for storage energy sums.",
        "period_objective": "Weight of this period in the objective function.",
        "years": "Number of years this period represents.",
    }
    for name in SNAPSHOT_WEIGHTINGS:
        _register(
            attributes,
            name,
            AttributeSpec(
                dtype=nw.Float64(),
                dims=frozenset({TIMESTEP}),
                default=1.0,
                description=_weighting_descriptions[name],
            ),
        )
    for name in PERIOD_WEIGHTINGS.values():
        _register(
            attributes,
            name,
            AttributeSpec(
                dtype=nw.Float64(),
                dims=frozenset({PERIOD}),
                default=1.0,
                description=_weighting_descriptions[name],
            ),
        )
    _register(
        attributes,
        SCENARIO_WEIGHTINGS["weight"],
        AttributeSpec(dtype=nw.Float64(), dims=frozenset({SCENARIO})),
    )

    return Schema(
        dimensions=dimensions,
        attributes=attributes,
        results=results,
        groups=groups,
        types=types,
        partial=frozenset({SCENARIO}),
    )


def declare_custom(
    schema: Schema,
    ctype: str,
    attr: str,
    dtype: nw.dtypes.DType,
    *,
    varying: bool,
    multiperiod: bool,
) -> None:
    """Register a custom attribute record-wide, granted to `ctype` with no default.

    Carrier and Shape attributes go over their own dim instead, under
    `custom_dim_attr_name`, and are never granted to a type.

    Raises `DatarecordExportError` when `attr` already names a registry
    attribute of a different dtype or dims. Two custom declarations of the
    same name merge by unioning their dims. A dtype mismatch between them
    raises only when at least one is time-varying, so two purely static
    columns that coincide in name may keep their own dtype.
    """
    from pypsa.network.io.datarecord.record import (  # noqa: PLC0415
        DatarecordExportError,
    )

    dim = _TYPE_DIMS.get(ctype)
    if dim is not None:
        record_attr = custom_dim_attr_name(dim, attr)
        schema.attributes[record_attr] = AttributeSpec(
            dtype=dtype, dims=frozenset({dim}), default=None
        )
        return

    varying_dims = {_ENTITY, SCENARIO}
    if varying:
        varying_dims.add(TIMESTEP)
        if multiperiod:
            varying_dims.add(PERIOD)
    dims = frozenset(varying_dims)

    custom_names = schema.meta.setdefault(_CUSTOM_ATTRS_META_KEY, set())
    existing = schema.attributes.get(attr)
    if existing is not None:
        if attr not in custom_names:
            if existing.dtype != dtype or existing.dims != dims:
                msg = (
                    f"{ctype} cannot declare {attr!r} as a custom attribute, "
                    f"it is already a registry attribute of a different shape"
                )
                raise DatarecordExportError(msg)
        elif existing.dtype != dtype:
            if varying or TIMESTEP in existing.dims:
                msg = (
                    f"{ctype} declares custom attribute {attr!r} as {dtype}, "
                    f"conflicting with its existing declaration as {existing.dtype}"
                )
                raise DatarecordExportError(msg)
            # Coincidental name clash between two static columns, neither
            # written to a shared file, so the dtype here is only documentation.
            dtype = existing.dtype
            dims = existing.dims | dims
        else:
            dims = existing.dims | dims

    schema.attributes[attr] = AttributeSpec(dtype=dtype, dims=dims, default=None)
    custom_names.add(attr)
    schema.types[ctype].attributes[attr] = TypeAttribute(default=None)
