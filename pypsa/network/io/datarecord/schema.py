# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The canonical datarecord schema built from PyPSA's component registry.

Built from `pypsa.components.types.all_components`, never from a network's
contents, so the schema is the same for every network of a given shape
(multi-period or not, integer or datetime snapshots).
"""

from __future__ import annotations

import functools
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


class DatarecordExportError(ValueError):
    """A network cannot be exported to the datarecord format as-is."""


TIMESTEP, PERIOD, SCENARIO, ENTITY_TYPE, PORT, SHAPE_TYPE = (
    "timestep",
    "period",
    "scenario",
    "entity_type",
    "port",
    "shape_type",
)
# The two groups beside `connection`. `carrier` relates a component to the
# Carrier entity it names, `shape` keys a component's geometry by its kind.
CARRIER, SHAPE = "carrier", "shape"
# The `shape` group's payload columns beside any custom Shape column.
SHAPE_GEOMETRY, SHAPE_NAME = "geometry", "shape_name"
# The two narwhals dtype names `build_schema` accepts for `timestep_dtype`.
TIMESTEP_DTYPES = ("Int64", "Datetime")

SNAPSHOT_WEIGHTINGS = ("objective", "generators", "stores")
PERIOD_WEIGHTINGS = {"objective": "period_objective", "years": "years"}
SCENARIO_WEIGHTINGS = {"weight": "scenario_weight"}

# The entity axis and its bus attachment aren't part of the public interface,
# addressed record-wide only through the `connection` group.
_ENTITY, _BUS, _CONNECTION = "entity", "bus", "connection"

# Component types the schema does not export as entity types: templates and
# library rows, derived non-schema types, and Shape, whose rows are the
# `shape` group keyed by the component they describe.
_EXCLUDED_TYPES = {
    "LineType",
    "TransformerType",
    "SubNetwork",
    "Network",
    "Shape",
}

# A component's `carrier` column names a Carrier entity. It is written as the
# `carrier` group's rows, never declared as an attribute.
CARRIER_ATTR = "carrier"

# Bus, Line and Transformer's `sub_network`, and Bus's `generator`, dropped
# from the schema and never written. Both are set by
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
}
_PYPSA_NAME_OVERRIDES = {
    (ctype, record_attr): attr
    for (ctype, attr), record_attr in _RECORD_NAME_OVERRIDES.items()
}


def record_name(ctype: str, attr: str) -> str:
    """PyPSA attribute -> record-wide attribute name."""
    return _RECORD_NAME_OVERRIDES.get((ctype, attr), attr)


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


def build_schema(*, multiperiod: bool, timestep_dtype: str) -> Schema:
    """Build the canonical datarecord schema for a PyPSA network of this shape.

    Parameters
    ----------
    multiperiod
        Whether `timestep` nests within `period`.
    timestep_dtype
        One of `TIMESTEP_DTYPES`: the narwhals dtype name for the snapshot
        axis, integer or datetime.

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
        SHAPE_TYPE: Dimension(
            dtype=nw.String(),
            description="A kind of geographic shape, e.g. onshore or offshore.",
        ),
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
        CARRIER: Group(
            over={_ENTITY: _ENTITY, CARRIER: _ENTITY},
            description="A component's energy carrier, itself a Carrier entity.",
        ),
        SHAPE: Group(
            over={_ENTITY: _ENTITY, SHAPE_TYPE: SHAPE_TYPE},
            description="A geographic shape of a component, one per kind.",
        ),
    }

    breakpoint_stems = {y for ctype in type_names for y in piecewise_attrs(ctype)["y"]}
    shape_defaults = all_by_name["Shape"].defaults

    attributes: dict[str, AttributeSpec] = {}
    results: dict[str, AttributeSpec] = {}
    types: dict[str, TypeSpec] = {}

    for ctype, ct in types_by_name.items():
        defaults = ct.defaults
        stems = _port_stems(ctype, defaults)
        grants: dict[str, TypeAttribute] = {}
        for attr, row in defaults.iterrows():
            if attr in ("name", CARRIER_ATTR) or (ctype, attr) in _TOPOLOGY_OUTPUTS:
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
        # A shape can describe a component of any type, so every type is
        # granted the shape group's payload, as every ported type is `port`.
        grants[SHAPE_GEOMETRY] = TypeAttribute(
            description=_text(shape_defaults.at[SHAPE_GEOMETRY, "description"])
        )
        grants[SHAPE_NAME] = TypeAttribute(
            description="The name the shape has in PyPSA."
        )
        types[ctype] = TypeSpec(attributes=grants, description=_text(ct.description))

    # The shape group's payload: the geometry as WKT, and the Shape's PyPSA
    # name, carried so a round-trip restores it rather than deriving one.
    # `component`, `idx` and `type` are the group's coordinates.
    for name in (SHAPE_GEOMETRY, SHAPE_NAME):
        _register(
            attributes, name, AttributeSpec(dtype=nw.String(), dims=frozenset({SHAPE}))
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


@functools.cache
def _registry_attribute_names() -> frozenset[str]:
    """Every attribute name `build_schema` registers on its own, before any custom attribute is declared.

    The name set is invariant to `multiperiod`/`timestep_dtype` (only the
    per-attribute shape differs), so one cached call tells `declare_custom`
    whether an existing name is a registry attribute or an earlier custom
    declaration of the same name.
    """
    schema = build_schema(multiperiod=False, timestep_dtype="Int64")
    return frozenset(schema.attributes)


def custom_fallback_name(ctype: str, attr: str) -> str:
    """Record-wide name for a custom attribute that cannot share a registry attribute's file.

    `declare_custom` falls back to it when `attr` is a registry attribute of
    an incompatible shape, e.g. ac_dc_meshed's static Carrier `efficiency`
    against the connection-addressed `efficiency`. The renames are recorded
    in `schema.meta["pypsa"]["custom_names"]` so import never has to guess.
    """
    return f"{ctype.lower()}_{attr}"


def _registry_shape_fits(
    existing: AttributeSpec, dtype: nw.dtypes.DType, dims: frozenset[str]
) -> bool:
    """Whether a custom declaration can join a registry attribute's file.

    Same dtype and no wider dims: a static column on a name some type varies
    in time is a timestep-NULL row of that file, but a time-varying `p_nom`
    would widen Bus's static one, and an entity-addressed column can never
    share a connection-addressed file.
    """
    return existing.dtype == dtype and dims <= existing.dims


def declare_custom(
    schema: Schema,
    ctype: str,
    attr: str,
    dtype: nw.dtypes.DType,
    *,
    varying: bool,
    multiperiod: bool,
) -> str:
    """Register a custom attribute record-wide, granted to `ctype` with no default.

    Returns the record-wide name it was declared under: `attr` itself, or
    `custom_fallback_name` when `attr` is a registry attribute whose shape the
    column cannot share. A custom static column on a name the registry
    declares time-varying elsewhere (ac_dc_meshed's Carrier `marginal_cost`)
    joins that spec instead, its values timestep-NULL rows of the same file.

    A custom Shape column is a payload column of the `shape` group, granted
    to every type like `geometry` and never time-varying (the group has no
    `timestep` axis).

    Raises `DatarecordExportError` when the fallback name is itself a
    registry attribute of an unfit shape, when a Shape attribute is
    `varying`, or when two custom declarations of one name disagree on dtype,
    since one attribute has one dtype, per the format's invariant. Two
    custom declarations of the same name otherwise merge by unioning dims.
    """
    if ctype == "Shape":
        if varying:
            msg = (
                f"Shape cannot declare custom attribute {attr!r} as time-varying, "
                f"shape columns have no timestep axis"
            )
            raise DatarecordExportError(msg)
        dims = frozenset({SHAPE})
    else:
        varying_dims = {_ENTITY, SCENARIO}
        if varying:
            varying_dims.add(TIMESTEP)
            if multiperiod:
                varying_dims.add(PERIOD)
        dims = frozenset(varying_dims)

    name = attr
    existing = schema.attributes.get(name)
    registry = _registry_attribute_names()
    if (
        existing is not None
        and name in registry
        and not _registry_shape_fits(existing, dtype, dims)
    ):
        name = custom_fallback_name(ctype, attr)
        existing = schema.attributes.get(name)
        if (
            existing is not None
            and name in registry
            and not _registry_shape_fits(existing, dtype, dims)
        ):
            msg = (
                f"{ctype} cannot declare {attr!r} as a custom attribute: both "
                f"it and {name!r} are registry attributes of a different shape"
            )
            raise DatarecordExportError(msg)
    if existing is not None:
        if name in registry:
            dims = existing.dims
        elif existing.dtype != dtype:
            msg = (
                f"{ctype} declares custom attribute {attr!r} as {dtype}, "
                f"conflicting with its existing declaration as {existing.dtype}"
            )
            raise DatarecordExportError(msg)
        else:
            dims = existing.dims | dims

    schema.attributes[name] = AttributeSpec(dtype=dtype, dims=dims, default=None)
    if ctype == "Shape":
        for type_spec in schema.types.values():
            type_spec.attributes[name] = TypeAttribute(default=None)
    else:
        schema.types[ctype].attributes[name] = TypeAttribute(default=None)
    return name
