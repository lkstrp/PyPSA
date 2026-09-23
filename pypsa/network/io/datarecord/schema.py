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

TIMESTEP, PERIOD, SCENARIO, ENTITY_TYPE, PORT = (
    "timestep",
    "period",
    "scenario",
    "entity_type",
    "port",
)
# The two narwhals dtype names `build_schema` accepts for `timestep_dtype`.
TIMESTEP_DTYPES = ("Int64", "Datetime")

SNAPSHOT_WEIGHTINGS = ("objective", "generators", "stores")
PERIOD_WEIGHTINGS = {"objective": "period_objective", "years": "years"}
SCENARIO_WEIGHTINGS = {"weight": "scenario_weight"}

# The entity axis and its bus attachment aren't part of the public interface,
# addressed record-wide only through the `connection` group.
_ENTITY, _BUS, _CONNECTION = "entity", "bus", "connection"

# Component types the schema does not export: templates/library rows and
# derived, non-schema types.
_EXCLUDED_TYPES = {"LineType", "TransformerType", "SubNetwork", "Network"}

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


def _column_record_name(ctype: str, stem: str, port: str | None) -> str:
    """Resolve the record-wide name for one column, `record_name` unless it is a port column.

    A `_RECORD_NAME_OVERRIDES` entry disambiguates a type's own,
    component-addressed column from the connection-addressed one every
    per-port flow shares (`Link.p` -> `p_activity`, leaving per-port `p0`/`p1`
    named plain `p`). Applying it to a port column instead would rename the
    connection-addressed flow away, so it is scoped to `port is None`.
    """
    return record_name(ctype, stem) if port is None else stem


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
    for port in ports:
        for stem in ("bus", "p", "q"):
            col = f"{stem}{port}"
            if col in defaults.index:
                result[col] = (stem, port)
        if single_port and coefficient_attr is None and "efficiency" in defaults.index:
            result["efficiency"] = ("efficiency", port)
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

    E.g. `{"bus0": ("bus", "0"), "efficiency": ("efficiency", "1"), "p": ("p", "")}`
    for a Link, or `{"bus": ("bus", ""), "p": ("p", ""), "q": ("q", "")}` for a
    single-port type such as Generator.
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
        _ENTITY: Dimension(dtype=nw.String(), description="A component."),
        _BUS: Dimension(dtype=nw.String(), description="A node of the network."),
    }
    types_by_name = {
        ct.name: ct for ct in all_components.values() if ct.name not in _EXCLUDED_TYPES
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
            if attr == "name":
                continue
            stem, port = stems.get(attr, (attr, None))
            name = _column_record_name(ctype, stem, port)
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
                    unit=_text(row.get("unit")),
                    description=_text(row.get("description")),
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
