# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Registry-derived schema, constants and naming helpers of the PyPSA tool."""

from __future__ import annotations

import math
from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import pandas as pd
from datarecord.duck import fn
from datarecord.schema.schema import Schema as RecordSchema
from datarecord.schema.specs import (
    AttributeSpec,
    Dimension,
    PropertySpec,
)
from duckdb import Expression

from pypsa.components.ports import port_table
from pypsa.components.types import all_components
from pypsa.constants import piecewise_stems

if TYPE_CHECKING:
    import pypsa


# The axes a PyPSA network is built from; all three must be declared in the
# schema's `dimensions`, since a network shape is built from them (https://energy-models.github.io/datarecord/design/tools/).
# A declared axis with no rows is fine - that is just a deterministic or
# single-period network.
SNAPSHOT, PERIOD, SCENARIO = "snapshot", "period", "scenario"


#: PyPSA's name for a multi-period snapshot's inner level: the dim of those
#: timesteps, a coordinate of the multi-dim `snapshot` beside `period`.
TIMESTEP = "timestep"


# Not PyPSA's own `weight` - that name is a Generator attribute (its clustering
# weight), and a record has one attribute per name record-wide - so the scenario
# weighting is renamed on the way in and back to `weight` on build.
SCENARIO_WEIGHT = "scenario_weight"


ENTITY, BUS, CARRIER, CONNECTION = "entity", "bus", "carrier", "connection"


GLOBAL_CONSTRAINT = "global_constraint"


SUB_NETWORK = "sub_network"
LINE_TYPE, TRANSFORMER_TYPE = "line_type", "transformer_type"


CTYPE = "ctype"


#: `ctype` as a `default_by` names it, by its subject.
BY_CTYPE = f"{ENTITY}.{CTYPE}"


#: The PyPSA types of catalog metadata, each a dim whose every column is a property.
_CATALOG_TYPES = {"LineType": LINE_TYPE, "TransformerType": TRANSFORMER_TYPE}


#: The PyPSA types whose members are labels of a dim of their own, not entities.
_AXIS_TYPES = {
    "Bus": BUS,
    "Carrier": CARRIER,
    "SubNetwork": SUB_NETWORK,
    "GlobalConstraint": GLOBAL_CONSTRAINT,
    **_CATALOG_TYPES,
}


#: PyPSA's type whose members are no labels, each a property of the label it outlines.
_SHAPE = "Shape"


#: What names a property as a shape's geometry, the shape's `type` after it.
_SHAPE_PREFIX = "shape_"


#: How a shape's geometry and a dim shapes make are described.
_GEOMETRY_DESCRIPTION = "A shape outlining the label, as WKT in the CRS `meta` names."
_SHAPE_DIM = "A label that shapes outline."


#: The registry inputs declared as properties, by subject: what describes a
#: label rather than data a model optimises against. `entity` also has
#: `ctype`, a `carrier` property is one of the `carrier` dim, and a
#: `_CATALOG_TYPES` dim takes every registry input as a property.
_PROPERTIES = {
    ENTITY: ("carrier", "sign", "type"),
    BUS: ("carrier", "type", "x", "y", "unit", "location"),
    CARRIER: ("color", "nice_name"),
    SUB_NETWORK: (CARRIER,),
    GLOBAL_CONSTRAINT: ("type", BUS),
}


#: How a SubNetwork's `carrier` is described, PyPSA registering it as an output.
_SUB_NETWORK_CARRIER = "The carrier of a sub-network's buses."


#: The integer dimension a bus port of a component is labelled on.
PORT = "port"


REQUIRED_DIMS = frozenset({SNAPSHOT, PERIOD, SCENARIO})


#: The attributes PyPSA keeps the same in every scenario: mirrors the local
#: `INVARIANT_ATTRS` of `pypsa.consistency.check_scenario_invariant_attributes`.
_SCENARIO_INVARIANT = frozenset(
    {
        "name",
        "bus",
        "type",
        "p_nom_extendable",
        "s_nom_extendable",
        "e_nom_extendable",
        "p_nom_mod",
        "s_nom_mod",
        "e_nom_mod",
        "committable",
        "sign",
        "carrier",
        "weight",
        "p_nom_opt",
        "s_nom_opt",
        "e_nom_opt",
        "build_year",
        "lifetime",
        "active",
    }
)


#: The dims a record's rows are keyed on, which no custom column may be named like.
_KEYS = frozenset({SNAPSHOT, TIMESTEP, PERIOD, SCENARIO, ENTITY, CONNECTION})


#: The `meta` key naming, per entity type, the custom static columns it was
#: exported with: the part of `_taken`'s vocabulary the registry lacks.
_CUSTOM = "custom_columns"


@dataclass(frozen=True)
class _Weighting:
    """One column of the weighting frame PyPSA keeps beside an axis."""

    dim: str
    column: str
    description: str


#: Every axis weighting, by the attribute over its axis it is stored as.
_WEIGHTINGS = {
    SCENARIO_WEIGHT: _Weighting(
        SCENARIO, "weight", "How much this scenario counts in the expectation."
    ),
    "snapshot_weighting_objective": _Weighting(
        SNAPSHOT, "objective", "How much this snapshot counts in the objective."
    ),
    "snapshot_weighting_stores": _Weighting(
        SNAPSHOT, "stores", "How long this snapshot lasts for a store's energy."
    ),
    "snapshot_weighting_generators": _Weighting(
        SNAPSHOT,
        "generators",
        "How much this snapshot counts towards a generator's energy limits.",
    ),
    "period_weighting_objective": _Weighting(
        PERIOD, "objective", "How much this period counts in the objective."
    ),
    "period_weighting_years": _Weighting(
        PERIOD, "years", "How many years this period lasts."
    ),
}


# PyPSA's `defaults["typ"]` (the attribute's Python type) mapped to the mathspec
# dtype the long schema stores, `str` for anything unlisted. Keyed on `typ` -
# what a value *is* - not `type`, whose `"static or series"`-style strings
# describe only where it lives and grow a new variant per PyPSA release.
_DTYPES = {bool: "bool", int: "int", float: "float", str: "str"}


def _default(value: Any) -> Any:
    """One `defaults["default"]` cell as JSON-storable, NaN as absent.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/model/attributes/#attributespec)

    """
    if isinstance(value, float) and math.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _same(a: Any, b: Any) -> bool:
    """Whether two `_default` values are one default, `True` and `1` being two."""
    return type(a) is type(b) and a == b


def _text(value: Any) -> str | None:
    """One `unit`/`description` cell as text, or None where PyPSA has none.

    Its registry leaves both NaN rather than empty, and `None` is the schema's
    "undeclared" - `""` would claim the attribute is documented as blank, or
    dimensionless, neither of which a missing cell says.

    Notes
    -----
    - [unit and description](https://energy-models.github.io/datarecord/design/model/schema/#unit-and-description)

    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _output_attributes(c: pypsa.Components) -> frozenset[str]:
    """A component type's result attributes, from PyPSA's own registry.

    `defaults["status"]` marks them ("Output"), so a PyPSA upgrade that adds a
    result attribute is picked up rather than needing a list kept here. Read off
    `c` rather than the bare registry: a network's extra ports (`p2` on a Link
    with `bus2`) exist only in its own `defaults`.
    """
    defaults = c.defaults
    is_output = defaults["status"].astype(str).str.startswith("Output")
    return frozenset(defaults.index[is_output])


_PER_UNIT = frozenset({"x_pu", "r_pu", "g_pu", "b_pu", "x_pu_eff", "r_pu_eff"})


"""A branch's per-unit impedances, which `calculate_dependent_values` sets."""


_DERIVED: dict[str, frozenset[str]] = {
    "Bus": frozenset({"q", "control", "generator", "sub_network"}),
    "Line": frozenset({"v_nom", *_PER_UNIT}),
    "ShuntImpedance": frozenset({"v_nom", "g_pu", "b_pu"}),
    "SubNetwork": frozenset({"obj"}),
    "Transformer": _PER_UNIT,
}


"""Columns not written, by type: PyPSA derives each from what is written.

`Bus.q` is the signed sum of the `q` results on the connections at that bus;
`control`, `generator` and `sub_network` follow from the connections and the
topology. `Bus.p` is written, as `p_bus` (`_RESULT_RENAMES`), since PyPSA's
solve sets it and a bus-level `p` would collide with the connection-level one. `Line.v_nom` and `ShuntImpedance.v_nom`
are no registry column: `calculate_dependent_values` copies them from the
bus, and as a custom column over `entity` they would clash with the `v_nom`
over `bus`. `calculate_dependent_values` also recomputes the per-unit
impedances from `x`, `r`, `g`, `b`, the bus `v_nom` and `tap_ratio`, so a
written copy goes stale once a child layer changes one of those inputs;
PyPSA recomputes them before a solve or a power flow, and `build` leaves
them at the registry default.
"""


_RESULT_RENAMES: dict[str, dict[str, str]] = {
    "Bus": {"p": "p_bus"},
}


"""Per type, the results renamed off the per-port result whose name they share."""


def _renames(ctype: str) -> dict[str, str]:
    """`ctype`'s PyPSA result -> record name renames; other types rename `p`."""
    return _RESULT_RENAMES.get(ctype, {"p": "p_internal"})


#: Per type, the inputs renamed off the name another type's input shares. `Bus.scaling` and
#: `GlobalConstraint.scaling` would be one record attribute over two dims.
_INPUT_RENAMES: dict[str, dict[str, str]] = {
    "GlobalConstraint": {"scaling": "constraint_scaling"},
}


def record_input(ctype: str, attr: str) -> str:
    """The record name of `ctype`'s PyPSA input `attr`."""
    return _INPUT_RENAMES.get(ctype, {}).get(attr, attr)


def pypsa_attribute(ctype: str, name: str) -> str:
    """The PyPSA attribute a record `name` of `ctype`, a result or an input, came from."""
    for renames in (_renames(ctype), _INPUT_RENAMES.get(ctype, {})):
        for attr, renamed in renames.items():
            if renamed == name:
                return attr
    return name


def _result_attributes(c: pypsa.Components, inputs: Container[str]) -> dict[str, str]:
    """`c`'s PyPSA outputs -> the result each lands in, the rule schema and frames share.

    A per-port output (`p0`, `p1`, `p2`, a one-port type's `p`) lands in its
    stem, one row per connection, exactly as a per-port input does; any other
    keeps PyPSA's name, but for `_RESULT_RENAMES`. Dropped: `name`, which is
    the member list rather than a value, `_DERIVED`, and any result
    named in `inputs`, the record's input declarations: one attribute holds
    one spec record-wide, and PyPSA registers some names (`SubNetwork.carrier`,
    `Line.x` beside `Bus.x`) as an output on one type and an input on another.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    - [results on connections](https://energy-models.github.io/datarecord/design/tools/pypsa/#results-on-connections)

    """
    per_port = _port_stems(c)
    derived = _DERIVED.get(c.name, frozenset())
    results = {}
    for attr in _output_attributes(c):
        if attr == "name" or attr in derived:
            continue
        renames = _renames(c.name)
        name = per_port.get(attr) or renames.get(attr, attr)
        if name not in inputs:
            results[attr] = name
    return results


@cache
def _record_stems() -> frozenset[str]:
    """Every attribute stem some PyPSA type carries once per port, inputs and results alike.

    A record stores each as one bus-keyed attribute, so these are the names
    whose port suffix is undone on write. Multiport types fill the union, which
    also bus-keys a one-port type's `p`.

    Notes
    -----
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

    """
    return frozenset().union(*(port_table(t).stems for t in all_components))


def _is_output(defaults: pd.DataFrame, attr: str) -> bool:
    """Whether `attr` is an output (custom attrs with no status are inputs).

    Mirrors `pypsa.common._is_output`.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

    """
    if attr not in defaults.index:
        return False
    return str(defaults.at[attr, "status"]).startswith("Output")


def _port_names(c: pypsa.Components) -> dict[str, tuple[str, int]]:
    """Per-port attribute name -> `(stem, port)`, for this type's actual ports."""
    table = c.port_table
    registered = c.defaults.index
    return {
        name: (stem, port)
        for stem in sorted(_record_stems())
        for port in table.ports_of(c)
        if (name := table.column(stem, port)) in registered
    }


def _port_stems(c: pypsa.Components) -> dict[str, str]:
    """Per-port attribute name -> its stem, for this type's actual ports."""
    return {name: stem for name, (stem, _) in _port_names(c).items()}


def _attribute_spec(row: pd.Series, *, over: str) -> AttributeSpec:
    """One `c.defaults` row as the spec a record declares, input or result.

    `over` is the dim its labels are: `entity` for a component's,
    the `connection` multi-dim for a per-port one, whose rows carry it as one
    `{entity, bus}` struct, `bus` or `carrier` for an axis type's
    (`_AXIS_TYPES`). A varying attribute varies over time too, beside its
    `_static_dims`.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/model/attributes/#attributespec)
    - [multi-dims](https://energy-models.github.io/datarecord/design/model/multi-dims/)

    """
    dims = _static_dims(row.name, over)
    return AttributeSpec(
        dtype=_DTYPES.get(row["typ"], "str"),  # type: ignore[arg-type]
        dims=dims | ({SNAPSHOT} if row["varying"] else set()),
        default=_default(row["default"]),
        unit=_text(row.get("unit")),
        description=_text(row.get("description")),
    )


def _port_spec(
    defaults: pd.DataFrame, stem: str, per_port: Mapping[str, str]
) -> AttributeSpec:
    """A per-port stem's spec, from its first port's row.

    The stem's port rows agree but for the default: a Process's `rate0` is
    `-1` and `rate1` `1`, the input and output ends. One stem-wide default
    would misstate a port, so where they differ none is declared, and `build`
    leaves an uncovered port to PyPSA's own.

    Parameters
    ----------
    defaults
        The type's `c.defaults`.
    per_port
        Per-port attribute name -> its stem, as `_port_stems` returns it.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/model/attributes/#attributespec)
    - [ports](https://energy-models.github.io/datarecord/design/tools/pypsa/#ports)

    """
    columns = [name for name, of in per_port.items() if of == stem]
    spec = _attribute_spec(defaults.loc[columns[0]], over=CONNECTION)
    if len({_default(defaults.at[name, "default"]) for name in columns}) > 1:
        return spec.model_copy(update={"default": None})
    return spec


def _with_default_by(
    attributes: dict[str, AttributeSpec], by_type: dict[str, dict[str, Any]]
) -> dict[str, AttributeSpec]:
    """`attributes`, each given `default_by` over `ctype` where a type's default differs.

    Parameters
    ----------
    by_type
        Attribute name -> component type -> that type's registry default.

    Notes
    -----
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/model/attributes/#attributespec)

    """
    widened = dict(attributes)
    for name, defaults_of in by_type.items():
        spec = attributes[name]
        differing = {
            ct: v for ct, v in defaults_of.items() if not _same(v, spec.default)
        }
        if differing:
            widened[name] = spec.model_copy(
                update={"default_by": {BY_CTYPE: differing}}
            )
    return widened


def _property(name: str, spec: AttributeSpec) -> PropertySpec:
    """Return the property spec of registry input `name`, where a `carrier` or `bus` names a label of that dim.

    PyPSA's `""` names none, so it is no such property's default.

    Notes
    -----
    - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)

    """
    fields = spec.model_dump(exclude={"dims", "breakpoints"})
    if name in (CARRIER, BUS):
        fields |= {
            "dtype": None,
            "one_of": name,
            "default": fields["default"] or None,
        }
    return PropertySpec(**fields)


def _registry_schema() -> RecordSchema:
    """The record schema PyPSA's whole component registry declares (`_registry`)."""
    return _registry()[0]


def _registry_carries() -> dict[str, frozenset[str]]:
    """Per entity type, the names PyPSA's registry declares for it (`_registry`)."""
    return _registry()[1]


def _input_names() -> frozenset[str]:
    """Every input the registry schema declares, attribute or property of any subject (`_registry`)."""
    return _registry()[2]


@cache
def _registry() -> tuple[RecordSchema, dict[str, frozenset[str]], frozenset[str]]:
    """The record schema PyPSA's whole component registry declares, what each entity type takes, and its inputs.

    Every network exports against this one schema: its dims and
    attributes describe every PyPSA component type and attribute, not
    only the ones a given network holds. Declaring a type or attribute a network
    lacks writes no file and is rejected by nothing - `ctype` only rejects an
    *unknown* type - so a network-independent schema is both correct and the
    stabler contract. Derived once, cached: the registry
    does not change within a process.

    `c.defaults` already declares what an `AttributeSpec` asks for, so this reads
    it rather than restating it. Results are attributes too, read off the same
    registry (`status` starting "Output"), so a PyPSA upgrade adding one is
    picked up rather than needing a list kept here; which names they are is
    `_result_attributes`, the rule `to_datarecord` keys its frames by, a
    per-port one declared over the `connection` multi-dim like a per-port input.

    The second item names, per entity type, the names it declares - stems,
    `entity` properties and results alike - which `build` keeps its values to
    (`_taken`). The third is every input, which no result may be named
    (`_result_attributes`).
    One spec per name record-wide: the first type
    declaring a name gives its spec, and a default differing per type is
    `default_by` (`_with_default_by`). A per-port column collapses to its stem
    (`_port_spec`), and a port's bus is the `connection` tuple itself.

    `Bus`, `Carrier` and `SubNetwork` are no entity types (`_AXIS_TYPES`):
    their members are the labels of `bus`, `carrier` and `sub_network`, and
    their inputs and results are over that dim. The properties of each subject
    are `_PROPERTIES`; a SubNetwork's `carrier` is declared here, PyPSA
    registering it as an output. `entity` also
    has `ctype`, which refers to the dim `ctype`, whose `one_of` is the
    entity types, so a type outside it is refused; `bus` is the one property of
    `connection`, keyed by `(entity, port)`. Every other input,
    each axis weighting (`_WEIGHTINGS`) over its axis included, is an
    attribute.

    Notes
    -----
    - [where a value lives](https://energy-models.github.io/datarecord/design/storage/format/#where-a-value-lives)
    - [the schema](https://energy-models.github.io/datarecord/design/model/schema/)
    - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)
    - [AttributeSpec](https://energy-models.github.io/datarecord/design/model/attributes/#attributespec)
    - [writing results](https://energy-models.github.io/datarecord/design/tools/pypsa/#writing-results)
    - [results on connections](https://energy-models.github.io/datarecord/design/tools/pypsa/#results-on-connections)

    """
    import pypsa

    n = pypsa.Network()
    attributes: dict[str, AttributeSpec] = {
        name: AttributeSpec(
            dtype="float",
            dims=frozenset({weighting.dim}),
            description=weighting.description,
        )
        for name, weighting in _WEIGHTINGS.items()
    }
    carries: dict[str, frozenset[str]] = {}
    by_type: dict[str, dict[str, Any]] = {}
    exported = sorted(n.all_components - {_SHAPE})
    ctypes = [ctype for ctype in exported if ctype not in _AXIS_TYPES]
    for ctype in ctypes:
        c = n.components[ctype]
        defaults = c.defaults
        per_port = _port_stems(c)
        is_output = _output_attributes(c)
        carried: set[str] = set()
        for attr in defaults.index:
            if attr == "name" or attr in is_output:
                continue
            stem = per_port.get(attr, attr)
            if stem in carried or stem == BUS:
                continue
            carried.add(stem)
            if attr not in per_port:
                by_type.setdefault(stem, {})[c.name] = _default(
                    defaults.at[attr, "default"]
                )
            attributes.setdefault(
                stem,
                _port_spec(defaults, stem, per_port)
                if attr in per_port
                else _attribute_spec(defaults.loc[attr], over=ENTITY),
            )
        if carried:
            carries[c.name] = frozenset(carried)
    attributes = _with_default_by(attributes, by_type)
    for name in piecewise_stems():
        if name in attributes:
            attributes[name] = attributes[name].model_copy(update={"breakpoints": True})
    properties = {
        ENTITY: {
            CTYPE: PropertySpec(
                one_of=CTYPE, description="What kind of entity an entity is."
            ),
            **{
                name: _property(name, attributes.pop(name))
                for name in _PROPERTIES[ENTITY]
            },
        }
    }
    for ctype, dim in _AXIS_TYPES.items():
        c = n.components[ctype]
        specs = {
            attr: _attribute_spec(c.defaults.loc[attr], over=dim)
            for attr in c.defaults.index
            if attr != "name" and attr not in _output_attributes(c)
        }
        renames = _INPUT_RENAMES.get(ctype, {})
        held = tuple(specs) if ctype in _CATALOG_TYPES else _PROPERTIES.get(dim, ())
        properties[dim] = (
            {CARRIER: PropertySpec(one_of=CARRIER, description=_SUB_NETWORK_CARRIER)}
            if dim == SUB_NETWORK
            else {name: _property(name, specs.pop(name)) for name in held}
        )
        attributes |= {renames.get(name, name): spec for name, spec in specs.items()}
    inputs = frozenset(
        {*attributes, *(name for named in properties.values() for name in named)}
    )
    results: dict[str, AttributeSpec] = {}
    for ctype in exported:
        c = n.components[ctype]
        per_port = _port_stems(c)
        named = _result_attributes(c, inputs)
        for attr, name in sorted(named.items()):
            over = CONNECTION if attr in per_port else _dim_of(ctype)
            results.setdefault(name, _attribute_spec(c.defaults.loc[attr], over=over))
        if ctype not in _AXIS_TYPES and named:
            carries[ctype] = carries.get(ctype, frozenset()) | set(named.values())
    properties[CONNECTION] = {BUS: PropertySpec(one_of=BUS)}
    schema = RecordSchema(
        dimensions={
            SNAPSHOT: Dimension(
                dtype="datetime",
                description="A point in the operational time series.",
            ),
            TIMESTEP: Dimension(
                dtype="datetime",
                description="A timestep of a multi-period network, within a period.",
            ),
            PERIOD: Dimension(
                dtype="int",
                unit="year",
                description="An investment period, labelled by its year.",
            ),
            SCENARIO: Dimension(
                dtype="str",
                description="One realisation of a stochastic problem.",
            ),
            ENTITY: Dimension(dtype="str", description="An entity."),
            CTYPE: Dimension(
                one_of=ctypes,  # type: ignore[arg-type]
                description="A kind of entity.",
            ),
            BUS: Dimension(dtype="str", description="A node of the network."),
            CARRIER: Dimension(
                dtype="str", description="An energy carrier or technology."
            ),
            SUB_NETWORK: Dimension(
                dtype="str",
                description="A connected part of the network, as PyPSA numbers them.",
            ),
            GLOBAL_CONSTRAINT: Dimension(
                dtype="str",
                description="A constraint over the whole network, as PyPSA names it.",
            ),
            LINE_TYPE: Dimension(dtype="str", description="A standard type of line."),
            TRANSFORMER_TYPE: Dimension(
                dtype="str", description="A standard type of transformer."
            ),
            PORT: Dimension(
                dtype="int",
                description=(
                    "A bus port of a component, PyPSA's index in bus0, p1, efficiency2."
                ),
            ),
            CONNECTION: Dimension(
                over={"entity": ENTITY, "port": PORT},
                description="An entity's attachment to a bus at one port.",
            ),
        },
        properties=properties,
        attributes=attributes | results,
        partial=frozenset(
            {ENTITY, BUS, PORT, CARRIER, GLOBAL_CONSTRAINT, SCENARIO, CONNECTION}
        ),
    )
    return schema, carries, inputs


def _taken(schema: RecordSchema, ctype: str, names: Iterable[str]) -> list[str]:
    """Those of `names` a `ctype` takes, in order: what no other type claims, or its own.

    A type claims its registry names (`_registry_carries`), its `_DERIVED`
    names, which an earlier record may still hold, and the custom columns its
    network was exported with (`meta`). A name no type claims - an attribute a
    record adds - reaches every type.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/pypsa/#a-value-its-type-does-not-take)

    """
    custom = schema.meta.get(_CUSTOM) or {}
    claimed = {
        t: carried | _DERIVED.get(t, frozenset()) | frozenset(custom.get(t, ()))
        for t, carried in _registry_carries().items()
    }
    everyone = frozenset().union(*claimed.values(), *map(frozenset, custom.values()))
    own = claimed.get(ctype, frozenset(custom.get(ctype, ())))
    return [n for n in names if n in own or n not in everyone]


def _shape_properties(schema: RecordSchema, dim: str) -> list[str]:
    """The properties of `dim` that hold a shape's geometry, named `shape_{type}`."""
    held = schema.properties.get(dim, {})
    return [name for name in held if name.startswith(_SHAPE_PREFIX)]


def _dim_of(ctype: str) -> str:
    """The dim `ctype`'s members are labels of: `entity`, or their own (`_AXIS_TYPES`)."""
    return _AXIS_TYPES.get(ctype, ENTITY)


def _static_dims(name: str, over: str) -> frozenset[str]:
    """The dims of attribute `name` over `over`, less `snapshot`.

    `scenario` is among them unless PyPSA keeps `name` the same in every
    scenario (`_SCENARIO_INVARIANT`).
    """
    return frozenset({over} if name in _SCENARIO_INVARIANT else {over, SCENARIO})


def _connection_struct(entity: Expression, port: Expression) -> Expression:
    """The `connection` column packed from `entity` and `port`, typed as declared.

    Notes
    -----
    - [multi-dims](https://energy-models.github.io/datarecord/design/model/multi-dims/)

    """
    schema = _registry_schema()
    struct = schema.duck_type(schema.dimensions[CONNECTION])
    return fn.struct_pack(**{ENTITY: entity, PORT: port}).cast(struct).alias(CONNECTION)
