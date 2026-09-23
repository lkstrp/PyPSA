# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Guard invariants of the component attribute registry.

A record-wide schema is built from `pypsa.components.types.all_components`, so
every attribute name must agree on its type, its varying-ness and whether it is
an input or output across all component types that define it.
"""

from pypsa.components.types import all_components

# Types the datarecord schema does not export: templates/library rows
# (LineType, TransformerType) and derived, non-schema types (SubNetwork, Network).
_SCHEMA_EXCLUDED_TYPES = {"LineType", "TransformerType", "SubNetwork", "Network"}


def test_attribute_typ_varying_status_agree_across_types() -> None:
    """Each attribute name has one `typ`, one `varying` and one `status` word."""
    typ_by_attr: dict[str, dict[str, object]] = {}
    varying_by_attr: dict[str, dict[str, object]] = {}
    status_by_attr: dict[str, dict[str, str]] = {}

    for component_name, component_type in all_components.items():
        if component_type.name in _SCHEMA_EXCLUDED_TYPES:
            continue
        defaults = component_type.defaults
        for attr, row in defaults.iterrows():
            if attr == "name":  # the index, not an attribute
                continue
            typ_by_attr.setdefault(attr, {})[component_name] = row["typ"]
            varying_by_attr.setdefault(attr, {})[component_name] = row["varying"]
            status_by_attr.setdefault(attr, {})[component_name] = row["status"].split()[
                0
            ]

    errors = []
    for attr, by_type in typ_by_attr.items():
        values = set(by_type.values())
        if len(values) > 1:
            errors.append(
                f"attribute '{attr}' has disagreeing 'typ' across types: {by_type}"
            )
    for attr, by_type in varying_by_attr.items():
        values = set(by_type.values())
        if len(values) > 1:
            errors.append(
                f"attribute '{attr}' has disagreeing 'varying' across types: {by_type}"
            )
    for attr, by_type in status_by_attr.items():
        values = set(by_type.values())
        if len(values) > 1:
            errors.append(
                f"attribute '{attr}' has disagreeing leading 'status' word across "
                f"types: {by_type}"
            )

    assert not errors, "\n".join(errors)
