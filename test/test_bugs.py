# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import pypsa


def test_rename_component_names_of_non_bus_type_does_not_raise():
    """Renaming names of a component type other than `Bus` raised `KeyError`:
    the cross-reference update built column names from every port label found
    on the OTHER component's own bus columns (e.g. Line's "0"/"1"), combined
    with the renamed type's name (e.g. "generator0"), a column no type
    actually carries.
    """
    n = pypsa.examples.ac_dc_meshed()

    generator = n.c.generators.static.index[0]
    n.c.generators.rename_component_names(**{generator: f"{generator}_renamed"})
    assert f"{generator}_renamed" in n.c.generators.static.index

    line = n.c.lines.static.index[0]
    n.c.lines.rename_component_names(**{line: f"{line}_renamed"})
    assert f"{line}_renamed" in n.c.lines.static.index
