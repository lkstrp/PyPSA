<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Global Constraints

The [`GlobalConstraint`][pypsa.components.GlobalConstraints] components describe constraints in the optimisation
problem that apply to multiple components at once.

{{ read_csv('../../../pypsa/data/component_attrs/global_constraints.csv') }}

## Flow-Based Domains

A flow-based domain for [flow-based market coupling](../optimization/global-constraints.md#flow-based-market-coupling) is stored as one global constraint per critical network element and contingency (CNEC) with `type="flow_based"`.

- `ram` is the remaining available margin, the right-hand side of the row. It is static in `n.global_constraints.ram` or time-varying in `n.global_constraints_t.ram`.
- The zonal PTDF lives in custom columns `ptdf_<name>`, static (`n.global_constraints["ptdf_A"]`) or time-varying (`n.global_constraints_t["ptdf_A"]`). Static and time-varying columns may be mixed.
- `<name>` is either a [`Bus`][pypsa.components.Buses] (a bidding zone) or a [`Link`][pypsa.components.Links] (a corridor whose flow enters the domain). A name that is both a bus and a link, or neither, raises an error when the model is built.
- A row with no value in a zone column reads 0 for that zone.
- `sense` and `constant` are ignored for these rows. The helper below writes `sense="<="`.
- `investment_period` restricts a row to one investment period.
- `mu_ram` holds the shadow price of each row per snapshot. `mu` stays 0 for these rows.

The helper [`add_flow_based`][pypsa.components.GlobalConstraints.add_flow_based] writes the rows from a zonal PTDF. A `CNEC x name` DataFrame gives a static domain, a DataFrame with a `(snapshot, CNEC)` MultiIndex a time-varying one, and a Series a single CNEC named by the Series name. `ram` takes a scalar, a Series indexed by CNEC or a `snapshot x CNEC` DataFrame. The property [`zonal_ptdf`][pypsa.components.GlobalConstraints.zonal_ptdf] and its xarray counterpart `n.c.global_constraints.da.zonal_ptdf` read the stacked matrix back, with the bare zone and link names as columns.

``` py
>>> n_fb = pypsa.Network()
>>> n_fb.add("Bus", ["A", "B"])
>>> ptdf = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=["AB (+)"])
>>> n_fb.c.global_constraints.add_flow_based(ptdf, ram=1000)
>>> n_fb.c.global_constraints.zonal_ptdf
bus       A    B
name
AB (+)  0.5 -0.5
```

The optimisation problem is described in [Flow-Based Market Coupling](../optimization/global-constraints.md#flow-based-market-coupling). The examples [Flow-Based Market Coupling](../../examples/fbmc-simple-example.ipynb) and [Determining Flow-Based Domains](../../examples/fbmc-building-domain.ipynb) walk through a three-zone system.
