<!--
SPDX-FileCopyrightText: PyPSA Contributors

SPDX-License-Identifier: CC-BY-4.0
-->

# Global Constraints

Global constraints apply to more than one component at once and are stored under a unique `name` in `n.global_constraints`. Several pre-defined types of global constraints are available (`type`).
For instance, for defining emission limits, transmission expansion limits, or technology capacity expansion limits using the `sense` and `constant` attributes.

By default, global constraints span across all investment periods. For models with multiple investment periods, global constraints can be limited to affect only single investment period (e.g. an emission limit for a specific year) by specifying the in the attribute `investment_period`. The shadow price of each global constraint is stored in the attribute `mu`. Flow-based rows store theirs per snapshot in `mu_ram` instead.

!!! note "Naming Convention"

    Global constraints carry the name scheme `GlobalConstraint-{name}` in the Linopy model. All `flow_based` rows share the one constraint `GlobalConstraint-ram`, indexed by `name`. Their net positions and the balance of the net positions are `FlowBased-net_position` and `FlowBased-balance`.

## Primary Energy

The primary energy constraints (`type="primary_energy"`) are used to limit byproducts of primary energy consumption of a carrier (e.g. CO~2~ emissions) in generators, storage units and stores. They depend on the generator efficiency and carrier-specific attributes (e.g. `co2_emissions`).

Suppose there is a global constraint defined for CO~2~ emissions (`carrier_attribute`) with sense "<=" (`sense`) and a limit $\Gamma$ (`constant`). Emissions can come from generators whose energy carriers have CO~2~ emissions, and from stores and storage units whose storage medium releases or absorbs CO~2~ when it is converted. Only stores and storage units with non-cyclic state of charge that is different at the start ($t=-1$) and end ($t=|T|-1$) of the optimisation period contribute.

If the specific emissions of carrier $s$ is $\rho_s$ (`n.carriers.co2_emissions`) and the generator with carrier $s$ at node $n$ and snapshot $t$ has efficiency $\eta_{n,s,t}$ then the CO~2~ constraint is

$$\begin{gather*}\sum_{n,s,t}  w_t^g \cdot \eta_{n,s,t}^{-1} \cdot g_{n,s,t}\cdot \rho_s + \sum_{n,s}\left(e_{n,s,t=-1} - e_{n,s,t=|T|-1}\right) \cdot \rho_s\\
+ \sum_{n,s}\left(soc_{n,s,t=-1} - soc_{n,s,t=|T|-1}\right) \cdot \rho_s \leq  \Gamma  \quad \leftrightarrow  \quad \mu\end{gather*}$$

The first sum is over generators; the second sum is over stores; the third over storage units. The shadow price $\mu$ would represent the CO~2~ price in this case.

This global constraint is added in the function `define_primary_energy_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $g_{n,s,t}$ | `n.generators_t.p` | Decision Variable |
    | $e_{n,s,t}$ | `n.stores_t.e` | Decision Variable |
    | $soc_{n,s,t}$ | `n.storage_units_t.soc` | Decision Variable |
    | $\mu$ | `n.global_constraints.mu` | Dual Variable |
    | $w_t^g$ | `n.snapshot_weightings.generators` | Parameter |
    | $\eta_{n,s,t}$ | `n.generators.efficiency` | Parameter |
    | $\rho_s$ | `n.carriers.co2_emissions` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |


## Operational Limit

The operational constraints can limit the net production of a carrier taking
into account generator, storage units and stores (`type="operational_limit"`).
For example, this can be used to limit the usage of gas in the system to a
certain amount $\Gamma$ (`constant` in MWh). With sense "<=" (`sense`), the
constraint would be given by

$$\begin{gather*}\sum_{n,s,t}  w_t^g \cdot g_{n,s,t}+ \sum_{n,s}\left(e_{n,s,t=-1} - e_{n,s,t=|T|-1}\right) \\
+ \sum_{n,s}\left(soc_{n,s,t=-1} - soc_{n,s,t=|T|-1}\right) \leq  \Gamma  \quad \leftrightarrow  \quad \mu\end{gather*}$$

The first sum is over generators; the second sum is over stores; the third over
storage units. Structurally, it is similar to the primary energy limit, but
without the consideration of specific emissions and efficiencies. The shadow
price $\mu$ (in currency/MWh) would represent the reduction in system costs if
the operational limit were relaxed by one unit.

This global constraint is added in the function `define_operational_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $g_{n,s,t}$ | `n.generators_t.p` | Decision Variable |
    | $e_{n,s,t}$ | `n.stores_t.e` | Decision Variable |
    | $soc_{n,s,t}$ | `n.storage_units_t.soc` | Decision Variable |
    | $\mu$ | `n.global_constraints.mu` | Dual Variable |
    | $w_t^g$ | `n.snapshot_weightings.generators` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |

## Volume Limit on Transmission Expansion

This global constraint can be used to limit the expansion volume in MWkm of transmission lines and links (`type="transmission_volume_expansion_limit"`). The `carrier_attribute` specifies the subset of carriers to consider. These can be individual carriers or concatenated by commas, e.g. "AC", "DC", "AC,DC", or a [`Link`][pypsa.components.Links] carrier such as "H2 pipeline". With `sense="<="`, the constraint is defined as

$$\sum_{l\in L_{\textrm{carriers}}} d_{l} F_{l} \leq \Gamma \quad \leftrightarrow  \quad \mu$$

where $L_{\textrm{carriers}}$ is the set of lines and links with the specified carriers, $\Gamma$ is the maximum allowed volume expansion in MWkm, $d_{l}$ is the distance of line or link $l$ in km and $F_{l}$ is the capacity of line or link $l$ in MW. The shadow price $\mu$ represents the marginal benefit of expanding the transmission capacity in currency/MWkm/a.

This global constraint is added in the function `define_transmission_volume_expansion_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $F_l$  | `n.{lines,links}.p_nom_opt` | Decision Variable |
    | $\mu$  | `n.global_constraints.mu` | Dual Variable |
    | $d_l$  | `n.{lines,links}.length` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |

## Cost Limit on Transmission Expansion

This global constraint can be used to limit the total investment cost in currency/a of transmission lines and links (`type="transmission_expansion_cost_limit"`). The `carrier_attribute` specifies the subset of carriers to consider. These can be individual carriers or concatenated by commas, e.g. "AC", "DC", "AC,DC", or a [`Link`][pypsa.components.Links] carrier such as "H2 pipeline". With `sense="<="`, the constraint is defined as

$$\sum_{l\in L_{\textrm{carriers}}} c_{l} F_{l} \leq \Gamma \quad \leftrightarrow  \quad \mu$$

where $L_{\textrm{carriers}}$ is the set of lines and links with the specified carriers, $c_{l}$ is the capital cost of line or link $l$ in currency/MW/a, $F_{l}$ is the capacity of line or link $l$ in MW and $\Gamma$ is the maximum allowed cost of line expansion in currency/a. The shadow price $mu$ represents how much the total system cost could be reduced if the spending limit was increased by one currency/a.

This global constraint is added in the function `define_transmission_expansion_cost_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $F_l$  | `n.{lines,links}.p_nom_opt` | Decision Variable |
    | $\mu$  | `n.global_constraints.mu` | Dual Variable |
    | $c_l$  | `n.{lines,links}.capital_cost` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |

## Expansion Limit

This global constraint can be used to limit the total capacity of components of a carrier (`type=tech_capacity_expansion_limit`). This global constraint can be specific to an investment period by setting the `investment_period` attribute and specific to a bus by setting the `bus` attribute. This constraint is mainly used for networks with multiple investment periods, where land usage and building rate restrictions need to be applied for a range of active components of a particular carrier (`carrier_attribute`) in a certain region.

!!! warning

    Currently, only the capacities of extendable components are considered, i.e. generators, storage units and stores with `extendable=True`. The capacities of non-extendable components are not considered in this constraint.

For example, the capacities of all onshore wind generators (`carrier_attribute="onshore wind"`) at a certain bus (`bus="DE"`) should be smaller (`sense="<="`) than a hypothetical technical potential of 200 GW for onshore wind in the specific region (`constant=200e3`). Then the technology capacity expansion constraint across all *active* components is given by

$$\sum_{s | b_s<=a<b_s+L_s} G_{n,s} \leq  \Gamma \quad a \in A \quad \leftrightarrow \quad \mu$$

where $A$ are the investment periods, $s$ are all extendable generators of the specified carrier, $b_s$ is the build year of an asset $s$ with lifetime $L_s$. In this example, the shadow price $\mu$ would represent the marginal benefit of expanding the capacity of onshore wind in currency/MW/a.

In general, the constraint would iterate over all investment variables for generators $G_{n,s}$, lines and transformers $P_{l}$, links $F_{l}$, stores $E_{n,s}$ and storage units $H_{n,s}$ for the specified carrier and bus. For components connecting two buses, the bus selection is done by `bus0`.

This global constraint is added in the function `define_tech_capacity_expansion_limit()`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $G_{n,s}$ | `n.generators.p_nom_opt` | Decision Variable |
    | $H_{n,s}$ | `n.storage_units.p_nom_opt` | Decision Variable |
    | $E_{n,s}$ | `n.stores.e_nom_opt` | Decision Variable |
    | $F_l$  | `n.links.p_nom_opt` | Decision Variable |
    | $P_l$  | `n.{lines,transformers}.s_nom_opt` | Decision Variable |
    | $\mu$  | `n.global_constraints.mu` | Dual Variable |
    | $b_s$  | `n.{<component>}.build_year` | Parameter |
    | $L_s$  | `n.{<component>}.lifetime` | Parameter |
    | $a$, $A$    | `n.investment_periods` | Parameter |
    | $\Gamma$ | `n.global_constraints.constant` | Parameter |


## Growth Limit per Carrier

This carrier-specific constraint type implements absolute and relative growth limits per carrier which constrains new installed capacities for each investment period. It can be defined by providing the attributes `n.carriers.max_growth` and `n.carriers.max_relative_growth`.

Suppose the absolute growth limit for a specific carrier $s$ is $\Gamma_s$ (`max_growth`) and the relative growth limit is $\gamma_s$ (`max_relative_growth`). With the growth limit constraint, for each investment period $a$, the new installed capacity $G_{a,s}$ of all components with carrier $s$ is limited by:

$$G_{a,s} \leq \gamma_s \cdot G_{a-1,s} + \Gamma_s \quad \forall a, s \quad \leftrightarrow \quad \mu_{a,s}$$

where $G_{a,s}$ represents the sum of all newly built nominal capacities of extendable components with carrier $s$ in period $a$. The relative growth limit $\gamma_s$ allows the new capacity in period $a$ to be proportional to the capacity added in the previous period $a-1$.

This constraint only applies to networks with multiple investment periods and only considers components that are newly activated in each period. For the constraint to take effect, at least one of the attributes `max_growth` or `max_relative_growth` must be set to a finite value for the carrier.

!!! warning

    The relative and abolute growth limits are additive, i.e. the absolute growth limit applies in addition to the relative growth limit.

In general, the constraint would iterate over all investment variables for generators $G_{n,s}$, lines and transformers $P_{l}$, links $F_{l}$, stores $E_{n,s}$ and storage units $H_{n,s}$ for the specified carrier. For components connecting two buses, the bus selection is done by `bus0`.

This global constraint is added in the function `define_growth_limit()` and carries the name `Carrier-growth_limit`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $G_{n,s}$ | `n.generators.p_nom_opt` | Decision Variable |
    | $H_{n,s}$ | `n.storage_units.p_nom_opt` | Decision Variable |
    | $E_{n,s}$ | `n.stores.e_nom_opt` | Decision Variable |
    | $F_l$  | `n.links.p_nom_opt` | Decision Variable |
    | $P_l$  | `n.{lines,transformers}.s_nom_opt` | Decision Variable |#
    | $\gamma_s$ | `n.carriers.max_relative_growth` | Parameter |
    | $\Gamma_s$ | `n.carriers.max_growth` | Parameter |

## Flow-Based Market Coupling

Global constraints with `type="flow_based"` implement flow-based market coupling. Each row is one critical network element and contingency (CNEC), with the zonal power transfer distribution factors (PTDF) in the columns `ptdf_<name>` and the remaining available margin in `ram`. See the [component page](../components/global-constraints.md#flow-based-domains) for how to store a domain.

The columns name zones $z \in Z$, which are buses, and corridors $l \in L$, which are links. Each zone gets a net position $NP_{z,t}$ that enters its nodal balance, so the zonal price $\pi_{z,t}$ stays the dual of the balance:

$$\sum \text{injections}_{z,t} - NP_{z,t} = 0 \quad \forall z, t \quad \leftrightarrow \quad \pi_{z,t}$$

The net positions balance across all zones,

$$\sum_{z} NP_{z,t} = 0 \quad \forall t,$$

and every CNEC $c$ limits the flow they cause, together with the flows $p_{l,t}$ of the corridor links:

$$\sum_{z} \mathrm{PTDF}_{c,z,t}\, NP_{z,t} + \sum_{l} \mathrm{PTDF}_{c,l,t}\, p_{l,t} \leq \mathrm{RAM}_{c,t} \quad \forall c, t \quad \leftrightarrow \quad \mu_{c,t}$$

$NP_{z,t}$ is the net injection of the zone bus, including the flows of its links, and has no bounds. The shadow price $\mu_{c,t} \leq 0$, and $|\mu_{c,t}|$ is the welfare gained per MW of additional RAM on the CNEC. The zonal price differences follow from the binding CNECs:

$$\pi_{z_1,t} - \pi_{z_2,t} = \sum_{c} \mu_{c,t} \left(\mathrm{PTDF}_{c,z_1,t} - \mathrm{PTDF}_{c,z_2,t}\right)$$

**Net-position convention and link columns.** The flow of a link touching a zone is already inside the net positions of the zones it connects. A link column therefore carries

$$\mathrm{PTDF}_{c,l} = h_{c,l} - \left(\eta_l\, \mathrm{PTDF}_{c,\text{bus1}} - \mathrm{PTDF}_{c,\text{bus0}}\right),$$

where $h_{c,l}$ is the sensitivity of the CNEC to the link flow $p_l$ (positive from `bus0` to `bus1`), and the PTDF of a bus that is no zone is 0. Two cases:

- A lossless corridor link ($\eta_l = 1$) from zone `A` to zone `B` with its own CNECs `L (+)` and `L (-)` has $\mathrm{PTDF}_{c,l} = \pm 1$ on these (where the zone PTDFs are 0) and $\mathrm{PTDF}_{c,l} = 0$ on the other CNECs. There $h_{c,l} = \mathrm{PTDF}_{c,B} - \mathrm{PTDF}_{c,A}$: the corridor flow reaches these CNECs only through the net positions it moves, $-p_l$ at `A` and $+p_l$ at `B`.
- Advanced hybrid coupling: a hub link `H` from an external bus `X` into zone `B` with $h_{c,H} = 0$, the PTDF of the reference zone, has `ptdf_H = -ptdf_B`. This reproduces a model in which the external generator sits at the reference zone. Standard hybrid coupling instead subtracts the forecast flow over such a border from `ram`.

A link from a zone to a bus that is no zone does not need a column. Its flow then only moves the net position of the zone.

**Time variation.** `ram` and every `ptdf_<name>` column may be time series in `n.global_constraints_t`, and static and time-varying columns can be mixed. The MultiIndex form of [`add_flow_based`][pypsa.components.GlobalConstraints.add_flow_based] writes them.

**Validation.** Building the model raises a `ValueError` when

- a `ptdf_<name>` column names neither a bus nor a link, or both;
- a line or transformer connects two zone buses;
- a link connects two zone buses but has no `ptdf_<name>` column;
- a link column touches no zone bus;
- a zone bus has no attached components;
- `ram` is NaN on a snapshot where the row binds;
- `investment_period` is set on a model optimised without `multi_investment_periods=True`, or lies outside `n.investment_periods`.

Indirect paths between zones through buses that are no zones are not detected, and the [consistency check][pypsa.Network.consistency_check] does not cover the domain.

**Investment periods.** With `multi_investment_periods=True`, a row with `investment_period` binds only in the snapshots of that period. Rows of a period outside the optimised snapshots are skipped without a message, so rolling horizons work.

**Results.** The shadow prices $\mu_{c,t}$, the duals of `GlobalConstraint-ram`, are stored in `n.global_constraints_t.mu_ram`. The zonal prices are in `n.buses_t.marginal_price`, and the net position of a zone is the net injection `n.buses_t.p` of its bus. The net-position variable and the dual of its balance are only kept in the model, as `n.model["FlowBased-net_position"].solution` and `n.model.constraints["FlowBased-balance"].dual`. Nothing is written to `n.buses`.

**Stochastic networks.** The domain applies to every scenario, and the net positions get a `scenario` dimension. Domains that differ between scenarios are not supported. `n.buses_t.p` holds zeros on stochastic networks, so read the net positions from `n.model["FlowBased-net_position"].solution`.

This global constraint is added in the functions `define_flow_based_variables()` and `define_flow_based_constraints()` and carries the names `GlobalConstraint-ram`, `FlowBased-net_position` and `FlowBased-balance`.

??? note "Mapping symbols to component attributes"

    | Symbol | Attribute | Type |
    |--------|-----------|------|
    | $NP_{z,t}$ | `n.model["FlowBased-net_position"]` | Decision Variable |
    | $p_{l,t}$ | `n.links_t.p0` | Decision Variable |
    | $\pi_{z,t}$ | `n.buses_t.marginal_price` | Dual Variable |
    | $\mu_{c,t}$ | `n.global_constraints_t.mu_ram` | Dual Variable |
    | $\mathrm{PTDF}_{c,z}$, $\mathrm{PTDF}_{c,l}$ | `n.global_constraints.ptdf_<name>` / `n.global_constraints_t.ptdf_<name>` | Parameter |
    | $\mathrm{RAM}_{c,t}$ | `n.global_constraints.ram` / `n.global_constraints_t.ram` | Parameter |
    | $\eta_l$ | `n.links.efficiency` | Parameter |

The examples [Flow-Based Market Coupling](../../examples/fbmc-simple-example.ipynb) and [Determining Flow-Based Domains](../../examples/fbmc-building-domain.ipynb) clear and build a domain on a three-zone system.
