# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Components transform module.

Contains single mixin class which is used to inherit to [pypsa.Components][] class.
Should not be used directly.

Transform methods are methods which modify and restructure data.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from pypsa.network.names import (
    NAMESPACE_ORDER,
    _name_level,
    format_clashes,
    name_owners,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pypsa.definitions.structures import Dict

logger = logging.getLogger(__name__)


class ComponentsTransformMixin:
    """Mixin class for components descriptors methods.

    Class inherits to [pypsa.Components][]. All attributes and methods can be used
    within any Components instance.
    """

    static: pd.DataFrame
    dynamic: Dict
    piecewise: Dict
    attached: Any
    n_save: Any
    name: Any

    def add(
        self,
        name: str | int | Sequence[int | str],
        suffix: str | Sequence[str] = "",
        overwrite: bool = False,
        return_names: bool | None = None,
        **kwargs: Any,
    ) -> pd.Index | None:
        """Add new components.

        <!-- md:badge-version v0.33.0 -->

        Handles addition of single and multiple components along with their attributes.
        Pass a list of names to add multiple components at once or pass a single name
        to add a single component.

        When a single component is added, all non-scalar attributes are assumed to be
        time-varying and indexed by snapshots.
        When multiple components are added, all non-scalar attributes are assumed to be
        static and indexed by names. A single value sequence is treated as scalar and
        broadcasted to all components. It is recommended to explicitly pass a scalar
        instead.
        If you want to add time-varying attributes to multiple components, you can pass
        a 2D array/ DataFrame where the first dimension is snapshots and the second
        dimension is names.

        Any attributes which are not specified will be given the default
        value from <!-- md:guide components.md -->.

        Parameters
        ----------
        name : str or int or list of str or list of int
            Component name(s)
        suffix : str or list of str, default ""
            Suffix added to each name. Pass a list together with a single `name`
            to add one component per suffix.
        overwrite : bool, default False
            If True, existing components with the same names as in `name` will be
            overwritten. Otherwise only new components will be added and others will be
            ignored.
        return_names : bool | None, default=None
            Whether to return the names of the new components. Defaults to module wide
            option (default: False). See `https://go.pypsa.org/options-params` for more
            information.
        kwargs : Any
            Component attributes, e.g. x=[0.1, 0.2], can be list, pandas.Series
            of pandas.DataFrame for time-varying

        Returns
        -------
        new_names : pandas.index or None
            Names of new components (including suffix) if return_names is `True`,
            otherwise `None`.

        Examples
        --------
        The example is shown for Generator component, but the same applies to all
        component types.

        >>> n = pypsa.Network()
        >>> c = n.components.generators
        >>> c
        Empty 'Generator' Components

        Add a single component:

        >>> c.add("my-generator-1", carrier="AC")

        A new generator is added to the components instance:
        >>> c
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'Unnamed Network'
        Components: 1

        With static data (and default values for all attributes):
        >>> c.static[["carrier", "p_nom"]]
                       carrier  p_nom
        name
        my-generator-1      AC    0.0

        Add multiple components with static attributes:

        >>> c.add(["my-generator-2", "my-generator-3"],
        ...       carrier=["AC", "DC"],
        ...       p_nom=10)

        A new generator is added to the components instance:
        >>> c
        'Generator' Components
        ----------------------
        Attached to PyPSA Network 'Unnamed Network'
        Components: 3

        With static data:
        >>> c.static[["carrier", "p_nom"]]
                   carrier  p_nom
        name
        my-generator-1      AC    0.0
        my-generator-2      AC   10.0
        my-generator-3      DC   10.0

        The single value for `p_nom` is broadcasted to all components. So you could also
        pass `[10, 10]` instead of `10`.

        See Also
        --------
        [pypsa.Network.add][]

        """
        if not self.attached:
            msg = (
                "Currently new components can only be added when the components "
                "are already attached to a network."
            )
            raise NotImplementedError(msg)

        return self.n_save.add(
            self.name,
            name,
            suffix=suffix,
            overwrite=overwrite,
            return_names=return_names,
            **kwargs,
        )

    def rename_component_names(self, **kwargs: str) -> None:
        """Rename component names.

        Rename components and also update all cross-references of the component in
        the network.

        Parameters
        ----------
        **kwargs
            Mapping of old names to new names.

        Examples
        --------
        Define some network
        >>> n = pypsa.Network()
        >>> n.add("Bus", ["bus1"])
        >>> n.add("Generator", ["gen1"], bus="bus1")
        >>> c = n.c.buses

        Now rename the bus

        >>> c.rename_component_names(bus1="bus2")

        Which updates the bus components

        >>> c.static.index
        Index(['bus2'], dtype='str', name='name')

        and all references in the network

        >>> n.generators.bus
        name
        gen1    bus2
        Name: bus, dtype: object

        """
        if not all(isinstance(v, str) for v in kwargs.values()):
            msg = "New names must be strings."
            raise ValueError(msg)

        self._refuse_taken_target_names(kwargs)

        # `level="name"` renames only the name level, so a stochastic
        # scenario label matching a new name stays untouched.
        self.static = self.static.rename(index=kwargs, level="name")
        for store in (self.dynamic, self.piecewise):
            for k, v in store.items():  # Modify in place
                level = "name" if isinstance(v.columns, pd.MultiIndex) else None
                store[k] = v.rename(columns=kwargs, level=level)

        # Rename cross references in network (if attached to one)
        if self.attached:
            n = self.n_save
            for component in n.components:
                if self.name == "Carrier":
                    # One `carrier` column per type, never port-suffixed.
                    cols = ["carrier"] if "carrier" in component.static.columns else []
                else:
                    col_name = self.name.lower()  # TODO: Generalize
                    cols = [
                        f"{col_name}{port}"
                        for port in component.ports
                        if f"{col_name}{port}" in component.static.columns
                    ]
                if cols and not component.static.empty:
                    component.static[cols] = component.static[cols].replace(kwargs)

            # Bus.generator names the slack generator attached to a bus.
            if self.name == "Generator":
                buses = n.components["Bus"].static
                if not buses.empty and "generator" in buses.columns:
                    buses["generator"] = buses["generator"].replace(kwargs)

            # Shape.idx names the component a shape describes.
            shapes = n.components["Shape"].static
            if not shapes.empty:
                own_shapes = shapes["component"] == self.name
                if own_shapes.any():
                    shapes.loc[own_shapes, "idx"] = shapes.loc[
                        own_shapes, "idx"
                    ].replace(kwargs)

    def _refuse_taken_target_names(self, kwargs: dict[str, str]) -> None:
        """Raise if a rename target name is already taken.

        Every type refuses a target already present within its own type,
        including a target reused by more than one rename in the same call.
        Namespace types (Generator, Bus, Carrier, ...) additionally refuse a
        target already held by another namespace type, since Shape and the
        other exempt types sit outside that namespace.
        """
        own_names = _name_level(self.static.index).unique().difference(kwargs.keys())
        same_type_clashes = own_names.intersection(kwargs.values())
        targets = pd.Index(kwargs.values())
        duplicate_targets = targets[targets.duplicated()].unique()
        clashes = same_type_clashes.union(duplicate_targets)
        if not clashes.empty:
            names = ", ".join(sorted(clashes))
            msg = f"name(s) already present in '{self.name}': {names}"
            raise ValueError(msg)

        if self.attached and self.name in NAMESPACE_ORDER:
            owners = name_owners(self.n_save, targets, exclude=self.name)
            if owners:
                owners = {
                    name: sorted({self.name, *types}) for name, types in owners.items()
                }
                msg = format_clashes(owners)
                raise ValueError(msg)
