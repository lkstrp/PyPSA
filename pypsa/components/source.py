# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Sources that feed the data containers of a Components object."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from pypsa.constants import DEFAULT_EPSG, DEFAULT_TIMESTAMP

if TYPE_CHECKING:
    from pypsa.definitions.components import ComponentType

#: Returned by every `InMemorySource.names` call, so never mutate it.
_NO_NAMES = pd.Index([], name="name", dtype=object)


@runtime_checkable
class Source(Protocol):
    """Hands out the frames of a component type on request.

    Every method returns a frame the caller owns: a Components object keeps it as its
    live container and mutates it in place.
    """

    def static(self, ctype: ComponentType) -> pd.DataFrame:
        """Return the static frame of `ctype`, indexed by component name."""
        ...

    def dynamic(self, ctype: ComponentType, attr: str) -> pd.DataFrame:
        """Return the time-varying frame of `attr`, indexed by snapshot."""
        ...

    def piecewise(self, ctype: ComponentType, attr: str) -> pd.DataFrame:
        """Return the piecewise breakpoint frame of the y-axis attribute `attr`."""
        ...

    def varying_attrs(self, ctype: ComponentType) -> list[str]:
        """Return the attributes `dynamic` can be asked for, without reading data."""
        ...

    def names(self, ctype: ComponentType) -> pd.Index:
        """Return the component names `static` would be indexed by, without building it.

        For a stochastic network, the names without the scenario level.
        """
        ...


class InMemorySource:
    """Empty frames shaped by the component type's defaults, the source of a network built in memory."""

    def static(self, ctype: ComponentType) -> pd.DataFrame:
        """Return an empty static frame with one typed column per static attribute."""
        static_dtypes = ctype.defaults.loc[ctype.defaults.static, "dtype"].drop(
            ["name"]
        )
        static: pd.DataFrame
        if ctype.name == "Shape":
            static = gpd.GeoDataFrame(
                {k: gpd.GeoSeries(dtype=d) for k, d in static_dtypes.items()},
                columns=static_dtypes.index,
                crs=CRS.from_epsg(DEFAULT_EPSG),
            )
        else:
            static = pd.DataFrame(
                {k: pd.Series(dtype=d) for k, d in static_dtypes.items()},
                columns=static_dtypes.index,
            )
        static.index.name = "name"
        return static

    def dynamic(self, ctype: ComponentType, attr: str) -> pd.DataFrame:
        """Return an empty float frame with the single default snapshot."""
        df = pd.DataFrame(index=pd.Index([DEFAULT_TIMESTAMP]), columns=[], dtype=float)
        df.index.name = "snapshot"
        df.columns.name = "name"
        return df

    def piecewise(self, ctype: ComponentType, attr: str) -> pd.DataFrame:
        """Return an empty breakpoint frame with `(name, attribute)` columns."""
        cols = pd.MultiIndex.from_tuples([], names=["name", "attribute"])
        return pd.DataFrame(
            index=pd.Index([], name="breakpoint", dtype=int),
            columns=cols,
            dtype=float,
        )

    def varying_attrs(self, ctype: ComponentType) -> list[str]:
        """Return the attributes flagged as varying in the defaults."""
        return list(ctype.defaults.index[ctype.defaults.varying])

    def names(self, ctype: ComponentType) -> pd.Index:
        """Return no names, the static frame starts empty."""
        return _NO_NAMES
