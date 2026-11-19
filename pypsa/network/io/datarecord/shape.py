# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The network shape and axis frames of the PyPSA tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import pandas as pd
from datarecord.duck import ex_all
from datarecord.protocols import RecordLike, Relations
from datarecord.schema.schema import Schema as RecordSchema
from datarecord.schema.schema import unpacked
from duckdb import ColumnExpression as col
from duckdb import (
    DuckDBPyConnection,
    DuckDBPyRelation,
    row_number,
)
from duckdb import StarExpression as star

from pypsa.network.io.datarecord.schema import (
    PERIOD,
    SCENARIO,
    SNAPSHOT,
)

#: A row's position in its relation, from 1, as `Index` orders by.
_POS = row_number().over().alias("_pos")


@dataclass(frozen=True)
class NetworkShape:
    """PyPSA's reading of a record's axis frames.

    The axis names and the index convention live here rather than on the
    record, which stays schema-generic: a record may declare any dims, and it is
    this tool that decides `snapshot` is the time axis and that a non-empty
    `scenario` axis means a stochastic network.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

    """

    schema: RecordSchema
    dims: Relations

    def _axis(self, dim: str) -> pd.DataFrame:
        """`dim`'s axis as a DataFrame, empty if the record declares none.

        A dim with no rows anywhere is absent from the mapping rather than
        present-and-empty, and both read the same way here: an empty
        frame, which is a deterministic or single-period network.

        Notes
        -----
        - [the long schema](https://energy-models.github.io/datarecord/design/storage/format/#the-long-schema)

        """
        rel = self.dims.get(dim)
        return rel.df() if rel is not None else pd.DataFrame(columns=[dim])

    @cached_property
    def snapshots(self) -> pd.DataFrame:
        """The `snapshot` axis in axis order, a multi-dim's tuples as a column per coordinate."""
        rel = self.dims.get(SNAPSHOT)
        if rel is None:
            return pd.DataFrame(columns=[SNAPSHOT])
        return unpacked(self.schema, SNAPSHOT, rel).df()

    @cached_property
    def periods(self) -> pd.DataFrame:
        return self._axis(PERIOD)

    @cached_property
    def scenarios(self) -> pd.DataFrame:
        return self._axis(SCENARIO)

    @property
    def stochastic(self) -> bool:
        return not self.scenarios.empty

    @property
    def multiperiod(self) -> bool:
        return not self.periods.empty

    @property
    def index_names(self) -> list[str]:
        """The static frame's index: PyPSA carries `scenario` as a level.

        Notes
        -----
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

        """
        return [SCENARIO, "entity"] if self.stochastic else ["entity"]


def _multiperiod(record: RecordLike) -> bool:
    """Whether `record` is multi-period, its `snapshot` a multi-dim over `period`."""
    spec = record.schema.dimensions.get(SNAPSHOT)
    return spec is not None and PERIOD in (spec.over or {})


@dataclass(frozen=True)
class Index:
    """A pandas Index paired with a matching DuckDB relation.

    `rel` carries `_pos`, the index's row position, so a join against it can
    be restored to `index`'s order (not necessarily sorted) by ordering on
    `_pos` rather than by value.
    """

    index: pd.Index
    rel: DuckDBPyRelation

    @classmethod
    def of(
        cls,
        index: pd.Index,
        con: DuckDBPyConnection,
        keys: pd.DataFrame | None = None,
    ) -> Index:
        """`index` keyed by its own levels, or by `keys`, one row per entry in its order."""
        df = index.to_frame(index=False) if keys is None else keys
        return cls(index, con.from_df(df).project(star(), _POS))

    def left_join(self, other: DuckDBPyRelation, on: Sequence[str]) -> DuckDBPyRelation:
        """LEFT JOIN `other` onto this index, restored to `index`'s row order."""
        return self.rel.join(
            other,
            ex_all(col(self.rel.alias, f) == col(other.alias, f) for f in on),
            how="left",
        ).order(col(self.rel.alias, "_pos"))
