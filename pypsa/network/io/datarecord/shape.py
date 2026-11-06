# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The network shape and axis frames of the PyPSA tool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from datarecord.duck import ex_all
from datarecord.protocols import RecordLike, Relations
from datarecord.resolve.record import Record
from datarecord.schema.specs import (
    PropertySpec,
)
from duckdb import CoalesceOperator as coalesce
from duckdb import ColumnExpression as col
from duckdb import ConstantExpression as lit
from duckdb import (
    DuckDBPyConnection,
    DuckDBPyRelation,
)

from pypsa.network.io.datarecord.schema import (
    PERIOD,
    SCENARIO,
    SNAPSHOT,
    TIMESTEP,
)


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

    dims: Relations
    properties: Relations

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
        """The `snapshot` axis in axis order, with its properties where it has any.

        Notes
        -----
        - [axis order](https://energy-models.github.io/datarecord/design/model/dims/#axis-order)

        """
        axis = self._axis(SNAPSHOT)
        frame = self.properties.get(SNAPSHOT)
        if frame is None:
            return axis
        return axis.merge(frame.df(), on=SNAPSHOT, how="left")

    @property
    def snapshot_levels(self) -> pd.DataFrame:
        """`snapshots` as PyPSA's index levels: `(period, timestep)` where a snapshot has a period.

        The timestep is `snapshot.timestep` where the record declares it, else
        the label (`_snapshot_frame`).

        Notes
        -----
        - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

        """
        if PERIOD not in self.snapshots:
            return self.snapshots[[SNAPSHOT]]
        timestep = self.snapshots.get(TIMESTEP, self.snapshots[SNAPSHOT])
        return pd.DataFrame({PERIOD: self.snapshots[PERIOD], TIMESTEP: timestep})

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


def _connection(record: RecordLike) -> DuckDBPyConnection:
    """Return the DuckDB connection the relations of `record` belong to.

    Off the concrete backing, since the protocol names no connection. Needed
    because a relation exposes no reachable reference to its connection, and
    this tool's `PIVOT` - which `relation.query` refuses as a `MULTI`
    statement - needs one.

    Raises
    ------
    TypeError
        If `record` is not a `Record`.

    Notes
    -----
    - [relations](https://energy-models.github.io/datarecord/design/use/reading/#relations)

    """
    if not isinstance(record, Record):
        msg = (
            f"the PyPSA tool builds from a Record, not a {type(record).__name__}: "
            "it runs DuckDB SQL on the record's connection"
        )
        raise TypeError(msg)
    return record.con


def _snapshot_frame(snapshots: pd.Index) -> pd.DataFrame:
    """`n.snapshots` as the `snapshot` axis with its properties, in PyPSA's order.

    A single-period network's snapshots are the labels. A multi-period one's
    `(period, timestep)` give each snapshot its `period`, and its timestep is
    the label where none repeats across periods; else the label is the
    snapshot's position and the timestep its `timestep`.

    Notes
    -----
    - [consuming a record](https://energy-models.github.io/datarecord/design/tools/)

    """
    if not isinstance(snapshots, pd.MultiIndex):
        return snapshots.to_frame(index=False)
    period = snapshots.get_level_values(PERIOD)
    timestep = snapshots.get_level_values(TIMESTEP)
    if timestep.is_unique:
        return pd.DataFrame({SNAPSHOT: timestep, PERIOD: period})
    return pd.DataFrame(
        {SNAPSHOT: np.arange(len(snapshots)), PERIOD: period, TIMESTEP: timestep}
    )


def _snapshot_properties(snapshots: pd.DataFrame) -> dict[str, PropertySpec]:
    """The properties of a multi-period network's `snapshot`, by `_snapshot_frame`'s columns.

    Notes
    -----
    - [properties](https://energy-models.github.io/datarecord/design/model/properties/#properties)

    """
    properties = {
        PERIOD: PropertySpec(
            one_of=PERIOD,
            description="The investment period a snapshot falls in.",
        )
    }
    if TIMESTEP in snapshots:
        properties[TIMESTEP] = PropertySpec(
            one_of=TIMESTEP,
            description="PyPSA's timestep of a snapshot labelled by its position.",
        )
    return properties


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
        df = index.to_frame(index=False) if keys is None else keys  # noqa: F841 - queried by name below
        return cls(index, con.sql("SELECT *, row_number() OVER () AS _pos FROM df"))

    def expand(self, rel: DuckDBPyRelation, column: str) -> DuckDBPyRelation:
        """Broadcast rows with a null `column` across this index, others as-is.

        `rel` must already carry the alias the caller wants to select its
        own columns by - `expand` doesn't impose one.
        """
        if len(self.index) == 0:
            return rel
        # `IS NULL = true` hashes; the one-sided `IS NULL` is a nested loop.
        axis = self.rel.project(col(column), lit(True).alias("_broadcast"))  # noqa: FBT003
        return rel.join(
            axis.set_alias("_ax"),
            col(rel.alias, column).isnull() == col("_ax", "_broadcast"),
            how="left",
        ).project(
            *(col(rel.alias, c) for c in rel.columns if c != column),
            coalesce(col(rel.alias, column), col("_ax", column)).alias(column),
        )

    def left_join(self, other: DuckDBPyRelation, on: Sequence[str]) -> DuckDBPyRelation:
        """LEFT JOIN `other` onto this index, restored to `index`'s row order."""
        return self.rel.join(
            other,
            ex_all(col(self.rel.alias, f) == col(other.alias, f) for f in on),
            how="left",
        ).order(f"{self.rel.alias}._pos")
