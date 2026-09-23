# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The datarecord I/O format: an optional, experimental full-parity format.

Requires the `datarecord` extra (Python 3.12+, `pip install pypsa[datarecord]`).
Importing this subpackage requires `datarecord` to be installed; `import pypsa`
itself never touches it.
"""

from __future__ import annotations

from pypsa.network.io.datarecord.schema import (
    ENTITY_TYPE,
    PERIOD,
    PERIOD_WEIGHTINGS,
    PORT,
    SCENARIO,
    SNAPSHOT_WEIGHTINGS,
    TIMESTEP,
    TIMESTEP_DTYPES,
    build_schema,
    port_columns,
    pypsa_name,
    record_name,
)

__all__ = [
    "ENTITY_TYPE",
    "PERIOD",
    "PERIOD_WEIGHTINGS",
    "PORT",
    "SCENARIO",
    "SNAPSHOT_WEIGHTINGS",
    "TIMESTEP",
    "TIMESTEP_DTYPES",
    "build_schema",
    "port_columns",
    "pypsa_name",
    "record_name",
]
