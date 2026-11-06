# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""The datarecord tool for PyPSA networks.

Requires the optional `datarecord` package (`pip install 'pypsa[datarecord]'`,
Python 3.12+). The modules of this package import it at top level, so import
this package only where it is needed.
"""

from datarecord.tools.base import UnsupportedRecordError

from pypsa.network.io._common import require_datarecord
from pypsa.network.io.datarecord.tool import PyPSA, PyPSATool

__all__ = ["PyPSA", "PyPSATool", "UnsupportedRecordError", "require_datarecord"]
