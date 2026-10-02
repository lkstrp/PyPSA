# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Shared helpers and base classes for importing and exporting data."""

from __future__ import annotations

import functools
import logging
import tempfile
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, overload
from urllib.request import urlretrieve

import pandas as pd

from pypsa._options import options

try:
    from cloudpathlib import AnyPath as Path
except ImportError:
    from pathlib import Path
if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Self

    import xarray as xr

    from pypsa import Network
logger = logging.getLogger(__name__)

_legacy_string_dtype_warned = False


def _use_legacy_string_dtype() -> bool:
    """Resolve whether string data is converted to object dtype on import."""
    global _legacy_string_dtype_warned  # noqa: PLW0603

    value = options.api.legacy_string_dtype
    if value is None:
        if not _legacy_string_dtype_warned:
            warnings.warn(
                "pandas infers the `str` dtype for string data since its version 3.0. "
                "PyPSA still converts it back to numpy object dtype on import, but will "
                "keep it from PyPSA 2.0 on. Set "
                "`pypsa.options.api.legacy_string_dtype` explicitly to suppress this "
                "warning.",
                FutureWarning,
                stacklevel=3,
            )
            _legacy_string_dtype_warned = True
        return True
    return value


def _coerce_string_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce `StringDtype` indices, columns and values to `object` dtype.

    Pre-1.3 behaviour, kept behind `options.api.legacy_string_dtype` until 2.0.
    """
    if not _use_legacy_string_dtype():
        return df

    def _coerce_axis(axis: pd.Index) -> pd.Index:
        if isinstance(axis, pd.MultiIndex):
            new_levels = [
                level.astype(object)
                if isinstance(level.dtype, pd.StringDtype)
                else level
                for level in axis.levels
            ]
            if any(
                nl is not ol for nl, ol in zip(new_levels, axis.levels, strict=True)
            ):
                axis = axis.set_levels(new_levels)
            return axis
        if isinstance(axis.dtype, pd.StringDtype):
            return axis.astype(object)
        return axis

    df.index = _coerce_axis(df.index)
    df.columns = _coerce_axis(df.columns)
    str_cols = [
        col for col, dtype in df.dtypes.items() if isinstance(dtype, pd.StringDtype)
    ]
    if str_cols:
        df[str_cols] = df[str_cols].astype(object)
    return df


def _legacy_snapshots(index: pd.Index) -> pd.Index:
    """Convert legacy snapshot labels to a form `set_snapshots` accepts.

    Older PyPSA files may carry a single `"now"` label, string date labels, or
    other arbitrary string labels. For a `MultiIndex` only the `timestep`
    level is converted, period labels are left untouched so a non-integer
    period still raises from `set_snapshots`.
    """
    if isinstance(index, pd.MultiIndex):
        period = index.get_level_values("period")
        timestep = _legacy_snapshot_level(index.get_level_values("timestep"), by=period)
        return pd.MultiIndex.from_arrays([period, timestep], names=index.names)
    return _legacy_snapshot_level(index)


def _legacy_snapshot_level(level: pd.Index, by: pd.Index | None = None) -> pd.Index:
    """Convert one snapshot index level to integer or datetime labels.

    Integer and datetime levels pass through unchanged. A single `"now"`
    label maps to `0`. Other labels are tried as dates, and if that fails,
    replaced by positions (per `by` group, for a multi-period timestep level)
    with a warning listing the original labels.
    """
    if level.dtype.kind in "iuM":
        return level

    if len(level) == 1 and level[0] == "now":
        return pd.Index([0], name=level.name)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(pd.Index(level))
        return pd.DatetimeIndex(parsed, name=level.name)
    except (ValueError, TypeError):
        pass

    logger.warning(
        "Converting legacy snapshot labels to positions: %s",
        ", ".join(map(str, level)),
    )
    if by is None:
        positions = range(len(level))
    else:
        positions = (
            pd.Series(range(len(level))).groupby(by.to_numpy(), sort=False).cumcount()
        )
    return pd.Index(positions, name=level.name)


def _get_safe_excel_sheet_name(sheet_name: str) -> str:
    """Convert sheet name to/from safe version for Excel's 31-character limit.

    Works bidirectionally - converts long names to short and short back to long. Only
    built-in mappings are handled, other names are returned unchanged and need to be
    handled by the user (via UserWarning from openpyxl).
    """
    mappings = {
        "storage_units-state_of_charge_set": "storage_units-soc_set",
        "storage_units-efficiency_dispatch": "storage_units-eff_dispatch",
        "generators-marginal_cost_piecewise": "generators-marginal_cost_pw",
        "generators-capital_cost_piecewise": "generators-capital_cost_pw",
        "processes-capital_cost_piecewise": "processes-capital_cost_pw",
        "processes-marginal_cost_piecewise": "processes-marginal_cost_pw",
        "storage_units-capital_cost_piecewise": "storage_units-capital_cost_pw",
        "storage_units-marginal_cost_piecewise": "storage_units-marginal_cost_pw",
        "transformers-capital_cost_piecewise": "transformers-capital_cost_pw",
    }

    if sheet_name in mappings:
        return mappings[sheet_name]

    for long_name, short_name in mappings.items():
        if sheet_name == short_name:
            return long_name

    return sheet_name


@overload
def _retrieve_from_url(
    url: str, io_function: Callable[[Path], pd.read_excel]
) -> pd.DataFrame: ...


@overload
def _retrieve_from_url(
    url: str, io_function: Callable[[Path], pd.HDFStore | xr.Dataset]
) -> Network: ...


@functools.lru_cache(maxsize=128)
def _retrieve_from_url(url: str, io_function: Callable) -> pd.DataFrame | Network:
    # Check if network requests are allowed
    if not options.get_option("general.allow_network_requests"):
        msg = "Network requests are disabled. Set `pypsa.options.general.allow_network_requests = True` to enable URL loading."
        raise ValueError(msg)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file_path = Path(temp_file.name)
        logger.info("Retrieving network data from %s.", url)
        if not url.startswith("http"):
            msg = f"Invalid URL: {url}"
            raise ValueError(msg)
        try:
            urlretrieve(url, file_path)  # noqa: S310
        except Exception as e:
            msg = f"Failed to retrieve network data from {url}: {e}"
            raise ValueError(msg) from e
        return io_function(file_path)


class _ImpExper:
    """Base class for importers and exporters."""

    ds: Any = None

    def __enter__(self) -> Self:
        """Enter the context manager."""
        if self.ds is not None:
            self.ds = self.ds.__enter__()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Exit the context manager."""
        if exc_type is None:
            self.finish()

        if self.ds is not None:
            self.ds.__exit__(exc_type, exc_val, exc_tb)

    @abstractmethod
    def finish(self) -> None:
        """Post-processing when process is finished."""


class _Exporter(_ImpExper):
    """_Exporter class."""

    path: Path

    def remove_static(self, list_name: str) -> None:
        """Remove static components data."""

    def remove_series(self, list_name: str, attr: str) -> None:
        """Remove dynamic components data."""

    def remove_piecewise(self, list_name: str, attr: str) -> None:
        """Remove piecewise component data."""

    @abstractmethod
    def save_attributes(self, attrs: dict) -> None:
        """Save generic network attributes."""

    @abstractmethod
    def save_meta(self, meta: dict) -> None:
        """Save meta data (`n.meta`)."""

    @abstractmethod
    def save_crs(self, crs: dict) -> None:
        """Save CRS of shapes of network."""

    @abstractmethod
    def save_snapshots(self, snapshots: Sequence) -> None:
        """Save snapshots data."""

    @abstractmethod
    def save_investment_periods(self, investment_periods: pd.Index) -> None:
        """Save investment periods data."""

    @abstractmethod
    def save_scenarios(self, scenarios: pd.DataFrame) -> None:
        """Save scenarios data."""

    @abstractmethod
    def save_static(self, list_name: str, df: pd.DataFrame) -> None:
        """Save static components data."""

    @abstractmethod
    def save_series(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save dynamic components data."""

    @abstractmethod
    def save_piecewise(self, list_name: str, attr: str, df: pd.DataFrame) -> None:
        """Save piecewise component data."""


class _Importer(_ImpExper):
    """Importer class."""

    def get_scenarios(self) -> pd.DataFrame | None:
        return None


def _sort_attrs(
    axis_labels: pd.Index, attrs_list: Sequence[str] | pd.Index
) -> pd.Index:
    """Order axis labels to match a desired attribute sequence.

    Parameters
    ----------
    axis_labels : pandas.Index
        Original axis labels that should be reordered.
    attrs_list : Sequence[str] | pandas.Index
        Desired ordering given as an ordered collection of attribute names.

    Returns
    -------
    pandas.Index
        `axis_labels` with the attributes appearing in `attrs_list` first and
        in the same order. Attributes missing from `attrs_list` follow in their
        original order while names not present in `axis_labels` are ignored.

    """
    if axis_labels.empty or len(attrs_list) == 0:
        return axis_labels

    attrs_index = (
        attrs_list if isinstance(attrs_list, pd.Index) else pd.Index(attrs_list)
    )
    existing = attrs_index.intersection(axis_labels, sort=False)
    if existing.empty:
        return axis_labels

    remaining = axis_labels.difference(attrs_index, sort=False)
    target = existing.union(remaining, sort=False)

    if axis_labels.equals(target):
        return axis_labels

    return target
