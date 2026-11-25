# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

"""Descriptors for component attributes."""

from __future__ import annotations

import logging
import re
from collections.abc import KeysView
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable, ItemsView, Iterable, Iterator, ValuesView

logger = logging.getLogger(__name__)


class Dict(dict):
    """Subclass of dict, which allows to use attribute syntax to get and set items.

    Stripped down from addict `https://github.com/mewwts/addict/`.
    """

    def __copy__(self) -> Dict:  # e.g. copy.copy()
        """Return a shallow copy of the Dict."""
        return Dict(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> Dict:  # e.g. copy.deepcopy()
        """Return a deep copy of the Dict."""
        return Dict({k: deepcopy(v, memo) for k, v in self.items()})

    def copy(self) -> Dict:  # e.g. d.copy()
        """Return a shallow copy of the Dict, preserving the Dict type."""
        return Dict(self)

    def __setattr__(self, name: str, value: Any) -> None:
        """Setattr is called when the syntax a.b = 2 is used to set a value."""
        if hasattr(Dict, name):
            msg = f"'Dict' object attribute '{name}' is read-only"
            raise AttributeError(msg)
        self[name] = value

    def __getattr__(self, item: str) -> Any:
        """Getattr is called when the syntax a.b is used to get a value."""
        try:
            return self.__getitem__(item)
        except KeyError as e:
            raise AttributeError(e.args[0]) from e

    def __delattr__(self, name: str) -> None:
        """Is invoked when del some_addict.b is called."""
        del self[name]

    _re_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

    def __dir__(self) -> list[str]:
        """Return a list of object attributes.

        This includes key names of any dict entries, filtered to the
        subset of valid attribute names (e.g. alphanumeric strings
        beginning with a letter or underscore).  Also includes
        attributes of parent dict class.
        """
        dict_keys = [
            m.string
            for k in self.keys()
            if isinstance(k, str) and (m := self._re_pattern.match(k))
        ]

        obj_attrs = list(dir(Dict))

        return dict_keys + obj_attrs


class _NotLoaded:
    def __repr__(self) -> str:
        return "<not loaded>"


_NOT_LOADED = _NotLoaded()


class LazyDict(Dict):
    """Dict whose keys are known up front and whose values load on first access.

    `load(key)` fetches one value. Membership, iteration, `len` and `keys` cover
    pending keys without loading them; item and attribute access, `get`, `pop` and
    `setdefault` load one key; `values`, `items`, copies, pickling and comparisons
    load every pending key first. Writing or deleting a pending key never loads it.
    Iteration lists loaded keys in insertion order, then pending keys in the order
    they were announced.
    """

    __slots__ = ("_load", "_pending")

    _load: Callable[[str], Any] | None
    _pending: dict[str, None]

    def __init__(
        self, *, load: Callable[[str], Any] | None = None, keys: Iterable[str] = ()
    ) -> None:
        """Create a dict announcing `keys`, each fetched with `load` when needed."""
        super().__init__()
        pending = dict.fromkeys(keys)
        if pending and load is None:
            msg = "A loader is required to announce keys"
            raise ValueError(msg)
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_load", load if pending else None)

    @property
    def pending(self) -> frozenset[str]:
        """Keys announced but not loaded yet."""
        return frozenset(self._pending)

    def materialize(self) -> None:
        """Load every pending key."""
        for key in list(self._pending):
            self._load_key(key)

    def _load_key(self, key: str) -> Any:
        assert self._load is not None  # noqa: S101
        value = self._load(key)
        dict.__setitem__(self, key, value)
        self._discard(key)
        return value

    def _discard(self, key: Any) -> bool:
        if not self._pending or key not in self._pending:
            return False
        del self._pending[key]
        if not self._pending:
            object.__setattr__(self, "_load", None)
        return True

    @classmethod
    def _loaded(cls, items: Iterable[tuple[Any, Any]]) -> LazyDict:
        new = cls()
        dict.update(new, items)
        return new

    def __missing__(self, key: Any) -> Any:
        """Load a pending key when it is not stored yet."""
        if key in self._pending:
            return self._load_key(key)
        raise KeyError(key)

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value of `key`, loading it if pending, else `default`."""
        if key in self._pending:
            return self._load_key(key)
        return dict.get(self, key, default)

    def pop(self, key: Any, *default: Any) -> Any:
        """Remove `key` and return its value, loading it if pending."""
        if key in self._pending:
            self._load_key(key)
        return dict.pop(self, key, *default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Return the value of `key`, loading it if pending, else insert `default`."""
        if key in self._pending:
            return self._load_key(key)
        return dict.setdefault(self, key, default)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Store `value`, replacing a pending key without loading it."""
        self._discard(key)
        dict.__setitem__(self, key, value)

    def __delitem__(self, key: Any) -> None:
        """Delete `key`, dropping a pending key without loading it."""
        if not self._discard(key):
            dict.__delitem__(self, key)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Store every given item, replacing pending keys without loading them."""
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __ior__(self, other: Any) -> Self:
        """Update in place with `other`."""
        self.update(other)
        return self

    def clear(self) -> None:
        """Remove every loaded and pending key."""
        dict.clear(self)
        self._pending.clear()
        object.__setattr__(self, "_load", None)

    def popitem(self) -> tuple[Any, Any]:
        """Remove and return the last item, loading it if pending."""
        if self._pending:
            key = next(reversed(self._pending))
            return key, self.pop(key)
        return dict.popitem(self)

    def __contains__(self, key: object) -> bool:
        """Whether `key` is loaded or pending."""
        return dict.__contains__(self, key) or key in self._pending

    def __iter__(self) -> Iterator[Any]:
        """Iterate loaded keys, then pending keys, without loading."""
        if not self._pending:
            return dict.__iter__(self)
        return iter([*dict.keys(self), *self._pending])

    def __reversed__(self) -> Iterator[Any]:
        """Iterate keys in reverse order, without loading."""
        return reversed(list(self))

    def __len__(self) -> int:
        """Count loaded and pending keys."""
        return dict.__len__(self) + len(self._pending)

    def keys(self) -> KeysView[Any]:  # type: ignore[override]
        """Return a view over loaded and pending keys, without loading."""
        if not self._pending:
            return dict.keys(self)
        return KeysView(self)

    def __repr__(self) -> str:
        """Represent loaded items like a dict and pending keys as not loaded."""
        if not self._pending:
            return dict.__repr__(self)
        entries = [*dict.items(self), *((k, _NOT_LOADED) for k in self._pending)]
        return "{" + ", ".join(f"{k!r}: {v!r}" for k, v in entries) + "}"

    def values(self) -> ValuesView[Any]:  # type: ignore[override]
        """Return a view over every value, loading pending keys first."""
        self.materialize()
        return dict.values(self)

    def items(self) -> ItemsView[Any, Any]:  # type: ignore[override]
        """Return a view over every item, loading pending keys first."""
        self.materialize()
        return dict.items(self)

    def copy(self) -> LazyDict:
        """Return a shallow, fully loaded copy."""
        self.materialize()
        return self._loaded(dict.items(self))

    def __copy__(self) -> LazyDict:
        """Return a shallow, fully loaded copy."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> LazyDict:
        """Return a deep, fully loaded copy detached from the loader."""
        self.materialize()
        return self._loaded((k, deepcopy(v, memo)) for k, v in dict.items(self))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as a fully loaded dict without the loader."""
        self.materialize()
        return (self.__class__, (), None, None, iter(dict.items(self)))

    def __eq__(self, other: object) -> bool:
        """Compare after loading every pending key on both sides."""
        self.materialize()
        if isinstance(other, LazyDict):
            other.materialize()
        return dict.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        """Compare after loading every pending key on both sides."""
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    __hash__ = None  # type: ignore[assignment]

    def __or__(self, other: Any) -> LazyDict:
        """Return a fully loaded copy updated with `other`."""
        if not isinstance(other, dict):
            return NotImplemented
        new = self.copy()
        new.update(other)
        return new
