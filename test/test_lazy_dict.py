# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import copy
import pickle

import pandas as pd
import pytest

from pypsa.definitions.structures import Dict, LazyDict


class Loader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, key: str) -> pd.DataFrame:
        self.calls.append(key)
        return pd.DataFrame({key: [1.0, 2.0]})


@pytest.fixture
def loader() -> Loader:
    return Loader()


@pytest.fixture
def d(loader: Loader) -> LazyDict:
    return LazyDict(load=loader, keys=["a", "b"])


def test_keys_without_loading(d, loader):
    assert list(d) == ["a", "b"]
    assert list(d.keys()) == ["a", "b"]
    assert len(d) == 2
    assert "a" in d
    assert "c" not in d
    assert loader.calls == []
    assert d.pending == frozenset({"a", "b"})


def test_getitem_loads_once(d, loader):
    first = d["a"]
    second = d["a"]
    assert first is second
    assert loader.calls == ["a"]
    assert d.pending == frozenset({"b"})


def test_attribute_access_loads_same_object(d, loader):
    assert d.a is d["a"]
    assert loader.calls == ["a"]


def test_missing_key_raises(d):
    with pytest.raises(KeyError):
        d["c"]
    with pytest.raises(AttributeError):
        d.c


def test_get_pop_setdefault_load(d, loader):
    assert d.get("a") is d["a"]
    assert d.get("c", 5) == 5
    assert isinstance(d.setdefault("b", None), pd.DataFrame)
    assert loader.calls == ["a", "b"]
    popped = d.pop("a")
    assert isinstance(popped, pd.DataFrame)
    assert "a" not in d


def test_setitem_on_pending_does_not_load(d, loader):
    d["a"] = 1
    assert d["a"] == 1
    assert loader.calls == []
    assert d.pending == frozenset({"b"})


def test_delitem_on_pending_does_not_load(d, loader):
    del d["a"]
    assert "a" not in d
    assert list(d) == ["b"]
    assert loader.calls == []
    with pytest.raises(KeyError):
        del d["a"]


def test_update_clear_popitem(d, loader):
    d.update({"a": 1}, c=3)
    assert loader.calls == []
    assert d.pending == frozenset({"b"})
    d |= {"b": 2}
    assert d.pending == frozenset()
    assert dict.__len__(d) == 3
    d.clear()
    assert len(d) == 0
    assert d.pending == frozenset()

    e = LazyDict(load=Loader(), keys=["x", "y"])
    key, value = e.popitem()
    assert key == "y"
    assert isinstance(value, pd.DataFrame)
    assert list(e) == ["x"]


def test_iteration_order_loaded_first(d):
    d["c"] = 1
    assert list(d) == ["c", "a", "b"]
    assert list(reversed(d)) == ["b", "a", "c"]


def test_items_and_values_load_everything(d, loader):
    items = dict(d.items())
    assert set(items) == {"a", "b"}
    assert sorted(loader.calls) == ["a", "b"]
    assert d.pending == frozenset()
    assert len(list(d.values())) == 2


def test_dict_and_concat_see_every_value(loader):
    d = LazyDict(load=loader, keys=["a", "b"])
    plain = dict(d)
    assert set(plain) == {"a", "b"}
    assert all(isinstance(v, pd.DataFrame) for v in plain.values())

    d = LazyDict(load=loader, keys=["a", "b"])
    assert set({**d}) == {"a", "b"}

    d = LazyDict(load=loader, keys=["a", "b"])
    concat = pd.concat(d, axis=1)
    assert list(concat.columns.get_level_values(0)) == ["a", "b"]


def test_deepcopy_and_pickle_are_detached(d, loader):
    for other in (copy.deepcopy(d), pickle.loads(pickle.dumps(d))):
        assert type(other) is LazyDict
        assert other.pending == frozenset()
        assert other._load is None
        assert list(other) == ["a", "b"]
        for k in other:
            pd.testing.assert_frame_equal(other[k], d[k])


def test_copy_materializes(d):
    for other in (copy.copy(d), d.copy()):
        assert type(other) is LazyDict
        assert other.pending == frozenset()
        assert other["a"] is d["a"]


def test_loader_dropped_once_loaded(d):
    d.materialize()
    assert d.pending == frozenset()
    assert d._load is None


def test_repr_does_not_load(d, loader):
    d["c"] = 1
    text = repr(d)
    assert "'a': <not loaded>" in text
    assert "'c': 1" in text
    assert loader.calls == []


def test_eq_with_dict(loader):
    d = LazyDict(load=lambda k: k.upper(), keys=["a", "b"])
    assert d == Dict(a="A", b="B")
    assert Dict(a="A", b="B") == LazyDict(load=lambda k: k.upper(), keys=["a", "b"])
    assert d != Dict(a="A")
    assert LazyDict(load=lambda k: k.upper(), keys=["a"]) != Dict(a="B")


def test_or_materializes(loader):
    d = LazyDict(load=lambda k: k.upper(), keys=["a"])
    assert d | {"b": "B"} == {"a": "A", "b": "B"}
    assert {"b": "B"} | LazyDict(load=lambda k: k.upper(), keys=["a"]) == {
        "a": "A",
        "b": "B",
    }


def test_in_place_mutation_is_live(d):
    df = d["a"]
    df.loc[0, "a"] = 10.0
    df.index = ["x", "y"]
    assert d["a"] is df
    assert d["a"].loc["x", "a"] == 10.0


def test_dir_lists_pending_keys(d, loader):
    assert "a" in dir(d)
    assert loader.calls == []


def test_attribute_set_goes_to_item(d, loader):
    d.a = 1
    assert d["a"] == 1
    assert loader.calls == []


def test_empty_behaves_like_dict():
    d = LazyDict()
    assert len(d) == 0
    assert d.pending == frozenset()
    d["x"] = 1
    assert d.x == 1
    assert d == Dict(x=1)
    assert repr(d) == "{'x': 1}"


def test_positional_mapping_rejected():
    with pytest.raises(TypeError):
        LazyDict({"a": 1})
