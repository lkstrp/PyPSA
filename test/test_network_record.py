# SPDX-FileCopyrightText: PyPSA Contributors
#
# SPDX-License-Identifier: MIT

import copy
import pickle

import pytest

import pypsa


def test_record_is_none_on_fresh_network():
    assert pypsa.Network().record is None


def test_record_dropped_by_deepcopy_and_pickle():
    n = pypsa.Network()
    rec = object()
    n._record = rec
    assert copy.deepcopy(n).record is None
    assert pickle.loads(pickle.dumps(n)).record is None
    assert n.record is rec
    assert copy.copy(n).record is rec


def test_pickle_without_record_attribute_restores():
    n = pypsa.Network()
    del n._record
    state = n.__getstate__()
    state.pop("_record", None)
    fresh = pypsa.Network.__new__(pypsa.Network)
    fresh.__setstate__(state)
    assert fresh.record is None


def test_record_has_no_setter():
    with pytest.raises(AttributeError):
        pypsa.Network().record = object()
