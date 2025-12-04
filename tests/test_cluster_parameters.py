"""Tests for the cluster kernel module."""

import os
import sys

import numpy as np
import pytest


from crow.cluster_modules.parameters import Parameters


def is_par_eq_dict(params, test_dict):
    for p_test, p_ref in zip(params, test_dict):
        assert p_test == p_ref
    for key_test, key_ref in zip(params.keys(), test_dict.keys()):
        assert key_test == key_ref
    for value_test, value_ref in zip(params.values(), test_dict.values()):
        assert value_test == value_ref
    for item_test, item_ref in zip(params.items(), test_dict.items()):
        assert item_test == item_ref


def test_init_parameters():
    _test_input = {"a": 1, "b": 2}
    params = Parameters(_test_input)
    is_par_eq_dict(params, _test_input)
    np.testing.assert_raises(KeyError, params.__setitem__, "x", None)


def test_update_parameters():
    _test_input = {"a": 1, "b": 2}
    params = Parameters(_test_input)

    # manual update
    _test_input2 = {"a": 3, "b": 4}
    for key, value in _test_input2.items():
        params[key] = value
    is_par_eq_dict(params, _test_input2)

    # update function
    _test_input2 = {"a": 5, "b": 6}
    params.update(_test_input2)
    is_par_eq_dict(params, _test_input2)

    # failsafes
    np.testing.assert_raises(ValueError, params.update, None)
    _bad_input = {"a": 100, "x": None}
    np.testing.assert_raises(KeyError, params.update, {"x": None})

    # make sure no value was updated
    is_par_eq_dict(params, _test_input2)
