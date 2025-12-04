"""Tests for the cluster abundance module."""

import os
import sys

import numpy as np
import pyccl
import pytest


from crow import ClusterAbundance

_TEST_COSMO = pyccl.CosmologyVanillaLCDM()


@pytest.fixture(name="cluster_abundance")
def fixture_cluster_abundance():
    """Test fixture that represents an assembled cluster abundance class."""
    return ClusterAbundance(_TEST_COSMO, pyccl.halos.MassFuncBocquet16())


def test_cluster_abundance_init(cluster_abundance: ClusterAbundance):
    assert cluster_abundance is not None
    assert cluster_abundance.cosmo == _TEST_COSMO
    # pylint: disable=protected-access
    assert cluster_abundance._hmf_cache == {}
    assert isinstance(
        cluster_abundance.halo_mass_function, pyccl.halos.MassFuncBocquet16
    )


def test_cluster_update_ingredients(cluster_abundance: ClusterAbundance):
    # pylint: disable=protected-access
    assert cluster_abundance._hmf_cache == {}


def test_abundance_comoving_returns_value(cluster_abundance: ClusterAbundance):
    result = cluster_abundance.comoving_volume(
        np.linspace(0.1, 1, 10, dtype=np.float64), 360**2
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 10
    assert np.all(result > 0)


# @pytest.mark.slow
def test_abundance_massfunc_returns_value(cluster_abundance: ClusterAbundance):
    result = cluster_abundance.mass_function(
        np.linspace(13, 17, 5, dtype=np.float64),
        np.linspace(0.1, 1, 5, dtype=np.float64),
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)
