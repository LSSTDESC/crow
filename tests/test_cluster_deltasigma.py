"""Tests for the cluster deltasigma module."""

import os
import sys

import numpy as np
import pyccl
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crow.deltasigma import ClusterShearProfile


@pytest.fixture(name="cluster_shear_profile")
def fixture_cluster_shear_profile():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterShearProfile((13, 17), (0, 2), hmf, 4.0)
    ca.set_beta_parameters(5.0)
    return ca


def test_cluster_update_ingredients(cluster_shear_profile: ClusterShearProfile):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_shear_profile.cosmo = cosmo
    assert cluster_shear_profile.cosmo is not None
    assert cluster_shear_profile.cosmo == cosmo
    # pylint: disable=protected-access
    assert cluster_shear_profile._hmf_cache == {}


def test_cluster_shear_profile_init(cluster_shear_profile: ClusterShearProfile):
    assert cluster_shear_profile is not None
    assert cluster_shear_profile.cluster_concentration is not None
    assert cluster_shear_profile.cosmo is None
    # pylint: disable=protected-access
    assert cluster_shear_profile._hmf_cache == {}
    assert isinstance(
        cluster_shear_profile.halo_mass_function, pyccl.halos.MassFuncBocquet16
    )
    assert cluster_shear_profile.min_mass == 13.0
    assert cluster_shear_profile.max_mass == 17.0
    assert cluster_shear_profile.min_z == 0.0
    assert cluster_shear_profile.max_z == 2.0


def test_deltasigma_profile_returns_value(cluster_shear_profile: ClusterShearProfile):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_shear_profile.cosmo = cosmo

    result = cluster_shear_profile.delta_sigma(
        np.linspace(13, 17, 5, dtype=np.float64),
        np.linspace(0.1, 1, 5, dtype=np.float64),
        5.0,
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)
