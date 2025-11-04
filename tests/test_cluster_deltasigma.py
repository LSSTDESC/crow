"""Tests for the cluster deltasigma module."""

import os
import sys

import numpy as np
import pyccl
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crow.deltasigma import ClusterDeltaSigma


@pytest.fixture(name="cluster_deltasigma")
def fixture_cluster_deltasigma_deltasigma():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterDeltaSigma((13, 17), (0, 2), hmf, 4.0, True)
    ca.set_beta_parameters(10.0)
    return ca

@pytest.fixture(name="cluster_reduced")
def fixture_cluster_deltasigma_reduced():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterDeltaSigma((13, 17), (0, 2), hmf, 4.0, False)
    ca.set_beta_parameters(10.0)
    return ca

@pytest.fixture(name="cluster_reduced_interp")
def fixture_cluster_deltasigma_reduced_interp():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterDeltaSigma((13, 17), (0, 2), hmf, 4.0, False, True)
    ca.set_beta_parameters(10.0)
    return ca

def test_cluster_update_ingredients(cluster_deltasigma: ClusterDeltaSigma, cluster_reduced: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.cosmo = cosmo
    cluster_reduced.cosmo = cosmo
    assert cluster_deltasigma.cosmo is not None
    assert cluster_deltasigma.cosmo == cosmo
    # pylint: disable=protected-access
    assert cluster_deltasigma._hmf_cache == {}

    assert cluster_reduced.cosmo is not None
    assert cluster_reduced.cosmo == cosmo
    # pylint: disable=protected-access
    assert cluster_reduced._hmf_cache == {}


def test_cluster_deltasigma_init(cluster_deltasigma: ClusterDeltaSigma):
    assert cluster_deltasigma is not None
    assert cluster_deltasigma.cluster_concentration is not None
    assert cluster_deltasigma.cosmo is None
    # pylint: disable=protected-access
    assert cluster_deltasigma._hmf_cache == {}
    assert isinstance(
        cluster_deltasigma.halo_mass_function, pyccl.halos.MassFuncBocquet16
    )
    assert cluster_deltasigma.min_mass == 13.0
    assert cluster_deltasigma.max_mass == 17.0
    assert cluster_deltasigma.min_z == 0.0
    assert cluster_deltasigma.max_z == 2.0


def test_deltasigma_profile_returns_value(cluster_deltasigma: ClusterDeltaSigma, cluster_reduced: ClusterDeltaSigma):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.cosmo = cosmo

    result = cluster_deltasigma.delta_sigma(
        np.linspace(13, 17, 5, dtype=np.float64),
        np.linspace(0.1, 1, 5, dtype=np.float64),
        5.0,
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)

    cluster_reduced.cosmo = cosmo

    result = cluster_reduced.delta_sigma(
        np.linspace(13, 17, 5, dtype=np.float64),
        np.linspace(0.1, 1, 5, dtype=np.float64),
        5.0,
    )
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)


def test_deltasigma_profile_miscentering(cluster_deltasigma, cluster_reduced):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma.cosmo = cosmo
    cluster_reduced.cosmo = cosmo
    
    log_mass = np.linspace(13, 17, 5)
    redshifts = np.linspace(0.1, 1, 5)
    radius = 5.0
    miscentering_frac = 0.5

    baseline = cluster_deltasigma.delta_sigma(log_mass, redshifts, radius)
    cluster_deltasigma.set_miscentering(miscentering_frac)
    result_mis = cluster_deltasigma.delta_sigma(log_mass, redshifts, radius)
    cluster_deltasigma.set_miscentering(0.0)
    result_right_center = cluster_deltasigma.delta_sigma(log_mass, redshifts, radius)
    np.testing.assert_allclose(result_right_center, baseline, rtol=1e-12)
    assert result_mis.shape == baseline.shape
    assert np.all(result_mis <= baseline)
    assert np.all(result_mis >= 0)

    baseline = cluster_reduced.delta_sigma(log_mass, redshifts, radius)
    cluster_deltasigma.set_miscentering(miscentering_frac)
    result_mis = cluster_reduced.delta_sigma(log_mass, redshifts, radius)
    cluster_reduced.set_miscentering(0.0)
    result_right_center = cluster_reduced.delta_sigma(log_mass, redshifts, radius)
    assert result_right_center == baseline
    assert result_mis is not None
    assert result_mis <= baseline
