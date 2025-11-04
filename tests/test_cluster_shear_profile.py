"""Tests for the cluster deltasigma module."""

import os
import sys

import numpy as np
import pyccl
import pytest
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crow.shear_profile import ClusterShearProfile


@pytest.fixture(name="cluster_deltasigma_profile")
def fixture_cluster_deltasigma_profile():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterShearProfile((13, 17), (0, 2), hmf, 4.0, True)
    ca.set_beta_parameters(10.0)
    return ca


@pytest.fixture(name="cluster_reduced_profile")
def fixture_cluster_reduced_profile():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterShearProfile((13, 17), (0, 2), hmf, 4.0, False)
    ca.set_beta_parameters(10.0)
    return ca


@pytest.fixture(name="cluster_reduced_interp_profile")
def fixture_cluster_reduced_interp_profile():
    """Test fixture that represents an assembled cluster deltasigma class."""
    hmf = pyccl.halos.MassFuncBocquet16()
    ca = ClusterShearProfile((13, 17), (0, 2), hmf, 4.0, False, True)
    ca.set_beta_parameters(10.0)
    ca.set_beta_s_interp(0, 2)
    return ca


def test_cluster_update_ingredients(
    cluster_deltasigma_profile: ClusterShearProfile,
    cluster_reduced_profile: ClusterShearProfile,
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma_profile.cosmo = cosmo
    cluster_reduced_profile.cosmo = cosmo

    for cluster in [cluster_deltasigma_profile, cluster_reduced_profile]:
        assert cluster.cosmo is not None
        assert cluster.cosmo == cosmo  # pylint: disable=protected-access
        assert cluster._hmf_cache == {}  # pylint: disable=protected-access


def test_cluster_deltasigma_profile_init(
    cluster_deltasigma_profile: ClusterShearProfile,
):
    assert cluster_deltasigma_profile is not None
    assert cluster_deltasigma_profile.cluster_concentration is not None
    assert cluster_deltasigma_profile.cosmo is None  # pylint: disable=protected-access
    assert (
        cluster_deltasigma_profile._hmf_cache == {}
    )  # pylint: disable=protected-access
    assert isinstance(
        cluster_deltasigma_profile.halo_mass_function, pyccl.halos.MassFuncBocquet16
    )
    assert cluster_deltasigma_profile.min_mass == 13.0
    assert cluster_deltasigma_profile.max_mass == 17.0
    assert cluster_deltasigma_profile.min_z == 0.0
    assert cluster_deltasigma_profile.max_z == 2.0


# ---- Helpers for repeated checks ----
def _check_delta_sigma_output(result):
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 5
    assert np.all(result > 0)


def _check_miscentering_behavior(
    cluster, log_mass, redshifts, radius, miscentering_frac, pdf=None
):
    """Shared logic for testing miscentering behavior."""
    baseline = cluster.delta_sigma(log_mass, redshifts, radius)
    cluster.set_miscentering(miscentering_frac, miscentering_distribution_function=pdf)
    result_mis = cluster.delta_sigma(log_mass, redshifts, radius)
    cluster.set_miscentering(0.0, miscentering_distribution_function=pdf)
    result_right_center = cluster.delta_sigma(log_mass, redshifts, radius)

    np.testing.assert_allclose(result_right_center, baseline, rtol=1e-12)
    assert result_mis.shape == baseline.shape
    assert np.all(result_mis <= baseline)
    assert np.all(result_mis >= 0)


def test_shear_profile_returns_value(
    cluster_deltasigma_profile: ClusterShearProfile,
    cluster_reduced_profile: ClusterShearProfile,
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    log_mass = np.linspace(13, 17, 5, dtype=np.float64)
    redshifts = np.linspace(0.1, 1, 5, dtype=np.float64)
    radius = 5.0

    for cluster in [cluster_deltasigma_profile, cluster_reduced_profile]:
        cluster.cosmo = cosmo
        result = cluster.delta_sigma(log_mass, redshifts, radius)
        _check_delta_sigma_output(result)


def test_shear_profile_returns_value_interp(
    cluster_reduced_interp_profile: ClusterShearProfile,
    cluster_reduced_profile: ClusterShearProfile,
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    log_mass = np.linspace(13, 17, 5, dtype=np.float64)
    redshifts = np.linspace(0.1, 1, 5, dtype=np.float64)
    radius = 5.0

    cluster_reduced_interp_profile.cosmo = cosmo
    result = cluster_reduced_profile.delta_sigma(
        log_mass,
        redshifts,
        radius,
    )
    _check_delta_sigma_output(result)
    cluster_reduced_profile.cosmo = cosmo
    result_exact = cluster_reduced_profile.delta_sigma(
        log_mass,
        redshifts,
        radius,
    )
    np.testing.assert_allclose(result, result_exact, rtol=1e-12)


def test_shear_profile_returns_value_twoh_boost(
    cluster_deltasigma_profile: ClusterShearProfile,
    cluster_reduced_profile: ClusterShearProfile,
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    log_mass = np.linspace(13, 17, 5, dtype=np.float64)
    redshifts = np.linspace(0.1, 1, 5, dtype=np.float64)
    radius = 5.0

    cluster_deltasigma_profile.cosmo = cosmo
    result = cluster_deltasigma_profile.delta_sigma(
        log_mass,
        redshifts,
        radius,
    )
    _check_delta_sigma_output(result)

    cluster_reduced_profile.cosmo = cosmo
    result = cluster_reduced_profile.delta_sigma(
        log_mass,
        redshifts,
        radius,
    )
    _check_delta_sigma_output(result)


@pytest.mark.slow
def test_shear_profile_miscentering(
    cluster_deltasigma_profile, cluster_reduced_profile
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    log_mass = np.linspace(13, 17, 5)
    redshifts = np.linspace(0.1, 1, 5)
    radius = 5.0
    miscentering_frac = 0.5

    for cluster in [cluster_deltasigma_profile, cluster_reduced_profile]:
        cluster.cosmo = cosmo
        _check_miscentering_behavior(
            cluster, log_mass, redshifts, radius, miscentering_frac
        )

    # Gaussian miscentering case (only for reduced profile)
    def gaussian_pdf(r_mis_list, mean=0.0, sigma=0.1):
        return norm.pdf(r_mis_list, loc=mean, scale=sigma)

    _check_miscentering_behavior(
        cluster_reduced_profile,
        log_mass,
        redshifts,
        radius,
        miscentering_frac,
        pdf=gaussian_pdf,
    )
