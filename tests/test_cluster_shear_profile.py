"""Tests for the cluster deltasigma module."""

import os
import sys

import numpy as np
import pyccl
import pytest
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crow.shear_profile import ClusterShearProfile

_TEST_COSMO = pyccl.CosmologyVanillaLCDM()


@pytest.fixture(name="cluster_deltasigma_profile")
def fixture_cluster_deltasigma_profile():
    """Test fixture that represents an assembled cluster deltasigma class."""
    delta_sigma = ClusterShearProfile(
        _TEST_COSMO,
        pyccl.halos.MassFuncBocquet16(),
        None,  # completeness
        4.0,  # concentration
        True,  # is_delta_sigma
    )
    delta_sigma.set_beta_parameters(10.0)
    return delta_sigma


@pytest.fixture(name="cluster_reduced_profile")
def fixture_cluster_reduced_profile():
    """Test fixture that represents an assembled cluster deltasigma class."""
    gt = ClusterShearProfile(
        _TEST_COSMO,
        pyccl.halos.MassFuncBocquet16(),
        None,  # completeness
        4.0,  # concentration
        False,  # is_delta_sigma
    )
    gt.set_beta_parameters(10.0)
    return gt


@pytest.fixture(name="cluster_reduced_interp_profile")
def fixture_cluster_reduced_interp_profile():
    """Test fixture that represents an assembled cluster deltasigma class."""
    gt = ClusterShearProfile(
        _TEST_COSMO,
        pyccl.halos.MassFuncBocquet16(),
        None,  # completeness
        4.0,  # concentration
        False,  # is_delta_sigma
        True,  # use_beta_s_interp
    )
    gt.set_beta_parameters(10.0)
    gt.set_beta_s_interp(0, 2)
    return gt


def test_cluster_update_ingredients(
    cluster_deltasigma_profile: ClusterShearProfile,
    cluster_reduced_profile: ClusterShearProfile,
):
    cosmo = pyccl.CosmologyVanillaLCDM()
    cluster_deltasigma_profile.cosmo = cosmo
    cluster_reduced_profile.cosmo = cosmo

    for cluster in [cluster_deltasigma_profile, cluster_reduced_profile]:
        assert cluster.cosmo is not None
        assert cluster.cosmo == _TEST_COSMO  # pylint: disable=protected-access
        assert cluster._hmf_cache == {}  # pylint: disable=protected-access


def test_cluster_deltasigma_profile_init(
    cluster_deltasigma_profile: ClusterShearProfile,
):
    assert cluster_deltasigma_profile is not None
    assert cluster_deltasigma_profile.cluster_concentration is not None
    # pylint: disable=protected-access
    assert (
        cluster_deltasigma_profile._hmf_cache == {}
    )  # pylint: disable=protected-access
    assert isinstance(
        cluster_deltasigma_profile.halo_mass_function, pyccl.halos.MassFuncBocquet16
    )


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
    cluster.miscentering_parameters = None
    cluster.set_beta_parameters(10.0, approx="order1")
    baseline = cluster.compute_shear_profile(log_mass, redshifts, radius)

    cluster.set_beta_parameters(10.0, approx="order2")
    baseline2 = cluster.compute_shear_profile(log_mass, redshifts, radius)

    cluster.set_miscentering(miscentering_frac, miscentering_distribution_function=pdf)
    cluster.set_beta_parameters(10.0, approx="order1")
    result_mis = cluster.compute_shear_profile(log_mass, redshifts, radius)

    cluster.set_miscentering(0.0, miscentering_distribution_function=pdf)
    result_right_center = cluster.compute_shear_profile(log_mass, redshifts, radius)
    np.testing.assert_allclose(result_right_center, baseline, rtol=1e-12)

    cluster.set_miscentering(miscentering_frac, miscentering_distribution_function=pdf)
    cluster.set_beta_parameters(10.0, approx="order2")
    result_mis_order2 = cluster.compute_shear_profile(log_mass, redshifts, radius)

    assert result_mis.shape == baseline.shape
    assert np.all(result_mis <= baseline)
    assert np.all(result_mis >= 0)
    assert np.all(result_mis_order2 <= baseline2)
    assert np.all(result_mis_order2 >= 0)


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
        result = cluster.compute_shear_profile(log_mass, redshifts, radius)
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
    result = cluster_reduced_profile.compute_shear_profile(
        log_mass,
        redshifts,
        radius,
    )
    _check_delta_sigma_output(result)
    cluster_reduced_profile.cosmo = cosmo
    result_exact = cluster_reduced_profile.compute_shear_profile(
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
    result = cluster_deltasigma_profile.compute_shear_profile(
        log_mass,
        redshifts,
        radius,
    )
    _check_delta_sigma_output(result)

    cluster_reduced_profile.cosmo = cosmo
    result = cluster_reduced_profile.compute_shear_profile(
        log_mass,
        redshifts,
        radius,
    )
    _check_delta_sigma_output(result)


def test_shear_profile_miscentering_fast(cluster_reduced_profile):
    cosmo = pyccl.CosmologyVanillaLCDM()
    log_mass = np.linspace(13, 17, 1)
    redshifts = np.linspace(0.1, 1, 1)
    radius = 5.0
    miscentering_frac = 0.5

    cluster_reduced_profile.cosmo = cosmo
    _check_miscentering_behavior(
        cluster_reduced_profile, log_mass, redshifts, radius, miscentering_frac
    )


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
