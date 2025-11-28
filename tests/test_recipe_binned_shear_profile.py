"""Tests for the cluster delta sigma module."""

import os
import sys

import numpy as np
import pyccl
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crow import (
    ClusterAbundance,
    ClusterShearProfile,
    completeness_models,
    kernel,
    mass_proxy,
    purity_models,
)
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.properties import ClusterProperty
from crow.recipes.binned_exact import ExactBinnedClusterRecipe
from crow.recipes.binned_grid import GridBinnedClusterRecipe

# from firecrown.models.cluster import ClusterProperty


def get_base_binned_exact(
    completeness, purity, is_delta_sigma
) -> ExactBinnedClusterRecipe:
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    cluster_theory = ClusterShearProfile(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        cluster_concentration=4.0,
        is_delta_sigma=is_delta_sigma,
    )
    cluster_theory.set_beta_parameters(10.0)
    cluster_recipe = ExactBinnedClusterRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=mass_proxy.MurataBinned(pivot_mass, pivot_redshift),
        completeness=completeness,
        purity=purity,
        mass_interval=(13, 17),
        true_z_interval=(0, 2),
    )
    cluster_recipe.mass_distribution.parameters["mu0"] = 3.0
    cluster_recipe.mass_distribution.parameters["mu1"] = 0.86
    cluster_recipe.mass_distribution.parameters["mu2"] = 0.0
    cluster_recipe.mass_distribution.parameters["sigma0"] = 3.0
    cluster_recipe.mass_distribution.parameters["sigma1"] = 0.7
    cluster_recipe.mass_distribution.parameters["sigma2"] = 0.0
    return cluster_recipe


def get_base_binned_grid(
    completeness, purity, is_delta_sigma
) -> GridBinnedClusterRecipe:
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    cluster_theory = ClusterShearProfile(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        cluster_concentration=4.0,
        is_delta_sigma=is_delta_sigma,
    )
    cluster_theory.set_beta_parameters(10.0)
    cluster_recipe = GridBinnedClusterRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=mass_proxy.MurataUnbinned(pivot_mass, pivot_redshift),
        completeness=completeness,
        purity=purity,
        mass_interval=(13, 17),
        true_z_interval=(0, 2),
        redshift_points=20,
        log_mass_points=60,
        log_proxy_points=20,
    )
    cluster_recipe.mass_distribution.parameters["mu0"] = 3.0
    cluster_recipe.mass_distribution.parameters["mu1"] = 0.86
    cluster_recipe.mass_distribution.parameters["mu2"] = 0.0
    cluster_recipe.mass_distribution.parameters["sigma0"] = 3.0
    cluster_recipe.mass_distribution.parameters["sigma1"] = 0.7
    cluster_recipe.mass_distribution.parameters["sigma2"] = 0.0
    return cluster_recipe


@pytest.fixture(name="binned_exact_deltasigma")
def fixture_binned_exact() -> ExactBinnedClusterRecipe:
    return get_base_binned_exact(None, None, True)


@pytest.fixture(name="binned_grid_deltasigma")
def fixture_binned_grid() -> GridBinnedClusterRecipe:
    return get_base_binned_grid(None, None, True)


@pytest.fixture(name="binned_exact_gt")
def fixture_binned_exact_gt() -> ExactBinnedClusterRecipe:
    return get_base_binned_exact(None, None, False)


@pytest.fixture(name="binned_grid_gt")
def fixture_binned_grid_gt() -> GridBinnedClusterRecipe:
    return get_base_binned_grid(None, None, False)


def test_binned_exact_deltasigma_init(
    binned_exact_deltasigma: ExactBinnedClusterRecipe,
):

    assert binned_exact_deltasigma is not None
    assert isinstance(binned_exact_deltasigma, ExactBinnedClusterRecipe)
    assert binned_exact_deltasigma.integrator is not None
    assert isinstance(binned_exact_deltasigma.integrator, NumCosmoIntegrator)
    assert binned_exact_deltasigma.redshift_distribution is not None
    assert isinstance(
        binned_exact_deltasigma.redshift_distribution,
        kernel.SpectroscopicRedshift,
    )
    assert binned_exact_deltasigma.mass_distribution is not None
    assert isinstance(
        binned_exact_deltasigma.mass_distribution, mass_proxy.MurataBinned
    )


def test_binned_grid_deltasigma_init(
    binned_grid_deltasigma: GridBinnedClusterRecipe,
):

    assert binned_grid_deltasigma is not None
    assert isinstance(binned_grid_deltasigma, GridBinnedClusterRecipe)
    assert binned_grid_deltasigma.redshift_distribution is not None
    assert isinstance(
        binned_grid_deltasigma.redshift_distribution,
        kernel.SpectroscopicRedshift,
    )
    assert binned_grid_deltasigma.mass_distribution is not None
    assert isinstance(
        binned_grid_deltasigma.mass_distribution, mass_proxy.MurataUnbinned
    )


def test_get_theory_prediction_returns_value(
    binned_exact_deltasigma: ExactBinnedClusterRecipe,
):
    prediction_none = binned_exact_deltasigma._get_theory_prediction_shear_profile(
        average_on=None
    )
    prediction = binned_exact_deltasigma._get_theory_prediction_shear_profile(
        ClusterProperty.DELTASIGMA
    )
    prediction_c = binned_exact_deltasigma._get_theory_prediction_counts()

    assert prediction is not None
    assert prediction_c is not None
    assert callable(prediction)
    assert callable(prediction_c)

    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0.0, 5.0)
    sky_area = 360**2
    radius_center = 1.5
    binned_exact_deltasigma.cluster_theory.set_beta_s_interp(0.1, 1)
    with pytest.raises(
        ValueError,
        match=f"The property should be" f" {ClusterProperty.DELTASIGMA}.",
    ):
        result = prediction_none(mass, z, mass_proxy_limits, sky_area, radius_center)

    result = prediction(mass, z, mass_proxy_limits, sky_area, radius_center)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    result_c = prediction_c(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result_c, np.ndarray)
    assert np.issubdtype(result_c.dtype, np.float64)
    assert len(result_c) == 2
    assert np.all(result_c > 0)


def test_get_function_to_integrate_returns_value(
    binned_exact_deltasigma: ExactBinnedClusterRecipe,
):
    prediction = binned_exact_deltasigma._get_theory_prediction_shear_profile(
        ClusterProperty.DELTASIGMA
    )
    function_to_integrate = (
        binned_exact_deltasigma._get_function_to_integrate_shear_profile(prediction)
    )

    assert function_to_integrate is not None
    assert callable(function_to_integrate)

    int_args = np.array([[13.0, 0.1], [17.0, 1.0]])
    extra_args = np.array([0, 5, 360**2, 1.5])
    binned_exact_deltasigma.cluster_theory.set_beta_s_interp(0.1, 1)

    result = function_to_integrate(int_args, extra_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction_c = binned_exact_deltasigma._get_theory_prediction_counts()
    function_to_integrate = binned_exact_deltasigma._get_function_to_integrate_counts(
        prediction_c
    )

    assert function_to_integrate is not None
    assert callable(function_to_integrate)

    int_args = np.array([[13.0, 0.1], [17.0, 1.0]])
    extra_args = np.array([0, 5, 360**2])

    result = function_to_integrate(int_args, extra_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)


def test_evaluates_theory_prediction_returns_value(
    binned_exact_deltasigma: ExactBinnedClusterRecipe,
):

    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    radius_center = np.atleast_1d(1.5)
    average_on = ClusterProperty.DELTASIGMA

    prediction = binned_exact_deltasigma.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radius_center, 360**2, average_on
    )
    prediction_c = binned_exact_deltasigma.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, 360**2
    )
    assert prediction > 0
    assert prediction_c > 0


def test_grid_shear_matches_exact_within_tolerance(
    binned_exact_deltasigma: ExactBinnedClusterRecipe,
    binned_grid_deltasigma: GridBinnedClusterRecipe,
):
    """Compare grid evaluation to exact evaluation for the same inputs."""
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    radii = np.atleast_1d(1.5)
    sky_area = 360**2
    average_on = ClusterProperty.DELTASIGMA

    pred_exact = binned_exact_deltasigma.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )
    pred_grid = binned_grid_deltasigma.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )

    # Allow a modest relative tolerance for grid approximation
    rel_tol = 1.0e-4
    assert pred_exact.shape == pred_grid.shape
    # avoid division by zero
    denom = np.where(pred_exact == 0, 1.0, pred_exact)
    assert np.all(np.abs((pred_grid - pred_exact) / denom) < rel_tol)


def test_grid_reduced_shear_matches_exact_within_tolerance(
    binned_exact_gt: ExactBinnedClusterRecipe,
    binned_grid_gt: GridBinnedClusterRecipe,
):
    """Compare grid evaluation to exact evaluation for the same inputs."""
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    radii = np.atleast_1d(1.5)
    sky_area = 360**2
    average_on = ClusterProperty.DELTASIGMA

    # set beta_s interp so delta sigma calculation is available
    binned_exact_gt.cluster_theory.set_beta_s_interp(0.1, 1)
    binned_grid_gt.cluster_theory.set_beta_s_interp(0.1, 1)

    pred_exact = binned_exact_gt.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )
    pred_grid = binned_grid_gt.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )

    # Allow a modest relative tolerance for grid approximation
    rel_tol = 1.0e-4
    assert pred_exact.shape == pred_grid.shape
    # avoid division by zero
    denom = np.where(pred_exact == 0, 1.0, pred_exact)
    assert np.all(np.abs((pred_grid - pred_exact) / denom) < rel_tol)


def test_shear_grid_shape_and_cache_behavior(
    binned_grid_deltasigma: GridBinnedClusterRecipe,
):
    """Test shape of internal shear grid and that caching is used."""
    binned_grid_deltasigma.setup()  # clear caches
    z_points = np.linspace(0.1, 1.0, 5)
    log_mass_poits = np.array([14.0, 15.0])
    radius_centers = np.atleast_1d(1.0)
    shear_key = tuple(z_points)

    # call _get_shear_grid to populate cache
    grid = binned_grid_deltasigma._get_shear_grid(
        z_points, log_mass_poits, radius_centers, shear_key
    )
    # per implementation: shape is (n_z, n_mass, n_radius)
    n_z = len(z_points)
    n_m = len(log_mass_poits)
    n_r = len(radius_centers)
    assert grid.shape == (n_z, n_m, n_r)
    assert shear_key in binned_grid_deltasigma._shear_grids

    # Overwrite cache and ensure retrieval uses stored value (cache hit)
    binned_grid_deltasigma._shear_grids[shear_key] = np.zeros_like(grid)
    recalled = binned_grid_deltasigma._get_shear_grid(
        z_points, log_mass_poits, radius_centers, shear_key
    )
    assert np.all(recalled == 0.0)


def test_shear_positivity_and_monotonicity_with_radius(
    binned_grid_deltasigma: GridBinnedClusterRecipe,
):
    """Shear should be positive and (typically) non-increasing with radius."""
    z_edges = (0.5, 0.6)
    mass_proxy_edges = (2, 5)
    radii = np.array([0.5, 1.5, 3.0])  # increasing radii
    sky_area = 360**2
    average_on = ClusterProperty.DELTASIGMA

    # set beta interpolation for both recipes if needed
    binned_grid_deltasigma.cluster_theory.set_beta_s_interp(0.1, 1)
    shear_vals = binned_grid_deltasigma.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )

    # Result should be positive for all radii
    assert np.all(shear_vals > 0)

    # Check monotonic non-increasing behavior across radii (for the averaged profile)
    # We allow non-strict monotonicity and small numerical noise
    diffs = np.diff(shear_vals)
    assert np.all(diffs <= 1e-6 + np.abs(shear_vals[:-1]) * 1e-6)


def test_integration_vectorization_multiple_radii(
    binned_grid_deltasigma: GridBinnedClusterRecipe,
):
    """Evaluate shear for multiple radii and ensure vectorized output shape matches."""
    z_edges = (0.5, 1.0)
    mass_proxy_edges = (2, 5)
    radii = np.linspace(0.5, 2.5, 5)
    sky_area = 360**2
    average_on = ClusterProperty.DELTASIGMA

    binned_grid_deltasigma.cluster_theory.set_beta_s_interp(0.1, 1)
    shear_vals = binned_grid_deltasigma.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )

    # should return an array with one entry per radius
    assert isinstance(shear_vals, np.ndarray)
    assert shear_vals.shape == (len(radii),)
    assert np.all(np.isfinite(shear_vals))


def test_shear_respects_completeness_and_purity_effects(
    binned_grid_deltasigma: GridBinnedClusterRecipe,
):
    """Basic sanity checks: enabling completeness should not increase counts/shear;
    purity behavior checked for no blow-ups."""
    base = binned_grid_deltasigma
    comp_dist = completeness_models.CompletenessAguena16()
    pur_dist = purity_models.PurityAguena16()
    with_comp = get_base_binned_grid(comp_dist, None, True)
    print("with_comp:", with_comp)
    with_pur = get_base_binned_grid(None, pur_dist, True)
    z_edges = (0.5, 0.8)
    mass_proxy_edges = (2, 5)
    radii = np.atleast_1d(1.5)
    sky_area = 360**2
    average_on = ClusterProperty.DELTASIGMA

    base_val = base.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )
    comp_val = with_comp.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )
    pur_val = with_pur.evaluate_theory_prediction_shear_profile(
        z_edges, mass_proxy_edges, radii, sky_area, average_on
    )

    # completeness should not increase the predicted average shear (it reduces effective counts/kernel)
    assert comp_val <= base_val + 1e-12

    # purity model usually acts to reduce contamination; behavior depends on model details,
    # but it should produce finite, non-negative numbers (sanity)
    assert np.isfinite(pur_val)
    assert pur_val >= 0.0
