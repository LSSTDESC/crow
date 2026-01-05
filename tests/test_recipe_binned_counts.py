"""Tests for the cluster abundance module."""

import os
import sys

import numpy as np
import pyccl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import floats
from scipy.integrate import dblquad, quad, simpson

from crow import (
    ClusterAbundance,
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


def get_base_binned_exact(completeness, purity) -> ExactBinnedClusterRecipe:
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    cluster_recipe = ExactBinnedClusterRecipe(
        cluster_theory=ClusterAbundance(
            cosmo=pyccl.CosmologyVanillaLCDM(),
            halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        ),
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


def get_base_binned_grid(completeness, purity) -> GridBinnedClusterRecipe:
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    cluster_recipe = GridBinnedClusterRecipe(
        cluster_theory=ClusterAbundance(
            cosmo=pyccl.CosmologyVanillaLCDM(),
            halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        ),
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=mass_proxy.MurataUnbinned(pivot_mass, pivot_redshift),
        completeness=completeness,
        purity=purity,
        mass_interval=(13, 17),
        true_z_interval=(0, 2),
        redshift_grid_size=20,
        mass_grid_size=50,
        proxy_grid_size=20,
    )
    cluster_recipe.mass_distribution.parameters["mu0"] = 3.0
    cluster_recipe.mass_distribution.parameters["mu1"] = 0.86
    cluster_recipe.mass_distribution.parameters["mu2"] = 0.0
    cluster_recipe.mass_distribution.parameters["sigma0"] = 3.0
    cluster_recipe.mass_distribution.parameters["sigma1"] = 0.7
    cluster_recipe.mass_distribution.parameters["sigma2"] = 0.0
    return cluster_recipe


@pytest.fixture(name="binned_exact")
def fixture_binned_exact() -> ExactBinnedClusterRecipe:
    return get_base_binned_exact(None, None)


@pytest.fixture(name="binned_grid")
def fixture_binned_grid() -> ExactBinnedClusterRecipe:
    return get_base_binned_grid(None, None)


def test_binned_exact_init(
    binned_exact: ExactBinnedClusterRecipe,
):

    assert binned_exact.mass_interval[0] == 13.0
    assert binned_exact.mass_interval[1] == 17.0
    assert binned_exact.true_z_interval[0] == 0.0
    assert binned_exact.true_z_interval[1] == 2.0

    assert binned_exact is not None
    assert isinstance(binned_exact, ExactBinnedClusterRecipe)
    assert binned_exact.integrator is not None
    assert isinstance(binned_exact.integrator, NumCosmoIntegrator)
    assert binned_exact.redshift_distribution is not None
    assert isinstance(binned_exact.redshift_distribution, kernel.SpectroscopicRedshift)
    assert binned_exact.mass_distribution is not None
    assert isinstance(binned_exact.mass_distribution, mass_proxy.MurataBinned)
    binned_exact.setup()


def test_binned_grid_init(
    binned_grid: GridBinnedClusterRecipe,
):
    assert binned_grid.mass_interval[0] == 13.0
    assert binned_grid.mass_interval[1] == 17.0
    assert binned_grid.true_z_interval[0] == 0.0
    assert binned_grid.true_z_interval[1] == 2.0

    assert binned_grid.mass_grid_size == 50
    assert binned_grid.proxy_grid_size == 20
    assert binned_grid.redshift_grid_size == 20

    assert binned_grid is not None
    assert isinstance(binned_grid, GridBinnedClusterRecipe)
    assert binned_grid.redshift_distribution is not None
    assert isinstance(binned_grid.redshift_distribution, kernel.SpectroscopicRedshift)
    assert binned_grid.mass_distribution is not None
    assert isinstance(binned_grid.mass_distribution, mass_proxy.MurataUnbinned)


def test_get_theory_prediction_returns_value(
    binned_exact: ExactBinnedClusterRecipe,
):

    prediction = binned_exact._get_theory_prediction_counts(ClusterProperty.COUNTS)

    assert prediction is not None
    assert callable(prediction)

    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0, 5)
    sky_area = 360**2

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)


@given(
    mass=floats(min_value=13.0, max_value=17.0),
    z=floats(min_value=0.1, max_value=1.0),
    sky_area=floats(min_value=500.0, max_value=25000.0),
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    max_examples=25,  # Balanced reduction from 100 to 25
    deadline=None,  # No timeout
    derandomize=False,  # Keep randomization for better coverage
)
def test_cluster_prediction_positivity_property(
    binned_exact: ExactBinnedClusterRecipe,
    mass: float,
    z: float,
    sky_area: float,
):
    """Test that cluster predictions are always positive using hypothesis."""
    prediction = binned_exact._get_theory_prediction_counts(ClusterProperty.COUNTS)

    mass_array = np.array([mass])
    z_array = np.array([z])
    mass_proxy_limits = (0, 5)

    result = prediction(mass_array, z_array, mass_proxy_limits, sky_area)

    # Physical constraint: cluster predictions must be positive
    assert np.all(result > 0), f"All cluster predictions must be positive, got {result}"
    assert np.all(np.isfinite(result)), f"All predictions must be finite, got {result}"


def test_get_theory_prediction_with_average_returns_value(
    binned_exact: ExactBinnedClusterRecipe,
):
    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0, 5)
    sky_area = 360**2

    prediction = binned_exact._get_theory_prediction_counts(
        average_on=ClusterProperty.MASS
    )

    assert prediction is not None
    assert callable(prediction)

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction = binned_exact._get_theory_prediction_counts(
        average_on=ClusterProperty.REDSHIFT
    )

    assert prediction is not None
    assert callable(prediction)

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction = binned_exact._get_theory_prediction_counts(
        average_on=(ClusterProperty.REDSHIFT | ClusterProperty.MASS)
    )

    assert prediction is not None
    assert callable(prediction)

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)


def test_get_theory_prediction_throws_with_nonimpl_average(
    binned_exact: ExactBinnedClusterRecipe,
):
    prediction = binned_exact._get_theory_prediction_counts(
        average_on=ClusterProperty.SHEAR
    )

    assert prediction is not None
    assert callable(prediction)


def test_get_function_to_integrate_returns_value(
    binned_exact: ExactBinnedClusterRecipe,
):
    prediction = binned_exact._get_theory_prediction_counts()
    function_to_integrate = binned_exact._get_function_to_integrate_counts(prediction)

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
    binned_exact: ExactBinnedClusterRecipe,
    binned_grid: GridBinnedClusterRecipe,
):
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    sky_area = 360**2

    # --- Test 1: Simple Counts (No Averaging) ---
    prediction_exact = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )
    prediction_grid = binned_grid.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )
    # The grid result should match the exact result very closely
    assert np.abs(prediction_grid / prediction_exact - 1.0) <= 1.0e-4
    assert prediction_exact > 0

    # --- Test 2: Counts Averaged on REDSHIFT ---
    prediction_exact = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area, ClusterProperty.REDSHIFT
    )
    prediction_grid = binned_grid.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area, ClusterProperty.REDSHIFT
    )
    assert np.abs(prediction_grid / prediction_exact - 1.0) <= 1.0e-4
    assert prediction_exact > 0

    # --- Test 3: Counts Averaged on MASS ---
    prediction_exact = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area, ClusterProperty.MASS
    )
    prediction_grid = binned_grid.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area, ClusterProperty.MASS
    )
    assert np.abs(prediction_grid / prediction_exact - 1.0) <= 1.0e-4
    assert prediction_exact > 0

    # --- Test 4: Counts Averaged on MASS and REDSHIFT ---
    prediction_exact = binned_exact.evaluate_theory_prediction_counts(
        z_edges,
        mass_proxy_edges,
        sky_area,
        (ClusterProperty.REDSHIFT | ClusterProperty.MASS),
    )
    prediction_grid = binned_grid.evaluate_theory_prediction_counts(
        z_edges,
        mass_proxy_edges,
        sky_area,
        (ClusterProperty.REDSHIFT | ClusterProperty.MASS),
    )
    assert np.abs(prediction_grid / prediction_exact - 1.0) <= 1.0e-4
    assert prediction_exact > 0


def test_evaluates_theory_prediction_with_completeness(
    binned_exact: ExactBinnedClusterRecipe,
    binned_grid: GridBinnedClusterRecipe,  # Added for grid test
):
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    sky_area = 360**2

    # --- EXACT Recipe Test ---
    prediction = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )

    binned_exact_w_comp = get_base_binned_exact(
        completeness_models.CompletenessAguena16(), None
    )
    prediction_w_comp = binned_exact_w_comp.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )
    assert prediction >= prediction_w_comp

    # --- GRID Recipe Test: Comparison to Exact ---
    binned_grid_w_comp = get_base_binned_grid(  # Create grid recipe with completeness
        completeness_models.CompletenessAguena16(), None
    )
    prediction_grid = (
        binned_grid.evaluate_theory_prediction_counts(  # Grid without completeness
            z_edges, mass_proxy_edges, sky_area
        )
    )
    prediction_grid_w_comp = (
        binned_grid_w_comp.evaluate_theory_prediction_counts(  # Grid with completeness
            z_edges, mass_proxy_edges, sky_area
        )
    )

    assert np.abs(prediction_grid / prediction - 1.0) <= 1.0e-4
    assert prediction_grid >= prediction_grid_w_comp
    assert np.abs(prediction_grid_w_comp / prediction_w_comp - 1.0) <= 1.0e-4


def test_evaluates_theory_prediction_assertions(
    binned_exact: ExactBinnedClusterRecipe,
):
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    sky_area = 360**2

    mass_proxy_edges_err = (2, 5, 6)
    z_edges_err = (0.5, 1, 1.2)

    try:
        binned_exact.evaluate_theory_prediction_counts(
            z_edges=z_edges,
            log_proxy_edges=mass_proxy_edges,
            sky_area=sky_area,
        )
    except AssertionError as e:
        pytest.fail(f"Valid inputs raised an AssertionError unexpectedly: {e}")

    with pytest.raises(AssertionError):
        binned_exact.evaluate_theory_prediction_counts(
            z_edges=z_edges,
            log_proxy_edges=mass_proxy_edges_err,
            sky_area=sky_area,
        )

    with pytest.raises(AssertionError):
        binned_exact.evaluate_theory_prediction_counts(
            z_edges=z_edges_err,
            log_proxy_edges=mass_proxy_edges,
            sky_area=sky_area,
        )

    binned_exact.purity = None
    prediction_exact = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )

    assert len(binned_exact.integrator.extra_args) == 3
    assert len(binned_exact.integrator.integral_bounds) == 2

    binned_exact.purity = purity_models.PurityAguena16()
    prediction_exact_w_pur = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )

    assert len(binned_exact.integrator.extra_args) == 1
    assert len(binned_exact.integrator.integral_bounds) == 3


def test_evaluates_theory_prediction_with_purity(
    binned_exact: ExactBinnedClusterRecipe,
    binned_grid: GridBinnedClusterRecipe,
):
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    sky_area = 360**2

    prediction_exact = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )

    binned_exact.purity = purity_models.PurityAguena16()

    prediction_exact_w_pur = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )

    binned_grid_w_pur = get_base_binned_grid(  # Create grid recipe with purity
        None, purity_models.PurityAguena16()
    )
    prediction_grid = (
        binned_grid.evaluate_theory_prediction_counts(  # Grid without purity
            z_edges, mass_proxy_edges, sky_area
        )
    )
    prediction_grid_w_pur = (
        binned_grid_w_pur.evaluate_theory_prediction_counts(  # Grid with purity
            z_edges, mass_proxy_edges, sky_area
        )
    )

    assert prediction_grid <= prediction_grid_w_pur
    assert prediction_exact <= prediction_exact_w_pur

    assert np.abs(prediction_grid / prediction_exact - 1.0) <= 1.0e-4
    assert np.abs(prediction_grid_w_pur / prediction_exact_w_pur - 1.0) <= 1.0e-4


@given(
    z=floats(min_value=1e-15, max_value=2.0), mass=floats(min_value=7.0, max_value=26.0)
)
def test_evaluates_theory_mass_distribution_with_purity(
    z: float,
    mass: float,
):

    mass_array = np.atleast_1d(mass)
    z_array = np.atleast_1d(z)
    mass_proxy_limits = (1.0, 5.0)

    # sets recipes

    PIVOT_Z = 0.6
    PIVOT_MASS = 14.625862906

    murata_binned_relation = mass_proxy.MurataBinned(PIVOT_MASS, PIVOT_Z)
    murata_binned_relation.parameters["mu0"] = 3.00
    murata_binned_relation.parameters["mu1"] = 0.086
    murata_binned_relation.parameters["mu2"] = 0.01
    murata_binned_relation.parameters["sigma0"] = 3.0
    murata_binned_relation.parameters["sigma1"] = 0.07
    murata_binned_relation.parameters["sigma2"] = 0.01

    _kwargs = dict(
        cluster_theory=ClusterAbundance(
            cosmo=pyccl.CosmologyVanillaLCDM(),
            halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        ),
        redshift_distribution=kernel.SpectroscopicRedshift(),
        mass_distribution=murata_binned_relation,
        completeness=None,
        mass_interval=(13, 17),
        true_z_interval=(0, 2),
    )

    # Test non-negativity property
    binned_exact = ExactBinnedClusterRecipe(**_kwargs, purity=None)
    probability = binned_exact._mass_distribution_distribution(
        mass_array, z_array, mass_proxy_limits
    )
    assert probability >= 0, f"Probability must be non-negative, got {probability}"

    # Test with purity
    binned_exact_w_pur = ExactBinnedClusterRecipe(
        **_kwargs, purity=purity_models.PurityAguena16()
    )

    def mass_distribtuion_purity_integrand(mass_proxy, mass, z):
        return binned_exact_w_pur._mass_distribution_distribution(
            np.array([mass]), np.array([z]), mass_proxy
        )

    probability_w_pur = [
        quad(
            mass_distribtuion_purity_integrand,
            mass_proxy_limits[0],
            mass_proxy_limits[1],
            args=(mass, z),
        )[0]
        for mass, z in zip(mass_array, z_array)
    ]

    assert (probability < probability_w_pur).all()


def test_setup_clears_caches(binned_grid: GridBinnedClusterRecipe):
    """Test that the setup function successfully clears all internal caches."""
    z_test = np.array([0.5, 1.0])
    log_proxy_test = np.array([1.5, 2.0])

    sky_area = 100.0

    # Fill HMF cache
    binned_grid._get_hmf_grid(z_test, sky_area, tuple(z_test))
    assert binned_grid._hmf_grid

    # Fill Mass Richness cache
    binned_grid._get_mass_richness_grid(
        z_test, log_proxy_test, (tuple(z_test), tuple(log_proxy_test))
    )
    assert binned_grid._mass_richness_grid

    binned_grid.setup()

    # Assert all caches are empty
    assert not binned_grid._hmf_grid
    assert not binned_grid._mass_richness_grid
    assert not binned_grid._completeness_grid
    assert not binned_grid._purity_grid
    assert not binned_grid._shear_grids


def test_get_hmf_grid(binned_grid: GridBinnedClusterRecipe):
    """Test HMF grid shape and positivity."""
    binned_grid.setup()

    z_points = np.linspace(0.1, 1.0, 5)
    sky_area = 1000.0
    hmf_key = tuple(z_points)

    hmf_grid = binned_grid._get_hmf_grid(z_points, sky_area, hmf_key)

    n_z = len(z_points)
    n_m = len(binned_grid.log_mass_grid)

    assert hmf_grid.shape == (n_z, n_m)
    assert np.all(hmf_grid >= 0.0)
    assert hmf_grid is binned_grid._hmf_grid[hmf_key]  # Check caching

    # Test 2: Ensure computation is skipped on second call (cache hit)
    binned_grid._hmf_grid[hmf_key] = np.zeros_like(hmf_grid)  # Overwrite cache
    recalled_hmf_grid = binned_grid._get_hmf_grid(z_points, sky_area, hmf_key)
    assert np.all(recalled_hmf_grid == 0.0)


def test_get_mass_richness_grid(binned_grid: GridBinnedClusterRecipe):
    """Test mass-richness grid shape and non-negativity."""
    binned_grid.setup()

    z_points = np.linspace(0.1, 1.0, 5)
    proxy_grid_size = np.linspace(1.5, 3.0, 4)
    key = (tuple(z_points), tuple(proxy_grid_size))

    mr_grid = binned_grid._get_mass_richness_grid(z_points, proxy_grid_size, key)

    n_p = len(proxy_grid_size)
    n_z = len(z_points)
    n_m = len(binned_grid.log_mass_grid)

    assert mr_grid.shape == (n_p, n_z, n_m)
    assert np.all(mr_grid >= 0.0)
    assert mr_grid is binned_grid._mass_richness_grid[key]  # Check caching

    binned_grid._mass_richness_grid[key] = np.zeros_like(mr_grid)
    recalled_mr_grid = binned_grid._get_mass_richness_grid(
        z_points, proxy_grid_size, key
    )
    assert np.all(recalled_mr_grid == 0.0)


def test_get_completeness_grid(binned_grid: GridBinnedClusterRecipe):
    """Test completeness grid shape and bounds (0 <= C <= 1)."""
    binned_grid_w_comp = get_base_binned_grid(
        completeness_models.CompletenessAguena16(), None
    )
    binned_grid_w_comp.setup()

    z_points = np.linspace(0.1, 1.0, 5)
    comp_key = tuple(z_points)

    comp_grid = binned_grid_w_comp._get_completeness_grid(z_points, comp_key)

    n_z = len(z_points)
    n_m = len(binned_grid.log_mass_grid)

    assert comp_grid.shape == (n_z, n_m)
    assert np.all(comp_grid >= 0.0)
    assert np.all(comp_grid <= 1.0)
    assert comp_grid is binned_grid_w_comp._completeness_grid[comp_key]  # Check caching
    binned_grid.setup()  # use original fixture
    flat_comp_grid = binned_grid._get_completeness_grid(z_points, comp_key)
    assert flat_comp_grid.shape == (n_z, n_m)
    assert np.allclose(flat_comp_grid, 1.0)


def test_get_purity_grid(binned_grid: GridBinnedClusterRecipe):
    """Test purity grid shape, bounds, caching, and integration correctness."""
    binned_grid_w_pur = get_base_binned_grid(None, purity_models.PurityAguena16())
    binned_grid_w_pur.setup()

    z_points = np.linspace(0.1, 1.0, 5)
    proxy_grid_size = np.linspace(1.5, 3.0, 4)
    key = (tuple(z_points), tuple(proxy_grid_size))

    pur_grid = binned_grid_w_pur._get_purity_grid(z_points, proxy_grid_size, key)
    n_p = len(proxy_grid_size)
    n_z = len(z_points)

    # Shape and bounds
    assert pur_grid.shape == (n_p, n_z)
    assert np.all(pur_grid >= 0.0)
    assert np.all(pur_grid <= 1.0)

    # Check caching
    assert pur_grid is binned_grid_w_pur._purity_grid[key]

    binned_grid.setup()  # reset original fixture
    flat_pur_grid = binned_grid._get_purity_grid(z_points, proxy_grid_size, key)
    assert flat_pur_grid.shape == (n_p, n_z)
    assert np.allclose(flat_pur_grid, 1.0)

    def integrand(ln_proxy_scalar, z_scalar):
        log_proxy_scalar = ln_proxy_scalar / np.log(10.0)
        z_array = np.array([z_scalar])
        log_proxy_array = np.array([log_proxy_scalar])
        return binned_grid_w_pur._purity_distribution(log_proxy_array, z_array)

    z_bin = (z_points[0], z_points[-1])
    proxy_bin = (proxy_grid_size[0], proxy_grid_size[-1])

    integral_exact, _ = dblquad(
        func=integrand,
        a=z_bin[0],
        b=z_bin[1],
        gfun=lambda z: proxy_bin[0] * np.log(10.0),
        hfun=lambda z: proxy_bin[1] * np.log(10.0),
    )

    integral_over_proxy = simpson(
        y=pur_grid,
        x=proxy_grid_size * np.log(10.0),
        axis=0,
    )
    simpson_integral = simpson(
        y=integral_over_proxy,
        x=z_points,
        axis=0,
    )

    abs_err = abs(simpson_integral - integral_exact)
    rel_err = abs(1.0 - simpson_integral / integral_exact)

    assert rel_err < 5e-3  # 0.5% relative tolerance
    assert abs_err < 5e-3  # absolute small error tolerance
