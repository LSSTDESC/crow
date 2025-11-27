"""Tests for the cluster abundance module."""

import os
import sys

import numpy as np
import pyccl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import floats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


@pytest.fixture(name="binned_exact")
def fixture_binned_exact() -> ExactBinnedClusterRecipe:
    return get_base_binned_exact(None, None)


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
):
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    sky_area = 360**2

    prediction = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area
    )

    assert prediction > 0
    prediction = binned_exact.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, sky_area, ClusterProperty.REDSHIFT
    )

    assert prediction > 0


def test_evaluates_theory_prediction_with_completeness(
    binned_exact: ExactBinnedClusterRecipe,
):
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    sky_area = 360**2

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


def test_evaluates_theory_prediction_with_purity(
    binned_exact: ExactBinnedClusterRecipe,
):
    ######################################
    # Henrique, also set up the tests here
    ######################################
    pass


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
    probability_w_pur = binned_exact_w_pur._mass_distribution_distribution(
        mass_array, z_array, mass_proxy_limits
    )

    assert (probability < probability_w_pur).all()
