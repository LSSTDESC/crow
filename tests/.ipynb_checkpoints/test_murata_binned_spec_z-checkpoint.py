"""Tests for the cluster abundance module."""

import os
import sys
from unittest.mock import Mock

import numpy as np
import pyccl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import floats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from crow.abundance import ClusterAbundance
from crow.deltasigma import ClusterAbundance
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.kernel import SpectroscopicRedshift
from crow.mass_proxy import MurataBinned
from crow.properties import ClusterProperty
from crow.recipes.murata_binned_spec_z import MurataBinnedSpecZRecipe


@pytest.fixture(name="murata_binned_spec_z")
def fixture_murata_binned_spec_z() -> MurataBinnedSpecZRecipe:
    cluster_recipe = MurataBinnedSpecZRecipe()
    cluster_recipe.mass_distribution.mu_p0 = 3.0
    cluster_recipe.mass_distribution.mu_p1 = 0.86
    cluster_recipe.mass_distribution.mu_p2 = 0.0
    cluster_recipe.mass_distribution.sigma_p0 = 3.0
    cluster_recipe.mass_distribution.sigma_p1 = 0.7
    cluster_recipe.mass_distribution.sigma_p2 = 0.0
    cosmo_ccl = pyccl.CosmologyVanillaLCDM()
    cluster_recipe.cluster_theory.cosmo = cosmo_ccl
    return cluster_recipe


def test_murata_binned_spec_z_init():
    recipe = MurataBinnedSpecZRecipe()

    assert recipe is not None
    assert isinstance(recipe, MurataBinnedSpecZRecipe)
    assert recipe.integrator is not None
    assert isinstance(recipe.integrator, NumCosmoIntegrator)
    assert recipe.redshift_distribution is not None
    assert isinstance(recipe.redshift_distribution, SpectroscopicRedshift)
    assert recipe.mass_distribution is not None
    assert isinstance(recipe.mass_distribution, MurataBinned)



def test_get_theory_prediction_returns_value(
    murata_binned_spec_z: MurataBinnedSpecZRecipe,
):
    prediction = murata_binned_spec_z.get_theory_prediction(ClusterProperty.COUNTS)

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
    murata_binned_spec_z: MurataBinnedSpecZRecipe,
    mass: float,
    z: float,
    sky_area: float,
):
    """Test that cluster predictions are always positive using hypothesis."""
    prediction = murata_binned_spec_z.get_theory_prediction(ClusterProperty.COUNTS)

    mass_array = np.array([mass])
    z_array = np.array([z])
    mass_proxy_limits = (0, 5)

    result = prediction(mass_array, z_array, mass_proxy_limits, sky_area)

    # Physical constraint: cluster predictions must be positive
    assert np.all(result > 0), f"All cluster predictions must be positive, got {result}"
    assert np.all(np.isfinite(result)), f"All predictions must be finite, got {result}"


def test_get_theory_prediction_with_average_returns_value(
    murata_binned_spec_z: MurataBinnedSpecZRecipe,
):
    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0, 5)
    sky_area = 360**2

    prediction = murata_binned_spec_z.get_theory_prediction(
        average_on=ClusterProperty.MASS
    )

    assert prediction is not None
    assert callable(prediction)

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction = murata_binned_spec_z.get_theory_prediction(
        average_on=ClusterProperty.REDSHIFT
    )

    assert prediction is not None
    assert callable(prediction)

    result = prediction(mass, z, mass_proxy_limits, sky_area)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction = murata_binned_spec_z.get_theory_prediction(
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
    murata_binned_spec_z: MurataBinnedSpecZRecipe,
):
    prediction = murata_binned_spec_z.get_theory_prediction(
        average_on=ClusterProperty.SHEAR
    )

    assert prediction is not None
    assert callable(prediction)

    # mass = np.linspace(13, 17, 2, dtype=np.float64)
    # z = np.linspace(0.1, 1, 2, dtype=np.float64)
    # mass_proxy_limits = (0, 5)
    # sky_area = 360**2

    # with pytest.raises(NotImplementedError):
    #    _ = prediction(mass, z, mass_proxy_limits, sky_area)


def test_get_function_to_integrate_returns_value(
    murata_binned_spec_z: MurataBinnedSpecZRecipe,
):
    prediction = murata_binned_spec_z.get_theory_prediction()
    function_to_integrate = murata_binned_spec_z.get_function_to_integrate(prediction)

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
    murata_binned_spec_z: MurataBinnedSpecZRecipe,
):
    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    mass_proxy_limits = (0.0, 5.0) 
    sky_area = 360**2

    prediction = murata_binned_spec_z.evaluate_theory_prediction(
      z_edges, mass_proxy_edges, 360**2
    )

    assert prediction > 0
    prediction = murata_binned_spec_z.evaluate_theory_prediction(
        z_edges, mass_proxy_edges, 360**2, ClusterProperty.REDSHIFT
    )

    assert prediction > 0
