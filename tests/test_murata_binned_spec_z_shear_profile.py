"""Tests for the cluster delta sigma module."""

import os
import sys

import numpy as np
import pyccl
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.kernel import SpectroscopicRedshift
from crow.mass_proxy import MurataBinned
from crow.properties import ClusterProperty
from crow.recipes.murata_binned_spec_z import MurataBinnedSpecZRecipe
from crow.shear_profile import ClusterShearProfile

# from firecrown.models.cluster import ClusterProperty


@pytest.fixture(name="murata_binned_spec_z_deltasigma")
def fixture_murata_binned_spec_z_deltasigma() -> MurataBinnedSpecZRecipe:
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    cluster_theory = ClusterShearProfile(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
        cluster_concentration=4.0,
        is_delta_sigma=True,
    )
    cluster_theory.set_beta_parameters(10.0)
    cluster_recipe = MurataBinnedSpecZRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=SpectroscopicRedshift(),
        mass_distribution=MurataBinned(pivot_mass, pivot_redshift),
        mass_interval=(13, 17),
        true_z_interval=(0, 2),
    )
    cluster_recipe.mass_distribution.mu_p0 = 3.0
    cluster_recipe.mass_distribution.mu_p1 = 0.86
    cluster_recipe.mass_distribution.mu_p2 = 0.0
    cluster_recipe.mass_distribution.sigma_p0 = 3.0
    cluster_recipe.mass_distribution.sigma_p1 = 0.7
    cluster_recipe.mass_distribution.sigma_p2 = 0.0
    return cluster_recipe


def test_murata_binned_spec_z_deltasigma_init(
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZRecipe,
):

    assert murata_binned_spec_z_deltasigma is not None
    assert isinstance(murata_binned_spec_z_deltasigma, MurataBinnedSpecZRecipe)
    assert murata_binned_spec_z_deltasigma.integrator is not None
    assert isinstance(murata_binned_spec_z_deltasigma.integrator, NumCosmoIntegrator)
    assert murata_binned_spec_z_deltasigma.redshift_distribution is not None
    assert isinstance(
        murata_binned_spec_z_deltasigma.redshift_distribution, SpectroscopicRedshift
    )
    assert murata_binned_spec_z_deltasigma.mass_distribution is not None
    assert isinstance(murata_binned_spec_z_deltasigma.mass_distribution, MurataBinned)


def test_get_theory_prediction_returns_value(
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZRecipe,
):
    prediction_none = (
        murata_binned_spec_z_deltasigma.get_theory_prediction_shear_profile(
            average_on=None
        )
    )
    prediction = murata_binned_spec_z_deltasigma.get_theory_prediction_shear_profile(
        ClusterProperty.DELTASIGMA
    )
    prediction_c = murata_binned_spec_z_deltasigma.get_theory_prediction_counts()

    assert prediction is not None
    assert prediction_c is not None
    assert callable(prediction)
    assert callable(prediction_c)

    mass = np.linspace(13, 17, 2, dtype=np.float64)
    z = np.linspace(0.1, 1, 2, dtype=np.float64)
    mass_proxy_limits = (0.0, 5.0)
    sky_area = 360**2
    radius_center = 1.5
    murata_binned_spec_z_deltasigma.cluster_theory.set_beta_s_interp(0.1, 1)
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
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZRecipe,
):
    prediction = murata_binned_spec_z_deltasigma.get_theory_prediction_shear_profile(
        ClusterProperty.DELTASIGMA
    )
    function_to_integrate = (
        murata_binned_spec_z_deltasigma.get_function_to_integrate_shear_profile(
            prediction
        )
    )

    assert function_to_integrate is not None
    assert callable(function_to_integrate)

    int_args = np.array([[13.0, 0.1], [17.0, 1.0]])
    extra_args = np.array([0, 5, 360**2, 1.5])
    murata_binned_spec_z_deltasigma.cluster_theory.set_beta_s_interp(0.1, 1)

    result = function_to_integrate(int_args, extra_args)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.float64)
    assert len(result) == 2
    assert np.all(result > 0)

    prediction_c = murata_binned_spec_z_deltasigma.get_theory_prediction_counts()
    function_to_integrate = (
        murata_binned_spec_z_deltasigma.get_function_to_integrate_counts(prediction_c)
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
    murata_binned_spec_z_deltasigma: MurataBinnedSpecZRecipe,
):

    mass_proxy_edges = (2, 5)
    z_edges = (0.5, 1)
    radius_center = 1.5
    average_on = ClusterProperty.DELTASIGMA

    prediction = (
        murata_binned_spec_z_deltasigma.evaluate_theory_prediction_shear_profile(
            z_edges, mass_proxy_edges, radius_center, 360**2, average_on
        )
    )
    prediction_c = murata_binned_spec_z_deltasigma.evaluate_theory_prediction_counts(
        z_edges, mass_proxy_edges, 360**2
    )
    assert prediction > 0
    assert prediction_c > 0
