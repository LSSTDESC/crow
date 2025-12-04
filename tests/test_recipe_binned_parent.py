"""Tests for the cluster abundance module."""

import os
import sys

import numpy as np
import pyccl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import floats


from crow import ClusterAbundance, kernel, mass_proxy, purity_models
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.properties import ClusterProperty
from crow.recipes.binned_parent import BinnedClusterRecipe

# from firecrown.models.cluster import ClusterProperty


def test_binned_init():
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    cluster_theory = ClusterAbundance(
        cosmo=pyccl.CosmologyVanillaLCDM(),
        halo_mass_function=pyccl.halos.MassFuncTinker08(mass_def="200c"),
    )
    redshift_distribution = kernel.SpectroscopicRedshift()
    mass_distribution = mass_proxy.MurataBinned(pivot_mass, pivot_redshift)
    completeness = None
    purity = None
    mass_interval = (13, 17)
    true_z_interval = (0, 2)

    binned_class = BinnedClusterRecipe(
        cluster_theory=cluster_theory,
        redshift_distribution=redshift_distribution,
        mass_distribution=mass_distribution,
        completeness=completeness,
        purity=purity,
        mass_interval=mass_interval,
        true_z_interval=true_z_interval,
    )

    assert binned_class.cluster_theory == cluster_theory
    assert binned_class.redshift_distribution == redshift_distribution
    assert binned_class.mass_distribution == mass_distribution
    assert binned_class.completeness == completeness
    assert binned_class.purity == purity
    assert binned_class.mass_interval == mass_interval
    assert binned_class.true_z_interval == true_z_interval

    np.testing.assert_raises(NotImplementedError, binned_class.setup)
    np.testing.assert_raises(
        NotImplementedError, binned_class.evaluate_theory_prediction_counts, *[None] * 4
    )
    np.testing.assert_raises(
        NotImplementedError,
        binned_class.evaluate_theory_prediction_lensing_profile,
        *[None] * 5,
    )
