"""Likelihood factory function for cluster number counts."""

import os

import pyccl as ccl
import sacc

from firecrown.likelihood.gaussian import ConstGaussian

from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools

import sys

sys.path.append("/global/homes/a/aguena/git_codes/clump/")
from firecrown_like_examples.binned_cluster_number_counts_deltasigma import (
    BinnedClusterDeltaSigma,
)
from firecrown_like_examples.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)

from firecrown.models.cluster.properties import ClusterProperty
from clump.recipes.murata_binned_spec_z_deltasigma import (
    MurataBinnedSpecZDeltaSigmaRecipe,
)
from clump.recipes.murata_binned_spec_z import MurataBinnedSpecZRecipe


def build_likelihood(
    build_parameters: NamedParameters,
) -> tuple[Likelihood, ModelingTools]:
    """Builds the likelihood for Firecrown."""
    # Pull params for the likelihood from build_parameters
    average_on = ClusterProperty.NONE
    if build_parameters.get_bool("use_cluster_counts", True):
        average_on |= ClusterProperty.COUNTS
    if build_parameters.get_bool("use_mean_log_mass", True):
        average_on |= ClusterProperty.MASS
    if build_parameters.get_bool("use_mean_deltasigma", True):
        average_on |= ClusterProperty.DELTASIGMA

    survey_name = "numcosmo_simulated_redshift_richness_deltasigma"
    likelihood = ConstGaussian(
        [
            BinnedClusterNumberCounts(
                average_on, survey_name, MurataBinnedSpecZRecipe()
            ),
            BinnedClusterDeltaSigma(
                average_on, survey_name, MurataBinnedSpecZDeltaSigmaRecipe()
            ),
        ]
    )

    # Read in sacc data
    sacc_file_nm = "cluster_redshift_richness_deltasigma_sacc_data.fits"
    sacc_data = sacc.Sacc.load_fits(sacc_file_nm)
    likelihood.read(sacc_data)

    modeling_tools = ModelingTools()

    return likelihood, modeling_tools
