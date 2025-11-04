"""Likelihood factory function for cluster number counts."""

import os
import sys

import sacc
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools
from firecrown.models.cluster import ClusterProperty

# remove this line after crow becomes installable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crow.recipes.murata_binned_spec_z import MurataBinnedSpecZRecipe
from crow.recipes.murata_binned_spec_z_deltasigma import (
    MurataBinnedSpecZDeltaSigmaRecipe,
)
from crow.mass_proxy import MurataBinned
from crow.kernel import SpectroscopicRedshift

# to be moved to firecrown eventually
from firecrown_like_examples.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)
from firecrown_like_examples.binned_cluster_number_counts_deltasigma import (
    BinnedClusterShearProfile,
)


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

    hmf = ccl.halos.MassFuncTinker08(mass_def="200c")
    redshift_distribution = SpectroscopicRedshift()
    pivot_mass, pivot_redshift = 14.625862906, 0.6
    mass_distribution = MurataBinned(pivot_mass, pivot_redshift)
    survey_name = "numcosmo_simulated_redshift_richness_deltasigma"

    likelihood = ConstGaussian(
        [
            BinnedClusterNumberCounts(
                average_on,
                survey_name,
                MurataBinnedSpecZRecipe(hmf, redshift_distribution, mass_distribution),
            ),
            BinnedClusterShearProfile(
                average_on,
                survey_name,
                MurataBinnedSpecZDeltaSigmaRecipe(
                    hmf=hmf,
                    redshift_distribution=redshift_distribution,
                    mass_distribution=mass_distribution,
                    is_delta_sigma=True,
                ),
            ),
        ]
    )

    # Read in sacc data
    sacc_file_nm = "cluster_redshift_richness_deltasigma_sacc_data.fits"
    sacc_data = sacc.Sacc.load_fits(sacc_file_nm)
    likelihood.read(sacc_data)

    modeling_tools = ModelingTools()

    return likelihood, modeling_tools
