"""Likelihood factory function for cluster number counts."""

import os

import pyccl as ccl
import sacc

from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.likelihood.binned_cluster_number_counts_deltasigma import (
    BinnedClusterDeltaSigma,
)
from firecrown.likelihood.binned_cluster_number_counts import (
    BinnedClusterNumberCounts,
)

from firecrown.likelihood.likelihood import Likelihood, NamedParameters
from firecrown.modeling_tools import ModelingTools

import sys
sys.path.append("/global/homes/a/aguena/git_codes/clump/")
#from clump.abundance import ClusterAbundance
#from clump.deltasigma import ClusterDeltaSigma
#from clump.properties import ClusterProperty
from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.deltasigma import ClusterDeltaSigma
from firecrown.models.cluster.properties import ClusterProperty
from clump.recipes.murata_binned_spec_z_deltasigma import MurataBinnedSpecZDeltaSigmaRecipe
from clump.recipes.murata_binned_spec_z import MurataBinnedSpecZRecipe
#from firecrown.models.cluster.recipes.murata_binned_spec_z_deltasigma import MurataBinnedSpecZDeltaSigmaRecipe
#from firecrown.models.cluster.recipes.murata_binned_spec_z import MurataBinnedSpecZRecipe


from clump.updatable_wrapper import UpdatableParameters


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
    print(average_on, survey_name, MurataBinnedSpecZRecipe())
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

    #cluster_model = get_cluster_model()
    #cluster_abundance = cluster_model.cluster_objects_list[0] #get_cluster_abundance()
    #cluster_deltasigma = cluster_model.cluster_objects_list[2] #get_cluster_deltasigma()
    cluster_abundance = get_cluster_abundance()
    cluster_deltasigma = get_cluster_deltasigma()
    modeling_tools = ModelingTools(
        cluster_abundance=cluster_abundance, cluster_deltasigma=cluster_deltasigma
    )

    return likelihood, modeling_tools



def get_cluster_abundance() -> ClusterAbundance:
    """Creates and returns a ClusterAbundance object."""
    hmf = ccl.halos.MassFuncTinker08(mass_def="200c")
    min_mass, max_mass = 13.0, 16.0
    min_z, max_z = 0.2, 0.8
    cluster_abundance = ClusterAbundance((min_mass, max_mass), (min_z, max_z), hmf)

    return cluster_abundance


def get_cluster_deltasigma() -> ClusterDeltaSigma:
    """Creates and returns a ClusterAbundance object."""
    hmf = ccl.halos.MassFuncTinker08(mass_def="200c")
    min_mass, max_mass = 13.0, 16.0
    min_z, max_z = 0.2, 0.8
    cluster_deltasigma = ClusterDeltaSigma(
        (min_mass, max_mass), (min_z, max_z), hmf, True
    )

    return cluster_deltasigma


def build_likelihood0(
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
    cluster_abundance = get_cluster_abundance()
    cluster_deltasigma = get_cluster_deltasigma()
    modeling_tools = ModelingTools(
        cluster_abundance=cluster_abundance, cluster_deltasigma=cluster_deltasigma
    )

    return likelihood, modeling_tools
