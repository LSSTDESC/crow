"""Tests for the Costanzi cluster mass richness module."""

import numpy as np
import pytest

from crow.cluster_modules.mass_proxy.costanzi import CostanziBaseModel, CostanziBinned

# Pivot values (Using the same pivot values as the Murata test)
PIVOT_Z = 0.6
PIVOT_MASS = 14.625862906

@pytest.fixture(name="costanzi_binned_relation")
def fixture_costanzi_binned() -> CostanziBinned:
    """Initialize Costanzi cluster object."""
    mr = CostanziBinned(PIVOT_MASS, PIVOT_Z)
    mr.parameters["mu0"] = 3.00
    mr.parameters["mu1"] = 0.086
    mr.parameters["mu2"] = 0.01
    mr.parameters["sigma0"] = 3.0
    mr.parameters["sigma1"] = 0.07
    mr.parameters["sigma2"] = 0.01
    mr.parameters["tau"] = 0.10
    mr.parameters["delta_mu"] = -2.0
    mr.parameters["sig_pure_scatter"] = 0.15
    mr.parameters["fprj"] = 0.95
    mr.parameters["fmsk"] = 0.05
    return mr

def test_create_costanzi_kernel():
    """Test initialization of the Costanzi kernel."""
    cb = CostanziBinned(1, 1)
    assert cb.pivot_ln_mass == 1 * np.log(10)
    assert cb.pivot_redshift == 1
    assert cb.log1p_pivot_redshift == np.log1p(1)
    assert cb.parameters["tau"] == 0.10
    assert cb.parameters["delta_mu"] == -2.0
    assert cb.parameters["fprj"] == 0.95

def test_costanzi_base_model_core_math():
    """
    Test the core math functions of CostanziBaseModel using the 
    supervisor's original face values and grid configurations.
    """
    # 1. Setup the mesh grids
    # richness between 0.01 and 300
    rich_obs_edges = np.geomspace(1E-2, 300, 201)
    rich_obs_bins = np.sqrt(rich_obs_edges[1:] * rich_obs_edges[:-1])
    rich_tru = np.geomspace(1E-2, 250, 100)
    # redshift between 0.01 and 1.5
    redshift = np.geomspace(1.0 + 1E-2, 1.0 + 1.5, 30) - 1.0
    # mesh
    rich_tru_mesh, _ = np.meshgrid(rich_tru, redshift, indexing='ij')

    # Parameters - face values from Costanzi
    tau = 0.10 * np.ones_like(rich_tru_mesh)
    delta_mu = -2.0 * np.ones_like(rich_tru_mesh)
    sig_pure = 0.15 * rich_tru_mesh
    fprj = 0.95 * np.ones_like(rich_tru_mesh)
    fmsk = 0.05 * np.ones_like(rich_tru_mesh)

    # 2. Test prob_richobs_at_richtru
    prob = CostanziBaseModel.prob_richobs_at_richtru(
        rich_obs=rich_obs_bins,
        rich_tru=rich_tru_mesh.flatten(),
        tau=tau.flatten(),
        delta_mu=delta_mu.flatten(),
        sig_pure=sig_pure.flatten(),
        fprj=fprj.flatten(),
        fmsk=fmsk.flatten(),
    )
    prob = prob.reshape(rich_obs_bins.shape + rich_tru_mesh.shape)
    
    # Verify dimensions (200 bins, 100 rich_tru, 30 redshift) and non-negativity
    assert prob.shape == (200, 100, 30)
    assert np.all(prob >= 0.0)

    # 3. Test Sprob_at_richtru (Integration)
    # define integrals
    rich_obs_eds = [10.0, 15.0, 20.0, 30.0, 60.0, 100.0]
    Sprob = CostanziBaseModel.Sprob_at_richtru(
        rich_obs_eds=rich_obs_eds,
        rich_obs_res=0.015,
        rich_tru=rich_tru_mesh.flatten(),
        tau=tau.flatten(),
        delta_mu=delta_mu.flatten(),
        sig_pure=sig_pure.flatten(),
        fprj=fprj.flatten(),
        fmsk=fmsk.flatten(),
    )
    Sprob = Sprob.reshape((len(rich_obs_eds) - 1,) + rich_tru_mesh.shape)
    
    # Verify dimensions (5 intervals, 100 rich_tru, 30 redshift) and non-negativity
    assert Sprob.shape == (5, 100, 30)
    assert np.all(Sprob >= 0.0)

def test_cluster_costanzi_binned_distribution_execution(
    costanzi_binned_relation: CostanziBinned,
):
    """Test that the Costanzi distribution method executes without errors."""
    mass = np.array([13.5, 14.0, 14.5])
    z = np.array([0.5, 0.5, 0.5])
    mass_proxy_limits = (1.0, 2.0)  

    result = costanzi_binned_relation.distribution(mass, z, mass_proxy_limits)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == mass.shape
