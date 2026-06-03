"""Tests for the Costanzi cluster mass richness module."""

import numpy as np
import pytest

from crow.cluster_modules.projection_effects.costanzi import (
    CostanziBaseModel,
    CostanziBinned,
    CostanziUnBinned,
)

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


@pytest.fixture(name="costanzi_unbinned_relation")
def fixture_costanzi_unbinned() -> CostanziUnBinned:
    """Initialize Costanzi unbinned cluster object."""
    mr = CostanziUnBinned(PIVOT_MASS, PIVOT_Z)
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
    assert cb.ln1p_pivot_redshift == np.log1p(1)
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
    rich_obs_edges = np.geomspace(1e-2, 300, 201)
    rich_obs_bins = np.sqrt(rich_obs_edges[1:] * rich_obs_edges[:-1])
    rich_tru = np.geomspace(1e-2, 250, 100)
    # redshift between 0.01 and 1.5
    redshift = np.geomspace(1.0 + 1e-2, 1.0 + 1.5, 30) - 1.0
    # mesh
    rich_tru_mesh, _ = np.meshgrid(rich_tru, redshift, indexing="ij")

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


def test_costanzi_compute_probabilities_paired_shapes():
    """Test projection convolution helper output shapes."""
    model = CostanziBinned(PIVOT_MASS, PIVOT_Z)

    rich_obs_eds = np.array([10.0, 20.0, 40.0])
    log_rich_tru = np.linspace(0.5, 2.0, 6)
    rich_tru = 10.0**log_rich_tru
    log_mass = np.array([13.5, 14.0, 14.5])
    z = np.array([0.2, 0.5, 0.8])

    result = model.compute_probabilities(
        rich_obs_eds=rich_obs_eds,
        rich_obs_res=0.05,
        log_rich_tru=log_rich_tru,
        log_mass=log_mass,
        z=z,
        tau=0.10,
        delta_mu=-2.0,
        sig_pure=0.15 * rich_tru,
        fprj=0.95,
        fmsk=0.05,
    )

    assert result["Sprob_richobs_richtru"].shape == (
        len(rich_obs_eds) - 1,
        len(log_rich_tru),
    )
    assert result["prob_richtru_mass_redshift"].shape == (
        len(log_rich_tru),
        len(log_mass),
    )
    assert result["Sprob_richobs_mass_redshift"].shape == (
        len(rich_obs_eds) - 1,
        len(log_mass),
    )
    assert np.all(result["Sprob_richobs_richtru"] >= 0.0)
    assert np.all(result["prob_richtru_mass_redshift"] >= 0.0)
    assert np.all(result["Sprob_richobs_mass_redshift"] >= 0.0)


def test_costanzi_compute_probabilities_mesh_shapes():
    """Test projection convolution helper meshes different-length 1D axes."""
    model = CostanziBinned(PIVOT_MASS, PIVOT_Z)

    rich_obs_eds = np.array([10.0, 20.0, 40.0])
    log_rich_tru = np.linspace(0.5, 2.0, 6)
    rich_tru = 10.0**log_rich_tru
    log_mass = np.array([13.5, 14.0])
    z = np.array([0.2, 0.5, 0.8])

    result = model.compute_probabilities(
        rich_obs_eds=rich_obs_eds,
        rich_obs_res=0.05,
        log_rich_tru=log_rich_tru,
        log_mass=log_mass,
        z=z,
        tau=0.10,
        delta_mu=-2.0,
        sig_pure=0.15 * rich_tru,
        fprj=0.95,
        fmsk=0.05,
    )

    assert result["Sprob_richobs_richtru"].shape == (
        len(rich_obs_eds) - 1,
        len(log_rich_tru),
    )
    assert result["prob_richtru_mass_redshift"].shape == (
        len(log_rich_tru),
        len(z),
        len(log_mass),
    )
    assert result["Sprob_richobs_mass_redshift"].shape == (
        len(rich_obs_eds) - 1,
        len(z),
        len(log_mass),
    )


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


def test_cluster_costanzi_binned_distribution_broadcasts():
    """Test Costanzi distribution broadcasts arbitrary mass and redshift arrays."""
    model = CostanziBinned(PIVOT_MASS, PIVOT_Z)
    mass = np.array([13.5, 14.0, 14.5])
    z = np.array([0.3, 0.6])
    mass_proxy_limits = (1.0, 2.0)

    result = model.distribution(mass, z, mass_proxy_limits)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    assert np.all(result >= 0.0)


def test_cluster_costanzi_unbinned_distribution_execution(
    costanzi_unbinned_relation: CostanziUnBinned,
):
    """Test that the Costanzi unbinned distribution executes."""
    mass = np.array([13.5, 14.0, 14.5])
    z = np.array([0.5, 0.5, 0.5])
    log_mass_proxy = np.array([1.2, 1.5, 1.8])

    result = costanzi_unbinned_relation.distribution(mass, z, log_mass_proxy)

    assert isinstance(result, np.ndarray)
    assert result.shape == mass.shape
    assert np.all(result >= 0.0)


def test_cluster_costanzi_unbinned_distribution_meshes_mass_redshift():
    """Test Costanzi unbinned distribution meshes different-length 1D axes."""
    model = CostanziUnBinned(PIVOT_MASS, PIVOT_Z)
    mass = np.array([13.5, 14.0, 14.5])
    z = np.array([0.3, 0.6])
    log_mass_proxy = 1.5

    result = model.distribution(mass, z, log_mass_proxy)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    assert np.all(result >= 0.0)


def test_cluster_costanzi_unbinned_matches_narrow_binned_probability():
    """Test unbinned PDF is consistent with a narrow binned probability."""
    binned_model = CostanziBinned(
        PIVOT_MASS, PIVOT_Z, projection_richness_resolution=0.001
    )
    unbinned_model = CostanziUnBinned(PIVOT_MASS, PIVOT_Z)

    log_mass = np.array([14.0])
    z = np.array([0.5])
    log_mass_proxy = np.array([1.5])
    delta_ln_proxy = 0.05
    delta_log_proxy = delta_ln_proxy / np.log(10.0)
    log_mass_proxy_limits = (
        log_mass_proxy[0] - 0.5 * delta_log_proxy,
        log_mass_proxy[0] + 0.5 * delta_log_proxy,
    )

    binned_probability = binned_model.distribution(log_mass, z, log_mass_proxy_limits)
    unbinned_probability = unbinned_model.distribution(log_mass, z, log_mass_proxy)

    assert binned_probability[0] == pytest.approx(
        unbinned_probability[0] * delta_ln_proxy,
        rel=0.15,
        abs=0.0,
    )
