"""Tests for intrinsic/projected richness grids and binned count predictions."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad, simpson
from scipy.stats import lognorm, norm

from crow.cluster_modules.kernel import SpectroscopicRedshift
from crow.cluster_modules.mass_proxy.murata import MurataUnbinned
from crow.cluster_modules.projection_effects import CostanziRichnessBias
from crow.properties import ClusterProperty
from crow.recipes.binned_grid import GridBinnedClusterRecipe


@pytest.fixture(name="projection_recipe")
def fixture_projection_recipe() -> GridBinnedClusterRecipe:
    """Use small grids without evaluating cosmology or halo profiles."""
    mass_distribution = MurataUnbinned(14.3, 0.5)
    mass_distribution.parameters["sigma1"] = 0.02
    mass_distribution.parameters["sigma2"] = 0.01
    return GridBinnedClusterRecipe(
        cluster_theory=None,
        redshift_distribution=SpectroscopicRedshift(),
        mass_distribution=mass_distribution,
        mass_interval=(14.0, 14.6),
        mass_grid_size=3,
        redshift_grid_size=3,
        proxy_grid_size=5,
        projection_model=CostanziRichnessBias(),
        ln_rich_tru=np.linspace(np.log(1.0), np.log(500.0), 801),
    )


@pytest.fixture(name="counts_recipe")
def fixture_counts_recipe(projection_recipe) -> GridBinnedClusterRecipe:
    """Toy abundance and selection factors isolate quadrature from cosmology."""
    projection_recipe.cluster_theory = SimpleNamespace(
        mass_function=lambda log_mass, z: np.exp(-(log_mass - 14.0)) * (1.0 + z),
        comoving_volume=lambda z, sky_area: sky_area * (1.0 + z) ** 2,
    )
    projection_recipe.completeness = SimpleNamespace(
        distribution=lambda log_mass, z: 0.7 + 0.1 * (log_mass - 14.0) - 0.1 * z
    )
    projection_recipe.purity = SimpleNamespace(
        distribution=lambda log_proxy, z: 0.8 + 0.05 * log_proxy - 0.1 * z
    )
    return projection_recipe


def _projected_density_reference(
    recipe, rich_obs, z, log_mass, rich_tru_limits, tau_at_rich_tru=None
):
    """Reference density per ln(rich_obs), integrating in linear true richness."""
    mean = recipe.mass_distribution.get_ln_mass_proxy_mean(
        np.array([log_mass]), np.array([z])
    )[0]
    sigma = recipe.mass_distribution.get_ln_mass_proxy_sigma(
        np.array([log_mass]), np.array([z])
    )[0]
    projection_model = recipe.projection_model
    if tau_at_rich_tru is not None:
        # Adaptive quadrature needs scalar rates at its own true-richness points.
        projection_model = CostanziRichnessBias(**projection_model.parameters)

    def integrand(rich_tru):
        if tau_at_rich_tru is not None:
            projection_model.parameters["tau"] = tau_at_rich_tru(rich_tru)
        projection_pdf = projection_model.prob_richobs_at_richtru(
            rich_obs=np.array([rich_obs]),
            rich_tru=np.array([rich_tru]),
        )[0, 0]
        intrinsic_pdf = lognorm.pdf(rich_tru, s=sigma, scale=np.exp(mean))
        return projection_pdf * intrinsic_pdf

    result, _ = quad(
        integrand,
        *rich_tru_limits,
        epsabs=1e-11,
        epsrel=1e-9,
    )
    return rich_obs * result


def test_intrinsic_grid_matches_scalar_gaussian(projection_recipe):
    """Each axis corresponds to true richness, redshift, and mass, respectively."""
    z = np.array([0.2, 0.7])
    ln_rich_tru = np.log(np.array([5.0, 12.0, 20.0, 35.0, 60.0]))

    result = projection_recipe._get_mass_truerich_grid(z, ln_rich_tru, tuple(z))

    assert result.shape == (5, 2, 3)
    for j, redshift in enumerate(z):
        for k, log_mass in enumerate(projection_recipe.log_mass_grid):
            mean = projection_recipe.mass_distribution.get_ln_mass_proxy_mean(
                np.array([log_mass]), np.array([redshift])
            )[0]
            sigma = projection_recipe.mass_distribution.get_ln_mass_proxy_sigma(
                np.array([log_mass]), np.array([redshift])
            )[0]
            expected = norm.pdf(ln_rich_tru, loc=mean, scale=sigma)
            np.testing.assert_allclose(result[:, j, k], expected)


@pytest.mark.parametrize("rich_tru_limits", [(1.0, 500.0), (12.0, 30.0)])
@pytest.mark.parametrize("richness_dependent_tau", [False, True])
def test_projected_grid_matches_linear_quadrature(
    projection_recipe, rich_tru_limits, richness_dependent_tau
):
    """Check both Jacobians and preserve probability lost to finite true bounds."""
    z = np.array([0.2, 0.7])
    rich_obs = np.array([12.0, 25.0, 55.0])
    log_proxy = np.log10(rich_obs)
    ln_rich_tru = np.linspace(*np.log(rich_tru_limits), 801)
    key = (tuple(z), tuple(log_proxy))
    tau_at_rich_tru = None
    if richness_dependent_tau:
        tau_ref, rich_ref = 0.20, 20.0
        tau = tau_ref * rich_ref / np.exp(ln_rich_tru)
        projection_recipe.projection_model = CostanziRichnessBias(tau=tau)

        def tau_at_rich_tru(rich_tru):
            return tau_ref * rich_ref / rich_tru

    result = projection_recipe._get_projected_mass_richness_grid(
        z, log_proxy, ln_rich_tru, key
    )

    assert result.shape == (3, 2, 3)
    expected = np.empty_like(result)
    for i, observed in enumerate(rich_obs):
        for j, redshift in enumerate(z):
            for k, log_mass in enumerate(projection_recipe.log_mass_grid):
                expected[i, j, k] = _projected_density_reference(
                    projection_recipe,
                    observed,
                    redshift,
                    log_mass,
                    rich_tru_limits,
                    tau_at_rich_tru=tau_at_rich_tru,
                )

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-10)


def test_projected_grid_converges_with_resolution(projection_recipe):
    """Refining the true-richness grid approaches independent quadrature."""
    z = np.array([0.5])
    log_proxy = np.log10(np.array([25.0]))
    rich_tru_limits = (1.0, 500.0)
    key = (tuple(z), tuple(log_proxy))
    expected = _projected_density_reference(
        projection_recipe,
        25.0,
        z[0],
        projection_recipe.log_mass_grid[1],
        rich_tru_limits,
    )

    errors = []
    for size in (21, 81, 321):
        ln_rich_tru = np.linspace(*np.log(rich_tru_limits), size)
        result = projection_recipe._get_projected_mass_richness_grid(
            z, log_proxy, ln_rich_tru, key
        )[0, 0, 1]
        errors.append(abs(result - expected))

    assert errors[2] < errors[1] < errors[0]
    assert errors[2] / expected < 2e-6


def test_true_richness_coordinates_distinguish_caches(projection_recipe):
    """Equal-size true grids with different coordinates cannot share cached PDFs."""
    z = np.array([0.2, 0.7])
    log_proxy = np.log10(np.array([12.0, 25.0, 55.0]))
    key = (tuple(z), tuple(log_proxy))
    grids = []
    for limits in ((1.0, 500.0), (12.0, 30.0)):
        ln_rich_tru = np.linspace(*np.log(limits), 101)
        result = projection_recipe._get_projected_mass_richness_grid(
            z, log_proxy, ln_rich_tru, key
        )
        assert (
            projection_recipe._get_projected_mass_richness_grid(
                z, log_proxy, ln_rich_tru, key
            )
            is result
        )
        intrinsic = projection_recipe._get_mass_truerich_grid(z, ln_rich_tru, tuple(z))
        assert (
            projection_recipe._get_mass_truerich_grid(z, ln_rich_tru, tuple(z))
            is intrinsic
        )
        grids.append(result)

    assert len(projection_recipe._mass_truerich_grid) == 2
    assert len(projection_recipe._projected_mass_richness_grid) == 2
    assert not np.allclose(grids[0], grids[1])


@pytest.mark.parametrize("model_name", ["mass_distribution", "projection_model"])
def test_setup_refreshes_projection_caches(projection_recipe, model_name):
    """setup() must invalidate both grids after either model changes."""
    z = np.array([0.2, 0.7])
    log_proxy = np.log10(np.array([12.0, 25.0, 55.0]))
    ln_rich_tru = np.linspace(np.log(1.0), np.log(500.0), 101)
    key = (tuple(z), tuple(log_proxy))
    original = projection_recipe._get_projected_mass_richness_grid(
        z, log_proxy, ln_rich_tru, key
    )

    if model_name == "mass_distribution":
        projection_recipe.mass_distribution.parameters["mu0"] += 0.2
    else:
        projection_recipe.projection_model.parameters["tau"] *= 2.0
    projection_recipe.setup()

    assert projection_recipe._mass_truerich_grid == {}
    assert projection_recipe._projected_mass_richness_grid == {}
    updated = projection_recipe._get_projected_mass_richness_grid(
        z, log_proxy, ln_rich_tru, key
    )
    assert not np.allclose(original, updated)


@pytest.mark.parametrize(
    "ln_rich_tru",
    [
        np.array([]),
        np.array([1.0]),
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0, 3.0]]),
        np.array([3.0, 2.0, 1.0]),
        np.array([1.0, 1.0, 2.0]),
        np.array([1.0, np.nan, 3.0]),
        np.array([1.0, 2.0, np.inf]),
    ],
)
def test_projected_grid_rejects_invalid_true_coordinates(
    projection_recipe, ln_rich_tru
):
    """Invalid quadrature coordinates fail before a probability grid is cached."""
    with pytest.raises(ValueError, match="ln_rich_tru"):
        projection_recipe._get_projected_mass_richness_grid(
            np.array([0.5]), np.array([1.0]), ln_rich_tru, "invalid"
        )
    assert projection_recipe._mass_truerich_grid == {}
    assert projection_recipe._projected_mass_richness_grid == {}


def test_projection_model_is_optional_and_required_by_helper():
    """Existing constructors remain valid; the projected helper needs a model."""
    recipe = GridBinnedClusterRecipe(
        cluster_theory=None,
        redshift_distribution=SpectroscopicRedshift(),
        mass_distribution=MurataUnbinned(14.3, 0.5),
    )
    assert recipe.projection_model is None
    assert recipe.ln_rich_tru is None
    assert recipe._mass_truerich_grid == {}
    assert recipe._projected_mass_richness_grid == {}
    with pytest.raises(ValueError, match="projection_model is required"):
        recipe._get_projected_mass_richness_grid(
            np.array([0.5]), np.array([1.0]), np.array([1.0, 2.0, 3.0]), "no-model"
        )


def test_constructor_copies_true_richness_grid():
    """The caller's array cannot accidentally modify the recipe configuration."""
    ln_rich_tru = np.linspace(np.log(1.0), np.log(500.0), 101)
    recipe = GridBinnedClusterRecipe(
        cluster_theory=None,
        redshift_distribution=SpectroscopicRedshift(),
        mass_distribution=MurataUnbinned(14.3, 0.5),
        ln_rich_tru=ln_rich_tru,
    )
    np.testing.assert_array_equal(recipe.ln_rich_tru, ln_rich_tru)
    assert recipe.ln_rich_tru.dtype == np.float64
    assert not np.shares_memory(recipe.ln_rich_tru, ln_rich_tru)
    ln_rich_tru[:] = 0.0
    assert np.all(np.diff(recipe.ln_rich_tru) > 0.0)


def _counts_integrand_reference(recipe, z, log_proxy, sky_area, projection):
    """Assemble a scalar reference without using any recipe grid helpers."""
    integrand = np.empty((len(log_proxy), len(z), len(recipe.log_mass_grid)))
    for i, observed in enumerate(10.0**log_proxy):
        for j, redshift in enumerate(z):
            redshift_array = np.array([redshift])
            volume = recipe.cluster_theory.comoving_volume(redshift_array, sky_area)[0]
            purity = 1.0
            if recipe.purity is not None:
                purity = recipe.purity.distribution(
                    np.array([log_proxy[i]]), redshift_array
                )[0]
            for k, log_mass in enumerate(recipe.log_mass_grid):
                mass_array = np.array([log_mass])
                abundance = recipe.cluster_theory.mass_function(
                    mass_array, redshift_array
                )[0]
                completeness = 1.0
                if recipe.completeness is not None:
                    completeness = recipe.completeness.distribution(
                        mass_array, redshift_array
                    )[0]
                if projection:
                    density = _projected_density_reference(
                        recipe,
                        observed,
                        redshift,
                        log_mass,
                        np.exp(recipe.ln_rich_tru[[0, -1]]),
                    )
                else:
                    mean = recipe.mass_distribution.get_ln_mass_proxy_mean(
                        mass_array, redshift_array
                    )[0]
                    sigma = recipe.mass_distribution.get_ln_mass_proxy_sigma(
                        mass_array, redshift_array
                    )[0]
                    density = norm.pdf(np.log(observed), loc=mean, scale=sigma)
                integrand[i, j, k] = (
                    volume * abundance * completeness * density / purity
                )
    return integrand


@pytest.mark.parametrize("projection", [False, True])
@pytest.mark.parametrize("with_selection", [False, True])
def test_counts_match_independent_integral(counts_recipe, projection, with_selection):
    """Check density selection, integration measures, and every count weight."""
    recipe = counts_recipe
    if not with_selection:
        recipe.completeness = None
        recipe.purity = None
    z_edges = (0.2, 0.7)
    log_proxy_edges = np.log10([12.0, 55.0])
    sky_area = 440.0
    z = np.linspace(*z_edges, recipe.redshift_grid_size)
    log_proxy = np.linspace(*log_proxy_edges, recipe.proxy_grid_size)
    integrand = _counts_integrand_reference(recipe, z, log_proxy, sky_area, projection)

    for average_on in (
        None,
        ClusterProperty.COUNTS,
        ClusterProperty.MASS,
        ClusterProperty.REDSHIFT,
        ClusterProperty.MASS | ClusterProperty.REDSHIFT,
    ):
        weighted = integrand.copy()
        if average_on is not None:
            if average_on & ClusterProperty.MASS:
                weighted *= recipe.log_mass_grid[np.newaxis, np.newaxis, :]
            if average_on & ClusterProperty.REDSHIFT:
                weighted *= z[np.newaxis, :, np.newaxis]
        # Reverse the production integration order for the reference.
        over_proxy = simpson(weighted, x=log_proxy * np.log(10.0), axis=0)
        over_z = simpson(over_proxy, x=z, axis=0)
        expected = simpson(over_z, x=recipe.log_mass_grid)

        result = recipe.evaluate_theory_prediction_counts(
            z_edges,
            log_proxy_edges,
            sky_area,
            average_on=average_on,
            projection=projection,
        )
        assert np.isfinite(result)
        assert result > 0.0
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-10)
        if not projection:
            default = recipe.evaluate_theory_prediction_counts(
                z_edges, log_proxy_edges, sky_area, average_on=average_on
            )
            assert default == result


@pytest.mark.parametrize(
    "missing_model, missing_grid, error",
    [
        (True, False, "projection_model"),
        (False, True, "ln_rich_tru"),
        (True, True, "projection_model"),
    ],
)
def test_counts_require_projection_configuration(
    projection_recipe, missing_model, missing_grid, error
):
    """Missing configuration fails clearly, before accessing cluster theory."""
    if missing_model:
        projection_recipe.projection_model = None
    if missing_grid:
        projection_recipe.ln_rich_tru = None
    with pytest.raises(ValueError, match=f"projection=True requires.*{error}"):
        projection_recipe.evaluate_theory_prediction_counts(
            (0.2, 0.7), np.log10([12.0, 55.0]), 440.0, projection=True
        )
    assert projection_recipe._hmf_grid == {}


def test_counts_validate_stored_true_richness_grid(counts_recipe):
    """The public projected-count path retains the existing coordinate checks."""
    counts_recipe.ln_rich_tru = np.array([1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="finite, strictly increasing"):
        counts_recipe.evaluate_theory_prediction_counts(
            (0.2, 0.7), np.log10([12.0, 55.0]), 440.0, projection=True
        )
    assert counts_recipe._mass_truerich_grid == {}
    assert counts_recipe._projected_mass_richness_grid == {}


def test_counts_switch_uses_separate_density_caches(counts_recipe):
    """ON/OFF calls cannot contaminate one another or alter the observed grids."""
    recipe = counts_recipe
    z_edges = (0.2, 0.7)
    log_proxy_edges = np.log10([12.0, 55.0])
    sky_area = 440.0
    unprojected = recipe.evaluate_theory_prediction_counts(
        z_edges, log_proxy_edges, sky_area
    )
    assert recipe._projected_mass_richness_grid == {}
    projected = recipe.evaluate_theory_prediction_counts(
        z_edges, log_proxy_edges, sky_area, projection=True
    )
    assert not np.isclose(projected, unprojected)
    assert len(recipe._mass_richness_grid) == 1
    assert len(recipe._projected_mass_richness_grid) == 1
    unprojected_grid = next(iter(recipe._mass_richness_grid.values()))
    projected_grid = next(iter(recipe._projected_mass_richness_grid.values()))
    assert unprojected_grid.shape == projected_grid.shape
    assert (
        recipe.evaluate_theory_prediction_counts(
            z_edges, log_proxy_edges, sky_area, projection=False
        )
        == unprojected
    )
    assert (
        recipe.evaluate_theory_prediction_counts(
            z_edges, log_proxy_edges, sky_area, projection=True
        )
        == projected
    )

    # The original call needs neither projection configuration nor valid grids.
    recipe.projection_model = None
    recipe.ln_rich_tru = None
    recipe.setup()
    assert (
        recipe.evaluate_theory_prediction_counts(z_edges, log_proxy_edges, sky_area)
        == unprojected
    )


@pytest.mark.parametrize("model_name", ["mass_distribution", "projection_model"])
def test_setup_refreshes_projected_counts(counts_recipe, model_name):
    """Updating either parameter block and calling setup refreshes predictions."""
    recipe = counts_recipe
    z_edges = (0.2, 0.7)
    log_proxy_edges = np.log10([12.0, 55.0])
    sky_area = 440.0
    original_grid = recipe.ln_rich_tru
    original_model = recipe.projection_model
    original = recipe.evaluate_theory_prediction_counts(
        z_edges, log_proxy_edges, sky_area, projection=True
    )
    if model_name == "mass_distribution":
        recipe.mass_distribution.parameters.update({"mu0": 3.2})
    else:
        recipe.projection_model.parameters.update({"tau": 0.2})
    recipe.setup()
    assert recipe._mass_truerich_grid == {}
    assert recipe._projected_mass_richness_grid == {}
    assert recipe.ln_rich_tru is original_grid
    assert recipe.projection_model is original_model
    updated = recipe.evaluate_theory_prediction_counts(
        z_edges, log_proxy_edges, sky_area, projection=True
    )
    assert not np.isclose(original, updated)


def test_counts_constant_tau_matches_uniform_tau_array(counts_recipe):
    """A constant calibration accepts a scalar or an equal-valued grid array."""
    recipe = counts_recipe
    tau = 0.20
    recipe.projection_model = CostanziRichnessBias(tau=tau)
    z_edges = (0.2, 0.7)
    log_proxy_edges = np.log10([12.0, 55.0])
    original = recipe.evaluate_theory_prediction_counts(
        z_edges, log_proxy_edges, 440.0, projection=True
    )
    recipe.projection_model = CostanziRichnessBias(
        tau=np.full_like(recipe.ln_rich_tru, tau)
    )
    recipe.setup()
    expected = recipe.evaluate_theory_prediction_counts(
        z_edges, log_proxy_edges, 440.0, projection=True
    )
    np.testing.assert_allclose(original, expected, rtol=1e-12)


def test_counts_tau_array_updates_after_setup(counts_recipe):
    """Rebuilding the rate array refreshes ON counts without affecting OFF."""
    recipe = counts_recipe
    tau_ref, rich_ref = 0.20, 20.0
    tau = tau_ref * rich_ref / np.exp(recipe.ln_rich_tru)
    recipe.projection_model = CostanziRichnessBias(tau=tau)
    z_edges = (0.2, 0.7)
    log_proxy_edges = np.log10([12.0, 55.0])
    off = recipe.evaluate_theory_prediction_counts(z_edges, log_proxy_edges, 440.0)
    on = recipe.evaluate_theory_prediction_counts(
        z_edges, log_proxy_edges, 440.0, projection=True
    )
    updated_tau_ref = 0.40
    tau_updated = updated_tau_ref * rich_ref / np.exp(recipe.ln_rich_tru)
    recipe.projection_model.parameters.update({"tau": tau_updated})
    recipe.setup()
    assert recipe._mass_truerich_grid == {}
    assert recipe._projected_mass_richness_grid == {}
    updated = recipe.evaluate_theory_prediction_counts(
        z_edges, log_proxy_edges, 440.0, projection=True
    )
    assert not np.isclose(on, updated)
    np.testing.assert_array_equal(recipe.projection_model.parameters["tau"], tau_updated)
    np.testing.assert_allclose(
        recipe.evaluate_theory_prediction_counts(z_edges, log_proxy_edges, 440.0),
        off,
        rtol=1e-12,
    )
