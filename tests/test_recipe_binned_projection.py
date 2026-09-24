"""Tests for intrinsic and projected richness grids in the grid recipe."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import lognorm, norm

from crow.cluster_modules.kernel import SpectroscopicRedshift
from crow.cluster_modules.mass_proxy.murata import MurataUnbinned
from crow.cluster_modules.projection_effects import CostanziRichnessBias
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
        projection_model=CostanziRichnessBias(),
    )


def _projected_density_reference(recipe, rich_obs, z, log_mass, rich_tru_limits):
    """Reference density per ln(rich_obs), integrating in linear true richness."""
    mean = recipe.mass_distribution.get_ln_mass_proxy_mean(
        np.array([log_mass]), np.array([z])
    )[0]
    sigma = recipe.mass_distribution.get_ln_mass_proxy_sigma(
        np.array([log_mass]), np.array([z])
    )[0]

    def integrand(rich_tru):
        projection_pdf = recipe.projection_model.prob_richobs_at_richtru(
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
def test_projected_grid_matches_linear_quadrature(projection_recipe, rich_tru_limits):
    """Check both Jacobians and preserve probability lost to finite true bounds."""
    z = np.array([0.2, 0.7])
    rich_obs = np.array([12.0, 25.0, 55.0])
    log_proxy = np.log10(rich_obs)
    ln_rich_tru = np.linspace(*np.log(rich_tru_limits), 801)
    key = (tuple(z), tuple(log_proxy))

    result = projection_recipe._get_projected_mass_richness_grid(
        z, log_proxy, ln_rich_tru, key
    )

    assert result.shape == (3, 2, 3)
    expected = np.empty_like(result)
    for i, observed in enumerate(rich_obs):
        for j, redshift in enumerate(z):
            for k, log_mass in enumerate(projection_recipe.log_mass_grid):
                expected[i, j, k] = _projected_density_reference(
                    projection_recipe, observed, redshift, log_mass, rich_tru_limits
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
    assert recipe._mass_truerich_grid == {}
    assert recipe._projected_mass_richness_grid == {}
    with pytest.raises(ValueError, match="projection_model is required"):
        recipe._get_projected_mass_richness_grid(
            np.array([0.5]), np.array([1.0]), np.array([1.0, 2.0, 3.0]), "no-model"
        )
