"""Tests for projection-effect models."""

import math

import numpy as np
import pytest

from crow.cluster_modules.projection_effects.lensing_bias import (
    LENSING_BIAS_DEFAULT_PARAMETERS,
    CostanziLensingBias,
)
from crow.cluster_modules.projection_effects.richness_bias import (
    RICHNESS_BIAS_DEFAULT_PARAMETERS,
    CostanziRichnessBias,
)


def test_costanzi_default_parameters():
    """Test the default Costanzi projection parameters."""
    model = CostanziRichnessBias()

    for key, value in RICHNESS_BIAS_DEFAULT_PARAMETERS.items():
        assert model.parameters[key] == value


def test_costanzi_constructor_overrides_default_parameters():
    """Constructor arguments override the default richness-bias calibration."""
    model = CostanziRichnessBias(
        tau=np.array([0.2]),
        delta_mu=np.array([-1.5]),
        fractional_sig_pure=np.array([0.2]),
        fprj=np.array([0.8]),
        fmsk=np.array([0.1]),
    )

    np.testing.assert_allclose(model.parameters["tau"], np.array([0.2]))
    np.testing.assert_allclose(model.parameters["delta_mu"], np.array([-1.5]))
    np.testing.assert_allclose(model.parameters["fractional_sig_pure"], np.array([0.2]))
    np.testing.assert_allclose(model.parameters["fprj"], np.array([0.8]))
    np.testing.assert_allclose(model.parameters["fmsk"], np.array([0.1]))


def test_costanzi_parameters_are_independent_between_instances():
    """Each richness-bias object owns an independent calibration."""
    first_model = CostanziRichnessBias()
    second_model = CostanziRichnessBias()

    first_model.parameters["tau"] = 0.20

    assert first_model.parameters["tau"] == 0.20
    assert second_model.parameters["tau"] == 0.10


def test_costanzi_exp_times_erfc_matches_direct_expression():
    """The overflow-safe helper should match the direct expression when finite."""
    log_factor = np.array([-4.0, -1.0, 0.5, 2.0])
    erfc_arg = np.array([-1.0, 0.0, 0.5, 1.5])

    result = CostanziRichnessBias._exp_times_erfc(log_factor, erfc_arg)
    expected = np.exp(log_factor) * np.array([math.erfc(x) for x in erfc_arg])

    np.testing.assert_allclose(result, expected)


def test_costanzi_probability_shape_positivity_and_finiteness():
    """Evaluate P(rich_obs | rich_tru) on a 2D true-richness parameter grid."""
    rich_obs_edges = np.geomspace(1e-2, 300.0, 201)
    rich_obs = np.sqrt(rich_obs_edges[1:] * rich_obs_edges[:-1])
    rich_tru = np.geomspace(1e-2, 250.0, 100)
    redshift = np.geomspace(1.0 + 1e-2, 1.0 + 1.5, 30) - 1.0
    rich_tru_mesh, _ = np.meshgrid(rich_tru, redshift, indexing="ij")
    model = CostanziRichnessBias()

    prob = model.prob_richobs_at_richtru(
        rich_obs=rich_obs,
        rich_tru=rich_tru_mesh.flatten(),
    )
    prob = prob.reshape(rich_obs.shape + rich_tru_mesh.shape)

    assert prob.shape == (200, 100, 30)
    assert np.all(np.isfinite(prob))
    assert np.all(prob >= 0.0)


def test_costanzi_probability_accepts_parameter_arrays():
    """Projection parameters can vary along the rich_tru axis."""
    rich_obs = np.array([10.0, 20.0, 40.0])
    rich_tru = np.array([15.0, 30.0, 60.0])
    model = CostanziRichnessBias()
    model.parameters.update(
        {
            "tau": np.array([0.08, 0.10, 0.12]),
            "delta_mu": np.array([-1.0, -2.0, -3.0]),
            "fractional_sig_pure": np.array([0.12, 0.15, 0.18]),
            "fprj": np.array([0.80, 0.90, 0.95]),
            "fmsk": np.array([0.01, 0.03, 0.05]),
        }
    )

    prob = model.prob_richobs_at_richtru(
        rich_obs=rich_obs,
        rich_tru=rich_tru,
    )

    assert prob.shape == (len(rich_obs), len(rich_tru))
    assert np.all(np.isfinite(prob))
    assert np.all(prob >= 0.0)


def test_costanzi_observed_bin_probability_shape_positivity_and_finiteness():
    """Integrate P(rich_obs | rich_tru) over observed-richness bins."""
    rich_obs_eds = np.array([10.0, 15.0, 20.0, 30.0, 60.0, 100.0])
    rich_tru = np.geomspace(1e-2, 250.0, 100)
    redshift = np.geomspace(1.0 + 1e-2, 1.0 + 1.5, 30) - 1.0
    rich_tru_mesh, _ = np.meshgrid(rich_tru, redshift, indexing="ij")
    model = CostanziRichnessBias()

    sprob = model.Sprob_at_richtru(
        rich_obs_eds=rich_obs_eds,
        rich_obs_res=0.015,
        rich_tru=rich_tru_mesh.flatten(),
    )
    sprob = sprob.reshape((len(rich_obs_eds) - 1,) + rich_tru_mesh.shape)

    assert sprob.shape == (5, 100, 30)
    assert np.all(np.isfinite(sprob))
    assert np.all(sprob >= 0.0)


def test_costanzi_observed_bin_probability_is_smaller_than_total_probability():
    """A finite observed-richness bin should not exceed the broad-bin integral."""
    rich_tru = np.geomspace(10.0, 100.0, 12)
    model = CostanziRichnessBias()

    narrow = model.Sprob_at_richtru(
        rich_obs_eds=np.array([20.0, 40.0]),
        rich_obs_res=0.01,
        rich_tru=rich_tru,
    )
    broad = model.Sprob_at_richtru(
        rich_obs_eds=np.array([1.0, 300.0]),
        rich_obs_res=0.01,
        rich_tru=rich_tru,
    )

    assert narrow.shape == broad.shape == (1, len(rich_tru))
    assert np.all(narrow <= broad)


@pytest.mark.parametrize(
    ("parameter", "value", "match"),
    [
        ("tau", -0.1, "tau can only be positive"),
        (
            "fractional_sig_pure",
            -0.1,
            "fractional_sig_pure can only be positive",
        ),
        ("fprj", 1.1, "fprj can only be between 0 and 1"),
        ("fmsk", -0.1, "fmsk can only be between 0 and 1"),
    ],
)
def test_costanzi_probability_validates_parameters(parameter, value, match):
    """Invalid projection parameters should fail explicitly."""
    rich_obs = np.array([10.0, 20.0])
    rich_tru = np.array([20.0, 40.0])
    model = CostanziRichnessBias()
    model.parameters[parameter] = value

    with pytest.raises(ValueError, match=match):
        model.prob_richobs_at_richtru(
            rich_obs=rich_obs,
            rich_tru=rich_tru,
        )

    rich_tru_sig_test = np.array([0.0, 20.0])
    with pytest.raises(ValueError, match="sig_pure can only be positive"):
        CostanziRichnessBias().prob_richobs_at_richtru(
            rich_obs=rich_obs,
            rich_tru=rich_tru_sig_test,
        )


def test_costanzi_observed_bin_probability_validates_bin_edges():
    """Observed-richness bin edges must be 1D and monotonically increasing."""
    rich_tru = np.array([20.0, 40.0])
    model = CostanziRichnessBias()

    with pytest.raises(ValueError, match="ndim is not 1"):
        model.Sprob_at_richtru(
            rich_obs_eds=np.array([[10.0, 20.0]]),
            rich_obs_res=0.01,
            rich_tru=rich_tru,
        )

    with pytest.raises(ValueError, match="monotonically increasing"):
        model.Sprob_at_richtru(
            rich_obs_eds=np.array([20.0, 10.0]),
            rich_obs_res=0.01,
            rich_tru=rich_tru,
        )


def test_costanzi_lensing_bias_default_parameters():
    """Test the default Costanzi lensing-bias parameters."""
    radius = np.array([0.5, 1.0, 2.0])
    model = CostanziLensingBias()

    for key, value in LENSING_BIAS_DEFAULT_PARAMETERS.items():
        assert model.parameters[key] == value

    correction = model.bsel(radius, radial_units="Mpc/h")

    assert correction.shape == (1, len(radius))
    np.testing.assert_allclose(correction, 1.0)


def test_costanzi_lensing_bias_constant_correction():
    """Test the Costanzi lensing bias model for a simple constant correction."""
    radius = np.array([0.5, 1.0, 2.0])
    model = CostanziLensingBias(
        A_sel=1.0,
        alpha_sel=0.0,
        beta_sel=0.0,
        gamma_sel=1.0,
        R0_sel=1.0,
    )
    correction = model.bsel(radius, radial_units="Mpc/h")

    assert correction.shape == (1, len(radius))
    np.testing.assert_allclose(correction, 2.0)


def test_costanzi_lensing_bias_multiple_parameter_sets():
    """Test multiple selection-bias parameter sets on a shared radius grid."""
    radius = np.array([0.5, 1.0, 2.0])
    model = CostanziLensingBias(
        A_sel=np.array([1.0, 2.0]),
        alpha_sel=np.array([0.0, 0.0]),
        beta_sel=np.array([0.0, 0.0]),
        gamma_sel=np.array([1.0, 1.0]),
        R0_sel=np.array([1.0, 1.0]),
    )
    correction = model.bsel(radius, radial_units="Mpc/h")

    assert correction.shape == (2, len(radius))
    np.testing.assert_allclose(correction[0], 2.0)
    np.testing.assert_allclose(correction[1], 3.0)


def test_costanzi_lensing_bias_uses_parameters_container():
    """Parameters mutated after construction are used, and re-validated, by bsel."""
    model = CostanziLensingBias(
        A_sel=1.0,
        alpha_sel=0.0,
        beta_sel=0.0,
        gamma_sel=1.0,
        R0_sel=1.0,
    )
    model.parameters["A_sel"] = 2.0

    correction = model.bsel(
        np.array([0.5, 1.0, 2.0]),
        radial_units="Mpc/h",
    )

    np.testing.assert_allclose(correction, 3.0)

    model.parameters["A_sel"] = np.array([[1.0]])
    with pytest.raises(ValueError, match="scalars or 1D arrays"):
        model.bsel(np.array([1.0]), radial_units="Mpc/h")

    model.parameters["A_sel"] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        model.bsel(np.array([1.0]), radial_units="Mpc/h")

    model.parameters["A_sel"] = 1.0
    model.parameters["R0_sel"] = -1.0
    with pytest.raises(ValueError, match="R0_sel must contain only positive values"):
        model.bsel(np.array([1.0]), radial_units="Mpc/h")

    model.parameters["R0_sel"] = 1.0
    model.parameters["gamma_sel"] = 0.0
    with pytest.raises(ValueError, match="gamma_sel must be non-zero"):
        model.bsel(np.array([1.0]), radial_units="Mpc/h")

    mismatched_model = CostanziLensingBias(
        A_sel=1.0,
        alpha_sel=1.0,
        beta_sel=1.0,
        gamma_sel=1.0,
        R0_sel=1.0,
        reference_redshift=0.5,
    )
    mismatched_model.parameters["A_sel"] = np.array([1.0, 2.0])
    mismatched_model.parameters["alpha_sel"] = np.array([1.0, 1.0])
    mismatched_model.parameters["beta_sel"] = np.array([1.0, 1.0])
    mismatched_model.parameters["gamma_sel"] = np.array([1.0, 1.0])
    mismatched_model.parameters["R0_sel"] = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="must match the number of"):
        mismatched_model.bsel(np.array([1.0]), radial_units="Mpc", cosmo_h=0.7)


def test_costanzi_lensing_bias_physical_radius_conversion():
    """Test explicit physical-Mpc to comoving-Mpc/h radius conversion."""
    radius_physical = np.array([1.0, 2.0])
    model = CostanziLensingBias(
        A_sel=1.0,
        alpha_sel=1.0,
        beta_sel=1.0,
        gamma_sel=1.0,
        R0_sel=1.0,
        reference_redshift=0.5,
    )

    correction = model.bsel(
        radius_physical,
        radial_units="Mpc",
        cosmo_h=0.7,
    )

    assert correction.shape == (1, len(radius_physical))
    expected_radius_comoving = radius_physical * (1.0 + 0.5) * 0.7
    np.testing.assert_allclose(
        correction,
        1.0 + expected_radius_comoving[np.newaxis, :],
    )


def test_costanzi_lensing_bias_requires_matching_parameter_shapes():
    """Each selection-bias parameter entry corresponds to one bin."""
    with pytest.raises(ValueError, match="scalars or 1D arrays"):
        CostanziLensingBias(A_sel=np.array([[1.0]]))

    with pytest.raises(ValueError, match="same length"):
        CostanziLensingBias(
            A_sel=np.array([1.0, 1.0]),
            alpha_sel=0.0,
            beta_sel=np.array([0.0, 0.0]),
            gamma_sel=np.array([1.0, 1.0]),
            R0_sel=np.array([1.0, 1.0]),
        )


def test_costanzi_lensing_bias_requires_matching_reference_redshift_shape():
    """The radius-conversion redshift has the same per-bin axis."""
    with pytest.raises(
        ValueError, match="reference_redshift must be a scalar or 1D array"
    ):
        CostanziLensingBias(A_sel=1.0, reference_redshift=np.array([[0.5]]))

    with pytest.raises(ValueError, match="reference_redshift"):
        CostanziLensingBias(
            A_sel=np.array([1.0, 1.0]),
            alpha_sel=np.array([0.0, 0.0]),
            beta_sel=np.array([0.0, 0.0]),
            gamma_sel=np.array([1.0, 1.0]),
            R0_sel=np.array([1.0, 1.0]),
            reference_redshift=0.5,
        )


def test_costanzi_lensing_bias_requires_cosmo_h_for_physical_radius():
    """Physical-Mpc radius conversion requires h."""
    model = CostanziLensingBias(
        A_sel=1.0,
        alpha_sel=0.0,
        beta_sel=0.0,
        gamma_sel=1.0,
        R0_sel=1.0,
        reference_redshift=0.5,
    )

    with pytest.raises(ValueError, match="cosmo_h is required"):
        model.bsel(np.array([1.0]), radial_units="Mpc")


def test_costanzi_lensing_bias_requires_reference_redshift_for_physical_radius():
    """Physical-Mpc radius conversion requires a reference redshift."""
    model = CostanziLensingBias(
        A_sel=1.0,
        alpha_sel=0.0,
        beta_sel=0.0,
        gamma_sel=1.0,
        R0_sel=1.0,
    )

    with pytest.raises(ValueError, match="reference_redshift is required"):
        model.bsel(np.array([1.0]), radial_units="Mpc", cosmo_h=0.7)


def test_costanzi_lensing_bias_rejects_invalid_bsel_inputs():
    """bsel validates radial_units, radius, and cosmo_h before evaluating Bsel(R)."""
    model = CostanziLensingBias(
        A_sel=1.0,
        alpha_sel=0.0,
        beta_sel=0.0,
        gamma_sel=1.0,
        R0_sel=1.0,
        reference_redshift=0.5,
    )

    with pytest.raises(ValueError, match="radial_units must be"):
        model.bsel(np.array([1.0]), radial_units="arcmin")

    with pytest.raises(ValueError, match="radius must be a 1D array"):
        model.bsel(np.array([[0.5, 1.0]]), radial_units="Mpc/h")

    with pytest.raises(ValueError, match="radius must contain only positive values"):
        model.bsel(np.array([0.0, 1.0]), radial_units="Mpc/h")

    with pytest.raises(ValueError, match="cosmo_h must be positive"):
        model.bsel(np.array([1.0]), radial_units="Mpc", cosmo_h=-0.1)
