"""Projection-effect lensing profile bias models."""

import numpy as np
import numpy.typing as npt

arrayLike = int | float | npt.ArrayLike


def _as_1d_array(name: str, value: arrayLike) -> npt.NDArray[np.floating]:
    """Convert a scalar or 1D parameter to a 1D float array."""
    array = np.atleast_1d(np.asarray(value, dtype=float))
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D array.")
    return array


def _selection_parameter_arrays(
    A_sel: arrayLike,
    alpha_sel: arrayLike,
    beta_sel: arrayLike,
    gamma_sel: arrayLike,
    R0_sel: arrayLike,
) -> tuple[npt.NDArray[np.floating], ...]:
    """Return selection-bias parameters after strict per-bin shape checks."""
    parameters = (
        _as_1d_array("A_sel", A_sel),
        _as_1d_array("alpha_sel", alpha_sel),
        _as_1d_array("beta_sel", beta_sel),
        _as_1d_array("gamma_sel", gamma_sel),
        _as_1d_array("R0_sel", R0_sel),
    )
    parameter_lengths = {len(parameter) for parameter in parameters}
    if len(parameter_lengths) != 1:
        raise ValueError(
            "A_sel, alpha_sel, beta_sel, gamma_sel, and R0_sel must all be "
            "scalars or all be 1D arrays with the same length."
        )
    return parameters


def bsel(
    Rcmv: arrayLike,
    A_sel: arrayLike,
    alpha_sel: arrayLike,
    beta_sel: arrayLike,
    gamma_sel: arrayLike,
    R0_sel: arrayLike,
) -> npt.NDArray[np.floating]:
    """
    Evaluate the Costanzi selection-bias correction to stacked lensing profiles.

    This implements Equation C1 in Costanzi et al. (2026).  The correction is
    calibrated for stacked profiles in observed richness-redshift bins and is
    therefore a function of projected radius and bin-level calibration
    parameters.

    Parameters
    ----------
    Rcmv
        Projected radius in comoving Mpc/h.  This can be either a 1D radial
        array with shape ``(n_rad,)`` or a 2D array with shape
        ``(n_set, n_rad)`` for set-specific radius conversions.
    A_sel, alpha_sel, beta_sel, gamma_sel, R0_sel
        Selection-bias calibration parameters.  These must be all scalars for
        one richness-redshift bin, or all one-dimensional arrays with the same
        length for multiple bins.  ``R0_sel`` is in comoving Mpc/h.

    Returns
    -------
    ndarray
        Multiplicative bias ``Bsel(R)`` with shape ``(n_set, n_rad)``.
    """
    Rcmv = np.asarray(Rcmv, dtype=float)
    if Rcmv.ndim not in (1, 2):
        raise ValueError("Rcmv must be a 1D radius array or a 2D set-radius array.")
    if np.any(Rcmv <= 0.0):
        raise ValueError("Rcmv must contain only positive radii.")

    A_sel, alpha_sel, beta_sel, gamma_sel, R0_sel = _selection_parameter_arrays(
        A_sel,
        alpha_sel,
        beta_sel,
        gamma_sel,
        R0_sel,
    )
    n_set = len(A_sel)

    if np.any(R0_sel <= 0.0):
        raise ValueError("R0_sel must contain only positive values.")
    if np.any(gamma_sel == 0.0):
        raise ValueError("gamma_sel must be non-zero.")

    if Rcmv.ndim == 1:
        radius = np.broadcast_to(Rcmv[np.newaxis, :], (n_set, len(Rcmv)))
    else:
        if Rcmv.shape[0] != n_set:
            raise ValueError(
                "The first dimension of 2D Rcmv must match the number of "
                "selection-bias parameter sets."
            )
        radius = Rcmv

    normed_radius = radius / R0_sel[:, np.newaxis]
    bias_minus_one = (
        A_sel[:, np.newaxis]
        * normed_radius ** alpha_sel[:, np.newaxis]
        * (1.0 + normed_radius ** gamma_sel[:, np.newaxis])
        ** (
            (beta_sel[:, np.newaxis] - alpha_sel[:, np.newaxis])
            / gamma_sel[:, np.newaxis]
        )
    )
    return bias_minus_one + 1.0


class CostanziLensingBias:
    """
    Costanzi selection-bias correction for stacked DeltaSigma profiles.

    The correction is calibrated per observed richness-redshift bin and should
    be applied after the stacked profile has been integrated over the selected
    cluster population.
    """

    def __init__(
        self,
        A_sel: arrayLike,
        alpha_sel: arrayLike,
        beta_sel: arrayLike,
        gamma_sel: arrayLike,
        R0_sel: arrayLike,
        *,
        radius_is_comoving_mpc_over_h: bool = True,
        reference_redshift: arrayLike | None = None,
    ) -> None:
        """
        Initialize the lensing selection-bias correction.

        Parameters
        ----------
        A_sel, alpha_sel, beta_sel, gamma_sel, R0_sel
            Parameters of ``Bsel(R)``.  These must be all scalars for one
            richness-redshift bin, or all one-dimensional arrays with the same
            length for multiple bins.
        radius_is_comoving_mpc_over_h
            If True, radii passed to :meth:`distribution` are already in
            comoving Mpc/h.  If False, radii are interpreted as physical Mpc and
            converted as ``R_comoving[Mpc/h] = R_physical[Mpc] * (1 + z_ref) * h``.
        reference_redshift
            Redshift used for the physical-Mpc to comoving-Mpc/h conversion.
            Required when ``radius_is_comoving_mpc_over_h`` is False.  It must
            be a scalar for one parameter set, or a one-dimensional array with
            the same length as the selection-bias parameters.
        """
        (
            self.A_sel,
            self.alpha_sel,
            self.beta_sel,
            self.gamma_sel,
            self.R0_sel,
        ) = _selection_parameter_arrays(A_sel, alpha_sel, beta_sel, gamma_sel, R0_sel)
        self.radius_is_comoving_mpc_over_h = radius_is_comoving_mpc_over_h

        self.reference_redshift = None
        if reference_redshift is not None:
            reference_redshift = _as_1d_array("reference_redshift", reference_redshift)
            if len(reference_redshift) != len(self.A_sel):
                raise ValueError(
                    "reference_redshift must be a scalar for one parameter set "
                    "or a 1D array with the same length as the selection-bias "
                    "parameters."
                )
            self.reference_redshift = reference_redshift

    def _radius_to_comoving_mpc_over_h(
        self,
        radius: npt.ArrayLike,
        cosmo_h: float | None,
    ) -> npt.NDArray[np.floating]:
        radius = np.asarray(radius, dtype=float)
        if radius.ndim != 1:
            raise ValueError("radius must be a 1D array.")

        if self.radius_is_comoving_mpc_over_h:
            return radius

        if self.reference_redshift is None:
            raise ValueError(
                "reference_redshift is required when "
                "radius_is_comoving_mpc_over_h is False."
            )
        if cosmo_h is None:
            raise ValueError(
                "cosmo_h is required to convert physical Mpc to comoving Mpc/h."
            )

        reference_redshift = np.atleast_1d(
            np.asarray(self.reference_redshift, dtype=float)
        )
        if reference_redshift.ndim != 1:
            raise ValueError("reference_redshift must be a scalar or 1D array.")

        return (
            radius[np.newaxis, :] * (1.0 + reference_redshift[:, np.newaxis]) * cosmo_h
        )

    def distribution(
        self,
        radius: npt.ArrayLike,
        *,
        cosmo_h: float | None = None,
    ) -> npt.NDArray[np.floating]:
        """
        Evaluate ``Bsel(R)``.

        Parameters
        ----------
        radius
            Radius array.  Units are controlled by
            ``radius_is_comoving_mpc_over_h``.
        cosmo_h
            Dimensionless Hubble parameter.  Required only when converting from
            physical Mpc to comoving Mpc/h.

        Returns
        -------
        ndarray
            Multiplicative correction with shape ``(n_set, n_rad)``.
        """
        radius_comoving = self._radius_to_comoving_mpc_over_h(radius, cosmo_h)
        return bsel(
            radius_comoving,
            self.A_sel,
            self.alpha_sel,
            self.beta_sel,
            self.gamma_sel,
            self.R0_sel,
        )
