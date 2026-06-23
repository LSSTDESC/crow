"""Projection-effect lensing profile bias models."""

import numpy as np
import numpy.typing as npt

from ..parameters import Parameters


class CostanziLensingBias:
    """
    Costanzi selection-bias correction for stacked DeltaSigma profiles.

    The correction is calibrated per observed richness-redshift bin and should
    be applied after the stacked profile has been integrated over the selected
    cluster population.
    """

    def __init__(
        self,
        A_sel: npt.ArrayLike,
        alpha_sel: npt.ArrayLike,
        beta_sel: npt.ArrayLike,
        gamma_sel: npt.ArrayLike,
        R0_sel: npt.ArrayLike,
        *,
        reference_redshift: npt.ArrayLike | None = None,
    ) -> None:
        """
        Initialize the lensing selection-bias correction.

        Parameters
        ----------
        A_sel, alpha_sel, beta_sel, gamma_sel, R0_sel
            Parameters of ``Bsel(R)``.  These must be all scalars for one
            richness-redshift bin, or all one-dimensional arrays with the same
            length for multiple bins. ``R0_sel`` is in comoving Mpc/h.
        reference_redshift
            Redshift used for the physical-Mpc to comoving-Mpc/h conversion.
            Required when :meth:`bsel` receives ``radial_units="Mpc"``. It must
            be a scalar for one parameter set or a one-dimensional array with
            the same length as the selection-bias parameters.
        """
        parameter_values = {
            "A_sel": np.atleast_1d(np.asarray(A_sel, dtype=float)),
            "alpha_sel": np.atleast_1d(np.asarray(alpha_sel, dtype=float)),
            "beta_sel": np.atleast_1d(np.asarray(beta_sel, dtype=float)),
            "gamma_sel": np.atleast_1d(np.asarray(gamma_sel, dtype=float)),
            "R0_sel": np.atleast_1d(np.asarray(R0_sel, dtype=float)),
        }
        if any(parameter.ndim != 1 for parameter in parameter_values.values()):
            raise ValueError(
                "A_sel, alpha_sel, beta_sel, gamma_sel, and R0_sel must be "
                "scalars or 1D arrays."
            )
        if len({len(parameter) for parameter in parameter_values.values()}) != 1:
            raise ValueError(
                "A_sel, alpha_sel, beta_sel, gamma_sel, and R0_sel must all be "
                "scalars or all be 1D arrays with the same length."
            )

        self.parameters = Parameters(parameter_values)

        self.reference_redshift = None
        if reference_redshift is not None:
            reference_redshift = np.atleast_1d(
                np.asarray(reference_redshift, dtype=float)
            )
            if reference_redshift.ndim != 1:
                raise ValueError("reference_redshift must be a scalar or 1D array.")
            if len(reference_redshift) != len(parameter_values["A_sel"]):
                raise ValueError(
                    "reference_redshift must be a scalar for one parameter set "
                    "or a 1D array with the same length as the selection-bias "
                    "parameters."
                )
            self.reference_redshift = reference_redshift

    def _radius_to_comoving_mpc_over_h(
        self,
        radius: npt.ArrayLike,
        radial_units: str,
        cosmo_h: float | None,
    ) -> npt.NDArray[np.floating]:
        radius = np.asarray(radius, dtype=float)
        if radius.ndim != 1:
            raise ValueError("radius must be a 1D array.")

        if radial_units == "Mpc/h":
            return radius

        if radial_units != "Mpc":
            raise ValueError(
                "radial_units must be 'Mpc' for physical radii or 'Mpc/h' "
                "for comoving radii."
            )

        if self.reference_redshift is None:
            raise ValueError(
                "reference_redshift is required when radial_units is 'Mpc'."
            )
        if cosmo_h is None:
            raise ValueError(
                "cosmo_h is required to convert physical Mpc to comoving Mpc/h."
            )
        if cosmo_h <= 0.0:
            raise ValueError("cosmo_h must be positive.")

        return (
            radius[np.newaxis, :]
            * (1.0 + self.reference_redshift[:, np.newaxis])
            * cosmo_h
        )

    def bsel(
        self,
        radius: npt.ArrayLike,
        *,
        radial_units: str,
        cosmo_h: float | None = None,
    ) -> npt.NDArray[np.floating]:
        """
        Evaluate the Costanzi selection-bias correction ``Bsel(R)``.

        This implements Equation C1 in Costanzi et al. (2026). The correction
        is calibrated for stacked profiles in observed richness-redshift bins.

        Parameters
        ----------
        radius
            One-dimensional radius array.
        radial_units
            Units and coordinate convention of ``radius``. Use ``"Mpc/h"``
            for comoving Mpc/h or ``"Mpc"`` for physical Mpc. Other units,
            including angular units, are not supported.
        cosmo_h
            Dimensionless Hubble parameter.  Required only when converting from
            physical Mpc to comoving Mpc/h.

        Returns
        -------
        ndarray
            Multiplicative correction with shape ``(n_set, n_rad)``.

        Raises
        ------
        ValueError
            If ``radial_units`` is not ``"Mpc"`` or ``"Mpc/h"``, or if the
            physical-Mpc conversion lacks ``reference_redshift`` or ``cosmo_h``.
        """
        radius_comoving = self._radius_to_comoving_mpc_over_h(
            radius,
            radial_units,
            cosmo_h,
        )
        if np.any(radius_comoving <= 0.0):
            raise ValueError("radius must contain only positive values.")

        A_sel = np.atleast_1d(np.asarray(self.parameters["A_sel"], dtype=float))
        alpha_sel = np.atleast_1d(np.asarray(self.parameters["alpha_sel"], dtype=float))
        beta_sel = np.atleast_1d(np.asarray(self.parameters["beta_sel"], dtype=float))
        gamma_sel = np.atleast_1d(np.asarray(self.parameters["gamma_sel"], dtype=float))
        R0_sel = np.atleast_1d(np.asarray(self.parameters["R0_sel"], dtype=float))
        parameters = (A_sel, alpha_sel, beta_sel, gamma_sel, R0_sel)

        if any(parameter.ndim != 1 for parameter in parameters):
            raise ValueError(
                "A_sel, alpha_sel, beta_sel, gamma_sel, and R0_sel must be "
                "scalars or 1D arrays."
            )
        if len({len(parameter) for parameter in parameters}) != 1:
            raise ValueError(
                "A_sel, alpha_sel, beta_sel, gamma_sel, and R0_sel must all be "
                "scalars or all be 1D arrays with the same length."
            )
        if np.any(R0_sel <= 0.0):
            raise ValueError("R0_sel must contain only positive values.")
        if np.any(gamma_sel == 0.0):
            raise ValueError("gamma_sel must be non-zero.")

        n_set = len(A_sel)
        if radius_comoving.ndim == 1:
            radius_comoving = np.broadcast_to(
                radius_comoving[np.newaxis, :],
                (n_set, len(radius_comoving)),
            )
        elif radius_comoving.shape[0] != n_set:
            raise ValueError(
                "The first radius dimension must match the number of "
                "selection-bias parameter sets."
            )

        normed_radius = radius_comoving / R0_sel[:, np.newaxis]
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
