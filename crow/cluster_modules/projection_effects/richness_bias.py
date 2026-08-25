import math

import numpy as np
import numpy.typing as npt
import scipy.special as spc

from ..parameters import Parameters

RICHNESS_BIAS_DEFAULT_PARAMETERS = {
    "tau": 0.10,
    "delta_mu": -2.0,
    "fractional_sig_pure": 0.15,
    "fprj": 0.95,
    "fmsk": 0.05,
}


class CostanziRichnessBias:
    """
    Projection effects model from Costanzi+19.

    Each instance owns one calibration in ``parameters``. Parameter values may
    be scalars or one-dimensional arrays broadcastable to the true-richness
    grid. ``fractional_sig_pure`` is the dimensionless fractional scatter,
    such that ``sig_pure = fractional_sig_pure * rich_tru``.

    Implemented as a CROW module.
    """

    def __init__(
        self,
        tau: npt.NDArray[np.float64] | None = None,
        delta_mu: npt.NDArray[np.float64] | None = None,
        fractional_sig_pure: npt.NDArray[np.float64] | None = None,
        fprj: npt.NDArray[np.float64] | None = None,
        fmsk: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize the Costanzi projection model.

        Parameters
        ----------
        tau : ndarray
            The projection effect parameter. If None, use the default value.
        delta_mu : ndarray
            The bias in the mean projected richness relative to the true richness,
            as ``delta_mu = mu - rich_tru``, where ``mu`` is the mean projected
            richness. If None, use the default value.
        fractional_sig_pure : ndarray
            The fractional scatter of the projected richness. The absolute scatter
            is ``sig_pure = fractional_sig_pure * rich_tru``, with projected
            richness distributed as ``rich_tru + N(delta_mu, sig_pure)``. If None,
            use the default value.
        fprj : ndarray
            The fraction of clusters affected by projection. If None, use the
            default value.
        fmsk : ndarray
            The fraction of clusters being masked by others. If None, use the
            default value.
        """
        self.parameters = Parameters({**RICHNESS_BIAS_DEFAULT_PARAMETERS})
        if tau is not None:
            self.parameters["tau"] = tau
        if delta_mu is not None:
            self.parameters["delta_mu"] = delta_mu
        if fractional_sig_pure is not None:
            self.parameters["fractional_sig_pure"] = fractional_sig_pure
        if fprj is not None:
            self.parameters["fprj"] = fprj
        if fmsk is not None:
            self.parameters["fmsk"] = fmsk

    @staticmethod
    def _exp_times_erfc(
        log_factor: npt.NDArray[np.float64],
        erfc_arg: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate exp(log_factor) * erfc(erfc_arg) without overflow.

        The Costanzi projection model contains terms of the form
        ``exp(x) * erfc(y)``.  For high true-richness grid points, ``exp(x)``
        can overflow even when the product is finite because ``erfc(y)`` is
        exponentially small.  For positive ``y`` we rewrite the expression with
        ``erfcx(y) = exp(y**2) * erfc(y)``.
        """
        log_factor = np.asarray(log_factor, dtype=float)
        erfc_arg = np.asarray(erfc_arg, dtype=float)
        result = np.empty_like(log_factor, dtype=float)

        positive_arg = erfc_arg >= 0.0
        result[positive_arg] = np.exp(
            log_factor[positive_arg] - erfc_arg[positive_arg] ** 2
        ) * spc.erfcx(erfc_arg[positive_arg])
        result[~positive_arg] = np.exp(log_factor[~positive_arg]) * spc.erfc(
            erfc_arg[~positive_arg]
        )
        return result

    def prob_richobs_at_richtru(
        self,
        rich_obs: npt.NDArray[np.float64],
        rich_tru: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.floating]:
        """
        Calculate the probability of observing the observed richness given the true richness.
        Equation 15 in Costanzi+19.

        Parameters:
        ----------------------------------------------------------
        rich_obs: ndarray
            The observed richness
        rich_tru: ndarray
            The true richness

        Notes
        -----
        This calculation uses ``tau``, ``delta_mu``, ``fractional_sig_pure``,
        ``fprj``, and ``fmsk`` from :attr:`parameters`.

        Return:
        ----------------------------------------------------------
        prob: ndarray
            The probability of P(rich_obs | rich_tru).
            The shape is (len(rich_obs), len(rich_tru))
        """
        # sanitize
        rich_obs = np.asarray(rich_obs)
        rich_tru = np.asarray(rich_tru)

        # Broadcast model parameters over the true-richness grid.
        tau = np.broadcast_to(
            np.asarray(self.parameters["tau"], dtype=float), rich_tru.shape
        )
        delta_mu = np.broadcast_to(
            np.asarray(self.parameters["delta_mu"], dtype=float), rich_tru.shape
        )
        fractional_sig_pure = np.broadcast_to(
            np.asarray(self.parameters["fractional_sig_pure"], dtype=float),
            rich_tru.shape,
        )
        sig_pure = fractional_sig_pure * rich_tru
        fprj = np.broadcast_to(
            np.asarray(self.parameters["fprj"], dtype=float), rich_tru.shape
        )
        fmsk = np.broadcast_to(
            np.asarray(self.parameters["fmsk"], dtype=float), rich_tru.shape
        )

        # Safety checks
        if not np.all(tau > 0.0):
            raise ValueError("tau can only be positive.")
        if not np.all(fractional_sig_pure > 0.0):
            raise ValueError("fractional_sig_pure can only be positive.")
        if not np.all(sig_pure > 0.0):
            raise ValueError("sig_pure can only be positive.")
        if not np.all((fprj >= 0.0) & (fprj <= 1.0)):
            raise ValueError("fprj can only be between 0 and 1.")
        if not np.all((fmsk >= 0.0) & (fmsk <= 1.0)):
            raise ValueError("fmsk can only be between 0 and 1.")

        # Forcing the shapes in (n_obs,1,) or (1,n_tru,)
        rich_obs = rich_obs[:, np.newaxis]
        rich_tru = rich_tru[np.newaxis, :]
        tau = tau[np.newaxis, :]
        delta_mu = delta_mu[np.newaxis, :]
        sig_pure = sig_pure[np.newaxis, :]
        fprj = fprj[np.newaxis, :]
        fmsk = fmsk[np.newaxis, :]

        # Calculation logic
        sig2_l = sig_pure**2  # (n_tru,)
        mu = rich_tru + delta_mu  # (n_tru,)
        erfc_arg1 = (mu + tau * sig2_l - rich_obs) / np.sqrt(
            2.0 * sig2_l
        )  # (n_obs, n_tru,)
        erfc_arg2 = (mu - rich_obs - rich_tru) / np.sqrt(
            2.0 * sig2_l
        )  # (n_obs, n_tru,)
        erfc_arg3 = (mu - rich_obs) / np.sqrt(2.0 * sig2_l)  # (n_obs, n_tru,)
        erfc_arg4 = (mu + tau * sig2_l - rich_obs - rich_tru) / np.sqrt(
            2.0 * sig2_l
        )  # (n_obs, n_tru,)
        log_exptau = (
            0.5 * tau * (2.0 * mu + tau * sig2_l - 2.0 * rich_obs)
        )  # (n_obs, n_tru,)

        gauss = (
            (1.0 - fmsk)
            * (1.0 - fprj)
            * np.exp(-0.5 * (rich_obs - mu) ** 2 / sig2_l)
            / np.sqrt(2.0 * math.pi * sig2_l)
        ) * 2.0  # (n_obs, n_tru,)
        term1 = (
            (1.0 - fmsk) * fprj * tau + fmsk * fprj / rich_tru
        ) * CostanziRichnessBias._exp_times_erfc(log_exptau, erfc_arg1)
        term23 = fmsk / rich_tru * (spc.erfc(erfc_arg2) - spc.erfc(erfc_arg3))
        term4 = (
            fmsk
            * fprj
            / rich_tru
            * CostanziRichnessBias._exp_times_erfc(
                log_exptau - tau * rich_tru,
                erfc_arg4,
            )
        )
        pdf = 0.5 * (gauss + term1 + term23 - term4)  # (n_obs, n_tru,)
        return pdf

    # def Sprob_at_richtru(
    #    self,
    #    rich_obs_eds: npt.NDArray[np.float64],
    #    rich_obs_res: npt.NDArray[np.float64],
    #    rich_tru: npt.NDArray[np.float64],
    # ) -> npt.NDArray[np.floating]:
    #    """
    #    Integrate the observed richness over an interval defined by (rich_obs_low, rich_obs_hgh, rich_obs_res)

    #    Parameters:
    #    ----------------------------------------------------------
    #    rich_obs_eds: ndarray
    #        The boundaries defining the richness intervals.
    #        It has a dimension of 1 and length at least 2.
    #    rich_obs_res: ndarray
    #        The resolution used in the integration over each richness integral.
    #        If a constant is provided, all richness integrals use the same resolution.
    #        If an array provided, it has a dimension of 1 and length exactly of len(rich_obs_eds) - 1.
    #        The resolution is defined as the fractional increase of 1 + rich_obs_res.
    #        Smaller rich_obs_res means higher resolutions.
    #    rich_tru: ndarray
    #        The true richness

    #    Return:
    #    ----------------------------------------------------------
    #    Sprob_at_richtru: ndarray
    #        The probability of \\int drich_obs P(rich_obs | rich_tru).
    #        The shape is (len(rich_obs_eds) - 1, len(rich_tru))
    #    """
    #    # sanitize
    #    rich_obs_eds = np.asarray(rich_obs_eds)
    #    rich_obs_res = np.asarray(rich_obs_res)
    #    rich_tru = np.asarray(rich_tru)

    #    # define integral boundaries
    #    if rich_obs_eds.ndim != 1:
    #        raise ValueError(
    #            "The array rich_obs_eds ndim is not 1. "
    #            "Integral boundaries cannot be defined."
    #        )
    #    if not np.all(np.diff(rich_obs_eds) > 0):
    #        raise ValueError(
    #            "The array rich_obs_eds has to be monotonically increasing."
    #        )

    #    rich_obs_low = rich_obs_eds[:-1]
    #    rich_obs_hgh = rich_obs_eds[1:]

    #    # define integral resolution
    #    rich_obs_res = np.broadcast_to(
    #        np.asarray(rich_obs_res, dtype=float), rich_obs_low.shape
    #    )

    #    rich_obs_nst = (np.log(rich_obs_hgh / rich_obs_low) / rich_obs_res).astype(int)
    #    rich_obs_edges = np.hstack(
    #        [
    #            (
    #                np.geomspace(rich_obs_low[n], rich_obs_hgh[n], rich_obs_nst[n])
    #                if n == 0
    #                else np.geomspace(
    #                    rich_obs_low[n], rich_obs_hgh[n], rich_obs_nst[n]
    #                )[1:]
    #            )
    #            for n in range(len(rich_obs_low))
    #        ]
    #    )
    #    rich_obs_bins = np.sqrt(rich_obs_edges[:-1] * rich_obs_edges[1:])
    #    rich_obs_steps = np.diff(rich_obs_edges)
    #    rich_obs_digit = np.digitize(rich_obs_bins, bins=rich_obs_eds)

    #    # calc
    #    prob_obs = self.prob_richobs_at_richtru(
    #        rich_obs=rich_obs_bins,
    #        rich_tru=rich_tru,
    #    )

    #    Sprob_obs = np.array(
    #        [
    #            np.sum(
    #                rich_obs_steps[rich_obs_digit == ith][:, np.newaxis]
    #                * prob_obs[rich_obs_digit == ith],
    #                axis=0,
    #            )
    #            for ith in range(1, len(rich_obs_eds))
    #        ]
    #    )
    #    return Sprob_obs
