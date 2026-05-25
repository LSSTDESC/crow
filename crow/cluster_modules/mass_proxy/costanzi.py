import math

import numpy as np

# from typing import Union
import numpy.typing as npt
import scipy.special as spc

# define arrayLike
# arrayLike = Union[int, float, npt.NDArray[np.floating], npt.NDArray[np.integer]]
arrayLike = int | float | npt.ArrayLike


class CostanziModel:
    """
    Projection effects model from Costanzi+19.
    Implemented as a CROW module.
    """

    def __init__(self):
        # Initialize fixed parameters here if needed in the future
        pass

    @staticmethod
    def prob_richobs_at_richtru(
        rich_obs: arrayLike,
        rich_tru: arrayLike,
        *,
        tau: arrayLike,
        delta_mu: arrayLike,
        sig_pure: arrayLike,
        fprj: arrayLike,
        fmsk: arrayLike,
    ) -> arrayLike:
        """
        Calculate the probability of observing the observed richness given the true richness.
        Equation 15 in Costanzi+19.

        Parameters:
        ----------------------------------------------------------
        rich_obs: 1d array
            The observed richnes
        rich_tru: 1d array
            The true richness
        tau: flaot or 1d array
            The projection effect parameter
        delta_mu: float or 1d array
            The bias in the mean of the projected richness w.r.t. the true richness,
            as delta_mu = mu - rich_true, where mu is the mean projected richness
        sig_pure: float or 1d array
            The scatter of the projected richness,
            as the projected richness ~ rich_true + N(delta_mu, sig_pure)
        fprj: float or 1d array
            The fraction of clusters affected by prjection
        fmsk: float or 1d array
            The fraction of clusters being masked by others

        Return:
        ----------------------------------------------------------
        prob: ndarray
            The probability of P(rich_obs | rich_tru).
            The shape is (len(rich_obs), len(rich_tru))
        """
        # sanitize
        rich_obs = np.asarray(rich_obs)
        rich_tru = np.asarray(rich_tru)

        # Broadcast arrays
        tau = np.broadcast_to(np.asarray(tau, dtype=float), rich_tru.shape)
        delta_mu = np.broadcast_to(np.asarray(delta_mu, dtype=float), rich_tru.shape)
        sig_pure = np.broadcast_to(np.asarray(sig_pure, dtype=float), rich_tru.shape)
        fprj = np.broadcast_to(np.asarray(fprj, dtype=float), rich_tru.shape)
        fmsk = np.broadcast_to(np.asarray(fmsk, dtype=float), rich_tru.shape)

        # Safety checks
        assert np.all(tau > 0.0), "tau can only be positive."
        assert np.all(sig_pure > 0.0), "sig_pure can only be positive."
        assert np.all(
            (fprj >= 0.0) & (fprj <= 1.0)
        ), "fprj can only be between 0 and 1."
        assert np.all(
            (fmsk >= 0.0) & (fmsk <= 1.0)
        ), "fmsk can only be between 0 and 1."

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
        exptau = np.exp(
            0.5 * tau * (2.0 * mu + tau * sig2_l - 2.0 * rich_obs)
        )  # (n_obs, n_tru,)

        gauss = (
            (1.0 - fmsk)
            * (1.0 - fprj)
            * np.exp(-0.5 * (rich_obs - mu) ** 2 / sig2_l)
            / np.sqrt(2.0 * math.pi * sig2_l)
        ) * 2.0  # (n_obs, n_tru,)
        term1 = (
            ((1.0 - fmsk) * fprj * tau + fmsk * fprj / rich_tru)
            * exptau
            * spc.erfc(erfc_arg1)
        )
        term23 = fmsk / rich_tru * (spc.erfc(erfc_arg2) - spc.erfc(erfc_arg3))
        term4 = (
            fmsk
            * fprj
            / rich_tru
            * np.exp(-tau * rich_tru)
            * exptau
            * spc.erfc(erfc_arg4)
        )
        pdf = 0.5 * (gauss + term1 + term23 - term4)  # (n_obs, n_tru,)
        return pdf

    @staticmethod
    def Sprob_at_richtru(
        rich_obs_eds: arrayLike,
        rich_obs_res: arrayLike,
        rich_tru: arrayLike,
        *,
        tau: arrayLike,
        delta_mu: arrayLike,
        sig_pure: arrayLike,
        fprj: arrayLike,
        fmsk: arrayLike,
    ) -> arrayLike:
        """
        Integrate the observed richness over an interval defined by (rich_obs_low, rich_obs_hgh, rich_obs_res)

        Parameters:
        ----------------------------------------------------------
        rich_obs_eds: 1d array
            The boundaries defining the richness intervals.
            It has a dimension of 1 and length at least 2.
        rich_obs_res: 1d array
            The resolution used in the integration over each richness integral.
            If a onstant provided, all richness integrals use the same resolution.
            If an array provided, it has a dimension of 1 and length exactly of len(rich_obs_eds) - 1.
            The resolution is defined as the fractional increase of 1 + rich_obs_res.
            Smaller rich_obs_res means higher resolutions.
        rich_tru: 1d array
            The true richness
        tau: flaot or 1d array
            The projection effect parameter
        delta_mu: float or 1d array
            The bias in the mean of the projected richness w.r.t. the true richness,
            as delta_mu = mu - rich_true, where mu is the mean projected richness
        sig_pure: float or 1d array
            The scatter of the projected richness,
            as the projected richness ~ rich_true + N(delta_mu, sig_pure)
        fprj: float or 1d array
            The fraction of clusters affected by prjection
        fmsk: float or 1d array
            The fraction of clusters being masked by others

        Return:
        ----------------------------------------------------------
        Sprob_at_richtru: ndarray
            The probability of \int drich_obs P(rich_obs | rich_tru).
            The shape is (len(rich_obs_eds) - 1, len(rich_tru))
        """
        # sanitize
        rich_obs_eds = np.asarray(rich_obs_eds)
        rich_obs_res = np.asarray(rich_obs_res)
        rich_tru = np.asarray(rich_tru)

        # Broadcast arrays
        tau = np.broadcast_to(np.asarray(tau, dtype=float), rich_tru.shape)
        delta_mu = np.broadcast_to(np.asarray(delta_mu, dtype=float), rich_tru.shape)
        sig_pure = np.broadcast_to(np.asarray(sig_pure, dtype=float), rich_tru.shape)
        fprj = np.broadcast_to(np.asarray(fprj, dtype=float), rich_tru.shape)
        fmsk = np.broadcast_to(np.asarray(fmsk, dtype=float), rich_tru.shape)

        # define integral boundaries
        assert (
            rich_obs_eds.ndim == 1
        ), "The array rich_obs_eds ndim is not 1. Intergral boundaries cannot be defined."
        assert np.all(
            np.diff(rich_obs_eds) > 0
        ), "The array rich_obs_eds has to be monotonically increasing."

        rich_obs_low = rich_obs_eds[:-1]
        rich_obs_hgh = rich_obs_eds[1:]

        # define integral resolution
        rich_obs_res = np.broadcast_to(
            np.asarray(rich_obs_res, dtype=float), rich_obs_low.shape
        )

        rich_obs_nst = (np.log(rich_obs_hgh / rich_obs_low) / rich_obs_res).astype(int)
        rich_obs_edges = np.hstack(
            [
                (
                    np.geomspace(rich_obs_low[n], rich_obs_hgh[n], rich_obs_nst[n])
                    if n == 0
                    else np.geomspace(
                        rich_obs_low[n], rich_obs_hgh[n], rich_obs_nst[n]
                    )[1:]
                )
                for n in range(len(rich_obs_low))
            ]
        )
        rich_obs_bins = np.sqrt(rich_obs_edges[:-1] * rich_obs_edges[1:])
        rich_obs_steps = np.diff(rich_obs_edges)
        rich_obs_digit = np.digitize(rich_obs_bins, bins=rich_obs_eds)

        # calc
        prob_obs = CostanziModel.prob_richobs_at_richtru(
            rich_obs=rich_obs_bins,
            rich_tru=rich_tru,
            tau=tau,
            delta_mu=delta_mu,
            sig_pure=sig_pure,
            fprj=fprj,
            fmsk=fmsk,
        )

        Sprob_obs = np.array(
            [
                np.sum(
                    rich_obs_steps[rich_obs_digit == ith][:, np.newaxis]
                    * prob_obs[rich_obs_digit == ith],
                    axis=0,
                )
                for ith in range(1, len(rich_obs_eds))
            ]
        )
        return Sprob_obs
