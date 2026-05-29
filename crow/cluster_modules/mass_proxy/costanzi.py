import math
import numpy as np
import numpy.typing as npt
import scipy.special as spc

from ..parameters import Parameters
# Import the Gaussian protocol/base class, similar to Murata
from .gaussian_protocol import MassRichnessGaussian

# Type alias for functions accepting scalars or array-like inputs
arrayLike = int | float | npt.ArrayLike

# 1. Define default parameter values.
COSTANZI_DEFAULT_PARAMETERS = {
    # Baseline Mass-Richness scaling parameters (placeholders, similar to Murata)
    "mu0": 3.0, "mu1": 0.8, "mu2": -0.3,
    "sigma0": 0.3, "sigma1": 0.0, "sigma2": 0.0,
    
    # Costanzi-specific projection effect parameters from the test script
    "tau": 0.10,
    "delta_mu": -2.0,
    "sig_pure_scatter": 0.15,  # Percentage of rich_tru (0.15 * rich_tru)
    "fprj": 0.95,
    "fmsk": 0.05,
}



class CostanziBaseModel:
    """
    Projection effects model from Costanzi+19.
    Implemented as a CROW module.
    """


    def __init__(self, pivot_log_mass: float, pivot_redshift: float):
        super().__init__()
        self.pivot_redshift = pivot_redshift
        self.pivot_ln_mass = pivot_log_mass * np.log(10.0)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)
        # Manage parameters collectively using the Parameters object
        self.parameters = Parameters({**COSTANZI_DEFAULT_PARAMETERS})

     # Linear calculation helper as a function of mass and redshift, identical to MurataModel
    @staticmethod
    def observed_value(p, log_mass, z, pivot_ln_mass, log1p_pivot_redshift):
        ln_mass = log_mass * np.log(10)
        delta_ln_mass = ln_mass - pivot_ln_mass
        delta_z = np.log1p(z) - log1p_pivot_redshift
        return p[0] + p[1] * delta_ln_mass + p[2] * delta_z

    def get_ln_mass_proxy_mean(self, log_mass, z):
        return self.observed_value(
            (self.parameters["mu0"], self.parameters["mu1"], self.parameters["mu2"]),
            log_mass, z, self.pivot_ln_mass, self.log1p_pivot_redshift
        )

    def get_ln_mass_proxy_sigma(self, log_mass, z):
        return self.observed_value(
            (self.parameters["sigma0"], self.parameters["sigma1"], self.parameters["sigma2"]),
            log_mass, z, self.pivot_ln_mass, self.log1p_pivot_redshift
        )

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
        prob_obs = CostanziBaseModel.prob_richobs_at_richtru(
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
    


class CostanziBinned(CostanziBaseModel, MassRichnessGaussian):
    """
    Costanzi model implementation for binned data vectors.
    """
    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy_limits: tuple[float, float],
    ) -> npt.NDArray[np.float64]:

        # Retrieve model parameters
        tau = self.parameters["tau"]
        delta_mu = self.parameters["delta_mu"]
        sig_pure_scatter = self.parameters["sig_pure_scatter"]
        fprj = self.parameters["fprj"]
        fmsk = self.parameters["fmsk"]

        # Convert observed richness bin edges from log10 space to linear space
        rich_obs_eds = [10**log_mass_proxy_limits[0], 10**log_mass_proxy_limits[1]]

        # Grid for true richness defined based on the supervisor's test script (1E-2 to 250)
        rich_tru_grid = np.geomspace(1E-2, 250, 100)

        # Calculate sig_pure dynamically as it scales with rich_tru
        sig_pure = sig_pure_scatter * rich_tru_grid

        # Call integration logic using the resolution from the supervisor's script (0.015)
        # Note: Depending on the pipeline requirement, rich_tru_grid may need adjustment
        sprob = self.Sprob_at_richtru(
            rich_obs_eds=rich_obs_eds,
            rich_obs_res=0.015,
            rich_tru=rich_tru_grid,
            tau=tau, delta_mu=delta_mu, sig_pure=sig_pure, fprj=fprj, fmsk=fmsk
        )

        # TODO: Implement the final marginalization layer over rich_tru_grid
        # Currently returning a dummy array to allow initial testing of imports
        return np.ones_like(log_mass) * 0.1
