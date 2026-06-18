import math

import numpy as np
import numpy.typing as npt
import scipy.special as spc
from scipy.integrate import simpson

# Import the Gaussian protocol/base class, similar to Murata
from ..mass_proxy.gaussian_protocol import MassRichnessGaussian
from ..parameters import Parameters

# Type alias for functions accepting scalars or array-like inputs
arrayLike = int | float | npt.ArrayLike

GAUSSIAN_DEFAULT_PARAMETERS = {
    "mu0": 3.0,
    "mu1": 0.8,
    "mu2": -0.3,
    "sigma0": 0.3,
    "sigma1": 0.0,
    "sigma2": 0.0,
}

COSTANZI_DEFAULT_PARAMETERS = {
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
        ) * CostanziBaseModel._exp_times_erfc(log_exptau, erfc_arg1)
        term23 = fmsk / rich_tru * (spc.erfc(erfc_arg2) - spc.erfc(erfc_arg3))
        term4 = (
            fmsk
            * fprj
            / rich_tru
            * CostanziBaseModel._exp_times_erfc(
                log_exptau - tau * rich_tru,
                erfc_arg4,
            )
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
            The probability of \\int drich_obs P(rich_obs | rich_tru).
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
    Costanzi projection effects on the standard mass-richness relation.
    """

    def __init__(
        self,
        pivot_log_mass: float,
        pivot_redshift: float,
        tru_proxy_grid_size: int = 100,
        tru_proxy_log_padding: float = 0.5,
        projection_richness_resolution: float = 0.015,
    ):
        """
        Initialize the Costanzi projection model with a Gaussian richness-mass relation.

        Parameters:
        ----------------------------------------------------------
        pivot_log_mass: float
            The pivot halo mass in log10 space.
        pivot_redshift: float
            The pivot redshift.
        tru_proxy_grid_size: int
            The number of grid points used for the true richness integration.
        tru_proxy_log_padding: float
            The padding added to both sides of the observed richness interval
            when defining the true richness grid in log10 space.
        projection_richness_resolution: float
            The fractional resolution used when integrating over observed
            richness intervals in the projection-effect model.
        """
        self.pivot_redshift = pivot_redshift
        self.pivot_ln_mass = pivot_log_mass * np.log(10.0)
        self.ln1p_pivot_redshift = np.log1p(self.pivot_redshift)
        self.parameters = Parameters(
            {**GAUSSIAN_DEFAULT_PARAMETERS, **COSTANZI_DEFAULT_PARAMETERS}
        )
        self.tru_proxy_grid_size = tru_proxy_grid_size
        self.tru_proxy_log_padding = tru_proxy_log_padding
        self.projection_richness_resolution = projection_richness_resolution

    @staticmethod
    def observed_value(p, log_mass, z, pivot_ln_mass, ln1p_pivot_redshift):
        """
        Evaluate a linear observable model in natural-log mass and redshift.

        Parameters:
        ----------------------------------------------------------
        p: tuple
            Three model parameters: normalization, mass slope, and redshift slope.
        log_mass: array_like
            The halo mass values in log10 space.
        z: array_like
            The redshift values.
        pivot_ln_mass: float
            The pivot halo mass in natural-log space.
        ln1p_pivot_redshift: float
            The natural logarithm of 1 + pivot_redshift.

        Return:
        ----------------------------------------------------------
        observed_value: ndarray
            The model value evaluated at log_mass and z.
        """
        ln_mass = log_mass * np.log(10)
        delta_ln_mass = ln_mass - pivot_ln_mass
        delta_ln1p_redshift = np.log1p(z) - ln1p_pivot_redshift
        return p[0] + p[1] * delta_ln_mass + p[2] * delta_ln1p_redshift

    def get_ln_mass_proxy_mean(self, log_mass, z):
        """
        Calculate the mean of the Gaussian richness-mass relation.

        Parameters:
        ----------------------------------------------------------
        log_mass: array_like
            The halo mass values in log10 space.
        z: array_like
            The redshift values.

        Return:
        ----------------------------------------------------------
        ln_mass_proxy_mean: ndarray
            The mean of the mass proxy in natural-log richness space.
        """
        return self.observed_value(
            (self.parameters["mu0"], self.parameters["mu1"], self.parameters["mu2"]),
            log_mass,
            z,
            self.pivot_ln_mass,
            self.ln1p_pivot_redshift,
        )

    def get_ln_mass_proxy_sigma(self, log_mass, z):
        """
        Calculate the scatter of the Gaussian richness-mass relation.

        Parameters:
        ----------------------------------------------------------
        log_mass: array_like
            The halo mass values in log10 space.
        z: array_like
            The redshift values.

        Return:
        ----------------------------------------------------------
        ln_mass_proxy_sigma: ndarray
            The Gaussian scatter in natural-log richness space.
        """
        return self.observed_value(
            (
                self.parameters["sigma0"],
                self.parameters["sigma1"],
                self.parameters["sigma2"],
            ),
            log_mass,
            z,
            self.pivot_ln_mass,
            self.ln1p_pivot_redshift,
        )

    def compute_probabilities(
        self,
        rich_obs_eds: arrayLike,
        rich_obs_res: arrayLike,
        log_rich_tru: arrayLike,
        log_mass: arrayLike,
        z: arrayLike,
        *,
        tau: arrayLike,
        delta_mu: arrayLike,
        sig_pure: arrayLike,
        fprj: arrayLike,
        fmsk: arrayLike,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Integrate the true richness to calculate the projected richness-mass relation.

        Parameters:
        ----------------------------------------------------------
        rich_obs_eds: 1d array
            The boundaries defining the observed richness intervals.
            It has a dimension of 1 and length at least 2.
        rich_obs_res: 1d array
            The resolution used in the integration over each observed richness interval.
            If a constant is provided, all observed richness integrals use the same resolution.
            If an array is provided, it has a dimension of 1 and length exactly of len(rich_obs_eds) - 1.
            The resolution is defined as the fractional increase of 1 + rich_obs_res.
            Smaller rich_obs_res means higher resolutions.
        log_rich_tru: 1d array
            The true richness grid in log10 space.
        log_mass: 1d array
            The halo mass values in log10 space.
            If log_mass and z are 1D arrays with different lengths, they are
            evaluated as a full redshift-mass mesh.
        z: 1d array
            The redshift values.
            If log_mass and z are 1D arrays with the same length, they are
            evaluated as paired arrays. Otherwise the shapes must be
            broadcastable with log_mass.
        tau: float or 1d array
            The projection effect parameter.
        delta_mu: float or 1d array
            The bias in the mean of the projected richness w.r.t. the true richness,
            as delta_mu = mu - rich_true, where mu is the mean projected richness.
        sig_pure: float or 1d array
            The scatter of the projected richness,
            as the projected richness ~ rich_true + N(delta_mu, sig_pure).
        fprj: float or 1d array
            The fraction of clusters affected by projection.
        fmsk: float or 1d array
            The fraction of clusters being masked by others.

        Return:
        ----------------------------------------------------------
        projection_probabilities: dict
            Dictionary containing the probability grids:
            Sprob_richobs_richtru has shape (len(rich_obs_eds) - 1, len(log_rich_tru)).
            prob_richtru_mass_redshift has shape (len(log_rich_tru), *pair_shape).
            Sprob_richobs_mass_redshift has shape (len(rich_obs_eds) - 1, *pair_shape).
        """
        rich_obs_eds = np.asarray(rich_obs_eds, dtype=float)
        log_rich_tru = np.asarray(log_rich_tru, dtype=float)
        rich_tru = 10.0**log_rich_tru
        log_mass = np.asarray(log_mass, dtype=float)
        z = np.asarray(z, dtype=float)

        if log_rich_tru.ndim != 1:
            raise ValueError("log_rich_tru must be a 1D array.")

        if log_mass.ndim == 1 and z.ndim == 1 and len(log_mass) != len(z):
            log_mass, z = np.meshgrid(log_mass, z, indexing="xy")
        else:
            log_mass, z = np.broadcast_arrays(log_mass, z)

        pair_shape = log_mass.shape
        log_mass = log_mass.reshape(-1)
        z = z.reshape(-1)

        Sprob_richobs_richtru = CostanziBaseModel.Sprob_at_richtru(
            rich_obs_eds=rich_obs_eds,
            rich_obs_res=rich_obs_res,
            rich_tru=rich_tru,
            tau=tau,
            delta_mu=delta_mu,
            sig_pure=sig_pure,
            fprj=fprj,
            fmsk=fmsk,
        )

        prob_richtru_mass_redshift = self.gaussian_kernel(
            np.broadcast_to(
                log_mass[np.newaxis, :], (len(log_rich_tru), len(log_mass))
            ),
            np.broadcast_to(z[np.newaxis, :], (len(log_rich_tru), len(z))),
            np.broadcast_to(
                log_rich_tru[:, np.newaxis], (len(log_rich_tru), len(log_mass))
            ),
        )
        ln_rich_tru = log_rich_tru * np.log(10.0)
        Sprob_richobs_mass_redshift = simpson(
            y=(
                Sprob_richobs_richtru[:, :, np.newaxis]
                * prob_richtru_mass_redshift[np.newaxis, :, :]
            ),
            x=ln_rich_tru,
            axis=1,
        )
        Sprob_richobs_mass_redshift = Sprob_richobs_mass_redshift.reshape(
            (len(rich_obs_eds) - 1,) + pair_shape
        )
        prob_richtru_mass_redshift = prob_richtru_mass_redshift.reshape(
            (len(log_rich_tru),) + pair_shape
        )

        return {
            "Sprob_richobs_richtru": Sprob_richobs_richtru,
            "prob_richtru_mass_redshift": prob_richtru_mass_redshift,
            "Sprob_richobs_mass_redshift": Sprob_richobs_mass_redshift,
        }

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy_limits: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the projected probability of the observed richness interval.

        Parameters:
        ----------------------------------------------------------
        log_mass: 1d array
            The halo mass values in log10 space.
        z: 1d array
            The redshift values.
        log_mass_proxy_limits: tuple
            The lower and upper boundaries of the observed richness interval
            in log10 space.

        Return:
        ----------------------------------------------------------
        probability: ndarray
            The probability of P(rich_obs | mass, redshift) integrated over
            the observed richness interval defined by log_mass_proxy_limits.
            The shape is the same as log_mass and z.
        """
        log_rich_tru = np.linspace(
            log_mass_proxy_limits[0] - self.tru_proxy_log_padding,
            log_mass_proxy_limits[1] + self.tru_proxy_log_padding,
            self.tru_proxy_grid_size,
        )
        rich_tru = 10.0**log_rich_tru
        result = self.compute_probabilities(
            rich_obs_eds=10.0 ** np.asarray(log_mass_proxy_limits),
            rich_obs_res=self.projection_richness_resolution,
            log_rich_tru=log_rich_tru,
            log_mass=np.asarray(log_mass),
            z=np.asarray(z),
            tau=self.parameters["tau"],
            delta_mu=self.parameters["delta_mu"],
            sig_pure=self.parameters["sig_pure_scatter"] * rich_tru,
            fprj=self.parameters["fprj"],
            fmsk=self.parameters["fmsk"],
        )
        probability = result["Sprob_richobs_mass_redshift"][0]
        assert isinstance(probability, np.ndarray)
        return probability


class CostanziUnBinned(CostanziBinned):
    """
    Costanzi projection effects for unbinned data vectors.

    This implementation evaluates the projected richness probability density
    directly at observed richness values. The returned density is with respect
    to natural-log observed richness, matching the MurataUnbinned convention.
    """

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the projected PDF at the observed richness values.

        Parameters:
        ----------------------------------------------------------
        log_mass: array_like
            The halo mass values in log10 space.
        z: array_like
            The redshift values.
        log_mass_proxy: array_like
            The observed richness values in log10 space.

        Return:
        ----------------------------------------------------------
        probability: ndarray
            The probability density P(ln rich_obs | mass, redshift).
            The shape follows the broadcasted log_mass, z, and log_mass_proxy
            inputs. If log_mass and z are different-length 1D arrays, they are
            evaluated as a full redshift-mass mesh before broadcasting with
            log_mass_proxy.
        """
        log_mass = np.asarray(log_mass, dtype=float)
        z = np.asarray(z, dtype=float)
        log_mass_proxy = np.asarray(log_mass_proxy, dtype=float)

        if log_mass.ndim == 1 and z.ndim == 1 and len(log_mass) != len(z):
            log_mass, z = np.meshgrid(log_mass, z, indexing="xy")

        log_mass, z, log_mass_proxy = np.broadcast_arrays(log_mass, z, log_mass_proxy)
        pair_shape = log_mass.shape
        log_mass = log_mass.reshape(-1)
        z = z.reshape(-1)
        log_mass_proxy = log_mass_proxy.reshape(-1)
        rich_obs = 10.0**log_mass_proxy

        log_rich_tru = np.linspace(
            np.min(log_mass_proxy) - self.tru_proxy_log_padding,
            np.max(log_mass_proxy) + self.tru_proxy_log_padding,
            self.tru_proxy_grid_size,
        )
        rich_tru = 10.0**log_rich_tru

        prob_richobs_richtru = CostanziBaseModel.prob_richobs_at_richtru(
            rich_obs=rich_obs,
            rich_tru=rich_tru,
            tau=self.parameters["tau"],
            delta_mu=self.parameters["delta_mu"],
            sig_pure=self.parameters["sig_pure_scatter"] * rich_tru,
            fprj=self.parameters["fprj"],
            fmsk=self.parameters["fmsk"],
        )
        prob_lnrichobs_richtru = rich_obs[:, np.newaxis] * prob_richobs_richtru

        prob_richtru_mass_redshift = self.gaussian_kernel(
            np.broadcast_to(
                log_mass[np.newaxis, :], (len(log_rich_tru), len(log_mass))
            ),
            np.broadcast_to(z[np.newaxis, :], (len(log_rich_tru), len(z))),
            np.broadcast_to(
                log_rich_tru[:, np.newaxis], (len(log_rich_tru), len(log_mass))
            ),
        )

        ln_rich_tru = log_rich_tru * np.log(10.0)
        probability = simpson(
            y=prob_lnrichobs_richtru.T * prob_richtru_mass_redshift,
            x=ln_rich_tru,
            axis=0,
        ).reshape(pair_shape)

        assert isinstance(probability, np.ndarray)
        return probability
