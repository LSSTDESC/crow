"""Module to compute the cluster excess surface mass density (delta sigma).

The galaxy cluster delta sigma integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""

from typing import Callable, Optional, Tuple

import clmm  # pylint: disable=import-error
import numpy as np
import numpy.typing as npt
import pyccl
from clmm.utils.beta_lens import (
    compute_beta_s_mean_from_distribution,
    compute_beta_s_square_mean_from_distribution,
)
from pyccl.cosmology import Cosmology
from scipy.interpolate import interp1d
from scipy.stats import gamma

from crow import ClusterAbundance
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator

from . import _clmm_patches
from .parameters import Parameters

##############################
# Monkeypatch CLMM functions #
##############################
clmm.theory.miscentering.integrate_azimuthially_miscentered_mean_surface_density = (  # pragma: no cover
    _clmm_patches.numcosmo_miscentered_mean_surface_density
)
clmm.Modeling._eval_2halo_term_generic = (  # pragma: no cover
    # _clmm_patches._eval_2halo_term_generic_orig
    # _clmm_patches._eval_2halo_term_generic_new
    _clmm_patches._eval_2halo_term_generic_vec
)


class ClusterShearProfile(ClusterAbundance):
    """The class that calculates the predicted delta sigma of galaxy clusters.

    The excess density surface mass density is a function of a specific cosmology,
    a mass and redshift range, an area on the sky, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    shear integrand.
    """

    def __init__(
        self,
        cosmo: Cosmology,
        halo_mass_function: pyccl.halos.MassFunc,
        cluster_concentration: float | None = None,
        is_delta_sigma: bool = False,
        use_beta_s_interp: bool = False,
        two_halo_term: bool = False,
        boost_factor: bool = False,
    ) -> None:
        super().__init__(cosmo, halo_mass_function)
        self.is_delta_sigma = is_delta_sigma
        self.parameters = Parameters({"cluster_concentration": cluster_concentration})

        self.two_halo_term = two_halo_term
        self.boost_factor = boost_factor

        self._clmm_cosmo = clmm.Cosmology(be_cosmo=self._cosmo, validate_input=False)

        self._beta_parameters = None
        self._beta_s_mean_interp = None
        self._beta_s_square_mean_interp = None

        self.use_beta_s_interp = use_beta_s_interp
        self.miscentering_parameters = None
        self.approx = None
        self.vectorized = False

    @property
    def cluster_concentration(self):
        return self.parameters["cluster_concentration"]

    @cluster_concentration.setter
    def cluster_concentration(self, value):
        self.parameters["cluster_concentration"] = value

    @property
    def use_beta_s_interp(self):
        return self.__use_beta_s_interp

    @use_beta_s_interp.setter
    def use_beta_s_interp(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"value (={value}) for use_beta_s_interp must be boolean.")
        self.__use_beta_s_interp = value
        if value:
            self.eval_beta_s_mean = self._beta_s_mean_interp
            self.eval_beta_s_square_mean = self._beta_s_square_mean_interp
        else:
            self.eval_beta_s_mean = self._beta_s_mean_exact
            self.eval_beta_s_square_mean = self._beta_s_square_mean_exact

    def set_beta_parameters(
        self,
        z_inf,
        zmax=10.0,
        delta_z_cut=0.1,
        zmin=None,
        z_distrib_func=None,
        approx="order1",
    ):
        r"""Set parameters to comput mean value of the geometric lensing efficicency

        .. math::
           \left<\beta_s\right> = \frac{\int_{z = z_{min}}^{z_{max}}\beta_s(z)N(z)}
           {\int_{z = z_{min}}^{z_{max}}N(z)}

        Parameters
        ----------
        z_inf: float
            Redshift at infinity
        zmax: float, optional
            Maximum redshift to be set as the source of the galaxy when performing the sum.
            Default: 10
        delta_z_cut: float, optional
            Redshift interval to be summed with :math:`z_{cl}` to return :math:`z_{min}`.
            This feature is not used if :math:`z_{min}` is provided by the user. Default: 0.1
        zmin: float, None, optional
            Minimum redshift to be set as the source of the galaxy when performing the sum.
            Default: None
        z_distrib_func: one-parameter function, optional
            Redshift distribution function. Default is Chang et al (2013) distribution function.
        approx : str, optional
            Type of computation to be made for reduced tangential shears, options are:

                * 'order1' : Same approach as in Weighing the Giants - III (equation 6 in
                  Applegate et al. 2014; https://arxiv.org/abs/1208.0605). `z_src_info` must be
                  'beta':

                  .. math::
                      g_t\approx\frac{\left<\beta_s\right>\gamma_{\infty}}
                      {1-\left<\beta_s\right>\kappa_{\infty}}

                * 'order2' : Same approach as in Cluster Mass Calibration at High
                  Redshift (equation 12 in Schrabback et al. 2017;
                  https://arxiv.org/abs/1611.03866).
                  `z_src_info` must be 'beta':

                  .. math::
                      g_t\approx\frac{\left<\beta_s\right>\gamma_{\infty}}
                      {1-\left<\beta_s\right>\kappa_{\infty}}
                      \left(1+\left(\frac{\left<\beta_s^2\right>}
                      {\left<\beta_s\right>^2}-1\right)\left<\beta_s\right>\kappa_{\infty}\right)
        Returns
        -------
        float
            Mean value of the geometric lensing efficicency
        """
        self._beta_parameters = {
            "z_inf": z_inf,
            "zmax": zmax,
            "delta_z_cut": delta_z_cut,
            "zmin": zmin,
            "z_distrib_func": z_distrib_func,
        }
        self.approx = approx.lower()

    def _beta_s_mean_exact(self, z_cl):
        z_cl = np.asarray(z_cl)
        if z_cl.ndim == 0:
            return compute_beta_s_mean_from_distribution(
                z_cl, cosmo=self._clmm_cosmo, **self._beta_parameters
            )
        return np.array(
            [
                compute_beta_s_mean_from_distribution(
                    float(z), cosmo=self._clmm_cosmo, **self._beta_parameters
                )
                for z in z_cl
            ]
        )

    def _beta_s_square_mean_exact(self, z_cl):
        z_cl = np.asarray(z_cl)
        if z_cl.ndim == 0:
            return compute_beta_s_square_mean_from_distribution(
                z_cl, cosmo=self._clmm_cosmo, **self._beta_parameters
            )
        return np.array(
            [
                compute_beta_s_square_mean_from_distribution(
                    float(z), cosmo=self._clmm_cosmo, **self._beta_parameters
                )
                for z in z_cl
            ]
        )

    def set_beta_s_interp(self, z_min, z_max, n_intep=3):

        # Note: this will set an interpolator with a fixed cosmology
        # must add check to verify consistency with main cosmology

        redshift_points = np.linspace(z_min, z_max, n_intep)
        beta_s_list = [self._beta_s_mean_exact(z_cl) for z_cl in redshift_points]
        self._beta_s_mean_interp = interp1d(
            redshift_points, beta_s_list, kind="quadratic", fill_value="extrapolate"
        )
        beta_s_square_list = [
            self._beta_s_square_mean_exact(z_cl) for z_cl in redshift_points
        ]
        self._beta_s_square_mean_interp = interp1d(
            redshift_points,
            beta_s_square_list,
            kind="quadratic",
            fill_value="extrapolate",
        )
        self.use_beta_s_interp = self.use_beta_s_interp

    def compute_shear_profile(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        radius_center: np.float64,
    ) -> npt.NDArray[np.float64]:
        """Delta sigma for cprint(new_pred)lusters."""
        mass_def = self.halo_mass_function.mass_def
        mass_type = mass_def.rho_type
        if mass_type == "matter":
            mass_type = "mean"
        moo = clmm.Modeling(
            massdef=mass_type,
            delta_mdef=mass_def.Delta,
            halo_profile_model="nfw",
            validate_input=False,
        )
        moo.set_cosmo(self._clmm_cosmo)
        moo.validate_input = False
        # NOTE: value set up not to break use in pyccl with firecronw
        # to be investigated
        moo.z_inf = 10.0
        return_vals = []
        for log_m, redshift in zip(log_mass, z):
            # pylint: disable=protected-access
            moo.set_concentration(self._get_concentration(log_m, redshift))
            moo.set_mass(10**log_m)
            val = self._one_halo_contribution(
                moo,
                radius_center,
                redshift,
            )
            if self.two_halo_term:
                val += self._two_halo_contribution(moo, radius_center, redshift)
            if self.boost_factor:
                val = self._correct_with_boost_nfw(val, radius_center)
            return_vals.append(val)
        return np.asarray(return_vals, dtype=np.float64)

    def compute_shear_profile_vectorized(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        radius_center: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Delta sigma for cprint(new_pred)lusters."""
        mass_def = self.halo_mass_function.mass_def
        mass_type = mass_def.rho_type
        if mass_type == "matter":
            mass_type = "mean"
        moo = clmm.Modeling(
            massdef=mass_type,
            delta_mdef=mass_def.Delta,
            halo_profile_model="nfw",
            validate_input=False,
        )
        moo.set_cosmo(self._clmm_cosmo)
        moo.validate_input = False
        # NOTE: value set up not to break use in pyccl with firecronw
        # to be investigated
        moo.z_inf = np.full_like(z, 10.0)
        moo._set_concentration(self._get_concentration(log_mass, z))
        moo._set_mass(10**log_mass)
        return_vals = self._one_halo_contribution(moo, radius_center, z)
        if self.two_halo_term:
            return_vals += moo._eval_excess_surface_density_2h(radius_center, z)
        if self.boost_factor:
            return_vals = self._correct_with_boost_nfw(return_vals, radius_center)
        return return_vals

    def _one_halo_contribution(
        self,
        clmm_model: clmm.Modeling,
        radius_center,
        redshift,
        sigma_offset=0.12,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Calculate the second halo contribution to the delta sigma."""
        beta_s_mean = None
        beta_s_square_mean = None
        if self.is_delta_sigma:
            first_halo_right_centered = clmm_model.eval_excess_surface_density(
                radius_center, redshift
            )
        else:
            beta_s_mean = self.eval_beta_s_mean(redshift)
            beta_s_square_mean = self.eval_beta_s_square_mean(redshift)
            first_halo_right_centered = clmm_model.eval_reduced_tangential_shear(
                radius_center,
                redshift,
                (beta_s_mean, beta_s_square_mean),
                z_src_info="beta",
                approx=self.approx,
            )

        if self.miscentering_parameters is not None:
            miscentering_integral, miscentering_frac = self.compute_miscentering(
                clmm_model, radius_center, redshift, beta_s_mean, beta_s_square_mean
            )
            return (
                (1.0 - miscentering_frac) * first_halo_right_centered
                + miscentering_frac * miscentering_integral
            )
        return first_halo_right_centered

    def _two_halo_contribution(
        self, clmm_model: clmm.Modeling, radius_center, redshift
    ) -> npt.NDArray[np.float64]:
        """Calculate the second halo contribution to the delta sigma."""
        # pylint: disable=protected-access
        if self.is_delta_sigma == False:
            raise Exception("Two halo contribution for gt is not suported yet.")

        second_halo_right_centered = clmm_model.eval_excess_surface_density_2h(
            np.atleast_1d(radius_center), redshift
        )

        return second_halo_right_centered[0]

    def _get_concentration(self, log_m: float, redshift: float) -> float:
        """Determine the concentration for a halo."""
        if self.cluster_concentration is not None:
            return self.cluster_concentration

        conc_model = pyccl.halos.concentration.ConcentrationBhattacharya13(
            mass_def=self.halo_mass_function.mass_def
        )
        a = 1.0 / (1.0 + redshift)
        return conc_model._concentration(
            self._cosmo, 10.0**log_m, a
        )  # pylint: disable=protected-access

    def _correct_with_boost_nfw(
        self, profiles: npt.NDArray[np.float64], radius_list: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Determine the nfw boost factor and correct the shear profiles."""
        boost_factors = clmm.utils.compute_powerlaw_boost(radius_list, 1.0)
        corrected_profiles = clmm.utils.correct_with_boost_values(
            profiles, boost_factors
        )
        return corrected_profiles

    def set_miscentering(
        self,
        miscentering_fraction: float,
        sigma: float = 0.12,
        miscentering_distribution_function: callable = None,
        integration_max: float = None,
    ) -> None:
        """Set the miscentering model parameters.

        Parameters
        ----------
        miscentering_fraction : float
            Fraction of miscentered clusters (required).
        sigma : float, optional
            Width of the miscentering distribution. Default is 0.12.
        miscentering_distribution_function : callable, optional
            Function describing the miscentering distribution (single-parameter). Default is None.
        integration_max : float, optional
            Maximum radius for integration in units of sigma. Default is 25 * sigma.
        """
        if integration_max is None:
            integration_max = 25.0 * sigma

        self.miscentering_parameters = {
            "miscentering_fraction": miscentering_fraction,
            "sigma": sigma,
            "miscentering_distribution_function": miscentering_distribution_function,
            "integration_max": integration_max,
        }

    def compute_miscentering(
        self, clmm_model, radius_center, redshift, beta_s_mean, beta_s_square_mean
    ):
        params = self.miscentering_parameters
        miscentering_frac = params["miscentering_fraction"]
        sigma = params["sigma"]
        miscentering_distribution_function = params[
            "miscentering_distribution_function"
        ]
        integration_max = params["integration_max"]

        integrator = NumCosmoIntegrator(
            relative_tolerance=1e-4,
            absolute_tolerance=1e-6,
        )

        def integration_func(int_args, extra_args):
            sigma_local = extra_args[0]
            r_mis_list = int_args[:, 0]

            esd_vals = np.array(
                [
                    clmm_model.eval_excess_surface_density(
                        np.array([radius_center]), redshift, r_mis=r_mis
                    )[0]
                    for r_mis in r_mis_list
                ]
            )
            if self.is_delta_sigma == False:
                sigma_c_inf = clmm_model.cosmo.eval_sigma_crit(
                    redshift, z_src=clmm_model.z_inf
                )
                sigma_mis_vals = np.array(
                    [
                        clmm_model.eval_surface_density(
                            np.array([radius_center]), redshift, r_mis=r_mis
                        )[0]
                        for r_mis in r_mis_list
                    ]
                )
                esd_vals = (beta_s_mean * esd_vals) / (
                    sigma_c_inf - beta_s_mean * sigma_mis_vals
                )
                if self.approx == "order2":
                    esd_vals = esd_vals * (
                        1.0
                        + (beta_s_square_mean / beta_s_mean**2 - 1.0)
                        * beta_s_mean
                        * sigma_mis_vals
                        / sigma_c_inf
                    )
            if miscentering_distribution_function is not None:
                pdf_vals = miscentering_distribution_function(r_mis_list)
            else:
                pdf_vals = gamma.pdf(r_mis_list, a=2.0, scale=sigma_local)

            return esd_vals * pdf_vals

        integrator.integral_bounds = [(0.0, integration_max)]
        integrator.extra_args = np.array([sigma])
        miscentering_integral = integrator.integrate(integration_func)
        return miscentering_integral, miscentering_frac
