"""Module to compute the cluster excess surface mass density (delta sigma).

The galaxy cluster delta sigma integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""

from typing import Optional, Tuple, Callable
from scipy.interpolate import interp1d
import clmm  # pylint: disable=import-error
from clmm.utils.beta_lens import (
    compute_beta_s_mean_from_distribution,
    compute_beta_s_square_mean_from_distribution,
)

import numpy as np
import numpy.typing as npt
import pyccl
from scipy.stats import gamma

from crow.abundance import ClusterAbundance
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator


class ClusterDeltaSigma(ClusterAbundance):
    """The class that calculates the predicted delta sigma of galaxy clusters.

    The excess density surface mass density is a function of a specific cosmology,
    a mass and redshift range, an area on the sky, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    shear integrand.
    """

    def __init__(
        self,
        mass_interval: tuple[float, float],
        z_interval: tuple[float, float],
        halo_mass_function: pyccl.halos.MassFunc,
        cluster_concentration: float | None = None,
        is_delta_sigma: bool = False,
        use_beta_s_interp: bool = False,
    ) -> None:
        super().__init__(mass_interval, z_interval, halo_mass_function)
        self.is_delta_sigma = is_delta_sigma
        self.cluster_concentration = cluster_concentration

        self._clmm_cosmo = clmm.Cosmology(be_cosmo=self._cosmo)

        self._beta_parameters = None
        self._beta_s_mean_interp = None
        self._beta_s_square_mean_interp = None

        self.use_beta_s_interp = use_beta_s_interp

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
        self, z_inf, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None
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

    def _beta_s_mean_exact(self, z_cl):
        return clmm.utils.compute_beta_s_mean_from_distribution(
            z_cl, cosmo=self._clmm_cosmo, **self._beta_parameters
        )

    def _beta_s_square_mean_exact(self, z_cl):
        return clmm.utils.compute_beta_s_mean_from_distribution(
            z_cl, cosmo=self._clmm_cosmo, **self._beta_parameters
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

    def delta_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        radius_center: np.float64,
        two_halo_term: bool = False,
        miscentering_frac: np.float64 = None,
        boost_factor: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Delta sigma for cprint(new_pred)lusters."""
        cosmo_clmm = clmm.Cosmology()
        # pylint: disable=protected-access
        cosmo_clmm._init_from_cosmo(self._cosmo)
        mass_def = self.halo_mass_function.mass_def
        mass_type = mass_def.rho_type
        if mass_type == "matter":
            mass_type = "mean"
        moo = clmm.Modeling(
            massdef=mass_type,
            delta_mdef=mass_def.Delta,
            halo_profile_model="nfw",
        )
        moo.set_cosmo(cosmo_clmm)
        return_vals = []
        for log_m, redshift in zip(log_mass, z):
            # pylint: disable=protected-access
            moo.set_concentration(self._get_concentration(log_m, redshift))
            moo.set_mass(10**log_m)
            val = self._one_halo_contribution(
                moo, radius_center, redshift, miscentering_frac
            )
            if two_halo_term:
                val += self._two_halo_contribution(moo, radius_center, redshift)
            if boost_factor:
                val = self._correct_with_boost_nfw(val, radius_center)
            return_vals.append(val)
        return np.asarray(return_vals, dtype=np.float64)

    def _one_halo_contribution(
        self,
        clmm_model: clmm.Modeling,
        radius_center,
        redshift,
        miscentering_frac=None,
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
            beta_s_mean = float(self.eval_beta_s_mean(redshift))
            beta_s_square_mean = float(self.eval_beta_s_square_mean(redshift))
            first_halo_right_centered = clmm_model.eval_tangential_shear(
                radius_center,
                redshift,
                (beta_s_mean, beta_s_square_mean),
                z_src_info="beta",
            )

        if miscentering_frac is not None:
            integrator = NumCosmoIntegrator(
                relative_tolerance=1e-2,
                absolute_tolerance=1e-6,
            )

            def integration_func(int_args, extra_args):
                sigma = extra_args[0]
                r_mis_list = int_args[:, 0]
                if self.is_delta_sigma:
                    esd_vals = np.array(
                        [
                            clmm_model.eval_excess_surface_density(
                                np.array([radius_center]), redshift, r_mis=r_mis
                            )[0]
                            for r_mis in r_mis_list
                        ]
                    )

                else:
                    esd_vals = np.array(
                        [
                            clmm_model.eval_tangential_shear(
                                np.array([radius_center]),
                                redshift,
                                r_mis,
                                z_src=(beta_s_mean, beta_s_square_mean),
                                z_src_info="beta",
                            )[0]
                            for r_mis in r_mis_list
                        ]
                    )

                gamma_vals = gamma.pdf(r_mis_list, a=2.0, scale=sigma)
                return esd_vals * gamma_vals

            integrator.integral_bounds = [(0.0, 25.0 * sigma_offset)]
            integrator.extra_args = np.array(
                [sigma_offset]
            )  ## From https://arxiv.org/pdf/2502.08444, we are using 0.12
            miscentering_integral = integrator.integrate(integration_func)
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
            np.array([radius_center]), redshift
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
        # print(boost_factors, radius_list, profiles)
        corrected_profiles = clmm.utils.correct_with_boost_values(
            profiles, boost_factors
        )
        return corrected_profiles
