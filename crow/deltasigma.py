"""Module to compute the cluster excess surface mass density (delta sigma).

The galaxy cluster delta sigma integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""

import time
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
from scipy.integrate import simpson
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
        is_delta_sigma: bool = False,
        cluster_concentration: float | None = None,
        beta_zbin_cl_edges: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(mass_interval, z_interval, halo_mass_function)
        self.is_delta_sigma = is_delta_sigma
        self.cluster_concentration = cluster_concentration
        self.beta_zbin_cl_edges = beta_zbin_cl_edges
        self.beta_interp = None
        
    def delta_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        radius_center: np.float64,
        two_halo_term: bool = False,
        miscentering_frac: np.float64 = None,
        boost_factor: bool = False,
        use_beta_interp: bool = False,
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
                moo, radius_center, redshift, miscentering_frac, use_beta_interp=use_beta_interp
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
        use_beta_interp: bool = False,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Calculate the second halo contribution to the delta sigma."""
        beta_s_mean = None
        beta_s_square_mean = None
        first_halo_right_centered = None
        clmm_model.z_inf = 10.0
        if self.is_delta_sigma:
            first_halo_right_centered = clmm_model.eval_excess_surface_density(
                radius_center, redshift
            )
        else:
            beta_s_mean, beta_s_square_mean = self._get_beta_lens_efficiency(clmm_model, redshift, use_beta_interp, **kwargs) 
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
                                z_src = (beta_s_mean, beta_s_square_mean),
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
        if self.is_delta_sigma ==False:
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
        
    def _get_beta_lens_efficiency(self, clmm_model: clmm.Modeling, redshift, use_beta_interp, **kwargs):
        beta_s_mean = None
        beta_s_square_mean = None
        zmax = kwargs.pop('zmax', 5.0)
        if use_beta_interp == False:
            beta_s_mean = compute_beta_s_mean_from_distribution(
                   z_cl=redshift,
                   z_inf=10,
                   cosmo=clmm_model.cosmo,
                   zmax=zmax,
                   **kwargs
            )
            beta_s_square_mean = compute_beta_s_square_mean_from_distribution(
                   z_cl=redshift,
                   z_inf=10,
                   cosmo=clmm_model.cosmo,
                   zmax=zmax,
                   **kwargs
            )
        else:
            if self.beta_interp == None:
                self.beta_interp = self._compute_beta_interp(clmm_model)
            beta_s_mean_interp = self.beta_interp[0]
            beta_s_square_mean_interp = self.beta_interp[1]
            beta_s_mean = beta_s_mean_interp(redshift)
            beta_s_square_mean = beta_s_square_mean_interp(redshift)
        return float(beta_s_mean), float(beta_s_square_mean)
        
    def _compute_beta_interp(self, clmm_model):
        clmm_cosmo = clmm_model.cosmo
        betaz_min = None
        betaz_max = None
        if self.beta_zbin_cl_edges == None:
            print(f"Warning, no redshift bin for beta estimation, using integration values ({self.min_z}, {self.max_z}) instead")
            betaz_min = self.min_z
            betaz_max = self.max_z
        else:
            betaz_min = self.beta_zbin_cl_edges[0]
            betaz_max = self.beta_zbin_cl_edges[-1]
        redshift_points = np.linspace(betaz_min, betaz_max, 3)
        beta_list = [clmm.utils.compute_beta_s_mean_from_distribution(z_cl, 10.0, clmm_cosmo, zmax=5.0) for z_cl in redshift_points]
        beta_sq_list = [clmm.utils.compute_beta_s_square_mean_from_distribution(z_cl, 10.0, clmm_cosmo, zmax=5.0) for z_cl in redshift_points]
        beta_mean_interp = interp1d(
            redshift_points,
            beta_list,
            kind='quadratic',
            fill_value='extrapolate'
        )
        beta_sq_mean_interp = interp1d(
            redshift_points,
            beta_sq_list,
            kind='quadratic',
            fill_value='extrapolate'
        )
        
        return (beta_mean_interp, beta_sq_mean_interp)
