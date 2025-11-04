"""Module for defining the classes used in the MurataBinnedSpecZ cluster recipe."""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt
import pyccl as ccl

from crow.deltasigma import ClusterShearProfile
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.kernel import SpectroscopicRedshift
from crow.mass_proxy import MurataBinned
from crow.properties import ClusterProperty


class MurataBinnedSpecZRecipe:
    """Cluster recipe with Murata19 mass-richness and spec-zs.

    This recipe uses the Murata 2019 binned mass-richness relation and assumes
    perfectly measured spec-zs.
    """

    def __init__(
        self,
        hmf,
        redshift_distribution,
        mass_distribution,
        min_mass=13.0,
        max_mass=16.0,
        min_z=0.2,
        max_z=0.8,
        is_delta_sigma=False,
        cluster_concentration=None,
        two_halo_term=False,
        miscentering_frac=None,
        boost_factor=False,
        use_beta_interp=False,
    ) -> None:

        self.integrator = NumCosmoIntegrator()

        self.redshift_distribution = redshift_distribution
        self.mass_distribution = mass_distribution
        self.two_halo_term = two_halo_term
        self.miscentering_frac = miscentering_frac
        self.boost_factor = boost_factor
        self.cluster_theory = ClusterShearProfile(
            mass_interval=(min_mass, max_mass),
            z_interval=(min_z, max_z),
            halo_mass_function=hmf,
            is_delta_sigma=is_delta_sigma,
            cluster_concentration=cluster_concentration,
            use_beta_s_interp=True,
        )
        self.cluster_theory.set_beta_parameters(
            z_inf=5.0, zmax=10.0, delta_z_cut=0.1, zmin=None, z_distrib_func=None
        )

    def get_theory_prediction_counts(
        self,
    ) -> Callable[
        [npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[float, float], float],
        npt.NDArray[np.float64],
    ]:
        """Get a callable that evaluates a cluster theory prediction.

        Returns a callable function that accepts mass, redshift, mass proxy limits,
        and the sky area of your survey and returns the theoretical prediction for the
        expected number of clusters.
        """

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy_limits: tuple[float, float],
            sky_area: float,
        ):
            prediction = (
                self.cluster_theory.comoving_volume(z, sky_area)
                * self.cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution(mass, z, mass_proxy_limits)
            )
            return prediction

        return theory_prediction

    def get_function_to_integrate_counts(
        self,
        prediction: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                tuple[float, float],
                float,
            ],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
        """Returns a callable function that can be evaluated by an integrator.

        This function is responsible for mapping arguments from the numerical integrator
        to the arguments of the theoretical prediction function.
        """

        def function_mapper(
            int_args: npt.NDArray, extra_args: npt.NDArray
        ) -> npt.NDArray[np.float64]:
            mass = int_args[:, 0]
            z = int_args[:, 1]

            mass_proxy_low = extra_args[0]
            mass_proxy_high = extra_args[1]
            sky_area = extra_args[2]

            return prediction(mass, z, (mass_proxy_low, mass_proxy_high), sky_area)

        return function_mapper

    def evaluate_theory_prediction_counts(
        self,
        z_edges,
        mass_proxy_edges,
        sky_area: float,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        self.integrator.integral_bounds = [
            (self.cluster_theory.min_mass, self.cluster_theory.max_mass),
            z_edges,
        ]
        self.integrator.extra_args = np.array([*mass_proxy_edges, sky_area])

        theory_prediction = self.get_theory_prediction_counts()
        prediction_wrapper = self.get_function_to_integrate_counts(theory_prediction)

        counts = self.integrator.integrate(prediction_wrapper)

        return counts

    def get_theory_prediction_shear_profile(
        self,
        average_on: None | ClusterProperty = None,  # pylint: disable=unused-argument
    ) -> Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            tuple[float, float],
            float,
            float,
        ],
        npt.NDArray[np.float64],
    ]:
        """Get a callable that evaluates a cluster theory prediction.

        Returns a callable function that accepts mass, redshift, mass proxy limits,
        and the sky area of your survey and returns the theoretical prediction for the
        expected number of clusters.
        """

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy_limits: tuple[float, float],
            sky_area: float,
            radius_center: float,
        ):
            prediction = (
                self.cluster_theory.comoving_volume(z, sky_area)
                * self.cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution(mass, z, mass_proxy_limits)
            )
            if average_on is None:
                # pylint: disable=no-member
                raise ValueError(
                    f"The property should be"
                    f" {ClusterProperty.DELTASIGMA} or {ClusterProperty.SHEAR}."
                )

            if average_on & (ClusterProperty.DELTASIGMA | ClusterProperty.SHEAR):
                prediction *= self.cluster_theory.delta_sigma(
                    log_mass=mass,
                    z=z,
                    radius_center=radius_center,
                    two_halo_term=self.two_halo_term,
                    miscentering_frac=self.miscentering_frac,
                    boost_factor=self.boost_factor,
                )
            return prediction

        return theory_prediction

    def get_function_to_integrate_shear_profile(
        self,
        prediction: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                tuple[float, float],
                float,
                float,
            ],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
        """Returns a callable function that can be evaluated by an integrator.

        This function is responsible for mapping arguments from the numerical integrator
        to the arguments of the theoretical prediction function.
        """

        def function_mapper(
            int_args: npt.NDArray, extra_args: npt.NDArray
        ) -> npt.NDArray[np.float64]:
            mass = int_args[:, 0]
            z = int_args[:, 1]

            mass_proxy_low = extra_args[0]
            mass_proxy_high = extra_args[1]
            sky_area = extra_args[2]
            radius_center = extra_args[3]
            return prediction(
                mass, z, (mass_proxy_low, mass_proxy_high), sky_area, radius_center
            )

        return function_mapper

    def evaluate_theory_prediction_shear_profile(
        self,
        z_edges,
        mass_proxy_edges,
        radius_center,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        self.integrator.integral_bounds = [
            (self.cluster_theory.min_mass, self.cluster_theory.max_mass),
            z_edges,
        ]
        radius_center = radius_center
        self.integrator.extra_args = np.array(
            [*mass_proxy_edges, sky_area, radius_center]
        )
        self.cluster_theory.set_beta_s_interp(*z_edges)
        theory_prediction = self.get_theory_prediction_shear_profile(average_on)
        prediction_wrapper = self.get_function_to_integrate_shear_profile(theory_prediction)
        deltasigma = self.integrator.integrate(prediction_wrapper)
        return deltasigma

