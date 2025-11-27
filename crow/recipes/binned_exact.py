"""Module for defining the classes used in the BinnedClusterRecipe cluster recipe."""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt
import pyccl as ccl

from crow import ClusterShearProfile
from crow import completeness as comp
from crow import kernel
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.properties import ClusterProperty

from .binned_parent import BinnedClusterRecipe

# To run with firecrown, use this import instead
# from firecrown.models.cluster import ClusterProperty


class ExactBinnedClusterRecipe(BinnedClusterRecipe):
    """Cluster recipe with Murata19 mass-richness and spec-zs.

    This recipe uses the Murata 2019 binned mass-richness relation and assumes
    perfectly measured spec-zs.
    """

    def __init__(
        self,
        cluster_theory,
        redshift_distribution,
        mass_distribution,
        completeness: comp.Completeness = None,
        mass_interval: tuple[float, float] = (11.0, 17.0),
        true_z_interval: tuple[float, float] = (0.0, 5.0),
    ) -> None:
        super().__init__(
            cluster_theory=cluster_theory,
            redshift_distribution=redshift_distribution,
            mass_distribution=mass_distribution,
            completeness=completeness,
            mass_interval=mass_interval,
            true_z_interval=true_z_interval,
        )

        self.integrator = NumCosmoIntegrator()

    def _get_theory_prediction_counts(
        self,
        average_on: None | ClusterProperty = None,
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
                * self.completeness_distribution(mass, z)
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution(mass, z, mass_proxy_limits)
            )

            if average_on is None:
                return prediction

            for cluster_prop in ClusterProperty:
                include_prop = cluster_prop & average_on
                if not include_prop:
                    continue
                if cluster_prop == ClusterProperty.MASS:
                    prediction *= mass
                if cluster_prop == ClusterProperty.REDSHIFT:
                    prediction *= z
            return prediction

        return theory_prediction

    def _get_function_to_integrate_counts(
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
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        self.integrator.integral_bounds = [
            self.mass_interval,
            z_edges,
        ]
        self.integrator.extra_args = np.array([*mass_proxy_edges, sky_area])

        theory_prediction = self._get_theory_prediction_counts(average_on)
        prediction_wrapper = self._get_function_to_integrate_counts(theory_prediction)

        counts = self.integrator.integrate(prediction_wrapper)

        return counts

    def _get_theory_prediction_shear_profile(
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
                prediction *= self.cluster_theory.compute_shear_profile(
                    log_mass=mass,
                    z=z,
                    radius_center=radius_center,
                )
            return prediction

        return theory_prediction

    def _get_function_to_integrate_shear_profile(
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
            self.mass_interval,
            z_edges,
        ]
        radius_center = radius_center
        self.integrator.extra_args = np.array(
            [*mass_proxy_edges, sky_area, radius_center]
        )
        if self.cluster_theory._beta_parameters is not None:
            self.cluster_theory.set_beta_s_interp(*z_edges)
        theory_prediction = self._get_theory_prediction_shear_profile(average_on)
        prediction_wrapper = self._get_function_to_integrate_shear_profile(
            theory_prediction
        )
        deltasigma = self.integrator.integrate(prediction_wrapper)
        return deltasigma
