"""Module for defining the classes used in the MurataBinnedSpecZ cluster recipe."""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt

from crow.abundance import ClusterAbundance
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
    ) -> None:
        super().__init__()

        self.integrator = NumCosmoIntegrator()
        self.redshift_distribution = redshift_distribution
        self.mass_distribution = mass_distribution

        self.hmf = hmf

        self.cluster_theory = ClusterAbundance(
            mass_interval=(min_mass, max_mass),
            z_interval=(min_z, max_z),
            halo_mass_function=self.hmf,
        )

    def get_theory_prediction(
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
                * self.redshift_distribution.distribution()
                * self.mass_distribution.distribution(
                    mass=mass, z=z, mass_proxy_limits=mass_proxy_limits
                )
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

    def get_function_to_integrate(
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

    def evaluate_theory_prediction(
        self,
        z_edges: tuple[float, float],
        mass_proxy_edges: tuple[float, float],
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

        self.integrator.extra_args = np.array(
            [mass_proxy_edges[0], mass_proxy_edges[1], sky_area]
        )

        theory_prediction = self.get_theory_prediction(average_on)
        prediction_wrapper = self.get_function_to_integrate(theory_prediction)

        counts = self.integrator.integrate(prediction_wrapper)

        return counts
