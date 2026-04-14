"""Module for defining the classes used in the BinnedClusterRecipe cluster recipe."""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt
import pyccl as ccl
from scipy.integrate import simpson

from crow import ClusterShearProfile, kernel
from crow.cluster_modules.completeness_models import Completeness
from crow.cluster_modules.purity_models import Purity
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.properties import ClusterProperty

from .binned_parent import BinnedClusterRecipe

# To run with firecrown, use this import instead
# from firecrown.models.cluster import ClusterProperty


class ExactBinnedClusterRecipe(BinnedClusterRecipe):
    """
    Concrete implementation of BinnedClusterRecipe using direct numerical integration.

    This recipe evaluates cluster observables by explicitly integrating over
    mass, redshift, and mass proxy using a numerical integrator. It assumes:

    - Murata et al. (2019) mass–richness relation
    - Perfectly measured spectroscopic redshifts
    - No precomputed grids or interpolation (fully numerical evaluation)

    The integration is performed using a configurable integrator
    (NumCosmoIntegrator), which maps the theoretical prediction into an
    integrable function over the relevant parameter space.

    Compared to other implementations, this class:
    - Does not rely on interpolation tables
    - Computes predictions on-the-fly via numerical quadrature
    - Handles purity/completeness through explicit modification of the integrand

    Notes
    -----
    This implementation is computationally more expensive but provides
    a reference "exact" evaluation of the model under the stated assumptions.
    """

    def __init__(
        self,
        cluster_theory,
        redshift_distribution,
        mass_distribution,
        completeness: Completeness = None,
        purity: Purity = None,
        mass_interval: tuple[float, float] = (11.0, 17.0),
        true_z_interval: tuple[float, float] = (0.0, 5.0),
    ) -> None:
        super().__init__(
            cluster_theory=cluster_theory,
            redshift_distribution=redshift_distribution,
            mass_distribution=mass_distribution,
            completeness=completeness,
            purity=purity,
            mass_interval=mass_interval,
            true_z_interval=true_z_interval,
        )

        self.integrator = NumCosmoIntegrator()

    def setup(self):
        pass

    def _setup_with_completeness(self):
        """
        Configure the completeness contribution to the integrand.

        If a completeness model is provided, its distribution function is used
        directly. Otherwise, completeness is assumed to be unity (no selection
        effects).
        """
        if self.completeness is None:
            self._completeness_distribution = lambda *args: 1
        else:
            self._completeness_distribution = self.completeness.distribution

    def _setup_with_purity(self):
        """
        Configure the completeness contribution to the integrand.

        If a completeness model is provided, its distribution function is used
        directly. Otherwise, completeness is assumed to be unity (no selection
        effects).
        """
        if self.purity is None:
            self._mass_distribution_distribution = self.mass_distribution.distribution
        else:
            self._mass_distribution_distribution = self._mass_distribution_purity

    def _mass_distribution_purity(self, log_mass, z, log_mass_proxy):

        return (
            self.mass_distribution.gaussian_kernel(log_mass, z, log_mass_proxy)
            / self.purity.distribution(log_mass_proxy, z)
            * np.log(10)
        )

    def _get_theory_prediction_counts(
        self,
        average_on: None | ClusterProperty = None,
    ) -> Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
        ],
        npt.NDArray[np.float64],
    ]:
        """
        Construct the integrand for cluster number counts.

        Returns
        -------
        callable
            Function of (mass, redshift, mass_proxy, sky_area) that evaluates
            the differential contribution to the number counts.

        Notes
        -----
        The integrand includes:
        - comoving volume element
        - halo mass function
        - completeness selection
        - redshift distribution
        - mass–proxy distribution (with optional purity correction)

        If `average_on` is specified, the integrand is weighted accordingly
        (e.g., by mass or redshift).
        """

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy: npt.NDArray[np.float64],
            sky_area: float,
        ):
            prediction = (
                self.cluster_theory.comoving_volume(z, sky_area)
                * self.cluster_theory.mass_function(mass, z)
                * self._completeness_distribution(mass, z)
                * self.redshift_distribution.distribution()
            )

            if self.purity == None:
                assert (
                    len(mass_proxy) == 2
                ), "mass_proxy with no purity should be size 2"
                prediction *= self._mass_distribution_distribution(
                    mass, z, (mass_proxy[0], mass_proxy[1])
                )
            else:
                prediction *= self._mass_distribution_distribution(mass, z, mass_proxy)

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
                npt.NDArray[np.float64],
                float,
            ],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
        """
        Map the theoretical prediction into the integrator's expected signature.

        This wrapper adapts the prediction function to the interface required by
        NumCosmoIntegrator by reorganizing input arguments.

        Returns
        -------
        callable
            Function compatible with the numerical integrator.
        """

        def function_mapper(
            int_args: npt.NDArray, extra_args: npt.NDArray
        ) -> npt.NDArray[np.float64]:
            mass = int_args[:, 0]
            z = int_args[:, 1]

            if self.purity == None:
                mass_proxy = np.array([extra_args[0], extra_args[1]])
                sky_area = extra_args[2]
            else:
                mass_proxy = int_args[:, 2]
                sky_area = extra_args[0]

            return prediction(mass, z, mass_proxy, sky_area)

        return function_mapper

    def evaluate_theory_prediction_counts(
        self,
        z_edges,
        log_proxy_edges,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """
        Compute predicted cluster number counts within a bin.

        This method performs a multidimensional numerical integral over:

        - halo mass
        - redshift
        - mass proxy

        Parameters
        ----------
        z_edges : array-like of shape (2,)
            Redshift bin limits.
        log_proxy_edges : array-like of shape (2,)
            Mass proxy bin limits in log10 space.
        sky_area : float
            Survey area in square degrees.
        average_on : ClusterProperty, optional
            Observable over which to average (e.g., mass, redshift).

        Returns
        -------
        float
            Expected number of clusters in the bin.

        Notes
        -----
        - Uses NumCosmoIntegrator for numerical quadrature.
        - Handles purity and completeness internally by modifying the integrand.
        - Assumes perfect redshift measurements.
        """
        assert len(log_proxy_edges) == 2, "log_proxy_edges should be size 2"
        assert len(z_edges) == 2, "z_edges should be size 2"

        if self.purity == None:
            self.integrator.integral_bounds = [
                self.mass_interval,
                z_edges,
            ]
            self.integrator.extra_args = np.array([*log_proxy_edges, sky_area])
        else:
            self.integrator.integral_bounds = [
                self.mass_interval,
                z_edges,
                log_proxy_edges,
            ]
            self.integrator.extra_args = np.array([sky_area])

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
            npt.NDArray[np.float64],
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
            mass_proxy: npt.NDArray[np.float64],
            sky_area: float,
            radius_center: float,
        ):
            prediction = (
                self.cluster_theory.comoving_volume(z, sky_area)
                * self.cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self._completeness_distribution(mass, z)
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
                if self.purity == None:
                    assert (
                        len(mass_proxy) == 2
                    ), "mass_proxy with no purity should be size 2"
                    prediction *= self._mass_distribution_distribution(
                        mass, z, (mass_proxy[0], mass_proxy[1])
                    )
                else:
                    prediction *= self._mass_distribution_distribution(
                        mass, z, mass_proxy
                    )

            return prediction

        return theory_prediction

    def _get_function_to_integrate_shear_profile(
        self,
        prediction: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
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

            if self.purity == None:
                mass_proxy = np.array([extra_args[0], extra_args[1]])
                sky_area = extra_args[2]
                radius_center = extra_args[3]
            else:
                mass_proxy = int_args[:, 2]
                sky_area = extra_args[0]
                radius_center = extra_args[1]

            return prediction(mass, z, mass_proxy, sky_area, radius_center)

        return function_mapper

    def evaluate_theory_prediction_lensing_profile(
        self,
        z_edges,
        log_proxy_edges,
        distance_centers,
        sky_area: float,
        average_on: None | ClusterProperty = None,
        distance_units: str = "mpc",
    ) -> float:
        """
        Compute the predicted stacked lensing (shear) profile.

        This method integrates the theoretical lensing signal over:

        - halo mass
        - redshift
        - mass proxy

        and evaluates it at multiple radial bins.

        Parameters
        ----------
        z_edges : array-like of shape (2,)
            Redshift bin limits.
        log_proxy_edges : array-like of shape (2,)
            Mass proxy bin limits in log10 space.
        radius_centers : array-like
            Radii at which the shear profile is evaluated.
        sky_area : float
            Survey area in square degrees.
        average_on : ClusterProperty, optional
            Observable defining the weighting of the prediction.
            Must include shear-related properties.

        Returns
        -------
        ndarray
            Predicted shear profile evaluated at each radius.

        Notes
        -----
        - Each radius is integrated independently.
        - The integrand includes the halo shear profile computed from the
            underlying cluster theory model.
        - Purity and completeness modify the integrand similarly to the
            number counts case.
        """
        assert len(log_proxy_edges) == 2, "log_proxy_edges should be size 2"
        assert len(z_edges) == 2, "z_edges should be size 2"
        radius_centers = self.cluster_theory.get_radius_centers_mpc(
            distance_centers, distance_units, z_edges
        )
        if self.purity == None:
            self.integrator.integral_bounds = [
                self.mass_interval,
                z_edges,
            ]
            extra_args = [*log_proxy_edges]
        else:
            self.integrator.integral_bounds = [
                self.mass_interval,
                z_edges,
                log_proxy_edges,
            ]
            extra_args = []

        deltasigma_list = []

        for radius_center in radius_centers:
            self.integrator.extra_args = np.concatenate(
                (extra_args, [sky_area, radius_center])
            )
            theory_prediction = self._get_theory_prediction_shear_profile(average_on)
            prediction_wrapper = self._get_function_to_integrate_shear_profile(
                theory_prediction
            )
            deltasigma = self.integrator.integrate(prediction_wrapper)
            deltasigma_list.append(deltasigma)
        return np.array(deltasigma_list).flatten()
