"""The NumCosmo integrator module.

This module holds the NumCosmo implementation of the integrator classes
"""

from enum import Enum
from typing import Callable

import numpy as np
import numpy.typing as npt

from crow.integrator.integrator import Integrator

try:
    from numcosmo_py import Ncm

    HAS_NUMCOSMO = True
except ImportError:
    HAS_NUMCOSMO = False  # pragma: no cover

if HAS_NUMCOSMO:

    class NumCosmoIntegralMethod(Enum):
        P = Ncm.IntegralNDMethod.P
        P_V = Ncm.IntegralNDMethod.P_V
        H = Ncm.IntegralNDMethod.H
        H_V = Ncm.IntegralNDMethod.H_V

    class CountsIntegralND(Ncm.IntegralND):
        """Integral subclass used to compute the integrals using NumCosmo."""

        def __init__(
            self,
            dim: int,
            fun: Callable[
                [npt.NDArray[np.float64], npt.NDArray[np.float64]],
                npt.NDArray[np.float64],
            ],
            args: npt.NDArray[np.float64],
        ) -> None:
            if not HAS_NUMCOSMO:  # pragma: no cover
                raise ImportError(
                    "numcosmo is required for NumcosmoIntegrator. "
                    "Install it with: conda install -c conda-forge numcosmo"
                )
            super().__init__()
            self.dim = dim
            self.fun = fun
            self.extra_args = args

        # pylint: disable-next=arguments-differ
        def do_get_dimensions(self) -> tuple[int, int]:
            """Returns the dimensionality of the integral."""
            return self.dim, 1

        # pylint: disable-next=arguments-differ
        def do_integrand(
            self,
            x: Ncm.Vector,
            dim: int,
            npoints: int,
            _fdim: int,
            fval: Ncm.Vector,
        ) -> None:
            """Called by NumCosmo to evaluate the integrand."""
            x_array = np.array(x.dup_array()).reshape(npoints, dim)
            fval.set_array(list(self.fun(x_array, self.extra_args)))

else:  # pragma: no cover
    NumCosmoIntegralMethod = None
    CountsIntegralND = None


class NumCosmoIntegrator(Integrator):
    """The NumCosmo implementation of the Integrator base class."""

    def __init__(
        self,
        method: None | NumCosmoIntegralMethod = None,
        relative_tolerance: float = 1e-4,
        absolute_tolerance: float = 1e-12,
    ) -> None:
        if not HAS_NUMCOSMO:  # pragma: no cover
            raise ImportError(
                "numcosmo is required for NumcosmoIntegrator. "
                "Install it with: conda install -c conda-forge numcosmo"
            )
        super().__init__()
        self.method = method or NumCosmoIntegralMethod.P_V
        self._relative_tolerance = relative_tolerance
        self._absolute_tolerance = absolute_tolerance

    def integrate(
        self,
        func_to_integrate: Callable[
            [npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ],
    ) -> float:
        """Integrate the provided integrand argument with NumCosmo."""
        Ncm.cfg_init()  # pylint: disable=no-value-for-parameter

        int_nd = CountsIntegralND(
            len(self.integral_bounds), func_to_integrate, self.extra_args
        )
        int_nd.set_method(self.method.value)
        int_nd.set_reltol(self._relative_tolerance)
        int_nd.set_abstol(self._absolute_tolerance)
        res = Ncm.Vector.new(1)
        err = Ncm.Vector.new(1)

        bl, bu = zip(*self.integral_bounds)
        int_nd.eval(Ncm.Vector.new_array(bl), Ncm.Vector.new_array(bu), res, err)
        return res.get(0)  # pylint: disable-msg=no-member
