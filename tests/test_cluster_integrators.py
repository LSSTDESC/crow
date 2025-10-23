"""Tests for the integrator module."""

import os
import sys

import numpy as np
import numpy.typing as npt
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.integrator.scipy_integrator import ScipyIntegrator


def test_numcosmo_integrator_integrate():
    integrator = NumCosmoIntegrator()

    def integrand(
        int_args: npt.NDArray[np.float64], _extra_args: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # xy
        a = int_args[:, 0]
        b = int_args[:, 1]
        result = a * b
        return np.atleast_1d(result).astype(np.float64)

    integrator.integral_bounds = [(0, 1), (0, 1)]
    integrator.extra_args = np.array([], dtype=np.float64)
    result = integrator.integrate(integrand)
    # \int_0^1 \int_0^1 xy dx dy = 1/4
    assert result == pytest.approx(0.25, rel=1e-15, abs=0)


def test_scipy_integrator_integrate():
    integrator = ScipyIntegrator()

    # TODO: should we just remove this?
    def integrand(
        int_args: npt.NDArray[np.float64], _extra_args: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        # xy
        a = int_args[0]
        b = int_args[1]
        result = a * b
        return np.atleast_1d(result).astype(np.float64)

    integrator.integral_bounds = [(0, 1), (0, 1)]
    integrator.extra_args = np.array([], dtype=np.float64)
    result = integrator.integrate(integrand)
    # \int_0^1 \int_0^1 xy dx dy = 1/4
    assert result == pytest.approx(0.25, rel=1e-15, abs=0)
