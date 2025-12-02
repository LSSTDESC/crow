import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import splev, splrep
from scipy.special import gamma, gammainc, jv

from crow.integrator.numcosmo_integrator import NumCosmoIntegrator


def numcosmo_miscentered_mean_surface_density(  # pragma: no cover
    r_proj, r_mis, integrand, norm, aux_args, extra_integral
):
    """
    NumCosmo replacement for `integrate_azimuthially_miscentered_mean_surface_density`.

    Integrates azimuthally and radially for the mean surface mass density kernel.
    """
    integrator = NumCosmoIntegrator(
        relative_tolerance=1e-6,
        absolute_tolerance=1e-3,
    )
    integrand = np.vectorize(integrand)
    r_proj = np.atleast_1d(r_proj)
    r_lower = np.full_like(r_proj, 1e-6)
    r_lower[1:] = r_proj[:-1]

    results = []
    args = (r_mis, *aux_args)
    integrator.extra_args = np.array(args)
    for r_low, r_high in zip(r_lower, r_proj):
        if extra_integral:
            integrator.integral_bounds = [
                (r_low, r_high),
                (1.0e-6, np.pi),
                (0.0, np.inf),
            ]

            def integrand_numcosmo(int_args, extra_args):
                r_local = int_args[:, 0]
                theta = int_args[:, 1]
                extra = int_args[:, 2]
                return integrand(theta, r_local, extra, *extra_args)

        else:
            integrator.integral_bounds = [(r_low, r_high), (1.0e-6, np.pi)]

            def integrand_numcosmo(int_args, extra_args):
                r_local = int_args[:, 0]
                theta = int_args[:, 1]
                return integrand(theta, r_local, *extra_args)

        res = integrator.integrate(integrand_numcosmo)
        results.append(res)

    results = np.array(results)
    mean_surface_density = np.cumsum(results) * norm * 2 / np.pi / r_proj**2
    if not np.iterable(r_proj):
        return res[0] * norm * 2 / np.pi / r_proj**2
    return mean_surface_density


def _eval_2halo_term_generic_orig(  # pragma: no cover
    self,
    sph_harm_ord,
    r_proj,
    z_cl,
    halobias=1.0,
    logkbounds=(-5, 5),
    ksteps=1000,
    loglbounds=(0, 6),
    lsteps=500,
):
    """eval excess surface density from the 2-halo term (original)"""
    # pylint: disable=protected-access
    da = self.cosmo.eval_da(z_cl)
    rho_m = self.cosmo._get_rho_m(z_cl)

    k_values = np.logspace(logkbounds[0], logkbounds[1], ksteps)
    pk_values = self.cosmo._eval_linear_matter_powerspectrum(k_values, z_cl)
    interp_pk = splrep(k_values, pk_values)
    theta = r_proj / da

    # calculate integral, units [Mpc]**-3
    def __integrand__(l_value, theta):
        k_value = l_value / ((1 + z_cl) * da)
        return l_value * jv(sph_harm_ord, l_value * theta) * splev(k_value, interp_pk)

    l_values = np.logspace(loglbounds[0], loglbounds[1], lsteps)
    kernel = np.array([simpson(__integrand__(l_values, t), x=l_values) for t in theta])
    return halobias * kernel * rho_m / (2 * np.pi * (1 + z_cl) ** 3 * da**2)


def _eval_2halo_term_generic_new(  # pragma: no cover
    ####################################################################################
    # NOTE: This function is just a small optimization of the one implemented on CLMM  #
    # here just to benchmark the difference due the restructuration of the integration #
    ####################################################################################
    self,
    sph_harm_ord,
    r_proj,
    z_cl,
    halobias=1.0,
    logkbounds=(-5, 5),
    ksteps=1000,
    loglbounds=(0, 6),
    lsteps=500,
):
    """eval excess surface density from the 2-halo term (updated integration)"""
    # pylint: disable=protected-access
    da = self.cosmo.eval_da(z_cl)
    rho_m = self.cosmo._get_rho_m(z_cl)
    theta = r_proj / da

    # interp pk
    _k_values = np.logspace(logkbounds[0], logkbounds[1], ksteps)
    interp_pk = splrep(
        _k_values, self.cosmo._eval_linear_matter_powerspectrum(_k_values, z_cl)
    )

    # integrate
    l_values = np.logspace(loglbounds[0], loglbounds[1], lsteps)
    kernel = simpson(
        (
            l_values
            * jv(sph_harm_ord, l_values * theta)
            * splev(l_values / ((1 + z_cl) * da), interp_pk)
        ),
        x=l_values,
    )
    return [halobias * kernel * rho_m / (2 * np.pi * (1 + z_cl) ** 3 * da**2)]


def _eval_2halo_term_generic_vec(  # pragma: no cover
    self,
    sph_harm_ord,
    r_proj,
    z_cl,
    halobias=1.0,
    logkbounds=(-5, 5),
    ksteps=1000,
    loglbounds=(0, 6),
    lsteps=500,
):
    """eval excess surface density from the 2-halo term (vectorized)"""
    z_cl = np.atleast_1d(z_cl)
    # pylint: disable=protected-access
    da = self.cosmo.eval_da(z_cl)
    rho_m = self.cosmo._get_rho_m(z_cl)

    # (n_z, n_r)
    theta = (r_proj / da[None, :]).T

    # calculate integral, units [Mpc]**-3
    l_values = np.logspace(loglbounds[0], loglbounds[1], lsteps)

    # (n_l, n_z)
    k_values = l_values[:, np.newaxis] / ((1 + z_cl) * da)
    pk_values = np.zeros(k_values.shape)
    for i in range(z_cl.size):
        pk_values[:, i] = self.cosmo._eval_linear_matter_powerspectrum(
            k_values[:, i], z_cl[i]
        )

    # (n_l, n_z, n_r)
    jv_values = jv(sph_harm_ord, l_values[:, None, None] * theta[None, :, :])
    kernel = l_values[:, None, None] * pk_values[:, :, None] * jv_values

    # (n_z, n_r)
    integ = simpson(kernel, x=l_values, axis=0)

    # (n_z, n_r)
    out = halobias * integ * (rho_m / (2 * np.pi * (1 + z_cl) ** 3 * da**2))[:, None]

    # (n_r, n_z)
    return out.T
