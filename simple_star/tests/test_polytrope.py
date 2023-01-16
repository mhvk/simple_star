# Licensed under the GPL v3 - see LICENSE.rst
"""Test that we can reproduce white dwarfs."""

import astropy.constants as const
import astropy.units as u
import numpy as np

from simple_star.polytrope import Polytrope


def alpha(k, gamma, rho_c):
    n = 1/(gamma - 1)
    return np.sqrt((n+1) / (4*np.pi*const.G) * k * rho_c ** ((1/n) - 1))


class TestB68:
    def setup_class(self):
        self.p_ism = 2.5e-12 * u.Pa  #((0.1 / u.cm**3) * const.k_B * 1e4 * u.K).to(u.Pa)
        self.t = 16 * u.K
        self.mu = 1/(0.7 / 2 + 0.3 / 4)  # 70% molecular hydrogen, 30% Helium+heavier
        self.gamma = 1
        self.k = const.k_B * self.t / (self.mu * const.m_p)
        self.poly = Polytrope(self.k, self.gamma)

    def p_gt_p_ism(self, r, mr, rho):
        return self.k * rho**self.gamma - self.p_ism


    def test_solution(self):
        rho_c = (2e5/u.cm**3 * self.mu * const.m_p).si
        result = self.poly(rho_c, np.linspace(0, 20000, 201) << u.AU,
                           condition=self.p_gt_p_ism)
        # Test condition is used properly.
        edge = result[-1]
        assert u.isclose(edge['p'], self.p_ism)
        # Test rough agreement with expectations.
        assert u.isclose(edge['mr'], 2.1*u.Msun, atol=0.1*u.Msun)
        assert u.isclose(edge['r'], 12500*u.AU, atol=1500*u.AU)


class TestNRCDWD:
    def setup_class(self):
        self.k1 = (3*np.pi**2)**(2/3)/5. * const.hbar**2 / const.m_e

    def test_wd(self):
        mu_e = 2
        rho_c = 2e5 * u.g / u.cm**3
        gamma = 5/3
        k = self.k1 / (mu_e * const.m_p) ** gamma
        poly = Polytrope(k, gamma)
        result = poly(rho_c, np.linspace(0, 0.05, 101) * u.Rsun)
        # Check we converged
        edge = result[-1]
        assert edge["r"] < 0.05 * u.Rsun
        # Check radius, mass, central density & pressure consistent with expectations.
        r_scale = alpha(k, gamma, rho_c)
        assert u.isclose(edge["r"], r_scale * 3.65375, atol=3e-5*u.Rsun)
        m_scale = 4*np.pi*r_scale**3 * rho_c
        assert u.isclose(edge["mr"], m_scale * 2.71406, atol=3e-5*u.Msun)
        mean_density = edge["mr"] / (4/3*np.pi*edge["r"]**3)
        assert u.isclose(result['rho'][0] / mean_density, 5.99071, rtol=2e-3)
        gm2byr4 = const.G * edge["mr"]**2 / edge["r"]**4
        assert u.isclose(result['p'][0] / gm2byr4, 0.77014, rtol=2e-3)
        # Check mass with Hamada-Salpeter
        assert u.isclose(edge["mr"], 0.21*u.Msun, atol=0.01*u.Msun)


class TestERCDWD:
    def setup_class(self):
        self.k2 = (3*np.pi**2)**(1/3)/4. * const.hbar * const.c

    def test_wd(self):
        mu_e = 2
        rho_c = 2e8 * u.g / u.cm**3
        gamma = 4/3
        k = self.k2 / (mu_e * const.m_p) ** gamma
        poly = Polytrope(k, gamma)
        result = poly(rho_c, np.linspace(0, 0.01, 101) * u.Rsun)
        # Check we converged
        edge = result[-1]
        assert edge["r"] < 0.01 * u.Rsun
        r_scale = alpha(k, gamma, rho_c)
        assert u.isclose(edge["r"], r_scale * 6.89685, atol=1e-3*u.Rsun)
        m_scale = 4*np.pi*r_scale**3 * rho_c
        assert u.isclose(edge["mr"], m_scale * 2.01824, atol=6e-3*u.Msun)
        mean_density = edge["mr"] / (4/3*np.pi*edge["r"]**3)
        assert u.isclose(result['rho'][0] / mean_density, 54.1825, rtol=0.05)
        gm2byr4 = const.G * edge["mr"]**2 / edge["r"]**4
        assert u.isclose(result['p'][0] / gm2byr4, 11.05066, rtol=0.07)
        # Check mass consistent with Chandrasekhar.
        assert u.isclose(result["mr"][-1], 1.44*u.Msun, atol=0.01*u.Msun)
