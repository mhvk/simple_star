"""Model a ball of gas with a polytropic equation of state."""

import warnings

import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.table import QTable
from scipy.integrate import solve_ivp


class Polytrope:
    """Model with a polytropic equation of state.

    P = K ρˠ

    Here, the polytropic exponent γ is related to the polytropic
    index n by γ = 1 + 1/n.

    Parameters
    ----------
    K : `~astropy.units.Quantity`
        Proportionality constant.  Should have appropriate units.
    gamma : float
        Polytropic exponent.
    """

    def __init__(self, k, gamma):
        self.k = k
        self.gamma = gamma

    def integrate(self, rho_c, r, condition=None):
        """Integrate MC and HE from a given starting density.

        Parameters
        ----------
        rho_c : `~astropy.units.Quantity`
            Central density to start integration at.
        r : `~astropy.units.Quantity`
            Radii at which to evaluate the integral.  Radii beyond the
            stopping condition will not be used.
        condition : callable, optional
            Function that tells when to stop integrating.  Will be pass on
            ``r, mr, rho`` and should return a value that changes sign when
            the integration should stop. By default, uses the density, i.e.,
            integration will stop once the density becomes negative.

        Returns
        -------
        result : `~astropy.table.QTable`
            A table with columns ``r, mr, rho, p`` evaluated at the given
            radii, as well as at the location where the integration stopped
            (if that was within the range of input radii).
        """
        self._rho_c = rho_c
        self._r_unit = r.unit
        self._condition = condition
        # It is easier to integrate with variables of the same order
        # of magnitude, so work relative to rho_c and an estimate
        # of the total mass (not critical to have it exactly right).
        self._mr_scale = (self._rho_c * (r.max() / 2) ** 3).to(u.Msun)
        # scipy integration routine cannot handle Quantity, so use
        # scaled values and multiply with scales as needed in .structure_eqs().
        sol = solve_ivp(
            self.structure_eqs,  # Function that calculates derivatives.
            t_span=r[[0, -1]].value,  # Range within which to integrate.
            y0=[0, 1],  # Initial values: Mr=0, rho/rho_c=1.
            t_eval=r.value,  # Evaluate at chosen points.
            events=self.terminate,  # Terminate when ρ<0 (or condition < 0).
        )
        if not sol.success:
            warnings.warn('Solver did not succeed.')
        # Put physical scales back on.
        r_sol = sol.t * self._r_unit
        mr = sol.y[0] * self._mr_scale
        rho = sol.y[1] * self._rho_c
        result = QTable([r_sol, mr, rho], names=['r', 'mr', 'rho'])
        if len(sol.t_events[0]):
            # If density got to 0, add a final row for that.
            assert len(sol.t_events) == 1 and len(sol.t_events[0]) == 1
            result.insert_row(
                len(result),
                dict(r=sol.t_events[0][0] * self._r_unit,
                     mr=sol.y_events[0][0, 0] * self._mr_scale,
                     rho=max(sol.y_events[0][0, 1], 0) * self._rho_c))
        # Add column with implied pressure (from polytropic EoS).
        result['p'] = (self.k * result['rho']**self.gamma).to(u.Pa)
        return result

    def structure_eqs(self, r, par):
        """Mass conservation and hydrostatic equilibrium (for internal use).

        dMᵣ/dr = 4πr²ρ
        dP/dr = -GMᵣρ/r²

        Latter transformed to dρ/dr by getting dP/dρ from EoS.

        Parameters
        ----------
        r : float
            Radius in units of ``self._r_unit``.
        par : list of float
            Current parameters: scaled enclosed mass and density.

        Returns
        -------
        derivatives : list of float
            Derivatives of scaled enclosed mass and density with radius.
        """
        if r == 0 or par[1] <= 0:
            return [0, 0]
        # Multiply with physical scale/unit.
        r = r * self._r_unit
        mr = par[0] * self._mr_scale
        rho = par[1] * self._rho_c
        # Mass conservation.
        dmr_dr = 4 * np.pi * r**2 * rho
        # Hydrostatic equibrium, use in terms of density.
        dp_dr = -const.G * mr / r**2 * rho
        # For polytropic equation of state, dP/dρ=γKρˠ⁻¹.
        dp_drho = self.k * rho ** (self.gamma - 1.) * self.gamma
        drho_dr = dp_dr / dp_drho
        # Bring back to correct unit and return just the values.
        return [(dmr_dr / self._mr_scale).to_value(1 / self._r_unit),
                (drho_dr / self._rho_c).to_value(1 / self._r_unit)]

    def terminate(self, r, par):
        """Stopping condition (for internal use).

        Returns the scaled density if no explicit ``condition`` was used in
        the call to ``integrate()``.  Otherwise, calls ``condition`` with
        ``r, mr, rho`` in physical units.

        Parameters
        ----------
        r : float
            Radius in units of ``self._r_unit``.
        par : list of float
            Current parameters: scaled enclosed mass and density.

        Returns
        -------
        value : float
            The value which, if it changes sign, signals that integration
            should stop.
        """
        if self._condition is None:
            # By default, just terminate if the density becomes less than 0.
            return par[1]

        condition = self._condition(
            r,
            mr=par[0]*self._mr_scale,
            rho=par[1]*self._rho_c
        )
        # Cannot pass back units, so get the number for any quantity output.
        return getattr(condition, 'value', condition)

    # Signal to solve_ivp to stop integration if the termination signal
    # changes sign.
    terminate.terminal = True
