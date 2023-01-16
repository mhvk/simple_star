****************************
Semi-analytic SN lightcurves
****************************

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy

.. image:: https://github.com/mhvk/simple_star/workflows/CI/badge.svg
    :target: https://github.com/mhvk/simple_star/actions
    :alt: Test Status

``simple_star`` solves the stellar structure equations for simple
equations of state.  It relies on `NumPy <http://www.numpy.org/>`_, `Astropy
<http://www.astropy.org/>`_, `Scipy <https://scipy.org/>`_ and
`matplotlib <https://matplotlib.org/>`_.

It was created for a University of Toronto stars class, and likely
contains bugs.

.. Installation

Installation instructions
=========================

The package and its dependencies can be installed with::

  pip install git+https://github.com/mhvk/simple_star.git#egg=simple_star

Basic examples
==============

The following is to model an isothermal interstellar cloud embedded in
a medium with a given pressure.

    >>> import numpy as np
    >>> import astropy.units as u
    >>> import astropy.constants as const
    >>> import matplotlib.pyplot as plt
    >>> from simple_star.polytrope import Polytrope

    >>> t = 16 * u.K
    >>> mu = 1/(0.7 / 2 + 0.3 / 4)  # 70% molecular hydrogen, 30% Helium+heavier
    >>> rho_c = (2e5/u.cm**3 * mu * const.m_p).si
    >>> pc = rho_c/(mu*const.m_p) * const.k_B * t
    >>> gamma = 1
    >>> k = pc / rho_c ** gamma
    >>> poly = Polytrope(k=k, gamma=gamma)
    >>> p_ism = 2.5e-12 * u.Pa  #((0.1 / u.cm**3) * const.k_B * 1e4 * u.K).to(u.Pa)
    >>> def p_gt_p_ism(r, mr, rho):
    ...     return k*rho**gamma - p_ism
    ...
    >>> result = poly.integrate(rho_c, np.linspace(0, 20000, 201) << u.AU,
    ...                         condition=p_gt_p_ism)
    >>> plt.plot(result["r"], result["p"])
    >>> plt.show()

Do inspect the ``results`` table.  The tests are another useful place
to get started.

Contributing
============

Please open a new issue for bugs, feedback or feature requests.

To add a contribution, please submit a pull request.  If you would
like assistance, please feel free to contact `@mhvk`_.

For more information on how to make code contributions, please see the `Astropy
developer documentation <http://docs.astropy.org/en/stable/index.html#developer-documentation)>`_.

License
=======

``simple_star`` is licensed under the GNU General Public License v3.0 - see the
``LICENSE`` file.

.. _@mhvk: https://github.com/mhvk
