Kiefer-Wolfowitz
=================

The Kiefer-Wolfowitz algorithm is a stochastic algorithm for finding an extremum of
an unknown function :math:`f` from :math:`\mathbb R^d` to :math:`\mathbb R`.

The algorithm do the following assumptions

- :math:`f` is strictly concave,
- :math:`f` is differentiable,

and is defined by

.. math::

    X_{n+1} = X_n + \frac{\gamma_n}{2c_n}(Y_{n+1} - Z_{n+1})

where the initial state :math:`X_0 \in \mathbb R^d` is chosen arbitrarily, and :math:`(\gamma_n)`, :math:`(c_n)`
are two deterministic, positive, and decreasing sequences converging to zero, such that

.. math::

    \sum_{n = 1}^{+∞}\gamma_n = +∞, \qquad \sum_{n = 1}^{+∞}\gamma_n c_n < +∞, \qquad \sum_{n = 1}^{+∞} \left (\frac{\gamma_n}{c_n} \right)^2 < +∞.

The implementation of this algorithm in Algosto is such that :math:`\gamma_n = \frac{a}{n^\alpha}` 
and :math:`c_n = \frac{b}{n^\beta}`.

The sequences :math:`(Y_n)` and :math:`(Z_n)` are two random vector sequences of :math:`\mathbb R^d` satisfying

.. math::
    \mathbb E[Y_{n+1}|F_n] = \begin{pmatrix}
    f(X_n + c_ne_1) \\
    f(X_n + c_ne_2) \\
    \vdots \\
    f(X_n + c_ne_d)
    \end{pmatrix}, \qquad \mathbb E[Z_{n+1}|F_n] = \begin{pmatrix}
    f(X_n - c_ne_1) \\
    f(X_n - c_ne_2) \\
    \vdots \\
    f(X_n - c_ne_d)
    \end{pmatrix}

where :math:`(e_1, e_2, \dots, e_d)` is the canonical basis of :math:`\mathbb R^d`.

.. autoclass:: algosto.solvers.KieferWolfowitzSolver
    :members:
