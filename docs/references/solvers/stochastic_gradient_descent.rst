Stochastic Gradient Descent (SGD)
=================================

The SGD is a famous algorithm to find the minimum of
a differentiable objective function :math:`f : I \subset \mathbb R^d \to \mathbb R`
of the form

.. math::

    f(x) = \frac{1}{M} \sum_{m=1}^M f_m(x)

where :math:`M \in \mathbb N^*` and :math:`f_m : \mathbb R^d \to \mathbb R`.

Iterations of the algorithm are of the form

.. math::

    X_{k+1} = X_k - \gamma_k \nabla f(X_k).

where :math:`(\gamma_k)_{k \in \mathbb N}` is a sequence of step such that

.. math::
    \sum_{k=1}^{+\infty}\gamma_k = +\infty \qquad \text{and} \qquad \sum_{k=1}^{+\infty}\gamma_k^2 < +\infty

We also assume that

- :math:`f` has a :math:`L`-Lipschitz gradient.

.. autoclass:: algosto.solvers.SGDSolver
    :members:
    :inherited-members:
