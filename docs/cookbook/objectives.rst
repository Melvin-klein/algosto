Objectives
==========

Create your objective function
------------------------------

To create your objective, you must define a function that takes a 
numpy array of shape ``(n, d)``, where ``n`` is the number of points
and ``d`` is the dimension of points, and returns a vector of length ``n``.
It means that the objective is able to handle multiple points at the same time.
It is mandatory to work with Algosto.

For instance, if you would define a convex fonction with the formula :

.. math::

    f(x) = e^{\langle\theta, x\rangle}

where :math:`\theta \in \mathbb R^d` is known, the following code do the job

.. code-block:: python

    theta = np.random.randn(d)

    def objective(x: np.array) -> np.array:
        return np.exp(x @ theta)

Then you can give this objective function to one of the Algosto's solvers.

.. code-block:: python

    from algosto.solvers import KieferWolfowitzSolver

    # ... define constraint (ct)

    solver = KieferWolfowitzSolver(ct, objective)

Sandbox functions
-----------------

Algosto implements some simple functions in the ``algosto.functions`` module.
The list of available functions is here : :doc:`function references <../references/functions/index>`

The following code is an example where we import the quadratic function.

.. code-block:: python

    from algosto.functions import quadratic

    objective, grad = quadratic()
