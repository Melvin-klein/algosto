Constraints
===========

Constraint is a main concept in Algosto. It defines the space where the solvers
will have to optimize objective functions.

Use an existing constraint
--------------------------

Currently, Algosto defines only two constraints which are a ball and a square on :math:`\mathbb R^d`.

Here is an example of how to use them.

.. code-block:: python

    from algosto.constraints import RdBallConstraint
    from algosto.solvers import SGDSolver
    from algosto.functions import quadratic

    objective, grad = quadratic()

    ct = RdBallConstraint(2, np.zeros(2), 5)

    solver = SGDSolver(ct, objective, grad)

    solver.fit()

This code minimizes the quadratic function on a ball in two dimensions centered at the origin with a radius of five.

.. note::

    You can find all details about the different constraints in 
    the :doc:`references <../references/constraints/index>` section of documentation.


Create your own constraint
--------------------------

If the built-in constraints does not fit your use case, you can define your own constraint.

Every constraint is a class and must inherits from ``algosto.constraints.AbstractConstraint``.
Thus, to create your constraint, write a class that inherits from ``AbstractConstraint``
and implement the following method : ``get_one_element()`` and ``get_grid()``.

For instance, if we want to create a constraint :math:`\mathcal C = \{x \in \mathbb R^d, ||x||_1 < s \}`
we should write the following code :

.. code-block:: python

    from algosto.constraints import AbstractConstraint

    class CustomConstraint(AbstractConstraint):
        def __init__(self, d: int, s: float) -> None:
            super().__init__(d)
            
            self._s = s
        
        def get_one_element(self) -> npt.NDArray[np.float64]:
            # x coordinates are between -1 and 1
            x = np.random.rand(self._d) * 2 - 1
            # Normalize the vector x
            x /= np.linalg.norm(x, 1)
            
            # Give a point randomly inside the set C
            return x * (self._s - np.random.uniform(0, self._s))

        def get_grid(self, num: int) -> Tuple[npt.NDArray, npt.NDArray]:
            # Essentialy get a mesh grid from numpy
            X, Y = super().get_grid(num)
            
            norm = np.abs(X) + np.abs(Y)
            
            X[norm > self._s] = np.nan
            Y[norm > self._s] = np.nan
            
            return X, Y

Once the new constraint is defined, you can use it like below :

.. code-block:: python

    from algosto.functions import quadratic
    from algosto.solvers import SGDSolver
    from algosto.evaluate import trajectory

    ct = CustomConstraint(5)

    objective, grad = quadratic()

    solver = SGDSolver(ct, objective, grad)

    solver.fit()

    trajectory(solver)
