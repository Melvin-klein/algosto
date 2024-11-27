Solvers
=======

Use an existing solver
----------------------

Algosto provides built-in solvers. They are all available inside the ``algosto.solvers`` module.
To use a solver, you first must define an :doc:`objective <objectives>` function and a :doc:`constraint <constraints>`.
Then pick one :doc:`solver available <../references/solvers/index>` and instanciate it.

.. code-block:: python

    from algosto.utils.functions import quadratic
    from algosto.constraints import RdBallConstraint
    from algosto.solvers import SGDSolver

    objective, grad = quadratic()

    ct = RdBallConstraint(2, np.zeros(2), 5)

    solver = SGDSolver(ct, objective, grad)
