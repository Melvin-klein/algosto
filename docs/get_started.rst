Get started
===============

Installation
------------

You can install Algosto with: 

.. prompt:: bash
    :prompts: $

    pip install algosto

Usage
-----

This section shows an example on how Algosto works by applying the
stochastic gradient descent algorithm (SGD) to a quadratic function.

Workflow
********

The basic workflow needs four elements :

An objective function
    This is the function we want to minimize.
    It is a Python function that takes a numpy matrix ``(n, d)``,
    where ``n`` is the number of points to handle and ``d`` is the dimension of points,
    and returns a result vector of length ``n``.

A constraint
    It is an object that defines the space in which the solver will optimize the function.
    Obviously, the objective function needs to be defined on this space.

A solver
    In Algosto, solvers are always classes that need an objective function and a constraint to be instanciated.
    Simply call the ``fit`` method to minimize the objective on the constraint.

A plot
    Algosto provides some functions to plot most used graph. You can build your own graph using *Matplotlib* or *Plotly*.

Objective function
******************

Based on the workflow given just before, we start by defining the quadratic objective function
and its gradient in order to use the SGD.

.. code-block:: python
    :caption: main.py

    import numpy as np

    def objective(x: np.array) -> float :
        return np.sum(x**2, axis=1)

    def grad(x: np.array) -> float :
        return 2*x

.. caution::

    As said before, objective functions and gradients need to be able to process multiple points simultaneously to work with Algosto.
    Specifically, if the function operates on points of dimension ``d``, it should accept a numpy array with shape ``(n, d)``
    and returns a numpy array of length ``n``, where ``n`` is the number of points provided to the objective function or gradient.


Algosto provides some toy objective functions, of which the quadratic function is a part, that you can import like this :

.. code-block:: python
    :caption: main.py

    from algosto.utils.functions import quadratic

    objective, grad = quadratic()

.. note::

    You can find a list of all available functions in the :doc:`references <references/utils/functions/index>` section of the documentation.

Constraint
**********

Now, we need to specify the definition space. 
To do that, Algosto provides object called *constraints* that you can import from the module ``algosto.constraints`` as follow :

.. code-block:: python
    :caption: main.py

    import numpy as np
    from algosto.constraints import RdBallConstraint

    ct = RdBallConstraint(2, np.zeros(2))

We define a two-dimensional ball in :math:`\mathbb{R}^d`, centered at the origin.
Constraints provide the solver with information about the space within which it can optimize the objective function.

.. note::

    You can find a list of all available constraints in the :doc:`references <references/constraints/index>` section of the documentation.

.. note::
    
    If your constraint is not yet implemented, you can define your own.
    Refer to the constraint chapter in the cookbook to learn how.

Solver
******

It's time to speak about the solver itself.
Solvers are avaible from the ``algosto.solvers`` module where you can find all the solvers implemented in Algosto.
In this example, we are going to use the stochastic gradient descent (SGD) to minimize the objective.

.. code-block:: python
    :caption: main.py

    from algosto.solvers import SGDSolver

    solver = SGDSolver(ct, objective)

.. note::

    You can find a list of all available solvers in the :doc:`references <references/solvers/index>` section of the documentation.

Finally, we can minimize the objective function with the help of the ``fit`` method :

.. code-block:: python
    :caption: main.py

    from algosto.utils import plot

    solver.fit()

    plot(solver)

Full workflow code
******************

The full Python code is avaible just below

.. code-block:: python
    :caption: main.py

    import numpy as np
    from algosto.utils.functions import quadratic
    from algosto.constraints import RdBallConstraint
    from algosto.solvers import SGDSolver
    from algosto.utils import plot

    objective, grad = quadratic()

    ct = RdBallConstraint(2, np.zeros(2))

    solver = SGD(ct, objective, grad)

    solver.fit()

    plot(solver)
