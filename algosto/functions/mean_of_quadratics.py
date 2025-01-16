import numpy as np
from typing import Tuple

def mean_of_quadratics() -> Tuple[callable, callable]:
    """
    Gives the mean of quadratics function and its gradient.

    Returns
    -------
        Objective : callable
            The mean of quadratics Python function that takes a vector ``x`` of size ``d`` as parameter
            and that returns the result as float.
        Grad : callable
            The gradient of the mean of quadratics as a python function that takes a vector ``x`` of size ``d``
            as parameter and returns a vector of size ``d``.
    
    Examples
    --------
    
    >>> from algosto.functions import mean_of_quadratic
    >>> from algosto.solvers import SGDSolver
    >>> objective, grad = mean_of_quadratic()
    >>> solver = SGDSolver(2, 2, objective, grad)
    """
    def f(x: np.array) -> float:
        return 1/x.shape[0] * np.sum(x**2)

    def grad(x, batch_filter) -> np.array:
        return 1/x.shape[0] * (2 * x) * batch_filter

    return f, grad
