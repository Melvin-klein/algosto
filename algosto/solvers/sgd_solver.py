from typing import Callable
import numpy as np
import numpy.typing as npt

from algosto.solvers import AbstractSolver
from algosto.constraints import AbstractConstraint

class SGDSolver(AbstractSolver):
    """
    Solver for the Stochastic Gradient Descent (SGD) algorithm.

    Parameters
    ----------
    d : int
        Dimension of the optimization problem. It means that it is the size of the vector ``x``
        given to the objective function and its gradient.
    M : int
        Number of chunks as shown in the mathematical description of the solver.
    objective : callable
        A callable that compute the objective functions values.
    grad : callable
        The gradient of the objective.
    gamma : float, default=0.1
        The step size as shown in the mathematical description of the solver.
    cst : AbstractConstraint, default=None
        A constraint that defines the space where the solver will optimize the objective.
    random_state : int, default=None
        A constant to fix all random behaviors of the solver and ensure reproducibility of the results.

    Examples
    --------
    
    >>> from algosto.functions import mean_of_quadratics
    >>> from algosto.solvers import SGDSolver
    >>> objective, grad = mean_of_quadratics()
    >>> solver = SGDSolver(d=2, M=2, objective, grad)
    >>> solver.fit(x_start=[-2, 1])
    """
    
    _M = None
    _grad = None
    _gamma = None
    
    def __init__(self, d: int, M: int, objective: callable, grad: callable, gamma: float = 0.1, cst: AbstractConstraint = None, random_state: int = None) -> None:
        super().__init__(d, objective, cst, random_state)

        self.set_name("SGD Solver")

        self.set_M(M)
        self.set_gradient(grad)
        self.set_gamma(gamma)

    def fit(self, x_start: np.ndarray = None, n_iter: int = 1000):
        """
        Run the SGD solver to approximate the solution of the optimization problem.

        Parameters
        ----------
        x_start : array_like
            The point where the algorithm will start running. If it is not filled,
            you must have defined a constraint using the ``cst`` parameter
            in which the solver will choose a random starting point.
        n_iter : int
            The number of iterations. The solver will run exactly this number of times.

        Raises
        ------
        ValueError
            It raises a ``ValueError`` if the dimension of ``x_start`` does not 
            match the dimension defined by ``d``.
        """
        super().fit(x_start, n_iter)
        
        x = self.get_x_start()

        for n in range(1, self._n_iter):
            batch_filter = self._make_batch_filter()

            #x = x - self.get_gamma() * self.get_gradient()(x, batch_filter=batch_filter)
            x = x - 0.1 * self.get_gradient()(x, batch_filter=batch_filter) # TODO : 0.1 -> 5/n

            self._save_position(x)

    def get_M(self) -> int:
        """
        Returns the value of ``M`` as shown in the mathematical description of the solver.

        Returns
        -------
        int
            Value of ``M``
        """
        return self._M

    def set_M(self, new_M: int) -> None:
        """
        Update the value of ``M``.

        Parameters
        ----------
        new_M : int
            The new value of ``M``.

        Raises
        ------
        ValueError
            Raise a value error if ``new_M`` is lower than 1.
        """
        if new_M < 1:
            raise ValueError("[M] parameter of SAGASolver must be higher or equal to 1.")
        
        self._M = new_M
    
    def get_gradient(self) -> callable:
        """
        Returns the gradient.

        Returns
        -------
        callable
            The gradient Python callable
        """
        return self._grad

    def set_gradient(self, new_grad: callable) -> None:
        """
        Updates the gradient function.

        Parameters
        ----------
        new_grad : callable
            A python callable object that takes a vector ``x`` of size ``d`` and a parameter ``batch_filter``.
        """
        self._grad = new_grad
    
    def get_gamma(self) -> float:
        """
        Returns the value of ``gamma``.

        Returns
        -------
        float
            Value of ``gamma``.
        """
        return self._gamma

    def set_gamma(self, new_gamma: float) -> None:
        """
        Updates the value of ``gamma``.

        Parameters
        ----------
        new_gamma : float
            New value of ``gamma``.

        Raises
        ------
        ValueError
            Raises a value error if gamma is lower or equal to 0.
        """
        if new_gamma <= 0:
            raise ValueError("[gamma] parameter mist be higher than 0.")
        
        self._gamma = new_gamma
            
    def _make_batch_filter(self, idx=None) -> np.array:
        batch_filter = np.zeros(self._M, dtype=int)
        idx = np.random.randint(0, self._M) if idx is None else idx
        batch_filter[idx] = 1
        
        return batch_filter
