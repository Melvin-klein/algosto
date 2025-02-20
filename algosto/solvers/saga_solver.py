import numpy as np

from algosto.constraints import AbstractConstraint
from algosto.solvers import AbstractSolver, SGDSolver


class SAGASolver(SGDSolver):
    """
    Solver using the Stochastic Average Gradient Augmented (SAGA) algorithm.

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
    alpha : float, default=1.0
        The alpha hyper-parameter as shown in the mathematical description of the solver.
    cst : AbstractConstraint, default=None
        A constraint that defines the space where the solver will optimize the objective.
    random_state : int, default=None
        A constant to fix all random behaviors of the solver and ensure reproducibility of the results.
    """
    
    _alpha = None
    _grad_memory = None

    def __init__(self, d: int, M: int, objective: callable, grad: callable, gamma: float = 0.1, alpha: float = 1., cst: AbstractConstraint = None, random_state: int = None) -> None:
        super().__init__(d, M, objective, grad, gamma, cst, random_state)
        
        self.set_name("SAGA Solver")
        
        self.set_alpha(alpha)
    
    def fit(self, x_start: np.array = None, n_iter: int = 1000):
        """
        Run the SAGA solver to approximate the solution of the optimization problem.

        Parameters
        ----------
            x_start : array_like, default=None
                Starting point for the algorithm. It must be a vector of dimension ``d``.
            n_iter : int, default=1000
                Number of iterations the solver will compute.

        Raises
        ------
            ValueError:
                ``x_start`` dimension and ``d`` must be equals.
        """
        AbstractSolver.fit(self, x_start, n_iter)
        
        x = self.get_x_start()
        
        self.set_grad_memory(np.zeros((self.get_M(), self.get_dimension())))

        for k in range(self.get_M()):
            batch_filter = self._make_batch_filter(k)
            self._grad_memory[k,] = self.get_gradient()(x, batch_filter=batch_filter)

        for n in range(1, n_iter):
            batch_filter = self._make_batch_filter()
            u = np.argmax(batch_filter)
            
            grad = self.get_gradient()(x, batch_filter=batch_filter)
            
            x = x - self.get_gamma() * (grad - self.get_alpha() * (self.get_grad_memory()[u] - (1/self.get_M()) * np.sum(self.get_grad_memory(), axis=0)))
            
            self.get_grad_memory()[k,] = grad

            self._save_position(x)

    def get_grad_memory(self) -> np.array:
        """
        Returns the gradient memory (called g in the mathematical description of the solver).

        Returns
        -------
        np.array
            A vector of size ``(M, d)``.
        """
        return self._grad_memory

    def set_grad_memory(self, new_grad_memory: np.array) -> None:
        """
        Update the gradient memory

        Parameters
        ----------
        new_grad_memory : np.array
            Must be an array of size ``(M, d)``.
        """
        self._grad_memory = new_grad_memory
    
    def get_alpha(self) -> float:
        """
        Returns the value of ``alpha`` as dicussed in the mathematical description of the solver.

        Returns
        -------
        float
            The value of ``alpha``.
        """
        return self._alpha
    
    def set_alpha(self, new_alpha: float) -> None:
        """
        Update the value of ``alpha``.

        Parameters
        ----------
        new_alpha : float
            The new value of ``alpha``.
        
        Raises
        ------
        ValueError
            Raise a value error is ``new_alpha`` is not between 0 and 1 included.
        """
        if not (0 <= new_alpha and new_alpha <= 1):
            raise ValueError("[alpha] parameter of SAGASolver must be between 0 and 1 included.")
        
        self._alpha = new_alpha
