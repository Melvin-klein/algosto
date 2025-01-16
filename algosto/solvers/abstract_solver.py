from abc import ABC
import numpy as np

from algosto.constraints import AbstractConstraint, RdConstraint

class AbstractSolver(ABC):
    
    _name = "Abstract Solver"
    
    _d = None
    _objective = None
    _cst = None
    _random_state = None
    
    _trajectory = list()
    _n_iter = None
    _x_start = None
    
    def __init__(self, d: int, objective: callable, cst: AbstractConstraint = None, random_state: int = None) -> None:
        super().__init__()

        self.set_dimension(d)
        self.set_objective(objective)
        self.set_constraint(cst)
        self.set_random_state(random_state)

    def fit(self, x_start: np.array = None, n_iter: int = 1000) -> np.array:
        self.set_n_iter(n_iter)
        
        self.set_x_start(self.get_constraint().get_one_element() if x_start is None else x_start)

        if self._random_state is not None:
            np.random.seed(self._random_state)

    def get_name(self) -> str:
        """
        Returns the solver ``name``.

        Returns
        -------
        out : str
            The ``name`` of the solver.
        """
        return self._name
    
    def set_name(self, new_name: str) -> None:
        """
        Updates the ``name`` of the solver.

        Parameters
        ----------
        new_name : str
            The new name to give to the solver.
        """
        if new_name is None or len(new_name) == 0:
            raise ValueError("[name] parameter can't be None or empty.")
            
        self._name = new_name
    
    def get_dimension(self) -> int:
        """
        Returns the dimension value ``d``.

        Returns
        -------
        out : int
            The value of ``d``.
        """
        return self._d
    
    def set_dimension(self, new_d: int) -> None:
        """
        Updates the value of ``d``.

        Parameters
        ----------
        new_d : int
            The new dimension of the optimization problem.

        Raises
        ------
        ValueError
            Raises value error if ``new_d`` is lower than 1.
        """
        if new_d < 1:
            raise ValueError("[new_d] parameter must be higher or equal to 1.")
        
        self._d = new_d

    def get_trajectory(self) -> np.array:
        """
        Returns the trajectory registered by the solver during the ``fit`` operation.
        
        It's a matrix of size ``(n_iter, d)``.

        Returns
        -------
            out : ndarray
                An array of dimension ``(n_iter, d)`` where d is the dimension defined in the constraint
        """
        return np.array(self._trajectory)

    def reset_trajectory(self) -> None:
        """
        Reset the trajectory of the solver using a new list only filled with x_start.
        """
        self._trajectory = list()
        self._trajectory.append(self.get_x_start())

    def get_objective(self) -> callable:
        """
        Returns the objective function.

        Returns
        -------
        callable
            The objective function.
        """
        return self._objective

    def set_objective(self, new_objective: callable) -> None:
        """
        Updates the objective of the solver.

        Parameters
        ----------
        new_objective : callable
            A python callable that takes a vector of size ``d`` as parameter and returns a float.
        """
        self._objective = new_objective

    def get_constraint(self) -> AbstractConstraint:
        """
        Returns the constraint.

        Returns
        -------
        AbstractConstraint
            A constraint object that defines the space where the solver minimize the objective function.
        """
        return self._cst
    
    def set_constraint(self, new_cst: AbstractConstraint) -> None:
        """
        Updates the constraint

        Parameters
        ----------
        new_cst : AbstractConstraint
            A constraint object.
        """
        if new_cst is None:
            new_cst = RdConstraint(self.get_dimension(), self)

        self._cst = new_cst

    def get_random_state(self) -> int:
        """
        Returns the ``random_state``.

        Returns
        -------
        out : int
            The ``random_state`` value.
        """
        return self._random_state
    
    def set_random_state(self, new_random_state: int) -> None:
        """
        Updates the value of ``random_state``.

        Parameters
        ----------
        new_random_state : int
            The new ``random_state`` value.

        Raises
        ------
        ValueError
            Raises a value error if the ``random_state`` is negative.
        """
        if new_random_state < 0:
            raise ValueError("[random_state] parameter must be higher or equal to 0.")
        
        self._random_state = new_random_state
    
    def get_n_iter(self) -> int:
        """
        Returns the value of ``n_iter``.

        Returns
        -------
        out : int
            The number of iteration done by the solver.
        """
        return self._n_iter

    def set_n_iter(self, new_n_iter: int) -> None:
        """
        Updates the value of ``n_iter``.

        Parameters
        ----------
        new_n_iter : int
            The number of iteration done by the solver.

        Raises
        ------
        ValueError
            Raises a value error if ``new_n_iter`` is lower than 1.
        """
        if new_n_iter < 1:
            raise ValueError("The value of ``n_iter`` must be higher or equal to 1.")
        
        self._n_iter = new_n_iter
    
    def get_x_start(self) -> np.array:
        """
        Returns the value of ``x_start`` which is a vector of size ``d``.

        Returns
        -------
        out : np.array
            A vector of size ``d``.
        """
        return self._x_start
    
    def set_x_start(self, new_x_start: np.array) -> None:
        """
        Updates the starting point ``x_start``. This method reset the trajectory by using ``reset_trajectory()``.

        Parameters
        ----------
        new_x_start : np.array
            A vector of size ``d``.

        Raises
        ------
        ValueError
            Raises a value error if the size of ``new_x_start`` does not match the dimension ``d``.
        """
        new_x_start = new_x_start if isinstance(new_x_start, np.ndarray) else np.array(new_x_start)
        
        if new_x_start.shape[0] != self.get_dimension():
            raise ValueError(f"The starting point must have the same "
                             f"dimension as the solver. "
                             f"Start point has {new_x_start.shape[0]} and solver as {self.get_dimension()}")

        self._x_start = new_x_start

        self.reset_trajectory()

    def _save_position(self, x: np.array) -> None:
        if x.shape[0] != self.get_dimension():
            raise ValueError("x must have the same dimension as the solver.")

        self._trajectory.append(x)
