from typing import Tuple, Callable
import numpy as np
import numpy.typing as npt

def quadratic(noise: str = None, custom_noise_fn: Callable = None) -> Tuple[Callable, Callable]:
    """
    Gives the quadratic function and its gradient.
    
    Parameters
    ----------
    noise : {'uniform', 'gaussian'}, default=None
        The type of noise that you want to add to the quadratic function. Values can be ``uniform`` which is a uniforme law on ``[-1, 1]``
        and ``gaussian`` which is a gaussian law of mean ``0`` and standard deviation ``1``.
        You can also add your own noise function by using the ``custom_noise_fn`` parameter.
    
    custom_noise_fn : callable, default=None
        Define your own noise function that will be added to the quadratic function.
        Its parameter is the vector ``x`` of size ``d`` and it must return a vector of noise of size ``d``.

    Returns
    -------
        objective : function
            The quadratic function. It takes a vector ``x`` of size ``d`` as parameter.

        grad : function
            The gradient of the quadratic function. It takes a vector ``x`` of size ``d`` as parameter.

    Examples
    --------
    An example without noise
    
    >>> from algosto.functions import quadratic
    >>> from algosto.solvers import KieferWolfowitzSolver
    >>> objective, _ = quadratic()
    >>> solver = KieferWolfowitzSolver(2, objective)

    An example with custom noise

    >>> import numpy as np
    >>> from algosto.functions import quadratic
    >>> from algosto.solvers import KieferWolfowitzSolver
    >>> def noise(x):
    ...     return np.random.normal(0, 5, x.shape[0])
    >>> objective, _ = quadratic(custom_noise_fn=noise)
    >>> solver = KieferWolfowitzSolver(2, objective)
    """
    if noise == 'uniform':
        noise_fn = lambda x: np.random.uniform(-1, 1, x.shape[0])
    elif noise == 'gaussian':
        noise_fn = lambda x: np.random.randn(x.shape[0])
    elif custom_noise_fn is not None:
        noise_fn = custom_noise_fn
    else:
        noise_fn = lambda x: np.zeros(x.shape[0])

    def objective(x: npt.NDArray) -> float :
        return np.sum(x**2, axis=1) + noise_fn(x)

    def grad(x: npt.NDArray):
        return 2*x

    return objective, grad
