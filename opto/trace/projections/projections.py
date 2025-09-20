from typing import Any


class Projection:
    """Abstract base class for parameter projection methods.

    Projections are used to constrain parameter updates during optimization,
    ensuring that parameters remain within valid or desired ranges/formats.

    Methods
    -------
    __call__(x)
        Apply projection to parameter (calls project method).
    project(x)
        Project parameter onto feasible set (must be implemented).

    Notes
    -----
    Projections are applied during optimization to:
    1. Enforce constraints (e.g., bounds, formats)
    2. Maintain parameter validity (e.g., proper code syntax)
    3. Apply regularization or normalization

    Common projection types:
    - Bound constraints: Clipping values to ranges
    - Format constraints: Ensuring proper syntax/structure
    - Semantic constraints: Maintaining meaning/validity

    Projections are applied sequentially if multiple are specified
    for a parameter.

    See Also
    --------
    ParameterNode : Parameters that can have projections
    Optimizer.project : Method that applies projections
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x: Any) -> Any:
        """Apply projection to a parameter.

        Parameters
        ----------
        x : Any
            The parameter value to project.

        Returns
        -------
        Any
            The projected parameter value.

        Notes
        -----
        This method simply delegates to project() for consistency
        with callable interface.
        """
        return self.project(x)

    def project(self, x: Any) -> Any:
        """Project parameter onto the feasible set.

        Parameters
        ----------
        x : Any
            The parameter value to project.

        Returns
        -------
        Any
            The projected parameter value that satisfies constraints.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.

        Notes
        -----
        Subclasses should implement this method to define specific
        projection logic. The projection should be idempotent:
        project(project(x)) = project(x).
        """
        raise NotImplementedError("Subclasses should implement this method.")
    