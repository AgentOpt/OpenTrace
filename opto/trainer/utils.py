from typing import List, Optional
import asyncio
import copy
import functools
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio
from opto import trace
from opto.trace.bundle import ALLOW_EXTERNAL_DEPENDENCIES
from opto.trace.modules import Module
from opto.trainer.guide import Guide

def safe_mean(x: List[float | None], missing_value=None) -> float | None:
    """Compute the mean of a nested list or nd.array of floats or None, returning missing_value (default None) for an empty list.

    Args:
        x (List[float | None]): List of floats or None
        missing_value (float | None, optional): Value to return if the list is empty or contains only None. Defaults to None.
    Returns:
        float | None: Mean of the list, or missing_value if the list is empty or contains only None
    """
    x = np.array(x)  # nd.array
    x = x[x != None] # filter out None values
    if x.size == 0:
        return missing_value
    return float(np.mean(x))

def async_run(runs, args_list = None, kwargs_list = None, max_workers = None, description = None, allow_sequential_run=True):
    """Run multiple functions in asynchronously.

    Args:
        runs (list): list of functions to run
        args_list (list): list of arguments for each function
        kwargs_list (list): list of keyword arguments for each function
        max_workers (int, optional): maximum number of worker threads to use.
            If None, the default ThreadPoolExecutor behavior is used.
        description (str, optional): description to display in the progress bar.
            This can indicate the current stage (e.g., "Evaluating", "Training", "Optimizing").
        allow_sequential_run (bool, optional): if True, runs the functions sequentially if max_workers is 1.
    """
    # if ALLOW_EXTERNAL_DEPENDENCIES is not False:
    #     warnings.warn(
    #         "Running async_run with external dependencies check enabled. "
    #         "This may lead to false positive errors. "
    #         "If such error happens, call disable_external_dependencies_check(True) before running async_run.",
    #         UserWarning,
    #     )

    if args_list is None:
        args_list = [[]] * len(runs)
    if kwargs_list is None:
        kwargs_list = [{}] * len(runs)

    if (max_workers == 1) and allow_sequential_run:  # run without asyncio
        print(f"{description} (Running sequentially).")
        return [run(*args, **kwargs) for run, args, kwargs in zip(runs, args_list, kwargs_list)]
    else:
        async def _run():
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                tasks = [loop.run_in_executor(executor, functools.partial(run, *args, **kwargs))
                         for run, args, kwargs, in zip(runs, args_list, kwargs_list)]

                # Use the description in the tqdm progress bar if provided
                if description:
                    return await tqdm_asyncio.gather(*tasks, desc=description)
                else:
                    return await tqdm_asyncio.gather(*tasks)

        # Handle Jupyter notebook
        try:
            return asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.get_running_loop()
            # We're in a loop (like Jupyter), so we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run())
                return future.result()


def batch_run(max_workers=None, description=None):
    """
    Create a function that runs in parallel using asyncio, with support for batching.
    The batch size is inferred as the length of the longest argument or keyword argument.

    Args:
        fun (callable): The function to run.

        max_workers (int, optional): Maximum number of worker threads to use.
            If None, the default ThreadPoolExecutor behavior is used.
        description (str, optional): Description to display in the progress bar.

    Returns:
        callable: A new function that processes batches of inputs.

    NOTE:
        If fun takes input that has __len__ (like lists or arrays), they won't be broadcasted.
        When using batch_run, be sure to pass list of such arguments of the same length.

    Example:
        >>> @batch_run(max_workers=4, description="Processing batch")
        >>> def my_function(x, y):
        >>>     return x + y
        >>> x = [1, 2, 3, 4, 5]
        >>> y = 10
        >>> outputs = my_function(x, y)
        >>> # outputs will be [11, 12, 13, 14, 15]
        >>> # This will run the function in asynchronously with 4 threads
    """

    def decorator(fun):
        """
        Decorator to create a function that runs in parallel using asyncio, with support for batching.

        Args:
            fun (callable): The function to run.

            max_workers (int, optional): Maximum number of worker threads to use.
                If None, the default ThreadPoolExecutor behavior is used.
            description (str, optional): Description to display in the progress bar.

        Returns:
            callable: A new function that processes batches of inputs.
        """
        def _fun(*args, **kwargs):

            # We try to infer the batch size from the args
            all_args = args + tuple(kwargs.values())
            # find all list or array-like arguments and use their length as batch size
            batch_size = max(len(arg) for arg in all_args if hasattr(arg, '__len__'))

            # broadcast the batch size to all args and record the indices that are broadcasted
            args = [arg if hasattr(arg, '__len__') else [arg] * batch_size for arg in args]
            kwargs = {k: v if hasattr(v, '__len__') else [v] * batch_size for k, v in kwargs.items()}

            # assert that all args and kwargs have the same length
            lengths = [len(arg) for arg in args] + [len(v) for v in kwargs.values()]
            if len(set(lengths)) != 1:
                raise ValueError("All arguments and keyword arguments must have the same length.")

            # deepcopy if it is a trace.Module (as they may have mutable state)
            # Module.copy() is used to create a new instance with the same parameters
            _args = [[a.copy() if isinstance(a, (Module, Guide)) else a for a in arg ] for arg in args ]
            _kwargs = {k: [a.copy() if isinstance(a, (Module, Guide)) else a  for a in v ] for k, v in kwargs.items() }

            # Run the forward function in parallel using asyncio with the same parameters.
            # Since trace.Node is treated as immutable, we can safely use the same instance.
            # The resultant graph will be the same as if we had called the function with the original arguments.

            # convert _args and _kwargs (args, kwargs of list) to lists of args and kwargs

            args_list = [tuple(aa[i] for aa in _args) for i in range(batch_size)]
            kwargs_list = [{k: _kwargs[k][i] for k in _kwargs} for i in range(batch_size)]

            outputs = async_run([fun] * batch_size, args_list=args_list, kwargs_list=kwargs_list,
                                max_workers=max_workers, description=description)
            return outputs

        return _fun

    return decorator


# ---------------------------------------------------------------------------
# Minibatch helpers
# ---------------------------------------------------------------------------

@trace.bundle()
def batchify(*items):
    """Concatenate multiple items into a formatted batch string.

    Parameters
    ----------
    *items : Any
        Variable number of items to concatenate into a batch.

    Returns
    -------
    str
        Formatted string with each item labeled by ID.

    Notes
    -----
    This function is decorated with @trace.bundle() and creates a formatted
    string where each item is prefixed with 'ID [i]:' for identification.
    """
    output = ''
    for i, item in enumerate(items):
        output += f'ID {[i]}: {item}\n'
    return output


# ---------------------------------------------------------------------------
# trace.Module graph helpers
# ---------------------------------------------------------------------------

def get_original_name(node):
    """Extract the original name from a node, removing all _copy suffixes."""
    py_name = node.py_name  # This removes colons: "param:0" -> "param0"

    # Find the first occurrence of "_copy" and remove it and everything after
    copy_index = py_name.find('_copy')
    if copy_index != -1:
        return py_name[:copy_index]
    else:
        return py_name


def is_node_copy(a, b):
    """Check if two nodes are copies of each other by comparing their original names.

    This function has transitivity: if A is a copy of B and B is a copy of C,
    then A is also considered a copy of C.
    """
    return get_original_name(a) == get_original_name(b)


def is_module_copy(a, b):
    """ Check if a and b (trace.Modules) are copies of each other. """
    parameters_a = a.parameters() # list of ParameterNode
    parameters_b = b.parameters() # list of ParameterNode
    # Check if all parameters of a are copies of b or vice versa
    # This might over count
    # need to check 1:1 correspondence
    matched = []
    for p_a in parameters_a:
        _matched = []
        for p_b in parameters_b:
            _matched.append(is_node_copy(p_a, p_b))
        matched.append(_matched)
    matched = np.array(matched)
    if np.all(np.sum(matched, axis=1) == 1) and np.all(np.sum(matched, axis=0) == 1):
        return True
    return False


def remap_update_dict(base_module, update_dict):
    """ Remap the update dict to the agent's parameters. update_dict might have keys which are copies of the base_module's parameters or visa versa.
        This function remaps the keys in update_dict to the original parameters of the base_module.

        The return dict is empty if no keys in update_dict matched any parameters of the base_module. This condition can be used to check if the update_dict contains non-trivial updates.
    """
    parameters = base_module.parameters()  # get the parameters of the base agent
    remapped_update_dict = {}
    for k, v in update_dict.items():
        for p in parameters:
            if is_node_copy(k, p): # Check if k is a copy of p or p is a copy of k
                remapped_update_dict[p] = v
                break # stop checking once we've found a match
    return remapped_update_dict


def set_module_parameters(agent, update_dict):
    """ Set the parameters of the agent based on the update_dict.
        The update_dict is a dictionary of ParameterNode: value pairs.
        The agent's parameters will be updated with the values from the update_dict.
    """
    remapped_update_dict = remap_update_dict(agent, update_dict)  # remap the update dict to the agent's parameters
    for k, v in remapped_update_dict.items():
        k._data = v  # set the parameter's data to the value in the update_dict


def create_module_from_update_dict(agent, update_dict):
    """ Create a new agent from the update_dict.
        The update_dict is a dictionary of ParameterNode: value pairs.
        A new agent will be created with the parameters set to the values from the update_dict.
    """
    new_agent = deepcopy_module(agent)  # create a copy of the agent
    set_module_parameters(new_agent, update_dict)  # set the parameters of the new agent
    return new_agent  # return the new agent


def deepcopy_module(agent):
    """ Create a deep copy of the agent, but reset the parameter names to remove the _copy suffixes.

        This is useful when we want to create a new agent for a new rollout,
        but we want to keep the parameter names consistent with the original agent
        so that the optimizer can recognize them across different rollouts.

        NOTE: This breaks the GRAPH's assumption on uniqueness of node names. Use with caution.
    """
    new_agent = copy.deepcopy(agent)
    for p_n in new_agent.parameters():
        for p_o in agent.parameters():
            if is_node_copy(p_n, p_o):
                p_n._name = p_o._name  # directly set the name to the original parameter's name
                break
    return new_agent


if __name__ == "__main__":

    def tester(t):  # regular time-consuming function
        import time
        print(t)
        time.sleep(t)
        return t, 2

    runs = [tester] * 10  # 10 tasks to demonstrate threading
    args_list = [(3,), (3,), (2,), (3,), (3,), (2,), (2,), (3,), (2,), (3,)]
    kwargs_list = [{}] * 10
    import time

    # Example with 1 thread (runs sequentially)
    print("Running with 1 thread (sequential):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=1)
    print(f"Time with 1 thread: {time.time()-start:.2f} seconds")

    # Example with limited workers (2 threads)
    print("\nRunning with 2 threads (parallel):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=2)
    print(f"Time with 2 threads: {time.time()-start:.2f} seconds")

    # Example with limited workers (4 threads)
    print("\nRunning with 4 threads (parallel):")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list, max_workers=4)
    print(f"Time with 4 threads: {time.time()-start:.2f} seconds")

    # Example with default number of workers
    print("\nRunning with default number of threads:")
    start = time.time()
    output = async_run(runs, args_list, kwargs_list)
    print(f"Time with default threads: {time.time()-start:.2f} seconds")
