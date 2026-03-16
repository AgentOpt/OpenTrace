import asyncio
import uuid
import time


class Worker:
    """
    A worker class that represents a stateless coroutine which processes inputs and updates state.
    """

    def __init__(self, func, args=None, kwargs=None, name=None):
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.name = name or f"Worker-{uuid.uuid4()}"
        self.result = None
        self.is_running = False
        self.is_done = False
        self.start_time = None
        self.end_time = None

    async def run(self, state, input_data):
        """
        Execute the worker's function asynchronously with the given state and input.

        If the function is a coroutine, it will be awaited. Otherwise,
        it will be executed in the default executor to avoid blocking the event loop.

        Parameters
        ----------
        state : Any
            The current state to be processed by the worker
        input_data : Any
            The input data to be processed by the worker

        Returns
        -------
        tuple
            A tuple containing (output, next_state)
        """
        self.is_running = True
        self.start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(self.func):
                # If the function is already a coroutine function
                output, next_state = await self.func(state, input_data, *self.args, **self.kwargs)
            else:
                # Run synchronous functions in the default executor
                output, next_state = await asyncio.to_thread(self.func, state, input_data, *self.args, **self.kwargs)

            self.result = output
            return output, next_state
        finally:
            self.is_running = False
            self.is_done = True
            self.end_time = time.time()

    async def __call__(self, state, input_data):
        """
        Make the worker callable directly.

        Parameters
        ----------
        state : Any
            The current state to be processed by the worker
        input_data : Any
            The input data to be processed by the worker

        Returns
        -------
        tuple
            A tuple containing (output, next_state)
        """
        return await self.run(state, input_data)

    @property
    def duration(self):
        """
        Calculate the execution duration of the worker.

        Returns
        -------
        float or None
            The duration in seconds if the worker has completed, otherwise None.
        """
        if self.start_time is None:
            return None
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def __repr__(self):
        status = "running" if self.is_running else "done" if self.is_done else "pending"
        return f"<{self.name} [{status}]>"
