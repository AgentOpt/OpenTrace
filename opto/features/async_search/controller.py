import asyncio


class Controller:
    """ Asynchronous Controller to manage multiple workers processing tasks concurrently.
    """
    async def run(self, num_workers, *args, **kwargs):

        await self.init(*args, **kwargs)

        # start all workers concurrently using asyncio.gather
        worker_tasks = await asyncio.gather(*(self.create_new_task() for _ in range(num_workers)))
        worker_tasks = list(worker_tasks)

        while worker_tasks:
            done, pending = await asyncio.wait(worker_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                worker_tasks.remove(task)
                result = task.result()
                await self.update(result)
                if self.should_stop():
                    for p in pending:
                        p.cancel()
                    return await self.post_process()
                else:
                    new_task = await self.create_new_task()
                    worker_tasks.append(new_task)

    async def create_new_task(self):
        new_worker = await self.create_worker()
        new_task = await self.create_task()
        return asyncio.create_task(new_worker(new_task))

    async def init(self , *args, **kwargs):
        # Initialize any required state before starting the controller
        pass

    async def post_process(self):
        # Final processing after all tasks are done
        pass

    async def create_worker(self):
        # return a coroutine function that can be called with a task
        raise NotImplementedError

    async def create_task(self):
        raise NotImplementedError

    async def update(self, result):
        # process the result and update internal state
        raise NotImplementedError

    def should_stop(self):
        # return True if stopping condition met else False
        raise NotImplementedError


if __name__ == "__main__":

    class TestController(Controller):

        async def run(self, num_workers):
            self.i = 0 # counter for tasks
            return await super().run(num_workers)

        async def create_worker(self):
            async def worker(task):
                await asyncio.sleep(task)
                return f"Completed task with sleep {task}"
            return worker

        async def create_task(self):
            return 1  # simple task: sleep for 1 second

        async def update(self, result):
            self.i += 1
            print(self.i, result)

        def should_stop(self):
            if self.i >= 5:
                return True
            return False  # never stop for testing

    import time
    controller = TestController()
    st = time.time()
    asyncio.run(controller.run(num_workers=3))
    used_time =  time.time() - st
    assert used_time < 2 + 0.5, "Should be less than 2 + eps seconds"