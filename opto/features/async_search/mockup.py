import asyncio
import random
from dataclasses import dataclass
from typing import Any
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: int
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None


@dataclass
class TaskResult:
    task_id: int
    status: TaskStatus
    result: Any
    duration: float


class Worker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.is_busy = False

    async def execute(self, task: Task) -> TaskResult:
        """Execute a task by sleeping for a random duration (1-10 seconds)."""
        self.is_busy = True
        task.status = TaskStatus.RUNNING

        sleep_duration = random.uniform(1, 10)
        print(f"Worker {self.worker_id}: Starting task {task.id}, will take {sleep_duration:.2f}s")

        try:
            await asyncio.sleep(sleep_duration)
            task.status = TaskStatus.COMPLETED
            task.result = f"Task {task.id} completed by worker {self.worker_id}"
            print(f"Worker {self.worker_id}: Completed task {task.id}")

            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=task.result,
                duration=sleep_duration
            )
        except Exception as e:
            task.status = TaskStatus.FAILED
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                result=str(e),
                duration=sleep_duration
            )
        finally:
            self.is_busy = False


class Master:
    def __init__(self, num_workers: int = 3):
        self.workers = [Worker(i) for i in range(num_workers)]
        self.task_counter = 0
        self.results: list[TaskResult] = []
        self._worker_available = asyncio.Condition()

    def create_task(self) -> Task:
        """Create a new task with a unique ID."""
        task = Task(id=self.task_counter)
        self.task_counter += 1
        return task

    async def dispatch_single(self, task: Task) -> TaskResult:
        """Dispatch a single task to the first available worker."""
        worker = await self._wait_for_available_worker()  # Actually waits for a free worker
        print(f"Master: Assigning task {task.id} to worker {worker.worker_id}")
        result = await worker.execute(task)
        self.results.append(result)
        await self._notify_worker_available()  # Signal that this worker is free again
        return result

    async def dispatch_batch(self, num_tasks: int) -> list[TaskResult]:
        """Dispatch multiple tasks and run them concurrently."""
        tasks = [self.create_task() for _ in range(num_tasks)]

        async def run_task(task: Task) -> TaskResult:
            worker = self.workers[task.id % len(self.workers)]
            return await worker.execute(task)

        results = await asyncio.gather(*[run_task(task) for task in tasks])
        self.results.extend(results)
        return list(results)

    def get_results(self) -> list[TaskResult]:
        """Get all completed task results."""
        return self.results

    def _get_available_worker(self) -> Worker | None:
        for worker in self.workers:
            if not worker.is_busy:
                return worker
        return None

    async def _wait_for_available_worker(self) -> Worker:
        async with self._worker_available:
            while True:
                worker = self._get_available_worker()
                if worker:
                    return worker
                await self._worker_available.wait()

    async def _notify_worker_available(self):
        async with self._worker_available:
            self._worker_available.notify_all()


async def main():
    print("=== Master/Worker Async System Demo ===\n")

    master = Master(num_workers=3)

    print("Dispatching 5 tasks concurrently...\n")
    results = await master.dispatch_batch(5)

    print("\n=== Results ===")
    for result in results:
        print(f"Task {result.task_id}: {result.status.value}, duration: {result.duration:.2f}s")

    total_time = max(r.duration for r in results)
    print(f"\nTotal wall-clock time: ~{total_time:.2f}s (tasks ran concurrently)")


if __name__ == "__main__":
    asyncio.run(main())
