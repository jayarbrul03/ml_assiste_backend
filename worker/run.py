import asyncio

from worker.tasks import WorkerSettings

if __name__ == "__main__":
    from arq.worker import run_worker

    asyncio.run(run_worker(WorkerSettings))
