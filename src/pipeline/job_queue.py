from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class PipelineJob:
    """A unit of work in the pipeline queue."""

    video_id: int
    url: str
    platform: str
    priority: int = 0  # lower = higher priority
    _id: int = field(default_factory=lambda: id(object()))

    def __lt__(self, other: "PipelineJob") -> bool:
        return self.priority < other.priority


ProcessorFn = Callable[[PipelineJob], Coroutine]


class JobQueue:
    """Async in-memory priority queue for pipeline jobs.

    Jobs are processed by the registered processor coroutine one at a time
    (no parallelism within the queue — add a pool if needed).

    Raises if a job is enqueued after the queue is stopped or if the
    processor raises — errors surface immediately (no silent swallowing).
    """

    def __init__(self, max_size: int = 0) -> None:
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._processor: ProcessorFn | None = None
        self._processed: list[PipelineJob] = []
        self._errors: list[tuple[PipelineJob, Exception]] = []

    def set_processor(self, fn: ProcessorFn) -> None:
        """Register the coroutine that processes each job."""
        self._processor = fn

    async def enqueue(self, job: PipelineJob) -> None:
        """Add a job to the queue.

        Raises RuntimeError if the queue is not running.
        """
        if not self._running:
            raise RuntimeError(
                "Cannot enqueue jobs — JobQueue is not running. Call start() first."
            )
        await self._queue.put((job.priority, job))
        logger.debug("Enqueued video_id=%d priority=%d", job.video_id, job.priority)

    async def enqueue_nowait(self, job: PipelineJob) -> None:
        """Non-blocking enqueue. Raises QueueFull if at capacity."""
        if not self._running:
            raise RuntimeError(
                "Cannot enqueue jobs — JobQueue is not running. Call start() first."
            )
        self._queue.put_nowait((job.priority, job))

    async def start(self) -> None:
        """Start the background worker task."""
        if self._running:
            raise RuntimeError("JobQueue is already running.")
        if self._processor is None:
            raise RuntimeError(
                "No processor registered. Call set_processor() before start()."
            )
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("JobQueue started")

    async def stop(self, wait: bool = True) -> None:
        """Stop the queue. If wait=True, drain remaining jobs first."""
        self._running = False
        if wait and self._worker_task:
            await self._queue.join()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("JobQueue stopped")

    async def _worker(self) -> None:
        while True:
            try:
                _, job = await self._queue.get()
            except asyncio.CancelledError:
                break

            logger.debug("Processing video_id=%d", job.video_id)
            try:
                await self._processor(job)
                self._processed.append(job)
            except Exception as exc:
                logger.error(
                    "Job failed for video_id=%d: %s", job.video_id, exc
                )
                self._errors.append((job, exc))
            finally:
                self._queue.task_done()

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def processed_count(self) -> int:
        return len(self._processed)

    @property
    def error_count(self) -> int:
        return len(self._errors)

    @property
    def errors(self) -> list[tuple[PipelineJob, Exception]]:
        return list(self._errors)
