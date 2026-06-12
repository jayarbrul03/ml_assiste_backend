"""Active learning: uncertainty scoring and task prioritization."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import PrelabelPrediction, Task


def compute_uncertainty_score(predictions: list[PrelabelPrediction]) -> float:
    """Return 0.0 (certain) – 1.0 (very uncertain).

    Uses 1 - min(confidence) across predictions. Tasks with no predictions
    are treated as maximally uncertain so they surface first in the queue.
    """
    if not predictions:
        return 1.0
    min_conf = min(p.confidence for p in predictions)
    return round(1.0 - min_conf, 3)


def needs_human_review(
    predictions: list[PrelabelPrediction],
    threshold: float,
) -> bool:
    """True when at least one prediction is below the autolabel threshold."""
    if not predictions:
        return True
    return any(p.confidence < threshold for p in predictions)


async def update_task_uncertainty(
    db: AsyncSession,
    asset_id: UUID,
    predictions: list[PrelabelPrediction],
) -> None:
    task_result = await db.execute(select(Task).where(Task.asset_id == asset_id))
    task = task_result.scalar_one_or_none()
    if task:
        task.uncertainty_score = compute_uncertainty_score(predictions)
