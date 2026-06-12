"""Autolabeling: auto-apply high-confidence predictions and optionally submit."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import (
    Annotation,
    AnnotationSource,
    Asset,
    PrelabelPrediction,
    Task,
    TaskStatus,
    User,
    UserRole,
)
from services.active_learning import compute_uncertainty_score, needs_human_review

logger = logging.getLogger(__name__)


async def _system_user_id(db: AsyncSession) -> UUID:
    result = await db.execute(
        select(User.id)
        .where(User.role.in_([UserRole.ml_engineer, UserRole.admin]))
        .limit(1)
    )
    user_id = result.scalar_one_or_none()
    if not user_id:
        result = await db.execute(select(User.id).limit(1))
        user_id = result.scalar_one()
    return user_id


async def process_autolabel_for_asset(
    db: AsyncSession,
    asset_id: UUID,
    *,
    confidence_threshold: float,
    auto_submit: bool,
) -> dict:
    """Apply autolabeling rules for one asset after pre-label predictions exist."""
    task_result = await db.execute(
        select(Task)
        .where(Task.asset_id == asset_id)
        .options(
            selectinload(Task.annotations),
            selectinload(Task.asset).selectinload(Asset.prelabel_predictions),
        )
    )
    task = task_result.scalar_one_or_none()
    if not task:
        return {"autolabeled": False, "auto_submitted": False, "applied": 0}

    pred_result = await db.execute(
        select(PrelabelPrediction).where(PrelabelPrediction.asset_id == asset_id)
    )
    predictions = list(pred_result.scalars().all())

    task.uncertainty_score = compute_uncertainty_score(predictions)

    # Remove prior autolabel annotations so re-runs stay idempotent.
    for ann in list(task.annotations):
        if ann.source == AnnotationSource.prelabel:
            db.delete(ann)

    system_user = await _system_user_id(db)
    applied = 0
    for pred in predictions:
        if pred.confidence >= confidence_threshold:
            db.add(
                Annotation(
                    task_id=task.id,
                    author_id=system_user,
                    label_class_id=pred.label_class_id,
                    type=pred.type,
                    geometry=pred.geometry,
                    source=AnnotationSource.prelabel,
                )
            )
            pred.accepted = True
            applied += 1
        else:
            pred.accepted = None

    fully_autolabeled = (
        bool(predictions)
        and applied == len(predictions)
        and not needs_human_review(predictions, confidence_threshold)
    )

    autolabeled = False
    auto_submitted = False

    if fully_autolabeled:
        task.autolabeled = True
        autolabeled = True
        if auto_submit and task.status in (TaskStatus.pending, TaskStatus.in_progress):
            task.status = TaskStatus.submitted
            task.submitted_at = datetime.now(UTC)
            task.autolabel_review_status = "pending"
            auto_submitted = True
            logger.info("Auto-submitted task %s (all predictions >= %.2f)", task.id, confidence_threshold)
    else:
        task.autolabeled = False
        if applied > 0:
            logger.info(
                "Partial autolabel on task %s: %d/%d predictions applied (uncertainty=%.2f)",
                task.id,
                applied,
                len(predictions),
                task.uncertainty_score or 0,
            )

    await db.flush()
    return {
        "autolabeled": autolabeled,
        "auto_submitted": auto_submitted,
        "applied": applied,
        "uncertainty_score": task.uncertainty_score,
    }
