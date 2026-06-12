"""Model versioning, batch re-run, and retraining feedback loop."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    Asset,
    Dataset,
    ModelVersion,
    ModelVersionStatus,
    PrelabelJob,
    PrelabelJobStatus,
    Project,
    Task,
    TrainingFeedback,
    TrainingFeedbackType,
)


async def get_or_create_default_model(db: AsyncSession, project_id: UUID) -> ModelVersion:
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.project_id == project_id, ModelVersion.status == ModelVersionStatus.active)
        .order_by(ModelVersion.created_at.desc())
        .limit(1)
    )
    mv = result.scalar_one_or_none()
    if mv:
        return mv

    from config import settings

    mv = ModelVersion(
        project_id=project_id,
        name="OpenAI Vision Detector",
        version=1,
        description="Default GPT-4o vision pre-label model",
        model_config={"provider": "openai", "model": settings.openai_model},
        status=ModelVersionStatus.active,
        metrics={"precision": 0.0, "recall": 0.0, "feedback_count": 0},
    )
    db.add(mv)
    await db.flush()
    return mv


async def create_model_version(
    db: AsyncSession,
    project_id: UUID,
    name: str,
    description: str | None = None,
    model_config: dict | None = None,
    parent_version_id: UUID | None = None,
) -> ModelVersion:
    max_ver = await db.scalar(
        select(func.max(ModelVersion.version)).where(ModelVersion.project_id == project_id)
    )
    mv = ModelVersion(
        project_id=project_id,
        name=name,
        version=(max_ver or 0) + 1,
        description=description,
        model_config=model_config or {},
        status=ModelVersionStatus.draft,
        parent_version_id=parent_version_id,
        metrics={},
    )
    db.add(mv)
    await db.flush()
    return mv


async def activate_model_version(db: AsyncSession, version_id: UUID) -> ModelVersion:
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == version_id))
    mv = result.scalar_one()

    # Deactivate other versions for same project
    others = await db.execute(
        select(ModelVersion).where(
            ModelVersion.project_id == mv.project_id,
            ModelVersion.status == ModelVersionStatus.active,
        )
    )
    for other in others.scalars().all():
        other.status = ModelVersionStatus.deprecated

    mv.status = ModelVersionStatus.active
    await db.flush()
    return mv


async def record_training_feedback(
    db: AsyncSession,
    *,
    task_id: UUID,
    feedback_type: TrainingFeedbackType,
    labeler_id: UUID,
    prediction_id: UUID | None = None,
    annotation_id: UUID | None = None,
    geometry_before: dict | None = None,
    geometry_after: dict | None = None,
    model_version_id: UUID | None = None,
) -> TrainingFeedback:
    fb = TrainingFeedback(
        task_id=task_id,
        prediction_id=prediction_id,
        annotation_id=annotation_id,
        feedback_type=feedback_type,
        geometry_before=geometry_before,
        geometry_after=geometry_after,
        labeler_id=labeler_id,
        model_version_id=model_version_id,
        used_in_training=False,
    )
    db.add(fb)
    await db.flush()
    return fb


async def trigger_retraining(
    db: AsyncSession,
    project_id: UUID,
    *,
    min_feedback: int = 5,
) -> ModelVersion:
    """Create a new model version from accumulated training feedback."""
    pending = await db.execute(
        select(TrainingFeedback)
        .join(Task, TrainingFeedback.task_id == Task.id)
        .join(Asset, Task.asset_id == Asset.id)
        .join(Dataset, Asset.dataset_id == Dataset.id)
        .where(Dataset.project_id == project_id, TrainingFeedback.used_in_training.is_(False))
    )
    feedback_rows = list(pending.scalars().all())
    if len(feedback_rows) < min_feedback:
        raise ValueError(f"Need at least {min_feedback} feedback samples, have {len(feedback_rows)}")

    active = await get_or_create_default_model(db, project_id)
    new_mv = await create_model_version(
        db,
        project_id,
        name=f"{active.name} (retrained)",
        description=f"Retrained from {len(feedback_rows)} feedback samples",
        model_config={**active.model_config, "retrained_at": datetime.now(UTC).isoformat()},
        parent_version_id=active.id,
    )
    new_mv.status = ModelVersionStatus.training

    accept_count = sum(1 for f in feedback_rows if f.feedback_type == TrainingFeedbackType.accept)
    reject_count = sum(1 for f in feedback_rows if f.feedback_type == TrainingFeedbackType.reject)
    correction_count = sum(
        1 for f in feedback_rows
        if f.feedback_type in (TrainingFeedbackType.correction, TrainingFeedbackType.prelabel_correction)
    )

    new_mv.metrics = {
        "feedback_count": len(feedback_rows),
        "accepts": accept_count,
        "rejects": reject_count,
        "corrections": correction_count,
        "estimated_precision": round(accept_count / max(accept_count + reject_count, 1), 3),
    }

    for fb in feedback_rows:
        fb.used_in_training = True
        fb.model_version_id = new_mv.id

    # Simulate training completion
    new_mv.status = ModelVersionStatus.active
    await activate_model_version(db, new_mv.id)

    return new_mv
