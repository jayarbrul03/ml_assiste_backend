"""GraphQL types and resolver logic for Auto-QA, quality, model versioning."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional
from uuid import UUID

import strawberry
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from strawberry.types import Info

from models import (
    Annotation,
    AnnotationSource,
    Asset,
    Dataset,
    LabelClass,
    ModelVersion,
    PrelabelJob,
    PrelabelJobStatus,
    PrelabelPrediction,
    QaCheck,
    Review,
    Task,
    TaskStatus,
    TrainingFeedback,
    TrainingFeedbackType,
    User,
    UserRole,
)
from services.auto_qa import run_auto_qa
from services.labeler_quality import get_labeler_quality, get_sla_summary
from services.model_registry import (
    activate_model_version,
    create_model_version,
    get_or_create_default_model,
    record_training_feedback,
    trigger_retraining,
)


@strawberry.type
class QaCheckType:
    id: strawberry.ID
    task_id: strawberry.ID
    iou_score: Optional[float]
    consensus_score: Optional[float]
    gold_standard_score: Optional[float]
    overall_score: Optional[float]
    passed: bool
    issues: list[str]
    created_at: datetime


@strawberry.type
class LabelerQualityType:
    labeler_id: strawberry.ID
    labeler_name: str
    labeler_email: str
    tasks_assigned: int
    tasks_completed: int
    tasks_approved: int
    tasks_rejected: int
    approval_rate: float
    avg_qa_score: Optional[float]
    sla_compliance_rate: float
    avg_turnaround_hours: Optional[float]
    quality_score: float


@strawberry.type
class SlaSummaryType:
    total_tracked: int
    overdue: int
    at_risk: int
    on_track: int
    sla_hours: int


@strawberry.type
class ModelVersionType:
    id: strawberry.ID
    project_id: strawberry.ID
    name: str
    version: int
    description: Optional[str]
    status: str
    metrics: strawberry.scalars.JSON
    model_config: strawberry.scalars.JSON
    created_at: datetime


@strawberry.type
class TrainingFeedbackStatsType:
    total: int
    pending: int
    accepts: int
    rejects: int
    corrections: int


@strawberry.input
class CreateModelVersionInput:
    project_id: strawberry.ID
    name: str
    description: Optional[str] = None
    model_config: Optional[strawberry.scalars.JSON] = None


@strawberry.input
class BatchRerunInput:
    dataset_id: strawberry.ID
    model_version_id: Optional[strawberry.ID] = None
    autolabel_enabled: bool = True
    confidence_threshold: float = 0.85
    auto_submit: bool = True


def qa_check_type(qa: QaCheck) -> QaCheckType:
    return QaCheckType(
        id=strawberry.ID(str(qa.id)),
        task_id=strawberry.ID(str(qa.task_id)),
        iou_score=qa.iou_score,
        consensus_score=qa.consensus_score,
        gold_standard_score=qa.gold_standard_score,
        overall_score=qa.overall_score,
        passed=bool(qa.passed),
        issues=qa.issues or [],
        created_at=qa.created_at,
    )


def model_version_type(mv: ModelVersion) -> ModelVersionType:
    return ModelVersionType(
        id=strawberry.ID(str(mv.id)),
        project_id=strawberry.ID(str(mv.project_id)),
        name=mv.name,
        version=mv.version,
        description=mv.description,
        status=mv.status.value,
        metrics=mv.metrics or {},
        model_config=mv.model_config or {},
        created_at=mv.created_at,
    )


async def _get_user(info: Info) -> User:
    user = info.context.get("user")
    if not user:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


async def resolve_autolabel_review_queue(info: Info):
    from gql_api.schema import _build_task_type

    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    if user.role not in (UserRole.reviewer, UserRole.admin, UserRole.ml_engineer):
        return []
    result = await db.execute(
        select(Task)
        .where(
            Task.autolabeled.is_(True),
            Task.status.in_([TaskStatus.submitted, TaskStatus.in_review]),
        )
        .options(
            selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
            selectinload(Task.annotations).selectinload(Annotation.label_class),
            selectinload(Task.assignee),
        )
        .order_by(Task.submitted_at.desc().nullslast())
    )
    return [await _build_task_type(db, t) for t in result.scalars().all()]


async def resolve_labeler_quality(info: Info, project_id: Optional[strawberry.ID] = None):
    db: AsyncSession = info.context["db"]
    await _get_user(info)
    pid = UUID(project_id) if project_id else None
    rows = await get_labeler_quality(db, pid)
    return [LabelerQualityType(**r) for r in rows]


async def resolve_sla_summary(info: Info, project_id: Optional[strawberry.ID] = None):
    db: AsyncSession = info.context["db"]
    await _get_user(info)
    pid = UUID(project_id) if project_id else None
    data = await get_sla_summary(db, pid)
    return SlaSummaryType(**data)


async def resolve_model_versions(info: Info, project_id: strawberry.ID):
    db: AsyncSession = info.context["db"]
    await _get_user(info)
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.project_id == UUID(project_id))
        .order_by(ModelVersion.version.desc())
    )
    return [model_version_type(mv) for mv in result.scalars().all()]


async def resolve_task_qa_check(info: Info, task_id: strawberry.ID):
    db: AsyncSession = info.context["db"]
    await _get_user(info)
    result = await db.execute(select(QaCheck).where(QaCheck.task_id == UUID(task_id)))
    qa = result.scalar_one_or_none()
    return qa_check_type(qa) if qa else None


async def resolve_training_feedback_stats(info: Info, project_id: strawberry.ID):
    db: AsyncSession = info.context["db"]
    await _get_user(info)
    pid = UUID(project_id)
    base = (
        select(TrainingFeedback)
        .join(Task, TrainingFeedback.task_id == Task.id)
        .join(Asset, Task.asset_id == Asset.id)
        .join(Dataset, Asset.dataset_id == Dataset.id)
        .where(Dataset.project_id == pid)
    )
    all_fb = list((await db.execute(base)).scalars().all())
    pending = [f for f in all_fb if not f.used_in_training]
    return TrainingFeedbackStatsType(
        total=len(all_fb),
        pending=len(pending),
        accepts=sum(1 for f in all_fb if f.feedback_type == TrainingFeedbackType.accept),
        rejects=sum(1 for f in all_fb if f.feedback_type == TrainingFeedbackType.reject),
        corrections=sum(
            1 for f in all_fb
            if f.feedback_type
            in (TrainingFeedbackType.correction, TrainingFeedbackType.prelabel_correction)
        ),
    )


async def resolve_accept_prelabel(info: Info, prediction_id: strawberry.ID, task_id: strawberry.ID):
    from gql_api.schema import AnnotationTypeGQL, GQLAnnotationSource, GQLAnnotationType

    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    pred_result = await db.execute(
        select(PrelabelPrediction).where(PrelabelPrediction.id == UUID(prediction_id))
    )
    pred = pred_result.scalar_one_or_none()
    if not pred:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Prediction not found")

    pred.accepted = True
    ann = Annotation(
        task_id=UUID(task_id),
        author_id=user.id,
        label_class_id=pred.label_class_id,
        type=pred.type,
        geometry=pred.geometry,
        source=AnnotationSource.prelabel,
    )
    db.add(ann)
    await record_training_feedback(
        db,
        task_id=UUID(task_id),
        feedback_type=TrainingFeedbackType.accept,
        labeler_id=user.id,
        prediction_id=pred.id,
        geometry_after=pred.geometry,
    )
    await db.commit()
    await db.refresh(ann)
    lc_result = await db.execute(select(LabelClass).where(LabelClass.id == ann.label_class_id))
    lc = lc_result.scalar_one()
    return AnnotationTypeGQL(
        id=strawberry.ID(str(ann.id)),
        task_id=strawberry.ID(str(ann.task_id)),
        label_class_id=strawberry.ID(str(ann.label_class_id)),
        label_class_name=lc.name,
        label_class_color=lc.color,
        type=GQLAnnotationType(ann.type.value),
        geometry=ann.geometry,
        source=GQLAnnotationSource(ann.source.value),
        version=ann.version,
    )


async def resolve_reject_prelabel(info: Info, prediction_id: strawberry.ID, task_id: strawberry.ID) -> bool:
    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    pred_result = await db.execute(
        select(PrelabelPrediction).where(PrelabelPrediction.id == UUID(prediction_id))
    )
    pred = pred_result.scalar_one_or_none()
    if not pred:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Prediction not found")

    pred.accepted = False
    await record_training_feedback(
        db,
        task_id=UUID(task_id),
        feedback_type=TrainingFeedbackType.reject,
        labeler_id=user.id,
        prediction_id=pred.id,
        geometry_before=pred.geometry,
    )
    await db.commit()
    return True


async def resolve_run_auto_qa(info: Info, task_id: strawberry.ID):
    db: AsyncSession = info.context["db"]
    await _get_user(info)
    qa = await run_auto_qa(db, UUID(task_id))
    await db.commit()
    return qa_check_type(qa)


async def resolve_approve_autolabel(info: Info, task_id: strawberry.ID):
    from gql_api.schema import _build_task_type

    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    if user.role not in (UserRole.reviewer, UserRole.admin, UserRole.ml_engineer):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Insufficient permissions")
    result = await db.execute(
        select(Task)
        .where(Task.id == UUID(task_id))
        .options(
            selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
            selectinload(Task.annotations).selectinload(Annotation.label_class),
            selectinload(Task.assignee),
        )
    )
    task = result.scalar_one()
    task.autolabel_review_status = "approved"
    task.status = TaskStatus.approved
    task.reviewed_at = datetime.now(UTC)
    await db.commit()
    return await _build_task_type(db, task)


async def resolve_reject_autolabel(info: Info, task_id: strawberry.ID, comment: str):
    from gql_api.schema import _build_task_type

    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    if user.role not in (UserRole.reviewer, UserRole.admin, UserRole.ml_engineer):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Insufficient permissions")
    result = await db.execute(
        select(Task)
        .where(Task.id == UUID(task_id))
        .options(
            selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
            selectinload(Task.annotations).selectinload(Annotation.label_class),
            selectinload(Task.assignee),
        )
    )
    task = result.scalar_one()
    task.autolabel_review_status = "rejected"
    task.status = TaskStatus.rejected
    task.rejection_reason = comment
    task.reviewed_at = datetime.now(UTC)
    db.add(Review(task_id=task.id, reviewer_id=user.id, status="rejected", comment=comment))
    await db.commit()
    return await _build_task_type(db, task)


async def resolve_create_model_version(info: Info, input: CreateModelVersionInput):
    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    if user.role not in (UserRole.admin, UserRole.ml_engineer):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Insufficient permissions")
    mv = await create_model_version(
        db,
        UUID(input.project_id),
        input.name,
        description=input.description,
        model_config=input.model_config,
    )
    await db.commit()
    await db.refresh(mv)
    return model_version_type(mv)


async def resolve_activate_model_version(info: Info, version_id: strawberry.ID):
    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    if user.role not in (UserRole.admin, UserRole.ml_engineer):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Insufficient permissions")
    mv = await activate_model_version(db, UUID(version_id))
    await db.commit()
    await db.refresh(mv)
    return model_version_type(mv)


async def resolve_trigger_retraining(info: Info, project_id: strawberry.ID):
    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    if user.role not in (UserRole.admin, UserRole.ml_engineer):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Insufficient permissions")
    try:
        mv = await trigger_retraining(db, UUID(project_id))
    except ValueError as exc:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await db.commit()
    await db.refresh(mv)
    return model_version_type(mv)


async def resolve_batch_rerun_prelabel(info: Info, input: BatchRerunInput):
    from gql_api.schema import _prelabel_job_type

    db: AsyncSession = info.context["db"]
    user = await _get_user(info)
    if user.role not in (UserRole.admin, UserRole.ml_engineer):
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Insufficient permissions")

    dataset_id = UUID(input.dataset_id)
    ds_result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = ds_result.scalar_one()

    if input.model_version_id:
        model_version_id = UUID(input.model_version_id)
    else:
        mv = await get_or_create_default_model(db, dataset.project_id)
        model_version_id = mv.id

    asset_count = await db.scalar(
        select(func.count(Asset.id)).where(Asset.dataset_id == dataset_id)
    )
    job = PrelabelJob(
        dataset_id=dataset_id,
        status=PrelabelJobStatus.pending,
        total_assets=asset_count or 0,
        confidence_threshold=input.confidence_threshold,
        autolabel_enabled=input.autolabel_enabled,
        auto_submit_enabled=input.auto_submit,
        model_version_id=model_version_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from worker.tasks import schedule_prelabel_job

    schedule_prelabel_job(str(job.id))
    return _prelabel_job_type(job)
