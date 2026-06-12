from datetime import datetime
from enum import Enum
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
    AnnotationType,
    Asset,
    Dataset,
    LabelClass,
    PrelabelJob,
    PrelabelJobStatus,
    PrelabelPrediction,
    Project,
    Review,
    Task,
    TaskStatus,
    User,
    UserRole,
)
from services.workflow import apply_transition, labeler_can_see_task

from gql_api.advanced import (
    BatchRerunInput,
    CreateModelVersionInput,
    LabelerQualityType,
    ModelVersionType,
    QaCheckType,
    SlaSummaryType,
    TrainingFeedbackStatsType,
    resolve_accept_prelabel,
    resolve_activate_model_version,
    resolve_approve_autolabel,
    resolve_autolabel_review_queue,
    resolve_batch_rerun_prelabel,
    resolve_create_model_version,
    resolve_labeler_quality,
    resolve_model_versions,
    resolve_reject_autolabel,
    resolve_reject_prelabel,
    resolve_run_auto_qa,
    resolve_sla_summary,
    resolve_task_qa_check,
    resolve_training_feedback_stats,
    resolve_trigger_retraining,
)


@strawberry.enum
class GQLUserRole(Enum):
    admin = "admin"
    ml_engineer = "ml_engineer"
    labeler = "labeler"
    reviewer = "reviewer"


@strawberry.enum
class GQLTaskStatus(Enum):
    pending = "pending"
    in_progress = "in_progress"
    submitted = "submitted"
    in_review = "in_review"
    approved = "approved"
    rejected = "rejected"


@strawberry.enum
class GQLAnnotationType(Enum):
    bbox = "bbox"
    polygon = "polygon"


@strawberry.enum
class GQLAnnotationSource(Enum):
    human = "human"
    prelabel = "prelabel"
    corrected = "corrected"


@strawberry.enum
class GQLPrelabelJobStatus(Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


@strawberry.type
class UserType:
    id: strawberry.ID
    email: str
    name: str
    role: GQLUserRole


@strawberry.type
class LabelClassType:
    id: strawberry.ID
    name: str
    color: str
    sort_order: int


@strawberry.type
class AssetType:
    id: strawberry.ID
    filename: str
    storage_key: str
    mime_type: str
    width: Optional[int]
    height: Optional[int]
    url: str


@strawberry.type
class AnnotationTypeGQL:
    id: strawberry.ID
    task_id: strawberry.ID
    label_class_id: strawberry.ID
    label_class_name: str
    label_class_color: str
    type: GQLAnnotationType
    geometry: strawberry.scalars.JSON
    source: GQLAnnotationSource
    version: int


@strawberry.type
class PrelabelPredictionType:
    id: strawberry.ID
    asset_id: strawberry.ID
    label_class_id: strawberry.ID
    label_class_name: str
    label_class_color: str
    type: GQLAnnotationType
    geometry: strawberry.scalars.JSON
    confidence: float
    accepted: Optional[bool]


@strawberry.type
class TaskType:
    id: strawberry.ID
    status: GQLTaskStatus
    asset: AssetType
    annotations: list[AnnotationTypeGQL]
    assignee_name: Optional[str]
    rejection_reason: Optional[str]
    project_id: strawberry.ID
    project_name: str
    submitted_at: Optional[datetime]
    reviewed_at: Optional[datetime]
    uncertainty_score: Optional[float]
    autolabeled: bool
    qa_score: Optional[float]
    qa_passed: Optional[bool]
    claimed_at: Optional[datetime]
    sla_deadline: Optional[datetime]
    autolabel_review_status: Optional[str]


@strawberry.type
class DatasetType:
    id: strawberry.ID
    name: str
    version: int
    asset_count: int
    created_at: datetime


@strawberry.type
class ProjectType:
    id: strawberry.ID
    name: str
    city: str
    description: Optional[str]
    label_classes: list[LabelClassType]
    datasets: list[DatasetType]
    task_counts: strawberry.scalars.JSON


@strawberry.type
class PrelabelJobType:
    id: strawberry.ID
    dataset_id: strawberry.ID
    status: GQLPrelabelJobStatus
    total_assets: int
    processed_assets: int
    autolabeled_assets: int
    auto_submitted_assets: int
    confidence_threshold: Optional[float]
    autolabel_enabled: bool
    auto_submit_enabled: bool
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


@strawberry.type
class DashboardSummary:
    total_tasks: int
    pending_tasks: int
    in_progress_tasks: int
    submitted_tasks: int
    approved_tasks: int
    rejected_tasks: int
    autolabeled_tasks: int
    uncertain_tasks: int
    review_pass_rate: float
    prelabel_acceptance_rate: float
    tasks_by_status: strawberry.scalars.JSON
    throughput_by_day: list[strawberry.scalars.JSON]
    project_progress: list[strawberry.scalars.JSON]


@strawberry.input
class LabelClassInput:
    name: str
    color: str = "#3b82f6"


@strawberry.input
class CreateProjectInput:
    name: str
    city: str
    description: Optional[str] = None
    label_classes: list[LabelClassInput]


@strawberry.input
class TriggerPrelabelInput:
    dataset_id: strawberry.ID
    autolabel_enabled: bool = True
    confidence_threshold: float = 0.85
    auto_submit: bool = True
    model_version_id: Optional[strawberry.ID] = None


@strawberry.input
class RegisterAssetInput:
    filename: str
    storage_key: str
    mime_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None


@strawberry.input
class AnnotationInput:
    id: Optional[str] = None
    label_class_id: str
    type: GQLAnnotationType
    geometry: strawberry.scalars.JSON
    source: GQLAnnotationSource = GQLAnnotationSource.human


def _user_type(u: User) -> UserType:
    return UserType(
        id=strawberry.ID(str(u.id)),
        email=u.email,
        name=u.name,
        role=GQLUserRole(u.role.value),
    )


def _label_class_type(lc: LabelClass) -> LabelClassType:
    return LabelClassType(
        id=strawberry.ID(str(lc.id)),
        name=lc.name,
        color=lc.color,
        sort_order=lc.sort_order,
    )


def _asset_type(a: Asset) -> AssetType:
    from services.storage import get_public_url

    return AssetType(
        id=strawberry.ID(str(a.id)),
        filename=a.filename,
        storage_key=a.storage_key,
        mime_type=a.mime_type,
        width=a.width,
        height=a.height,
        url=get_public_url(a.storage_key),
    )


async def _task_counts(db: AsyncSession, project_id: UUID) -> dict:
    result = await db.execute(
        select(Task.status, func.count(Task.id))
        .join(Asset, Task.asset_id == Asset.id)
        .join(Dataset, Asset.dataset_id == Dataset.id)
        .where(Dataset.project_id == project_id)
        .group_by(Task.status)
    )
    counts = {s.value: 0 for s in TaskStatus}
    for status, count in result.all():
        counts[status.value] = count
    return counts


async def _build_task_type(db: AsyncSession, task: Task) -> TaskType:
    asset = task.asset
    dataset = asset.dataset if asset else None
    project = dataset.project if dataset else None

    annotations = []
    for ann in task.annotations:
        lc = ann.label_class
        annotations.append(
            AnnotationTypeGQL(
                id=strawberry.ID(str(ann.id)),
                task_id=strawberry.ID(str(ann.task_id)),
                label_class_id=strawberry.ID(str(ann.label_class_id)),
                label_class_name=lc.name if lc else "",
                label_class_color=lc.color if lc else "#000",
                type=GQLAnnotationType(ann.type.value),
                geometry=ann.geometry,
                source=GQLAnnotationSource(ann.source.value),
                version=ann.version,
            )
        )

    return TaskType(
        id=strawberry.ID(str(task.id)),
        status=GQLTaskStatus(task.status.value),
        asset=_asset_type(asset),
        annotations=annotations,
        assignee_name=task.assignee.name if task.assignee else None,
        rejection_reason=task.rejection_reason,
        project_id=strawberry.ID(str(project.id)) if project else strawberry.ID(""),
        project_name=project.name if project else "",
        submitted_at=task.submitted_at,
        reviewed_at=task.reviewed_at,
        uncertainty_score=task.uncertainty_score,
        autolabeled=bool(task.autolabeled),
        qa_score=task.qa_score,
        qa_passed=task.qa_passed,
        claimed_at=task.claimed_at,
        sla_deadline=task.sla_deadline,
        autolabel_review_status=task.autolabel_review_status,
    )


async def _get_user(info: Info) -> User:
    user = info.context.get("user")
    if not user:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _prelabel_job_type(job: PrelabelJob) -> PrelabelJobType:
    return PrelabelJobType(
        id=strawberry.ID(str(job.id)),
        dataset_id=strawberry.ID(str(job.dataset_id)),
        status=GQLPrelabelJobStatus(job.status.value),
        total_assets=job.total_assets,
        processed_assets=job.processed_assets,
        autolabeled_assets=job.autolabeled_assets,
        auto_submitted_assets=job.auto_submitted_assets,
        confidence_threshold=job.confidence_threshold,
        autolabel_enabled=bool(job.autolabel_enabled),
        auto_submit_enabled=bool(job.auto_submit_enabled),
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@strawberry.type
class Query:
    @strawberry.field
    async def me(self, info: Info) -> UserType:
        user = await _get_user(info)
        return _user_type(user)

    @strawberry.field
    async def projects(self, info: Info) -> list[ProjectType]:
        db: AsyncSession = info.context["db"]
        await _get_user(info)
        result = await db.execute(
            select(Project)
            .options(selectinload(Project.label_classes), selectinload(Project.datasets))
            .order_by(Project.created_at.desc())
        )
        projects = result.scalars().all()
        out = []
        for p in projects:
            datasets = []
            for d in p.datasets:
                asset_count = await db.scalar(
                    select(func.count(Asset.id)).where(Asset.dataset_id == d.id)
                )
                datasets.append(
                    DatasetType(
                        id=strawberry.ID(str(d.id)),
                        name=d.name,
                        version=d.version,
                        asset_count=asset_count or 0,
                        created_at=d.created_at,
                    )
                )
            out.append(
                ProjectType(
                    id=strawberry.ID(str(p.id)),
                    name=p.name,
                    city=p.city,
                    description=p.description,
                    label_classes=[_label_class_type(lc) for lc in sorted(p.label_classes, key=lambda x: x.sort_order)],
                    datasets=datasets,
                    task_counts=await _task_counts(db, p.id),
                )
            )
        return out

    @strawberry.field
    async def project(self, info: Info, id: strawberry.ID) -> Optional[ProjectType]:
        db: AsyncSession = info.context["db"]
        await _get_user(info)
        result = await db.execute(
            select(Project)
            .where(Project.id == UUID(id))
            .options(selectinload(Project.label_classes), selectinload(Project.datasets))
        )
        p = result.scalar_one_or_none()
        if not p:
            return None
        datasets = []
        for d in p.datasets:
            asset_count = await db.scalar(select(func.count(Asset.id)).where(Asset.dataset_id == d.id))
            datasets.append(
                DatasetType(
                    id=strawberry.ID(str(d.id)),
                    name=d.name,
                    version=d.version,
                    asset_count=asset_count or 0,
                    created_at=d.created_at,
                )
            )
        return ProjectType(
            id=strawberry.ID(str(p.id)),
            name=p.name,
            city=p.city,
            description=p.description,
            label_classes=[_label_class_type(lc) for lc in sorted(p.label_classes, key=lambda x: x.sort_order)],
            datasets=datasets,
            task_counts=await _task_counts(db, p.id),
        )

    @strawberry.field
    async def my_task_queue(self, info: Info) -> list[TaskType]:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        result = await db.execute(
            select(Task)
            .join(Asset)
            .join(Dataset)
            .options(
                selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
                selectinload(Task.annotations).selectinload(Annotation.label_class),
                selectinload(Task.assignee),
            )
            .where(
                Task.status.in_([TaskStatus.pending, TaskStatus.in_progress, TaskStatus.rejected]),
            )
            .order_by(Task.uncertainty_score.desc().nulls_last(), Task.created_at)
        )
        tasks = result.scalars().all()
        if user.role == UserRole.labeler:
            tasks = [t for t in tasks if labeler_can_see_task(t, user)]
        return [await _build_task_type(db, t) for t in tasks]

    @strawberry.field
    async def review_queue(self, info: Info) -> list[TaskType]:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        if user.role not in (UserRole.reviewer, UserRole.admin):
            return []
        result = await db.execute(
            select(Task)
            .options(
                selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
                selectinload(Task.annotations).selectinload(Annotation.label_class),
                selectinload(Task.assignee),
            )
            .where(Task.status.in_([TaskStatus.submitted, TaskStatus.in_review]))
            .order_by(Task.submitted_at.desc().nullslast())
        )
        return [await _build_task_type(db, t) for t in result.scalars().all()]

    @strawberry.field
    async def task(self, info: Info, id: strawberry.ID) -> Optional[TaskType]:
        db: AsyncSession = info.context["db"]
        await _get_user(info)
        result = await db.execute(
            select(Task)
            .where(Task.id == UUID(id))
            .options(
                selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
                selectinload(Task.annotations).selectinload(Annotation.label_class),
                selectinload(Task.assignee),
            )
        )
        t = result.scalar_one_or_none()
        if not t:
            return None
        return await _build_task_type(db, t)

    @strawberry.field
    async def prelabel_predictions(self, info: Info, asset_id: strawberry.ID) -> list[PrelabelPredictionType]:
        db: AsyncSession = info.context["db"]
        await _get_user(info)
        result = await db.execute(
            select(PrelabelPrediction)
            .where(PrelabelPrediction.asset_id == UUID(asset_id))
            .options(selectinload(PrelabelPrediction.label_class))
        )
        out = []
        for p in result.scalars().all():
            lc = p.label_class
            out.append(
                PrelabelPredictionType(
                    id=strawberry.ID(str(p.id)),
                    asset_id=strawberry.ID(str(p.asset_id)),
                    label_class_id=strawberry.ID(str(p.label_class_id)),
                    label_class_name=lc.name if lc else "",
                    label_class_color=lc.color if lc else "#000",
                    type=GQLAnnotationType(p.type.value),
                    geometry=p.geometry,
                    confidence=p.confidence,
                    accepted=p.accepted,
                )
            )
        return out

    @strawberry.field
    async def prelabel_status(self, info: Info, dataset_id: strawberry.ID) -> Optional[PrelabelJobType]:
        db: AsyncSession = info.context["db"]
        await _get_user(info)
        result = await db.execute(
            select(PrelabelJob)
            .where(PrelabelJob.dataset_id == UUID(dataset_id))
            .order_by(PrelabelJob.created_at.desc())
            .limit(1)
        )
        job = result.scalar_one_or_none()
        if not job:
            return None
        return _prelabel_job_type(job)

    @strawberry.field
    async def dashboard_summary(self, info: Info, project_id: Optional[strawberry.ID] = None) -> DashboardSummary:
        db: AsyncSession = info.context["db"]
        await _get_user(info)

        base = select(Task.status, func.count(Task.id))
        if project_id:
            base = (
                base.join(Asset, Task.asset_id == Asset.id)
                .join(Dataset, Asset.dataset_id == Dataset.id)
                .where(Dataset.project_id == UUID(project_id))
            )
        base = base.group_by(Task.status)
        result = await db.execute(base)
        status_counts = {s.value: 0 for s in TaskStatus}
        total = 0
        for status, count in result.all():
            status_counts[status.value] = count
            total += count

        approved = status_counts.get("approved", 0)
        rejected = status_counts.get("rejected", 0)
        reviewed = approved + rejected
        pass_rate = (approved / reviewed * 100) if reviewed else 0.0

        prelabel_result = await db.execute(
            select(PrelabelPrediction.accepted).where(PrelabelPrediction.accepted.isnot(None))
        )
        accepted_vals = [r[0] for r in prelabel_result.all()]
        prelabel_rate = (sum(1 for v in accepted_vals if v) / len(accepted_vals) * 100) if accepted_vals else 0.0

        from config import settings

        uncertainty_cutoff = settings.active_learning_uncertainty_threshold

        autolabel_q = select(func.count(Task.id)).where(Task.autolabeled.is_(True))
        uncertain_q = select(func.count(Task.id)).where(
            Task.uncertainty_score.isnot(None),
            Task.uncertainty_score >= uncertainty_cutoff,
            Task.status.in_([TaskStatus.pending, TaskStatus.in_progress, TaskStatus.rejected]),
        )
        if project_id:
            pid = UUID(project_id)
            autolabel_q = (
                autolabel_q.join(Asset, Task.asset_id == Asset.id)
                .join(Dataset, Asset.dataset_id == Dataset.id)
                .where(Dataset.project_id == pid)
            )
            uncertain_q = (
                uncertain_q.join(Asset, Task.asset_id == Asset.id)
                .join(Dataset, Asset.dataset_id == Dataset.id)
                .where(Dataset.project_id == pid)
            )
        autolabeled_tasks = await db.scalar(autolabel_q) or 0
        uncertain_tasks = await db.scalar(uncertain_q) or 0

        from datetime import UTC, datetime, timedelta

        throughput = []
        for i in range(6, -1, -1):
            day = datetime.now(UTC).date() - timedelta(days=i)
            day_start = datetime.combine(day, datetime.min.time()).replace(tzinfo=UTC)
            day_end = day_start + timedelta(days=1)
            q = select(func.count(Task.id)).where(
                Task.submitted_at >= day_start, Task.submitted_at < day_end
            )
            if project_id:
                q = (
                    q.join(Asset, Task.asset_id == Asset.id)
                    .join(Dataset, Asset.dataset_id == Dataset.id)
                    .where(Dataset.project_id == UUID(project_id))
                )
            count = await db.scalar(q) or 0
            throughput.append({"date": day.isoformat(), "count": count})

        proj_result = await db.execute(select(Project))
        project_progress = []
        for p in proj_result.scalars().all():
            counts = await _task_counts(db, p.id)
            ptotal = sum(counts.values())
            project_progress.append(
                {
                    "project_id": str(p.id),
                    "project_name": p.name,
                    "approved": counts.get("approved", 0),
                    "total": ptotal,
                    "percent": round(counts.get("approved", 0) / ptotal * 100, 1) if ptotal else 0,
                }
            )

        return DashboardSummary(
            total_tasks=total,
            pending_tasks=status_counts.get("pending", 0),
            in_progress_tasks=status_counts.get("in_progress", 0),
            submitted_tasks=status_counts.get("submitted", 0),
            approved_tasks=approved,
            rejected_tasks=rejected,
            autolabeled_tasks=autolabeled_tasks,
            uncertain_tasks=uncertain_tasks,
            review_pass_rate=round(pass_rate, 1),
            prelabel_acceptance_rate=round(prelabel_rate, 1),
            tasks_by_status=status_counts,
            throughput_by_day=throughput,
            project_progress=project_progress,
        )

    @strawberry.field
    async def autolabel_review_queue(self, info: Info) -> list[TaskType]:
        return await resolve_autolabel_review_queue(info)

    @strawberry.field
    async def labeler_quality(
        self, info: Info, project_id: Optional[strawberry.ID] = None
    ) -> list[LabelerQualityType]:
        return await resolve_labeler_quality(info, project_id)

    @strawberry.field
    async def sla_summary(
        self, info: Info, project_id: Optional[strawberry.ID] = None
    ) -> SlaSummaryType:
        return await resolve_sla_summary(info, project_id)

    @strawberry.field
    async def model_versions(self, info: Info, project_id: strawberry.ID) -> list[ModelVersionType]:
        return await resolve_model_versions(info, project_id)

    @strawberry.field
    async def task_qa_check(self, info: Info, task_id: strawberry.ID) -> Optional[QaCheckType]:
        return await resolve_task_qa_check(info, task_id)

    @strawberry.field
    async def training_feedback_stats(
        self, info: Info, project_id: strawberry.ID
    ) -> TrainingFeedbackStatsType:
        return await resolve_training_feedback_stats(info, project_id)


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_project(self, info: Info, input: CreateProjectInput) -> ProjectType:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        if user.role not in (UserRole.admin, UserRole.ml_engineer):
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail="Insufficient permissions")

        project = Project(
            name=input.name,
            city=input.city,
            description=input.description,
            label_schema={"classes": [lc.name for lc in input.label_classes]},
        )
        db.add(project)
        await db.flush()
        for i, lc in enumerate(input.label_classes):
            db.add(LabelClass(project_id=project.id, name=lc.name, color=lc.color, sort_order=i))
        dataset = Dataset(project_id=project.id, name="Default Dataset", version=1)
        db.add(dataset)
        await db.commit()
        await db.refresh(project)
        result = await db.execute(
            select(Project)
            .where(Project.id == project.id)
            .options(selectinload(Project.label_classes), selectinload(Project.datasets))
        )
        p = result.scalar_one()
        return ProjectType(
            id=strawberry.ID(str(p.id)),
            name=p.name,
            city=p.city,
            description=p.description,
            label_classes=[_label_class_type(lc) for lc in sorted(p.label_classes, key=lambda x: x.sort_order)],
            datasets=[
                DatasetType(
                    id=strawberry.ID(str(d.id)),
                    name=d.name,
                    version=d.version,
                    asset_count=0,
                    created_at=d.created_at,
                )
                for d in p.datasets
            ],
            task_counts={s.value: 0 for s in TaskStatus},
        )

    @strawberry.mutation
    async def register_assets(
        self, info: Info, dataset_id: strawberry.ID, assets: list[RegisterAssetInput]
    ) -> int:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        if user.role not in (UserRole.admin, UserRole.ml_engineer):
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail="Insufficient permissions")

        count = 0
        for a in assets:
            asset = Asset(
                dataset_id=UUID(dataset_id),
                filename=a.filename,
                storage_key=a.storage_key,
                mime_type=a.mime_type,
                width=a.width,
                height=a.height,
                file_size=a.file_size,
            )
            db.add(asset)
            await db.flush()
            task = Task(asset_id=asset.id, status=TaskStatus.pending)
            db.add(task)
            count += 1
        await db.commit()
        return count

    @strawberry.mutation
    async def claim_task(self, info: Info, task_id: strawberry.ID) -> TaskType:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        result = await db.execute(
            select(Task)
            .where(Task.id == UUID(task_id))
            .options(
                selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
                selectinload(Task.annotations).selectinload(Annotation.label_class),
                selectinload(Task.assignee),
            )
        )
        task = result.scalar_one_or_none()
        if not task:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Task not found")

        action = "rework" if task.status == TaskStatus.rejected else "claim"
        apply_transition(task, action, user)
        await db.commit()
        await db.refresh(task)
        return await _build_task_type(db, task)

    @strawberry.mutation
    async def submit_task(self, info: Info, task_id: strawberry.ID) -> TaskType:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        result = await db.execute(
            select(Task)
            .where(Task.id == UUID(task_id))
            .options(
                selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
                selectinload(Task.annotations).selectinload(Annotation.label_class),
                selectinload(Task.assignee),
            )
        )
        task = result.scalar_one_or_none()
        if not task:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Task not found")
        apply_transition(task, "submit", user)
        from services.auto_qa import run_auto_qa

        await run_auto_qa(db, task.id)
        await db.commit()
        return await _build_task_type(db, task)

    @strawberry.mutation
    async def save_annotations(
        self, info: Info, task_id: strawberry.ID, annotations: list[AnnotationInput]
    ) -> list[AnnotationTypeGQL]:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        result = await db.execute(
            select(Task)
            .where(Task.id == UUID(task_id))
            .options(selectinload(Task.annotations))
        )
        task = result.scalar_one_or_none()
        if not task:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Task not found")

        existing_ids = {str(a.id) for a in task.annotations}
        incoming_ids = {a.id for a in annotations if a.id}
        for eid in existing_ids - incoming_ids:
            for ann in task.annotations:
                if str(ann.id) == eid:
                    db.delete(ann)

        saved = []
        for inp in annotations:
            if inp.id and inp.id in existing_ids:
                ann = next(a for a in task.annotations if str(a.id) == inp.id)
                ann.geometry = inp.geometry
                ann.label_class_id = UUID(inp.label_class_id)
                ann.type = AnnotationType(inp.type.value)
                ann.source = AnnotationSource(inp.source.value)
                ann.version += 1
            else:
                ann = Annotation(
                    task_id=task.id,
                    author_id=user.id,
                    label_class_id=UUID(inp.label_class_id),
                    type=AnnotationType(inp.type.value),
                    geometry=inp.geometry,
                    source=AnnotationSource(inp.source.value),
                )
                db.add(ann)
            saved.append(ann)

        await db.commit()
        out = []
        for ann in saved:
            await db.refresh(ann)
            lc_result = await db.execute(select(LabelClass).where(LabelClass.id == ann.label_class_id))
            lc = lc_result.scalar_one()
            out.append(
                AnnotationTypeGQL(
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
            )
        return out

    @strawberry.mutation
    async def claim_review(self, info: Info, task_id: strawberry.ID) -> TaskType:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        result = await db.execute(
            select(Task)
            .where(Task.id == UUID(task_id))
            .options(
                selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
                selectinload(Task.annotations).selectinload(Annotation.label_class),
                selectinload(Task.assignee),
            )
        )
        task = result.scalar_one_or_none()
        if not task:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Task not found")
        apply_transition(task, "claim_review", user)
        await db.commit()
        return await _build_task_type(db, task)

    @strawberry.mutation
    async def approve_task(self, info: Info, task_id: strawberry.ID, comment: Optional[str] = None) -> TaskType:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        result = await db.execute(
            select(Task)
            .where(Task.id == UUID(task_id))
            .options(
                selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
                selectinload(Task.annotations).selectinload(Annotation.label_class),
                selectinload(Task.assignee),
            )
        )
        task = result.scalar_one_or_none()
        if not task:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Task not found")
        apply_transition(task, "approve", user)
        db.add(Review(task_id=task.id, reviewer_id=user.id, status="approved", comment=comment))
        await db.commit()
        return await _build_task_type(db, task)

    @strawberry.mutation
    async def reject_task(self, info: Info, task_id: strawberry.ID, comment: str) -> TaskType:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        result = await db.execute(
            select(Task)
            .where(Task.id == UUID(task_id))
            .options(
                selectinload(Task.asset).selectinload(Asset.dataset).selectinload(Dataset.project),
                selectinload(Task.annotations).selectinload(Annotation.label_class),
                selectinload(Task.assignee),
            )
        )
        task = result.scalar_one_or_none()
        if not task:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Task not found")
        apply_transition(task, "reject", user)
        task.rejection_reason = comment
        db.add(Review(task_id=task.id, reviewer_id=user.id, status="rejected", comment=comment))
        await db.commit()
        return await _build_task_type(db, task)

    @strawberry.mutation
    async def trigger_prelabel(self, info: Info, input: TriggerPrelabelInput) -> PrelabelJobType:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        if user.role not in (UserRole.admin, UserRole.ml_engineer):
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail="Insufficient permissions")

        dataset_id = input.dataset_id
        ds_result = await db.execute(select(Dataset).where(Dataset.id == UUID(dataset_id)))
        dataset = ds_result.scalar_one()

        model_version_id = None
        if input.model_version_id:
            model_version_id = UUID(input.model_version_id)
        else:
            from services.model_registry import get_or_create_default_model

            mv = await get_or_create_default_model(db, dataset.project_id)
            model_version_id = mv.id

        asset_count = await db.scalar(
            select(func.count(Asset.id)).where(Asset.dataset_id == UUID(dataset_id))
        )
        job = PrelabelJob(
            dataset_id=UUID(dataset_id),
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

    @strawberry.mutation
    async def apply_prelabels(self, info: Info, task_id: strawberry.ID) -> list[AnnotationTypeGQL]:
        db: AsyncSession = info.context["db"]
        user = await _get_user(info)
        result = await db.execute(
            select(Task)
            .where(Task.id == UUID(task_id))
            .options(selectinload(Task.asset).selectinload(Asset.prelabel_predictions))
        )
        task = result.scalar_one_or_none()
        if not task:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Task not found")

        preds = task.asset.prelabel_predictions
        saved = []
        for pred in preds:
            if pred.accepted is False:
                continue
            ann = Annotation(
                task_id=task.id,
                author_id=user.id,
                label_class_id=pred.label_class_id,
                type=pred.type,
                geometry=pred.geometry,
                source=AnnotationSource.prelabel,
            )
            db.add(ann)
            pred.accepted = True
            saved.append(ann)

        await db.commit()
        out = []
        for ann in saved:
            await db.refresh(ann)
            lc_result = await db.execute(select(LabelClass).where(LabelClass.id == ann.label_class_id))
            lc = lc_result.scalar_one()
            out.append(
                AnnotationTypeGQL(
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
            )
        return out

    @strawberry.mutation
    async def accept_prelabel_prediction(
        self, info: Info, prediction_id: strawberry.ID, task_id: strawberry.ID
    ) -> AnnotationTypeGQL:
        return await resolve_accept_prelabel(info, prediction_id, task_id)

    @strawberry.mutation
    async def reject_prelabel_prediction(
        self, info: Info, prediction_id: strawberry.ID, task_id: strawberry.ID
    ) -> bool:
        return await resolve_reject_prelabel(info, prediction_id, task_id)

    @strawberry.mutation
    async def run_auto_qa(self, info: Info, task_id: strawberry.ID) -> QaCheckType:
        return await resolve_run_auto_qa(info, task_id)

    @strawberry.mutation
    async def approve_autolabel_review(self, info: Info, task_id: strawberry.ID) -> TaskType:
        return await resolve_approve_autolabel(info, task_id)

    @strawberry.mutation
    async def reject_autolabel_review(
        self, info: Info, task_id: strawberry.ID, comment: str
    ) -> TaskType:
        return await resolve_reject_autolabel(info, task_id, comment)

    @strawberry.mutation
    async def create_model_version(
        self, info: Info, input: CreateModelVersionInput
    ) -> ModelVersionType:
        return await resolve_create_model_version(info, input)

    @strawberry.mutation
    async def activate_model_version(
        self, info: Info, version_id: strawberry.ID
    ) -> ModelVersionType:
        return await resolve_activate_model_version(info, version_id)

    @strawberry.mutation
    async def trigger_retraining(self, info: Info, project_id: strawberry.ID) -> ModelVersionType:
        return await resolve_trigger_retraining(info, project_id)

    @strawberry.mutation
    async def batch_rerun_prelabel(self, info: Info, input: BatchRerunInput) -> PrelabelJobType:
        return await resolve_batch_rerun_prelabel(info, input)


schema = strawberry.Schema(query=Query, mutation=Mutation)
