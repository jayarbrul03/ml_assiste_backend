"""Per-labeler quality scoring and SLA tracking."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models import Review, Task, TaskStatus, User, UserRole


def compute_sla_deadline(claimed_at: datetime) -> datetime:
    from datetime import timedelta

    return claimed_at + timedelta(hours=settings.sla_hours)


def set_task_sla_on_claim(task: Task) -> None:
    from datetime import timedelta

    now = datetime.now(UTC)
    task.claimed_at = now
    task.sla_deadline = now + timedelta(hours=settings.sla_hours)


async def get_labeler_quality(db: AsyncSession, project_id: UUID | None = None) -> list[dict]:
    """Compute per-labeler quality metrics from task/review/QA data."""
    labelers_q = select(User).where(User.role == UserRole.labeler)
    labelers = list((await db.execute(labelers_q)).scalars().all())

    results = []
    for labeler in labelers:
        base = select(Task).where(Task.assignee_id == labeler.id)
        if project_id:
            from models import Asset, Dataset

            base = (
                base.join(Asset, Task.asset_id == Asset.id)
                .join(Dataset, Asset.dataset_id == Dataset.id)
                .where(Dataset.project_id == project_id)
            )

        tasks = list((await db.execute(base)).scalars().all())
        if not tasks:
            continue

        completed = [t for t in tasks if t.status in (TaskStatus.submitted, TaskStatus.in_review, TaskStatus.approved, TaskStatus.rejected)]
        approved = [t for t in tasks if t.status == TaskStatus.approved]
        rejected = [t for t in tasks if t.status == TaskStatus.rejected]

        reviewed = len(approved) + len(rejected)
        approval_rate = round(len(approved) / reviewed * 100, 1) if reviewed else 0.0

        qa_scores = [t.qa_score for t in tasks if t.qa_score is not None]
        avg_qa = round(sum(qa_scores) / len(qa_scores), 3) if qa_scores else None

        sla_met = 0
        sla_total = 0
        for t in completed:
            if t.claimed_at and t.submitted_at and t.sla_deadline:
                sla_total += 1
                if t.submitted_at <= t.sla_deadline:
                    sla_met += 1
        sla_compliance = round(sla_met / sla_total * 100, 1) if sla_total else 100.0

        # Composite quality score 0-100
        quality = 0.0
        parts = 0
        if reviewed:
            quality += approval_rate * 0.4
            parts += 0.4
        if avg_qa is not None:
            quality += avg_qa * 100 * 0.35
            parts += 0.35
        quality += sla_compliance * 0.25
        parts += 0.25
        quality_score = round(quality / parts, 1) if parts else 0.0

        # Avg turnaround hours
        turnarounds = []
        for t in completed:
            if t.claimed_at and t.submitted_at:
                hrs = (t.submitted_at - t.claimed_at).total_seconds() / 3600
                turnarounds.append(hrs)
        avg_turnaround = round(sum(turnarounds) / len(turnarounds), 2) if turnarounds else None

        results.append(
            {
                "labeler_id": str(labeler.id),
                "labeler_name": labeler.name,
                "labeler_email": labeler.email,
                "tasks_assigned": len(tasks),
                "tasks_completed": len(completed),
                "tasks_approved": len(approved),
                "tasks_rejected": len(rejected),
                "approval_rate": approval_rate,
                "avg_qa_score": avg_qa,
                "sla_compliance_rate": sla_compliance,
                "avg_turnaround_hours": avg_turnaround,
                "quality_score": quality_score,
            }
        )

    results.sort(key=lambda x: x["quality_score"], reverse=True)
    return results


async def get_sla_summary(db: AsyncSession, project_id: UUID | None = None) -> dict:
    """SLA tracking summary across active tasks."""
    base = select(Task).where(
        Task.status.in_([TaskStatus.in_progress, TaskStatus.pending, TaskStatus.rejected]),
        Task.sla_deadline.isnot(None),
    )
    if project_id:
        from models import Asset, Dataset

        base = (
            base.join(Asset, Task.asset_id == Asset.id)
            .join(Dataset, Asset.dataset_id == Dataset.id)
            .where(Dataset.project_id == project_id)
        )

    tasks = list((await db.execute(base)).scalars().all())
    now = datetime.now(UTC)
    overdue = [t for t in tasks if t.sla_deadline and t.sla_deadline < now]
    at_risk = [
        t for t in tasks
        if t.sla_deadline and t.sla_deadline >= now
        and (t.sla_deadline - now).total_seconds() < settings.sla_hours * 3600 * 0.25
    ]

    return {
        "total_tracked": len(tasks),
        "overdue": len(overdue),
        "at_risk": len(at_risk),
        "on_track": len(tasks) - len(overdue) - len(at_risk),
        "sla_hours": settings.sla_hours,
    }
