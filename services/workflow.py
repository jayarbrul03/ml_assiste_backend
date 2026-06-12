import uuid
from dataclasses import dataclass

from fastapi import HTTPException, status

from models import Task, TaskStatus, User, UserRole

ALLOWED_TRANSITIONS: dict[TaskStatus, dict[str, TaskStatus]] = {
    TaskStatus.pending: {"claim": TaskStatus.in_progress},
    TaskStatus.in_progress: {"submit": TaskStatus.submitted, "release": TaskStatus.pending},
    TaskStatus.submitted: {"claim_review": TaskStatus.in_review},
    TaskStatus.in_review: {"approve": TaskStatus.approved, "reject": TaskStatus.rejected},
    TaskStatus.rejected: {"rework": TaskStatus.in_progress},
    TaskStatus.approved: {},
}


@dataclass
class TransitionResult:
    task: Task
    previous_status: TaskStatus


def labeler_can_see_task(task: Task, user: User) -> bool:
    """Whether a task should appear in a labeler's queue."""
    if task.assignee_id == user.id:
        return True
    if task.status == TaskStatus.pending:
        return True
    if task.status == TaskStatus.rejected and task.assignee_id is None:
        return True
    return False


def validate_transition(task: Task, action: str, user: User) -> TaskStatus:
    allowed = ALLOWED_TRANSITIONS.get(task.status, {})
    new_status = allowed.get(action)
    if not new_status:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot {action} task in status {task.status.value}",
        )

    if action == "claim":
        if user.role not in (UserRole.labeler, UserRole.admin):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Labeler role required")
        if task.assignee_id and task.assignee_id != user.id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task already claimed")

    if action == "submit":
        if task.assignee_id != user.id and user.role != UserRole.admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not task assignee")
        if not task.annotations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task must have at least one annotation before submit",
            )

    if action == "claim_review":
        if user.role not in (UserRole.reviewer, UserRole.admin):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Reviewer role required")

    if action in ("approve", "reject"):
        if user.role not in (UserRole.reviewer, UserRole.admin):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Reviewer role required")
        if task.reviewer_id and task.reviewer_id != user.id and user.role != UserRole.admin:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task claimed by another reviewer")

    if action == "rework":
        if user.role not in (UserRole.labeler, UserRole.admin):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Labeler role required")

    return new_status


def apply_transition(task: Task, action: str, user: User) -> TransitionResult:
    previous = task.status

    # Idempotent: already claimed by same user (avoids double-call errors in React Strict Mode)
    if action == "claim_review" and task.status == TaskStatus.in_review:
        if user.role == UserRole.admin or task.reviewer_id == user.id:
            return TransitionResult(task=task, previous_status=previous)
        if task.reviewer_id and task.reviewer_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Task claimed by another reviewer",
            )

    if action == "claim" and task.status == TaskStatus.in_progress:
        if user.role == UserRole.admin or task.assignee_id == user.id:
            return TransitionResult(task=task, previous_status=previous)
        if task.assignee_id and task.assignee_id != user.id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task already claimed")

    if action == "rework" and task.status == TaskStatus.in_progress:
        if user.role == UserRole.admin or task.assignee_id == user.id:
            return TransitionResult(task=task, previous_status=previous)

    new_status = validate_transition(task, action, user)
    task.status = new_status

    if action == "claim":
        task.assignee_id = user.id
        from services.labeler_quality import set_task_sla_on_claim

        set_task_sla_on_claim(task)
    elif action == "release":
        task.assignee_id = None
    elif action == "submit":
        from datetime import UTC, datetime

        task.submitted_at = datetime.now(UTC)
    elif action == "claim_review":
        task.reviewer_id = user.id
    elif action == "approve":
        from datetime import UTC, datetime

        task.reviewed_at = datetime.now(UTC)
    elif action == "reject":
        from datetime import UTC, datetime

        task.reviewed_at = datetime.now(UTC)
    elif action == "rework":
        task.reviewer_id = None
        task.rejection_reason = None
        task.assignee_id = user.id
        from services.labeler_quality import set_task_sla_on_claim

        set_task_sla_on_claim(task)

    return TransitionResult(task=task, previous_status=previous)
