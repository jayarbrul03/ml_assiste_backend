"""Auto-QA: IoU checks, consensus scoring, gold-standard comparison."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from config import settings
from models import (
    Annotation,
    Asset,
    GoldStandardAnnotation,
    PrelabelPrediction,
    QaCheck,
    Task,
)
from services.geometry import match_and_score


def _ann_dict(ann: Annotation | GoldStandardAnnotation | PrelabelPrediction) -> dict:
    return {
        "label_class_id": str(ann.label_class_id),
        "type": ann.type.value if hasattr(ann.type, "value") else ann.type,
        "geometry": ann.geometry,
    }


async def run_auto_qa(db: AsyncSession, task_id: UUID) -> QaCheck:
    """Run full Auto-QA suite on a task and persist results."""
    result = await db.execute(
        select(Task)
        .where(Task.id == task_id)
        .options(
            selectinload(Task.annotations),
            selectinload(Task.asset).selectinload(Asset.prelabel_predictions),
        )
    )
    task = result.scalar_one_or_none()
    if not task:
        raise ValueError("Task not found")

    submitted = [_ann_dict(a) for a in task.annotations]

    # Gold standard comparison
    gs_result = await db.execute(
        select(GoldStandardAnnotation).where(GoldStandardAnnotation.task_id == task_id)
    )
    gold_standards = list(gs_result.scalars().all())
    gold_refs = [_ann_dict(g) for g in gold_standards]
    gold_score, gold_issues = (
        match_and_score(submitted, gold_refs, iou_threshold=settings.qa_iou_threshold)
        if gold_refs
        else (None, [])
    )

    # Consensus: submitted vs pre-label predictions
    preds = task.asset.prelabel_predictions if task.asset else []
    pred_refs = [_ann_dict(p) for p in preds]
    consensus_score, consensus_issues = (
        match_and_score(submitted, pred_refs, iou_threshold=settings.qa_iou_threshold)
        if pred_refs
        else (None, [])
    )

    # Self-consistency IoU (duplicate check among submitted — penalize overlaps)
    iou_score = 1.0
    iou_issues: list[str] = []
    if len(submitted) >= 2:
        overlaps = 0
        for i in range(len(submitted)):
            for j in range(i + 1, len(submitted)):
                from services.geometry import geometry_iou

                iou = geometry_iou(
                    submitted[i]["geometry"],
                    submitted[j]["geometry"],
                    submitted[i].get("type", "bbox"),
                )
                if iou > 0.7 and submitted[i]["label_class_id"] == submitted[j]["label_class_id"]:
                    overlaps += 1
        if overlaps:
            iou_score = max(0.0, 1.0 - overlaps * 0.2)
            iou_issues.append(f"{overlaps} overlapping duplicate annotation(s) detected")
    elif len(submitted) == 1 and gold_refs:
        iou_score = gold_score or 0.0
    elif len(submitted) == 0:
        iou_score = 0.0
        iou_issues.append("No annotations submitted")

    # Weighted overall score
    scores: list[tuple[float, float]] = [(iou_score, 0.3)]
    if consensus_score is not None:
        scores.append((consensus_score, 0.35))
    if gold_score is not None:
        scores.append((gold_score, 0.35))

    total_w = sum(w for _, w in scores)
    overall = round(sum(s * w for s, w in scores) / total_w, 4) if total_w else 0.0

    all_issues = iou_issues + consensus_issues + gold_issues
    passed = overall >= settings.qa_pass_threshold and not any(
        "Missing" in i or "Extra" in i or "No annotations" in i for i in all_issues
    )

    # Upsert QaCheck
    existing = await db.execute(select(QaCheck).where(QaCheck.task_id == task_id))
    qa = existing.scalar_one_or_none()
    if not qa:
        qa = QaCheck(task_id=task_id)
        db.add(qa)

    qa.iou_score = iou_score
    qa.consensus_score = consensus_score
    qa.gold_standard_score = gold_score
    qa.overall_score = overall
    qa.passed = passed
    qa.issues = all_issues

    task.qa_score = overall
    task.qa_passed = passed

    await db.flush()
    return qa
