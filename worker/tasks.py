import asyncio
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from config import settings
from database import async_session
from models import (
    Asset,
    Dataset,
    LabelClass,
    ModelVersion,
    ModelVersionStatus,
    PrelabelJob,
    PrelabelJobStatus,
    PrelabelPrediction,
    Project,
)
from services.active_learning import update_task_uncertainty
from services.autolabel import process_autolabel_for_asset
from services.prelabel import generate_openai_predictions


async def _resolve_openai_model(db, job: PrelabelJob) -> str:
    if job.model_version_id:
        result = await db.execute(
            select(ModelVersion).where(ModelVersion.id == job.model_version_id)
        )
        mv = result.scalar_one_or_none()
        if mv and mv.model_config.get("model"):
            return mv.model_config["model"]
    return settings.openai_model


async def run_prelabel_job(job_id: str) -> dict:
    async with async_session() as db:
        result = await db.execute(select(PrelabelJob).where(PrelabelJob.id == UUID(job_id)))
        job = result.scalar_one_or_none()
        if not job:
            return {"error": "Job not found"}

        job.status = PrelabelJobStatus.running
        await db.commit()

        confidence_threshold = (
            job.confidence_threshold
            if job.confidence_threshold is not None
            else settings.autolabel_confidence_threshold
        )
        autolabel_enabled = job.autolabel_enabled
        auto_submit = job.auto_submit_enabled

        autolabeled_count = 0
        auto_submitted_count = 0

        try:
            ds_result = await db.execute(
                select(Dataset)
                .where(Dataset.id == job.dataset_id)
                .options(
                    selectinload(Dataset.assets),
                    selectinload(Dataset.project).selectinload(Project.label_classes),
                )
            )
            dataset = ds_result.scalar_one()
            label_classes = list(dataset.project.label_classes)
            openai_model = await _resolve_openai_model(db, job)

            for i, asset in enumerate(dataset.assets):
                existing = await db.execute(
                    select(PrelabelPrediction).where(PrelabelPrediction.asset_id == asset.id)
                )
                for old in existing.scalars().all():
                    db.delete(old)

                predictions = await generate_openai_predictions(
                    asset, label_classes, model=openai_model
                )
                for pred in predictions:
                    db.add(
                        PrelabelPrediction(
                            asset_id=asset.id,
                            label_class_id=pred["label_class_id"],
                            type=pred["type"],
                            geometry=pred["geometry"],
                            confidence=pred["confidence"],
                        )
                    )
                await db.flush()

                pred_rows = list(
                    (
                        await db.execute(
                            select(PrelabelPrediction).where(PrelabelPrediction.asset_id == asset.id)
                        )
                    ).scalars().all()
                )

                if autolabel_enabled and pred_rows:
                    stats = await process_autolabel_for_asset(
                        db,
                        asset.id,
                        confidence_threshold=confidence_threshold,
                        auto_submit=auto_submit,
                    )
                    if stats["autolabeled"]:
                        autolabeled_count += 1
                    if stats["auto_submitted"]:
                        auto_submitted_count += 1
                elif pred_rows:
                    await update_task_uncertainty(db, asset.id, pred_rows)

                job.processed_assets = i + 1
                job.autolabeled_assets = autolabeled_count
                job.auto_submitted_assets = auto_submitted_count
                await db.commit()
                await asyncio.sleep(0.2)

            job.status = PrelabelJobStatus.completed
            job.completed_at = datetime.now(UTC)
            await db.commit()
            return {
                "status": "completed",
                "processed": job.processed_assets,
                "autolabeled": autolabeled_count,
                "auto_submitted": auto_submitted_count,
            }
        except Exception as exc:
            job.status = PrelabelJobStatus.failed
            job.error_message = str(exc)
            await db.commit()
            return {"error": str(exc)}


def schedule_prelabel_job(job_id: str) -> None:
    asyncio.create_task(run_prelabel_job(job_id))
