"""Lightweight SQLite schema patches for existing databases."""

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine


async def apply_schema_patches(engine: AsyncEngine) -> None:
    """Add columns introduced after initial MVP without requiring a full reset."""
    async with engine.begin() as conn:
        def _patch(sync_conn) -> None:
            inspector = inspect(sync_conn)
            tables = inspector.get_table_names()

            if "tasks" in tables:
                task_cols = {c["name"] for c in inspector.get_columns("tasks")}
                patches = {
                    "uncertainty_score": "ALTER TABLE tasks ADD COLUMN uncertainty_score FLOAT",
                    "autolabeled": "ALTER TABLE tasks ADD COLUMN autolabeled BOOLEAN DEFAULT 0",
                    "qa_score": "ALTER TABLE tasks ADD COLUMN qa_score FLOAT",
                    "qa_passed": "ALTER TABLE tasks ADD COLUMN qa_passed BOOLEAN",
                    "claimed_at": "ALTER TABLE tasks ADD COLUMN claimed_at DATETIME",
                    "sla_deadline": "ALTER TABLE tasks ADD COLUMN sla_deadline DATETIME",
                    "autolabel_review_status": "ALTER TABLE tasks ADD COLUMN autolabel_review_status VARCHAR(50)",
                }
                for col, sql in patches.items():
                    if col not in task_cols:
                        sync_conn.execute(text(sql))

            if "prelabel_jobs" in tables:
                job_cols = {c["name"] for c in inspector.get_columns("prelabel_jobs")}
                job_patches = {
                    "autolabeled_assets": "ALTER TABLE prelabel_jobs ADD COLUMN autolabeled_assets INTEGER DEFAULT 0",
                    "auto_submitted_assets": "ALTER TABLE prelabel_jobs ADD COLUMN auto_submitted_assets INTEGER DEFAULT 0",
                    "confidence_threshold": "ALTER TABLE prelabel_jobs ADD COLUMN confidence_threshold FLOAT",
                    "autolabel_enabled": "ALTER TABLE prelabel_jobs ADD COLUMN autolabel_enabled BOOLEAN DEFAULT 1",
                    "auto_submit_enabled": "ALTER TABLE prelabel_jobs ADD COLUMN auto_submit_enabled BOOLEAN DEFAULT 1",
                    "model_version_id": "ALTER TABLE prelabel_jobs ADD COLUMN model_version_id CHAR(32)",
                }
                for col, sql in job_patches.items():
                    if col not in job_cols:
                        sync_conn.execute(text(sql))

        await conn.run_sync(_patch)
