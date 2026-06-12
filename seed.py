"""Initialize database schema and default demo data."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select

from auth.password import hash_password
from config import settings
from database import Base, async_session, engine
from models import (
    Annotation,
    AnnotationSource,
    AnnotationType,
    Asset,
    Dataset,
    GoldStandardAnnotation,
    LabelClass,
    ModelVersion,
    ModelVersionStatus,
    Project,
    Task,
    TaskStatus,
    User,
    UserRole,
)

DEMO_USERS = [
    ("admin@demo.com", "Admin User", "admin123", UserRole.admin),
    ("ml@demo.com", "ML Engineer", "ml123", UserRole.ml_engineer),
    ("labeler@demo.com", "Labeler User", "label123", UserRole.labeler),
    ("reviewer@demo.com", "Reviewer User", "review123", UserRole.reviewer),
]

LABEL_CLASSES = [
    ("Pedestrian", "#ef4444"),
    ("Vehicle", "#3b82f6"),
    ("Cyclist", "#22c55e"),
]

# Vehicle / street-scene images (Unsplash, stable URLs)
VEHICLE_IMAGE_SOURCES = [
    ("vehicle_01.jpg", "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&h=600&fit=crop"),
    ("vehicle_02.jpg", "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=800&h=600&fit=crop"),
    ("vehicle_03.jpg", "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?w=800&h=600&fit=crop"),
    ("vehicle_04.jpg", "https://images.unsplash.com/photo-1619767886558-efdc259cde1a?w=800&h=600&fit=crop"),
    ("vehicle_05.jpg", "https://images.unsplash.com/photo-1550355291-bbee04a92027?w=800&h=600&fit=crop"),
    ("vehicle_06.jpg", "https://images.unsplash.com/photo-1605559424843-9e4c228bf1c2?w=800&h=600&fit=crop"),
    ("vehicle_07.jpg", "https://images.unsplash.com/photo-1568605117036-5fe5e7bab0b7?w=800&h=600&fit=crop"),
    ("vehicle_08.jpg", "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800?w=800&h=600&fit=crop"),
    ("vehicle_09.jpg", "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?w=800&h=600&fit=crop"),
    ("vehicle_10.jpg", "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=800&h=600&fit=crop"),
]


async def _download_vehicle_images() -> list[tuple[str, str]]:
    """Download vehicle images to local storage. Returns (filename, storage_key) pairs."""
    sample_dir = settings.local_storage_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str]] = []

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for filename, url in VEHICLE_IMAGE_SOURCES:
            storage_key = f"sample/{filename}"
            local_path = settings.local_storage_dir / storage_key

            if not local_path.exists():
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    local_path.write_bytes(response.content)
                    print(f"  Downloaded {filename}")
                except Exception as exc:
                    print(f"  Warning: could not download {filename} ({exc}), using remote URL")
                    results.append((filename, url))
                    continue

            results.append((filename, storage_key))

    return results


async def create_schema() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    from migrate import apply_schema_patches

    await apply_schema_patches(engine)


async def reset_schema() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)


async def seed_defaults() -> bool:
    """Insert default demo data. Returns True if data was inserted."""
    async with async_session() as db:
        existing = await db.execute(select(User).limit(1))
        if existing.scalar_one_or_none():
            return False

        users: dict[str, User] = {}
        for email, name, password, role in DEMO_USERS:
            user = User(email=email, name=name, hashed_password=hash_password(password), role=role)
            db.add(user)
            users[role.value] = user
        await db.flush()

        project = Project(
            name="San Francisco - Vehicles",
            city="San Francisco",
            description="Demo project for AV vehicle detection and street-scene labeling",
            label_schema={"classes": [lc[0] for lc in LABEL_CLASSES]},
        )
        db.add(project)
        await db.flush()

        label_class_objs = []
        for i, (name, color) in enumerate(LABEL_CLASSES):
            lc = LabelClass(project_id=project.id, name=name, color=color, sort_order=i)
            db.add(lc)
            label_class_objs.append(lc)
        await db.flush()

        dataset = Dataset(project_id=project.id, name="Vehicle Dataset v1", version=1)
        db.add(dataset)
        await db.flush()

        print("Downloading default vehicle images...")
        sample_images = await _download_vehicle_images()

        vehicle_class = next(lc for lc in label_class_objs if lc.name == "Vehicle")

        tasks = []
        for filename, storage_key in sample_images:
            asset = Asset(
                dataset_id=dataset.id,
                filename=filename,
                storage_key=storage_key,
                mime_type="image/jpeg",
                width=800,
                height=600,
            )
            db.add(asset)
            await db.flush()
            task = Task(asset_id=asset.id, status=TaskStatus.pending)
            db.add(task)
            tasks.append(task)

        await db.flush()

        labeler = users["labeler"]
        reviewer = users["reviewer"]

        for i, task in enumerate(tasks[:3]):
            task.status = TaskStatus.approved
            task.assignee_id = labeler.id
            task.reviewer_id = reviewer.id
            task.submitted_at = datetime.now(UTC) - timedelta(days=i + 1)
            task.reviewed_at = datetime.now(UTC) - timedelta(days=i)
            db.add(
                Annotation(
                    task_id=task.id,
                    author_id=labeler.id,
                    label_class_id=vehicle_class.id,
                    type=AnnotationType.bbox,
                    geometry={"x": 120 + i * 30, "y": 180, "width": 200, "height": 120},
                    source=AnnotationSource.human,
                )
            )

        tasks[3].status = TaskStatus.submitted
        tasks[3].assignee_id = labeler.id
        tasks[3].submitted_at = datetime.now(UTC)
        db.add(
            Annotation(
                task_id=tasks[3].id,
                author_id=labeler.id,
                label_class_id=vehicle_class.id,
                type=AnnotationType.bbox,
                geometry={"x": 200, "y": 220, "width": 280, "height": 140},
                source=AnnotationSource.human,
            )
        )

        tasks[4].status = TaskStatus.in_progress
        tasks[4].assignee_id = labeler.id

        tasks[5].status = TaskStatus.rejected
        tasks[5].assignee_id = labeler.id
        tasks[5].rejection_reason = "Vehicle bounding box too small — must cover full vehicle body"

        # Default model version for pre-labeling
        from config import settings

        db.add(
            ModelVersion(
                project_id=project.id,
                name="OpenAI Vision Detector",
                version=1,
                description="Default GPT-4o vision pre-label model",
                model_config={"provider": "openai", "model": settings.openai_model},
                status=ModelVersionStatus.active,
                metrics={"precision": 0.85, "recall": 0.82, "feedback_count": 0},
            )
        )

        # Gold standard annotations for Auto-QA demo (first 2 tasks)
        for i, task in enumerate(tasks[:2]):
            db.add(
                GoldStandardAnnotation(
                    task_id=task.id,
                    label_class_id=vehicle_class.id,
                    type=AnnotationType.bbox,
                    geometry={"x": 120 + i * 30, "y": 180, "width": 200, "height": 120},
                )
            )

        await db.commit()
        return True


def _print_demo_users() -> None:
    print("Demo users:")
    for email, _, password, role in DEMO_USERS:
        print(f"  {role.value}: {email} / {password}")


async def init_database(*, reset: bool = False, force_seed: bool = False) -> None:
    """Create schema and ensure default demo data exists."""
    if reset:
        print("Resetting database schema...")
        await reset_schema()
        inserted = await seed_defaults()
        print("Database reset to default demo state.")
        if inserted:
            _print_demo_users()
        return

    await create_schema()
    if force_seed:
        inserted = await seed_defaults()
        if inserted:
            print("Default demo data inserted.")
            _print_demo_users()
        else:
            print("Database already contains data; skipping seed.")
        return

    inserted = await seed_defaults()
    if inserted:
        print("Database initialized with default demo data.")
        _print_demo_users()
    else:
        print("Database ready (default data already present).")


async def seed() -> None:
    """Backward-compatible entry point used by main.py startup."""
    await init_database(reset=False, force_seed=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize AV platform database")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables and re-create default demo data",
    )
    args = parser.parse_args()
    asyncio.run(init_database(reset=args.reset))
