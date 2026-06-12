import io
import json
import zipfile
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import Annotation, Asset, Dataset, LabelClass, Project, Task, TaskStatus


async def export_project(
    db: AsyncSession, project_id: UUID, fmt: str = "json"
) -> tuple[bytes, str, str]:
    result = await db.execute(
        select(Project)
        .where(Project.id == project_id)
        .options(
            selectinload(Project.label_classes),
            selectinload(Project.datasets)
            .selectinload(Dataset.assets)
            .selectinload(Asset.task)
            .selectinload(Task.annotations)
            .selectinload(Annotation.label_class),
        )
    )
    project = result.scalar_one_or_none()
    if not project:
        raise ValueError("Project not found")

    approved_tasks: list[Task] = []
    for dataset in project.datasets:
        for asset in dataset.assets:
            if asset.task and asset.task.status == TaskStatus.approved:
                approved_tasks.append(asset.task)

    if fmt == "coco":
        return _export_coco(project, approved_tasks)
    return _export_json(project, approved_tasks)


def _export_json(project: Project, tasks: list[Task]) -> tuple[bytes, str, str]:
    data = {
        "project": {"id": str(project.id), "name": project.name, "city": project.city},
        "label_classes": [
            {"id": str(lc.id), "name": lc.name, "color": lc.color} for lc in project.label_classes
        ],
        "tasks": [],
    }
    for task in tasks:
        asset = task.asset
        data["tasks"].append(
            {
                "task_id": str(task.id),
                "asset": {
                    "id": str(asset.id),
                    "filename": asset.filename,
                    "storage_key": asset.storage_key,
                    "width": asset.width,
                    "height": asset.height,
                },
                "annotations": [
                    {
                        "id": str(a.id),
                        "type": a.type.value,
                        "label_class": a.label_class.name,
                        "geometry": a.geometry,
                        "source": a.source.value,
                    }
                    for a in task.annotations
                ],
            }
        )

    content = json.dumps(data, indent=2).encode()
    filename = f"{project.name.replace(' ', '_')}_export.json"
    return content, filename, "application/json"


def _export_coco(project: Project, tasks: list[Task]) -> tuple[bytes, str, str]:
    categories = [
        {"id": i + 1, "name": lc.name, "supercategory": "object"}
        for i, lc in enumerate(sorted(project.label_classes, key=lambda x: x.sort_order))
    ]
    class_id_map = {lc.id: i + 1 for i, lc in enumerate(sorted(project.label_classes, key=lambda x: x.sort_order))}

    images = []
    annotations = []
    ann_id = 1

    for img_id, task in enumerate(tasks, start=1):
        asset = task.asset
        images.append(
            {
                "id": img_id,
                "file_name": asset.filename,
                "width": asset.width or 800,
                "height": asset.height or 600,
            }
        )
        for ann in task.annotations:
            coco_ann: dict = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id_map.get(ann.label_class_id, 1),
                "iscrowd": 0,
            }
            if ann.type.value == "bbox":
                g = ann.geometry
                coco_ann["bbox"] = [g["x"], g["y"], g["width"], g["height"]]
                coco_ann["area"] = g["width"] * g["height"]
            else:
                pts = ann.geometry.get("points", [])
                flat = [coord for p in pts for coord in (p["x"], p["y"])]
                xs = [p["x"] for p in pts]
                ys = [p["y"] for p in pts]
                coco_ann["segmentation"] = [flat]
                coco_ann["area"] = (max(xs) - min(xs)) * (max(ys) - min(ys)) if pts else 0
            annotations.append(coco_ann)
            ann_id += 1

    coco = {
        "info": {"description": project.name, "version": "1.0"},
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.json", json.dumps(coco, indent=2))
    buf.seek(0)
    filename = f"{project.name.replace(' ', '_')}_coco.zip"
    return buf.read(), filename, "application/zip"
