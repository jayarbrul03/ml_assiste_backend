"""Geometry utilities for Auto-QA (IoU, matching)."""

from __future__ import annotations


def bbox_to_xyxy(geom: dict) -> tuple[float, float, float, float]:
    x = float(geom.get("x", 0))
    y = float(geom.get("y", 0))
    w = float(geom.get("width", 0))
    h = float(geom.get("height", 0))
    return x, y, x + w, y + h


def bbox_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = bbox_to_xyxy(a)
    bx1, by1, bx2, by2 = bbox_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return round(inter / union, 4)


def polygon_area(points: list[dict]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += float(points[i]["x"]) * float(points[j]["y"])
        area -= float(points[j]["x"]) * float(points[i]["y"])
    return abs(area) / 2.0


def polygon_bbox(geom: dict) -> dict:
    points = geom.get("points") or []
    if not points:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


def geometry_iou(a: dict, b: dict, ann_type: str = "bbox") -> float:
    ga = a if ann_type == "bbox" else polygon_bbox(a)
    gb = b if ann_type == "bbox" else polygon_bbox(b)
    return bbox_iou(ga, gb)


def match_and_score(
    submitted: list[dict],
    reference: list[dict],
    *,
    iou_threshold: float = 0.5,
) -> tuple[float, list[str]]:
    """Greedy IoU matching; returns mean IoU and issue messages."""
    if not submitted and not reference:
        return 1.0, []
    if not submitted:
        return 0.0, [f"Missing {len(reference)} expected annotation(s)"]
    if not reference:
        return 0.0, [f"Found {len(submitted)} extra annotation(s) with no reference"]

    used: set[int] = set()
    ious: list[float] = []
    issues: list[str] = []

    for i, ref in enumerate(reference):
        best_iou = 0.0
        best_j = -1
        ref_type = ref.get("type", "bbox")
        for j, sub in enumerate(submitted):
            if j in used:
                continue
            if ref.get("label_class_id") and sub.get("label_class_id"):
                if str(ref["label_class_id"]) != str(sub["label_class_id"]):
                    continue
            sub_type = sub.get("type", "bbox")
            iou = geometry_iou(ref["geometry"], sub["geometry"], ref_type if ref_type == sub_type else "bbox")
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            used.add(best_j)
            ious.append(best_iou)
        else:
            issues.append(f"Missing or low-IoU match for reference annotation #{i + 1}")

    for j, sub in enumerate(submitted):
        if j not in used:
            issues.append(f"Extra annotation #{j + 1} not matched to reference")

    mean_iou = sum(ious) / len(reference) if reference else 0.0
    return round(mean_iou, 4), issues
