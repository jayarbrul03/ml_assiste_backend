import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    ml_engineer = "ml_engineer"
    labeler = "labeler"
    reviewer = "reviewer"


class TaskStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    submitted = "submitted"
    in_review = "in_review"
    approved = "approved"
    rejected = "rejected"


class AnnotationType(str, enum.Enum):
    bbox = "bbox"
    polygon = "polygon"


class AnnotationSource(str, enum.Enum):
    human = "human"
    prelabel = "prelabel"
    corrected = "corrected"


class PrelabelJobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class ModelVersionStatus(str, enum.Enum):
    draft = "draft"
    active = "active"
    training = "training"
    deprecated = "deprecated"


class TrainingFeedbackType(str, enum.Enum):
    accept = "accept"
    reject = "reject"
    correction = "correction"
    prelabel_correction = "prelabel_correction"


class AutolabelReviewStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, native_enum=False), default=UserRole.labeler)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    assigned_tasks: Mapped[list["Task"]] = relationship(
        back_populates="assignee", foreign_keys="Task.assignee_id"
    )
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="author")
    reviews: Mapped[list["Review"]] = relationship(back_populates="reviewer")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    city: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    label_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    label_classes: Mapped[list["LabelClass"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class LabelClass(Base):
    __tablename__ = "label_classes"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100))
    color: Mapped[str] = mapped_column(String(20), default="#3b82f6")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    project: Mapped["Project"] = relationship(back_populates="label_classes")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped["Project"] = relationship(back_populates="datasets")
    assets: Mapped[list["Asset"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    prelabel_jobs: Mapped[list["PrelabelJob"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(500))
    storage_key: Mapped[str] = mapped_column(String(500))
    mime_type: Mapped[str] = mapped_column(String(100), default="image/jpeg")
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    dataset: Mapped["Dataset"] = relationship(back_populates="assets")
    task: Mapped["Task | None"] = relationship(back_populates="asset", uselist=False)
    prelabel_predictions: Mapped[list["PrelabelPrediction"]] = relationship(
        back_populates="asset", cascade="all, delete-orphan"
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), unique=True)
    status: Mapped[TaskStatus] = mapped_column(Enum(TaskStatus, native_enum=False), default=TaskStatus.pending)
    assignee_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    uncertainty_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    autolabeled: Mapped[bool] = mapped_column(default=False)
    qa_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    qa_passed: Mapped[bool | None] = mapped_column(nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sla_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    autolabel_review_status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    asset: Mapped["Asset"] = relationship(back_populates="task")
    assignee: Mapped["User | None"] = relationship(back_populates="assigned_tasks", foreign_keys=[assignee_id])
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    reviews: Mapped[list["Review"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    qa_checks: Mapped[list["QaCheck"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    gold_standards: Mapped[list["GoldStandardAnnotation"]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    label_class_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("label_classes.id"))
    type: Mapped[AnnotationType] = mapped_column(Enum(AnnotationType, native_enum=False))
    geometry: Mapped[dict] = mapped_column(JSON)
    source: Mapped[AnnotationSource] = mapped_column(
        Enum(AnnotationSource, native_enum=False), default=AnnotationSource.human
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    attributes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    task: Mapped["Task"] = relationship(back_populates="annotations")
    author: Mapped["User"] = relationship(back_populates="annotations")
    label_class: Mapped["LabelClass"] = relationship()


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    reviewer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(50))
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    task: Mapped["Task"] = relationship(back_populates="reviews")
    reviewer: Mapped["User"] = relationship(back_populates="reviews")


class PrelabelJob(Base):
    __tablename__ = "prelabel_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"))
    status: Mapped[PrelabelJobStatus] = mapped_column(
        Enum(PrelabelJobStatus, native_enum=False), default=PrelabelJobStatus.pending
    )
    total_assets: Mapped[int] = mapped_column(Integer, default=0)
    processed_assets: Mapped[int] = mapped_column(Integer, default=0)
    autolabeled_assets: Mapped[int] = mapped_column(Integer, default=0)
    auto_submitted_assets: Mapped[int] = mapped_column(Integer, default=0)
    confidence_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    autolabel_enabled: Mapped[bool] = mapped_column(default=True)
    auto_submit_enabled: Mapped[bool] = mapped_column(default=True)
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="prelabel_jobs")


class PrelabelPrediction(Base):
    __tablename__ = "prelabel_predictions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    label_class_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("label_classes.id"))
    type: Mapped[AnnotationType] = mapped_column(Enum(AnnotationType, native_enum=False))
    geometry: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float, default=0.85)
    accepted: Mapped[bool | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    asset: Mapped["Asset"] = relationship(back_populates="prelabel_predictions")
    label_class: Mapped["LabelClass"] = relationship()


class QaCheck(Base):
    __tablename__ = "qa_checks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    iou_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    consensus_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    gold_standard_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool] = mapped_column(default=False)
    issues: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    task: Mapped["Task"] = relationship(back_populates="qa_checks")


class GoldStandardAnnotation(Base):
    __tablename__ = "gold_standard_annotations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    label_class_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("label_classes.id"))
    type: Mapped[AnnotationType] = mapped_column(Enum(AnnotationType, native_enum=False))
    geometry: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    task: Mapped["Task"] = relationship(back_populates="gold_standards")
    label_class: Mapped["LabelClass"] = relationship()


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer, default=1)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[ModelVersionStatus] = mapped_column(
        Enum(ModelVersionStatus, native_enum=False), default=ModelVersionStatus.draft
    )
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    project: Mapped["Project"] = relationship()
    parent: Mapped["ModelVersion | None"] = relationship(remote_side="ModelVersion.id")


class TrainingFeedback(Base):
    __tablename__ = "training_feedback"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("prelabel_predictions.id"), nullable=True
    )
    annotation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("annotations.id"), nullable=True
    )
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    labeler_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    feedback_type: Mapped[TrainingFeedbackType] = mapped_column(
        Enum(TrainingFeedbackType, native_enum=False)
    )
    geometry_before: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    geometry_after: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    used_in_training: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
