from datetime import datetime
import uuid
from sqlalchemy import Integer, String, Float, DateTime, Text, Boolean, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base


class User(Base):
    """Portal user account — email + bcrypt password, verified by JWT."""
    __tablename__ = "users"

    id:              Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    name:            Mapped[str]      = mapped_column(String(256), nullable=False)
    email:           Mapped[str]      = mapped_column(String(256), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str]      = mapped_column(String(256), nullable=False)
    is_active:       Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:      Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    location: Mapped[str] = mapped_column(String(256), default="India")
    platforms: Mapped[str] = mapped_column(String(256), default="all")
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    searched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SavedJob(Base):
    __tablename__ = "saved_jobs"
    __table_args__ = (UniqueConstraint("session_id", "job_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512))
    company: Mapped[str] = mapped_column(String(256))
    location: Mapped[str] = mapped_column(String(256), nullable=True)
    min_salary: Mapped[float] = mapped_column(Float, nullable=True)
    max_salary: Mapped[float] = mapped_column(Float, nullable=True)
    salary_currency: Mapped[str] = mapped_column(String(8), default="INR")
    job_url: Mapped[str] = mapped_column(Text, nullable=True)
    platform: Mapped[str] = mapped_column(String(64), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    date_posted: Mapped[str] = mapped_column(String(32), nullable=True)
    saved_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrendingCache(Base):
    __tablename__ = "trending_cache"

    cache_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    data: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# PAYMENTS
# ─────────────────────────────────────────────────────────────────────────────
class PaymentOrder(Base):
    """
    One row per Razorpay order. We NEVER store card data — only the Razorpay
    order/payment IDs and our own metadata. State machine:
       created → paid | failed
    """
    __tablename__ = "payment_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Razorpay identifiers
    razorpay_order_id:   Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    razorpay_payment_id: Mapped[str] = mapped_column(String(64), nullable=True, index=True)

    # What was purchased
    product_type: Mapped[str] = mapped_column(String(32), nullable=False)   # plan_pro | plan_enterprise | resume_analysis | job_boost
    plan_code:    Mapped[str] = mapped_column(String(32), nullable=True)    # pro | enterprise | None

    # Money — stored in paise (integer) to avoid float rounding
    amount:   Mapped[int] = mapped_column(Integer, nullable=False)          # in paise
    currency: Mapped[str] = mapped_column(String(8), default="INR")

    # State machine
    status: Mapped[str] = mapped_column(String(16), default="created", index=True)  # created|paid|failed

    # Idempotency — client-supplied key prevents duplicate orders
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=True, unique=True, index=True)

    # Audit
    notes:      Mapped[str] = mapped_column(Text, nullable=True)            # JSON metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    paid_at:    Mapped[datetime] = mapped_column(DateTime, nullable=True)


class Subscription(Base):
    """
    Active subscription per session. One active row per session_id.
    Free users simply have no row (or status=expired).
    """
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    plan_code: Mapped[str] = mapped_column(String(32), default="free")      # free|pro|enterprise
    status:    Mapped[str] = mapped_column(String(16), default="active")    # active|expired|cancelled

    # Links to the order that activated this subscription
    last_order_id: Mapped[str] = mapped_column(String(64), nullable=True)

    current_period_start: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    current_period_end:   Mapped[datetime] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PostedJob(Base):
    """
    Jobs posted directly by employers / recruiters through the portal.
    Shown in search results alongside scraped jobs (platform = 'portal').
    """
    __tablename__ = "posted_jobs"

    id:       Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id:   Mapped[str] = mapped_column(String(64), unique=True, index=True,
                                          default=lambda: f"portal_{uuid.uuid4().hex[:12]}")

    # Core fields
    title:    Mapped[str] = mapped_column(String(512), nullable=False)
    company:  Mapped[str] = mapped_column(String(256), nullable=False)
    location: Mapped[str] = mapped_column(String(256), nullable=False)

    # Classification
    job_type:   Mapped[str] = mapped_column(String(64),  nullable=True)   # Full-time | Part-time | Contract | Internship | Freelance
    work_mode:  Mapped[str] = mapped_column(String(64),  nullable=True)   # On-site | Remote | Hybrid
    experience: Mapped[str] = mapped_column(String(64),  nullable=True)   # Fresher | 1-3 years | …

    # Compensation
    min_salary:      Mapped[float] = mapped_column(Float,      nullable=True)
    max_salary:      Mapped[float] = mapped_column(Float,      nullable=True)
    salary_currency: Mapped[str]   = mapped_column(String(8),  default="INR")

    # Content
    description: Mapped[str] = mapped_column(Text,         nullable=False)
    skills:      Mapped[str] = mapped_column(Text,         nullable=True)   # comma-separated

    # Contact / apply
    contact_email: Mapped[str] = mapped_column(String(256), nullable=False)
    apply_url:     Mapped[str] = mapped_column(Text,        nullable=True)
    company_url:   Mapped[str] = mapped_column(Text,        nullable=True)

    is_active:  Mapped[bool]     = mapped_column(Boolean,  default=True)
    posted_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WebhookEvent(Base):
    """
    Idempotent webhook log. Razorpay may deliver the same event multiple times;
    we dedupe on event_id so processing is exactly-once.
    """
    __tablename__ = "webhook_events"

    event_id:    Mapped[str] = mapped_column(String(64), primary_key=True)  # Razorpay x-razorpay-event-id
    event_type:  Mapped[str] = mapped_column(String(64), nullable=False)
    processed:   Mapped[bool] = mapped_column(Boolean, default=False)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
