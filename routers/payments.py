"""
Payment routes (Razorpay):

  GET  /api/v1/payments/plans          — public pricing catalogue
  POST /api/v1/payments/create-order   — server-side order creation
  POST /api/v1/payments/verify         — verify checkout signature, grant access
  POST /api/v1/payments/webhook        — Razorpay webhook (signature-verified, idempotent)
  GET  /api/v1/payments/subscription   — current subscription status for a session
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_db
from db.models import PaymentOrder, Subscription, WebhookEvent
from schemas.payment import (
    CreateOrderRequest, CreateOrderResponse,
    VerifyPaymentRequest, VerifyPaymentResponse,
    SubscriptionStatus,
)
from services.payment_config import get_product, PLANS_PUBLIC
from services import payment_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/payments", tags=["payments"])

DEFAULT_SESSION = "anonymous"


def _session(x_session_id: Optional[str] = Header(None)) -> str:
    return x_session_id or DEFAULT_SESSION


# ── Public pricing ────────────────────────────────────────────────────────────
@router.get("/plans")
async def get_plans():
    return {"plans": PLANS_PUBLIC}


# ── Create order ──────────────────────────────────────────────────────────────
@router.post("/create-order", response_model=CreateOrderResponse)
async def create_order(
    body: CreateOrderRequest,
    session_id: str = Depends(_session),
    db: AsyncSession = Depends(get_db),
):
    # 1. Resolve price from SERVER catalogue — never trust client amount
    product = get_product(body.product_key)
    if not product:
        raise HTTPException(status_code=400, detail="Unknown product")

    # 2. Idempotency — return existing order if this key was already used
    if body.idempotency_key:
        existing = await db.execute(
            select(PaymentOrder).where(PaymentOrder.idempotency_key == body.idempotency_key)
        )
        row = existing.scalar_one_or_none()
        if row and row.status == "created":
            return CreateOrderResponse(
                razorpay_order_id=row.razorpay_order_id,
                razorpay_key_id=payment_service.get_public_key(),
                amount=row.amount,
                currency=row.currency,
                product_type=row.product_type,
                label=product["label"],
                description=product["label"],
            )

    # 3. Create the order on Razorpay (sync SDK → thread)
    try:
        receipt = f"rcpt_{session_id[:8]}_{int(datetime.utcnow().timestamp())}"
        notes = {
            "session_id": session_id,
            "product_type": product["product_type"],
            "plan_code": product["plan_code"] or "",
        }
        rzp_order = await payment_service.create_razorpay_order(
            product["amount"], product["currency"], receipt, notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Razorpay order creation failed: {e}")
        raise HTTPException(status_code=502, detail="Could not create payment order")

    # 4. Persist our record
    order = PaymentOrder(
        session_id=session_id,
        razorpay_order_id=rzp_order["id"],
        product_type=product["product_type"],
        plan_code=product["plan_code"],
        amount=product["amount"],
        currency=product["currency"],
        status="created",
        idempotency_key=body.idempotency_key,
        notes=json.dumps(notes),
    )
    db.add(order)
    await db.commit()

    return CreateOrderResponse(
        razorpay_order_id=rzp_order["id"],
        razorpay_key_id=payment_service.get_public_key(),
        amount=product["amount"],
        currency=product["currency"],
        product_type=product["product_type"],
        label=product["label"],
        description=product["label"],
    )


# ── Verify payment (client callback after checkout) ───────────────────────────
@router.post("/verify", response_model=VerifyPaymentResponse)
async def verify_payment(
    body: VerifyPaymentRequest,
    session_id: str = Depends(_session),
    db: AsyncSession = Depends(get_db),
):
    # 1. Verify signature — the security gate
    try:
        valid = payment_service.verify_payment_signature(
            body.razorpay_order_id, body.razorpay_payment_id, body.razorpay_signature
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not valid:
        logger.warning(f"Invalid signature for order {body.razorpay_order_id}")
        raise HTTPException(status_code=400, detail="Payment signature verification failed")

    # 2. Find our order
    result = await db.execute(
        select(PaymentOrder).where(PaymentOrder.razorpay_order_id == body.razorpay_order_id)
    )
    order = result.scalar_one_or_none()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    # 3. Idempotent — if already paid, just return success
    if order.status == "paid":
        return VerifyPaymentResponse(
            success=True, status="paid", plan_code=order.plan_code,
            message="Payment already processed",
        )

    # 4. Mark paid + grant entitlement
    order.status = "paid"
    order.razorpay_payment_id = body.razorpay_payment_id
    order.paid_at = datetime.utcnow()

    await _grant_entitlement(db, order)
    await db.commit()

    return VerifyPaymentResponse(
        success=True, status="paid", plan_code=order.plan_code,
        message="Payment verified successfully",
    )


# ── Webhook (server-to-server, the source of truth) ───────────────────────────
@router.post("/webhook")
async def razorpay_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    raw_body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")
    event_id = request.headers.get("X-Razorpay-Event-Id", "")

    # 1. Verify webhook signature
    if not payment_service.verify_webhook_signature(raw_body, signature):
        logger.warning("Webhook signature verification failed")
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    # 2. Idempotency — skip if we've seen this event
    if event_id:
        seen = await db.execute(select(WebhookEvent).where(WebhookEvent.event_id == event_id))
        if seen.scalar_one_or_none():
            return {"status": "already_processed"}

    payload = json.loads(raw_body)
    event_type = payload.get("event", "")

    # 3. Record the event (dedup guard)
    if event_id:
        db.add(WebhookEvent(event_id=event_id, event_type=event_type, processed=False))
        await db.commit()

    # 4. Handle payment captured
    if event_type in ("payment.captured", "order.paid"):
        entity = payload.get("payload", {}).get("payment", {}).get("entity", {}) \
            or payload.get("payload", {}).get("order", {}).get("entity", {})
        rzp_order_id = entity.get("order_id") or entity.get("id")
        rzp_payment_id = entity.get("id")

        if rzp_order_id:
            result = await db.execute(
                select(PaymentOrder).where(PaymentOrder.razorpay_order_id == rzp_order_id)
            )
            order = result.scalar_one_or_none()
            if order and order.status != "paid":
                order.status = "paid"
                order.razorpay_payment_id = rzp_payment_id
                order.paid_at = datetime.utcnow()
                await _grant_entitlement(db, order)

    # 5. Mark event processed
    if event_id:
        ev = await db.execute(select(WebhookEvent).where(WebhookEvent.event_id == event_id))
        row = ev.scalar_one_or_none()
        if row:
            row.processed = True

    await db.commit()
    return {"status": "ok"}


# ── Subscription status ───────────────────────────────────────────────────────
@router.get("/subscription", response_model=SubscriptionStatus)
async def get_subscription(
    session_id: str = Depends(_session),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Subscription).where(Subscription.session_id == session_id))
    sub = result.scalar_one_or_none()

    if not sub:
        return SubscriptionStatus(plan_code="free", status="active", is_active=True)

    # Check expiry
    is_active = True
    if sub.plan_code != "free" and sub.current_period_end:
        is_active = sub.current_period_end > datetime.utcnow()
        if not is_active and sub.status == "active":
            sub.status = "expired"
            sub.plan_code = "free"
            await db.commit()

    return SubscriptionStatus(
        plan_code=sub.plan_code,
        status=sub.status,
        current_period_end=sub.current_period_end.isoformat() if sub.current_period_end else None,
        is_active=is_active,
    )


# ── Entitlement granting (shared by verify + webhook) ─────────────────────────
async def _grant_entitlement(db: AsyncSession, order: PaymentOrder):
    """Activate the subscription/perk for a paid order. Idempotent."""
    product = get_product(order.product_type)
    if not product:
        return

    # One-time products (resume analysis, job boost) → no subscription change
    if product["plan_code"] is None:
        logger.info(f"One-time purchase fulfilled: {order.product_type} for {order.session_id}")
        return

    # Subscription products → upsert subscription row
    result = await db.execute(select(Subscription).where(Subscription.session_id == order.session_id))
    sub = result.scalar_one_or_none()

    period_days = product["period_days"] or 30
    now = datetime.utcnow()
    period_end = now + timedelta(days=period_days)

    if sub:
        sub.plan_code = product["plan_code"]
        sub.status = "active"
        sub.last_order_id = order.razorpay_order_id
        sub.current_period_start = now
        sub.current_period_end = period_end
        sub.updated_at = now
    else:
        db.add(Subscription(
            session_id=order.session_id,
            plan_code=product["plan_code"],
            status="active",
            last_order_id=order.razorpay_order_id,
            current_period_start=now,
            current_period_end=period_end,
        ))
    logger.info(f"Subscription activated: {product['plan_code']} for {order.session_id}")
