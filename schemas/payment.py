from pydantic import BaseModel, Field
from typing import Optional


class CreateOrderRequest(BaseModel):
    # Client sends ONLY the product key — never the amount (prevents tampering)
    product_key: str = Field(..., description="plan_pro | plan_enterprise | resume_analysis | job_boost")
    idempotency_key: Optional[str] = Field(None, description="Client-generated UUID to dedupe retries")


class CreateOrderResponse(BaseModel):
    razorpay_order_id: str
    razorpay_key_id: str          # public key — safe to expose
    amount: int                    # paise
    currency: str
    product_type: str
    label: str
    name: str = "JobQuest India"
    description: str


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class VerifyPaymentResponse(BaseModel):
    success: bool
    status: str
    plan_code: Optional[str] = None
    message: str


class SubscriptionStatus(BaseModel):
    plan_code: str
    status: str
    current_period_end: Optional[str] = None
    is_active: bool
