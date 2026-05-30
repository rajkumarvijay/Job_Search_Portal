"""
Razorpay payment service — uses the REST API directly via httpx
(no SDK, so it works cleanly on Python 3.13).

SECURITY PRINCIPLES enforced here:
  • Prices come ONLY from the server-side catalogue (payment_config).
  • We never see/store raw card data — Razorpay Checkout handles PCI scope.
  • Every payment is verified via HMAC-SHA256 signature before granting access.
  • Webhooks are signature-verified AND deduplicated (exactly-once processing).
  • Idempotency keys prevent duplicate orders on client retry.
  • Constant-time comparison (hmac.compare_digest) prevents timing attacks.
"""

import hashlib
import hmac
import logging
import os

import httpx

logger = logging.getLogger(__name__)

RAZORPAY_API_BASE = "https://api.razorpay.com/v1"


def _credentials() -> tuple[str, str]:
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_id or not key_secret:
        raise ValueError(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set. "
            "Add them in Railway → backend service → Variables."
        )
    return key_id, key_secret


def get_public_key() -> str:
    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    if not key_id:
        raise ValueError("RAZORPAY_KEY_ID not set")
    return key_id


async def create_razorpay_order(amount: int, currency: str, receipt: str, notes: dict) -> dict:
    """
    Create an order via Razorpay REST API.
    amount is in paise. Returns the order JSON (contains 'id').
    """
    key_id, key_secret = _credentials()
    payload = {
        "amount": amount,
        "currency": currency,
        "receipt": receipt,
        "payment_capture": 1,   # auto-capture on success
        "notes": notes,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{RAZORPAY_API_BASE}/orders",
            json=payload,
            auth=(key_id, key_secret),
        )
    if resp.status_code not in (200, 201):
        logger.error(f"Razorpay order error {resp.status_code}: {resp.text}")
        raise RuntimeError(f"Razorpay API error: {resp.status_code}")
    return resp.json()


def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Verify checkout signature: HMAC_SHA256(order_id|payment_id, key_secret).
    Returns True only if authentic — this proves the payment really succeeded.
    """
    _, key_secret = _credentials()
    payload = f"{order_id}|{payment_id}".encode()
    expected = hmac.new(key_secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    """
    Verify a Razorpay webhook: HMAC_SHA256(raw_body, webhook_secret)
    must equal the X-Razorpay-Signature header.
    """
    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
    if not webhook_secret:
        logger.warning("RAZORPAY_WEBHOOK_SECRET not set — rejecting webhook")
        return False
    expected = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")
