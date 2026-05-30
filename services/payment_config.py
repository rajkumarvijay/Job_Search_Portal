"""
Single source of truth for pricing. Amounts are in PAISE (₹1 = 100 paise)
and defined server-side ONLY — the client never sends an amount, which
prevents price-tampering fraud.
"""

# ── Product catalogue (server-authoritative prices) ──────────────────────────
PRODUCTS = {
    "plan_pro": {
        "product_type": "plan_pro",
        "plan_code":    "pro",
        "amount":       49900,    # ₹499.00
        "currency":     "INR",
        "label":        "Pro Plan — Monthly",
        "period_days":  30,
    },
    "plan_enterprise": {
        "product_type": "plan_enterprise",
        "plan_code":    "enterprise",
        "amount":       199900,   # ₹1999.00
        "currency":     "INR",
        "label":        "Enterprise Plan — Monthly",
        "period_days":  30,
    },
    "resume_analysis": {
        "product_type": "resume_analysis",
        "plan_code":    None,
        "amount":       4900,     # ₹49.00
        "currency":     "INR",
        "label":        "Resume ATS Analysis",
        "period_days":  None,
    },
    "job_boost": {
        "product_type": "job_boost",
        "plan_code":    None,
        "amount":       99900,    # ₹999.00
        "currency":     "INR",
        "label":        "Featured Job Boost",
        "period_days":  None,
    },
}

# Public plan metadata for the pricing page (no secrets here)
PLANS_PUBLIC = [
    {
        "code": "free",
        "name": "Free",
        "price": 0,
        "price_label": "₹0",
        "period": "forever",
        "product_key": None,
        "features": [
            "5 AI job searches per day",
            "Browse all 6 job portals",
            "Save up to 10 jobs",
            "Basic search filters",
        ],
        "cta": "Current Plan",
        "highlight": False,
    },
    {
        "code": "pro",
        "name": "Pro",
        "price": 499,
        "price_label": "₹499",
        "period": "per month",
        "product_key": "plan_pro",
        "features": [
            "Unlimited AI job searches",
            "Unlimited resume ATS scoring",
            "Worldwide job results",
            "Priority email support",
            "Save unlimited jobs",
        ],
        "cta": "Upgrade to Pro",
        "highlight": True,
    },
    {
        "code": "enterprise",
        "name": "Enterprise",
        "price": 1999,
        "price_label": "₹1999",
        "period": "per month",
        "product_key": "plan_enterprise",
        "features": [
            "Everything in Pro",
            "Priority job results ranking",
            "API access for integrations",
            "Dedicated account manager",
            "Custom job alerts",
        ],
        "cta": "Go Enterprise",
        "highlight": False,
    },
]


def get_product(product_key: str) -> dict | None:
    return PRODUCTS.get(product_key)
