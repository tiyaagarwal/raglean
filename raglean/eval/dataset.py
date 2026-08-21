"""Small synthetic RAG QA dataset: a fictional SaaS company's support knowledge base.

Deliberately mixes three query shapes so the benchmark exercises every optimization
layer honestly:
  - `simple`  factual lookups (should route to the cheap tier, cache well)
  - `paraphrase` groups: same underlying question asked differently (tests whether the
    cache generalizes beyond exact string repeats — an honest stress test, since the
    default offline hash embedder only reliably catches near-exact repeats; swapping in
    a real sentence embedder is what closes this gap, see docs/ARCHITECTURE.md)
  - `complex` multi-part/reasoning queries (should route to mid/premium tier)

Not a public benchmark — built for this project so licensing/attribution isn't a
concern, and so the "gold answer" is unambiguous for the F1 metric. Swapping in a public
benchmark (HotpotQA / NaturalQuestions subset) is a natural upgrade once real embeddings
and a real judge LLM are wired in.
"""
from __future__ import annotations

REFUND_CTX = (
    "Refunds are available within 30 days of purchase for any unused product in its "
    "original packaging. Digital subscriptions can be refunded within 14 days if less "
    "than 10% of the included usage quota has been consumed. Refunds are issued to the "
    "original payment method and typically take 5-7 business days to appear. Shipping "
    "fees are non-refundable unless the item arrived damaged or defective."
)

SHIPPING_CTX = (
    "Standard shipping takes 5-7 business days within the country and is free on orders "
    "over $50. Express shipping takes 1-2 business days and costs a flat $15. "
    "International orders take 10-21 business days depending on customs processing and "
    "are subject to import duties charged by the destination country. Orders can be "
    "tracked using the link emailed at time of dispatch."
)

BILLING_CTX = (
    "Subscription plans renew automatically on the monthly or annual anniversary of "
    "signup. Customers can cancel anytime from Account Settings and will retain access "
    "until the end of the current billing period. Downgrading a plan takes effect at "
    "the next renewal, while upgrading takes effect immediately with a prorated charge "
    "for the remainder of the current period. Failed payments are retried three times "
    "over seven days before the account is suspended."
)

WARRANTY_CTX = (
    "Hardware products carry a 1-year limited warranty covering manufacturing defects, "
    "extendable to 3 years by registering the product within 30 days of purchase. The "
    "warranty does not cover accidental damage, water damage, or unauthorized repairs. "
    "Warranty claims require a proof of purchase and are processed within 10 business "
    "days of receiving the returned item."
)

SUPPORT_CTX = (
    "Support tickets submitted by Enterprise customers receive a first response within "
    "2 hours, 24/7. Pro-tier customers receive a first response within 8 business hours. "
    "Free-tier customers are supported through the community forum only, with no "
    "guaranteed response time. Priority escalation is available for outages affecting "
    "production systems, regardless of plan tier."
)


# Each item: id, query, context_chunks, gold_answer, complexity tag, paraphrase group id
# (items sharing a group id are near-duplicate questions, used to stress-test the cache).
EVAL_DATASET: list[dict] = [
    {"id": "refund-1", "group": "refund-window", "tag": "simple", "context_chunks": [REFUND_CTX],
     "query": "How many days do I have to request a refund?",
     "gold_answer": "30 days for physical products, 14 days for digital subscriptions."},
    {"id": "refund-2", "group": "refund-window", "tag": "simple", "context_chunks": [REFUND_CTX],
     "query": "What is the refund window for a purchase?",
     "gold_answer": "30 days for physical products, 14 days for digital subscriptions."},
    {"id": "refund-3", "group": "refund-window", "tag": "simple", "context_chunks": [REFUND_CTX],
     "query": "Within how many days can I get my money back?",
     "gold_answer": "30 days for physical products, 14 days for digital subscriptions."},
    {"id": "refund-4", "group": "refund-shipping", "tag": "simple", "context_chunks": [REFUND_CTX],
     "query": "Are shipping fees refundable?",
     "gold_answer": "No, unless the item arrived damaged or defective."},
    {"id": "refund-5", "group": "refund-reasoning", "tag": "complex", "context_chunks": [REFUND_CTX, SHIPPING_CTX],
     "query": "Why might a customer get a full refund including shipping, and how does that compare to the standard refund policy?",
     "gold_answer": "Standard refunds exclude shipping fees, but shipping is refunded too if the item arrived damaged or defective."},

    {"id": "ship-1", "group": "ship-time", "tag": "simple", "context_chunks": [SHIPPING_CTX],
     "query": "How long does standard shipping take?",
     "gold_answer": "5-7 business days."},
    {"id": "ship-2", "group": "ship-time", "tag": "simple", "context_chunks": [SHIPPING_CTX],
     "query": "What is the delivery time for standard shipping?",
     "gold_answer": "5-7 business days."},
    {"id": "ship-3", "group": "ship-cost", "tag": "simple", "context_chunks": [SHIPPING_CTX],
     "query": "How much does express shipping cost?",
     "gold_answer": "A flat $15."},
    {"id": "ship-4", "group": "ship-reasoning", "tag": "complex", "context_chunks": [SHIPPING_CTX],
     "query": "Why would an international order take longer than express shipping, and what causes the extra delay?",
     "gold_answer": "International orders depend on customs processing and import duties, taking 10-21 business days versus 1-2 for express."},

    {"id": "bill-1", "group": "bill-cancel", "tag": "simple", "context_chunks": [BILLING_CTX],
     "query": "What happens if I cancel my subscription?",
     "gold_answer": "You retain access until the end of the current billing period."},
    {"id": "bill-2", "group": "bill-cancel", "tag": "simple", "context_chunks": [BILLING_CTX],
     "query": "If I cancel, do I lose access immediately?",
     "gold_answer": "No, you retain access until the end of the current billing period."},
    {"id": "bill-3", "group": "bill-upgrade", "tag": "complex", "context_chunks": [BILLING_CTX],
     "query": "What is the difference between upgrading and downgrading a plan in terms of when it takes effect and billing?",
     "gold_answer": "Upgrading takes effect immediately with a prorated charge; downgrading takes effect at the next renewal."},
    {"id": "bill-4", "group": "bill-failed", "tag": "simple", "context_chunks": [BILLING_CTX],
     "query": "How many times is a failed payment retried?",
     "gold_answer": "Three times over seven days."},

    {"id": "warr-1", "group": "warr-length", "tag": "simple", "context_chunks": [WARRANTY_CTX],
     "query": "How long is the standard hardware warranty?",
     "gold_answer": "1 year, extendable to 3 years by registering within 30 days."},
    {"id": "warr-2", "group": "warr-length", "tag": "simple", "context_chunks": [WARRANTY_CTX],
     "query": "What is the default warranty period for hardware?",
     "gold_answer": "1 year, extendable to 3 years by registering within 30 days."},
    {"id": "warr-3", "group": "warr-exclusions", "tag": "complex", "context_chunks": [WARRANTY_CTX],
     "query": "What types of damage are excluded from warranty coverage and why might that matter for a claim?",
     "gold_answer": "Accidental damage, water damage, and unauthorized repairs are excluded."},

    {"id": "sup-1", "group": "sup-enterprise", "tag": "simple", "context_chunks": [SUPPORT_CTX],
     "query": "What is the response time for Enterprise support tickets?",
     "gold_answer": "First response within 2 hours, 24/7."},
    {"id": "sup-2", "group": "sup-free", "tag": "simple", "context_chunks": [SUPPORT_CTX],
     "query": "Do free-tier customers get a guaranteed support response time?",
     "gold_answer": "No, free-tier customers only get community forum support with no guaranteed response time."},
    {"id": "sup-3", "group": "sup-reasoning", "tag": "complex", "context_chunks": [SUPPORT_CTX, BILLING_CTX],
     "query": "Compare the support response times across plan tiers and explain why Enterprise customers might still get priority even on a lower plan during an outage.",
     "gold_answer": "Enterprise gets 2-hour 24/7 response, Pro gets 8 business hours, Free gets forum only, but priority escalation for production outages is available regardless of plan tier."},

    # Long-context items: the full knowledge base is dumped in as context (as a naive
    # RAG retriever with a generous top-k might do), even though only one section is
    # actually relevant to the question. These specifically exercise the compressor —
    # short single-paragraph items above stay under the token budget and are left
    # untouched, which is correct behavior, not a bug.
    {"id": "long-1", "group": "warr-length", "tag": "long_context",
     "context_chunks": [REFUND_CTX, SHIPPING_CTX, BILLING_CTX, WARRANTY_CTX, SUPPORT_CTX],
     "query": "How long is the standard hardware warranty?",
     "gold_answer": "1 year, extendable to 3 years by registering within 30 days."},
    {"id": "long-2", "group": "ship-cost", "tag": "long_context",
     "context_chunks": [REFUND_CTX, SHIPPING_CTX, BILLING_CTX, WARRANTY_CTX, SUPPORT_CTX],
     "query": "How much does express shipping cost?",
     "gold_answer": "A flat $15."},
]


def load_eval_dataset() -> list[dict]:
    return list(EVAL_DATASET)
