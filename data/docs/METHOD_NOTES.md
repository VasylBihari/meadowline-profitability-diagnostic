# Method notes

Demo project based on synthetic Shopify data.

## Metric definitions

- **Valid order:** not voided and not a test order (payment gateway `bogus` or code `TEST100`).
- **Gross sales:** quantity × list price, valid orders.
- **Net revenue:** gross sales − item discounts − refunds.
- **Contribution:** net revenue + shipping charged − product cost (restocked returns are recovered, replacements add cost again) − postage − packaging − payment fees − return shipping.
- **Contribution margin:** contribution / net revenue.
- **CAC:** channel spend / new buyers from that source in the same period.
- **Person:** one `email_hash`. One person can have several `customer_id` values.

## Comparing discounted and non-discounted orders

Discounted and non-discounted orders are different kinds of orders. So the comparison is made inside groups of the same order type:

- **Contribution per order (CX1):** groups = new or repeat buyer × number of units × set or no set. Only groups with at least 30 orders on both sides are used (9 of 12 groups, 98.3% of discounted orders). Weights = mix of the discounted orders. The 95% interval comes from 2,000 bootstrap resamples.
- **Return rate (CX3b):** groups = product × new or repeat order. Same 30-unit rule.
- **Welcome code and repeat (CX4):** groups = acquisition source. Unknown or unclear source labels are excluded (294 of 2,740 buyers). Result also checked without November–December.
- No result changed sign after matching (no Simpson reversal). Matching does not remove selection: the data cannot show cause.

## Right-censoring

Recent orders had less time to be returned or repeated. A fall in recent return or repeat rates is not read as improvement.

- Return rates: orders up to 2026-05-16 (last return date minus the 45-day longest delay).
- 90-day repeat: first orders up to 2026-03-31.

## Quality control

`src/07_qa_and_sizing.py` recomputes revenue, discounts, refunds and contribution from the raw files by a separate path. Result: $160,984.28 vs $160,981.36 in the main path. The $2.92 gap comes from the postage estimate (A3). 24 of 24 checks pass (`outputs/phase5/qa_results.csv`).

## Number Register

Every figure in the report has an ID with value, numerator/denominator, filter, period and source script: `outputs/phase5/master_register_P1_P4.csv` (P1–P4 and P3b) and `outputs/phase5/number_register_phase5.csv` (P5 sizing).
