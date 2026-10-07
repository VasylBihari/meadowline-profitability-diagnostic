# Assumptions log and limits

Demo project based on synthetic Shopify data. Fictional store.

## Assumptions

| # | Assumption | Why | Effect if wrong |
|---|---|---|---|
| A1 | Product P12 (set) has no cost in the file. Cost = P01 + P03 + P07 = $21.10. | Set is made of these three products. No extra packaging cost is added. | Below 1% of contribution. |
| A2 | Free gift P13 is not in the product file. Cost = **$1.00 per unit** (277 units). This is a placeholder, not data from the founder. | Cost is unknown. | $0–3 per unit changes contribution by about ±$554 (±0.34%). |
| A3 | 40 shipments have no postage cost. Estimated from parcel weight inside each service × zone group. Total $358.75 (alternative method: $355.84). | Missing values. | $2.92 on total contribution. |
| A4 | Source labels "IG" and "Google" (182 customer records) are mapped to Meta and Google Ads. | Labels are unclear. | Channel order did not change in the strict and generous cases. |
| A5 | Payment fee is not refunded when an order is returned. | Usual processor rule; not in the data. | Contribution is slightly lower than it could be. |
| A6 | Order-level costs (postage, packaging, fees) are split to lines by gross line value. | Needed for product economics. | A units-based split gave a similar picture (3 of 12 product ranks change between revenue and contribution). |

## Cleaning decisions

Every cleaning decision, with a row count, is in `outputs/cleaning_log.csv`. Examples: 61 voided orders and 6 test orders excluded (5,006 valid orders left); 11 duplicate order lines, 4 duplicate shipments, 2 discount rows, 3 shipments and 4 returns for orders that do not exist removed.

## Limits

- Synthetic data. Patterns come from a generator. Return rates by product, channel repeat rates and the November peak may be artifacts.
- No operating costs. Contribution is not profit.
- Observational data. No finding proves cause.
- Recent orders had less time to be returned or repeated. Return rates use orders up to 2026-05-16. 90-day repeat uses first orders up to 2026-03-31.
- Marketing: spend is by channel and month. Customer source labels are messy. Platform numbers are self-reported and not checked against the platforms.
- Segments with fewer than 30 observations are not used for conclusions.
