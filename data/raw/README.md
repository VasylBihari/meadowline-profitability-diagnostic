# Meadowline Skin Co. — Shopify export (SYNTHETIC / DEMO DATA)

**This dataset is fully synthetic.** "Meadowline Skin Co." is a fictional brand. No real customers, orders, or businesses are represented. Any analysis built on it must be labelled "Demo project based on synthetic Shopify data."

Period: 2025-07-01 to 2026-06-30 (UTC). Simulated extract date: 2026-07-02. Currency: USD. Files are raw exports and were intentionally **not cleaned** — treat them as you would a founder's real export.

## Files and grain

| File | Grain (one row = …) | Key | Notes |
|---|---|---|---|
| customers.csv | customer record | customer_id | `email_hash` identifies a person; the same person can have more than one customer_id (guest checkout). Includes newsletter-only subscribers. |
| orders.csv | order | order_id | `subtotal` = list-price items before discounts; excludes tax and shipping. `discount_total` = item discounts only. `financial_status` includes `voided`. `payment_fee` is the processor fee. |
| order_items.csv | order line | order_item_id | `unit_price` = list price at the time of sale (before discount). `line_discount` = order discount allocated to the line. A $0 line is a free gift. |
| products.csv | product | product_id | `cogs` = landed unit cost from the founder's spreadsheet. Bundles list `bundle_components`. `current_price` is today's price, not necessarily the price at the time of each order. |
| returns.csv | returned order line | return_id | `refund_type`: full / partial / replacement. `return_method`: shipped_back / returnless_refund. `restocked` = unit went back to sellable stock. |
| shipping_costs.csv | shipment (label) | shipment_id | Brand's cost: `postage_cost` + `packaging_cost`. An order can have more than one shipment (replacements). |
| discounts.csv | discount application | discount_application_id | One order can have several rows. `free_shipping` rows are recorded here only. |
| marketing_spend.csv | month × channel | month, channel | Spend and platform-reported (self-attributed) conversions. Platform-reported figures are not reconciled to Shopify. |

## Relationships

- orders 1—N order_items, discounts, shipping_costs
- order_items 1—N returns (via order_item_id)
- customers 1—N orders
- products 1—N order_items
- marketing_spend relates to customers only through acquisition channel and month; there is no order-level attribution.

## Known caveats of the export itself

- Data is a raw export; validate joins, duplicates and IDs before computing metrics.
- Customers' `acquisition_source` was captured at checkout and is not standardised.
- Returns are only included up to the extract date, so recent orders have had less time to generate returns.
- No operating expenses, platform subscriptions or salaries are included, so the dataset supports **contribution** analysis, not net profit.
