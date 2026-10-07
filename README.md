# Meadowline Skin Co.: Shopify Profitability Diagnostic

> **Demo project based on synthetic Shopify data.** Meadowline Skin Co. is a fictional brand. No real client, no real customers, no real results. All findings are about this dataset only.

A 12-month diagnostic of a fictional DTC skincare store (Jul 2025 – Jun 2026, USD). The question: **where does money stay in the business, and where does it leak?**

The metric is **contribution** (net revenue + shipping charged − product cost − postage − packaging − payment fees − return shipping). It is not net profit: the data has no operating costs.

**Read first:** [`docs/Meadowline_Profitability_Diagnostic.pdf`](docs/Meadowline_Profitability_Diagnostic.pdf) (7-page report).

## Main results (this dataset only)

| Result | Value | Register ID |
|---|---|---|
| Net revenue | $283,485 | P2-04 |
| Contribution / net revenue | $160,981 / 56.8% | P3-002, P3-003 |
| Contribution per order, discounted vs no code, same order type | $25.93 vs $38.38 (−$12.45) | P3-035, P3-036 |
| 90-day repeat, WELCOME15 vs no code, same source | 23.1% vs 29.9% | P4-072 |
| Unit return rate, mature orders: store / product P06 | 5.44% / 13.2% | P3-063, P3-065 |
| 90-day contribution per new buyer / CAC: Google / TikTok | 1.84× / 0.69× | P4-037, P4-040 |

No single large profit leak was found. The discount, welcome-code, return and channel patterns are **observational**: they do not prove cause. See the report for limits.

## How the analysis is organised

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1. Data quality | `src/01_data_quality.py` | Checks rows, keys, duplicates, orphans, dates, labels. Read-only. | log in `docs/logs/` |
| 2. Cleaning | `src/02_clean.py` | Applies each cleaning decision and counts it. Builds clean tables. | `outputs/*_clean.csv`, `cleaning_log.csv` |
| 3. Revenue baseline | `src/03_revenue_baseline.py` | Gross → discounts → refunds → net revenue. KPIs and months. | `outputs/phase2/` |
| 4. Product, discount, return economics | `src/04_product_discount_return_economics.py` | Contribution by product, discount groups, returns. Three cross-analyses with matched groups. | `outputs/phase3/` |
| 5. Follow-up checks | `src/05_followup_checks.py` | Return rate by discount status (product × new/repeat). Shipping recovery. | `outputs/phase3/` |
| 6. Customers and marketing | `src/06_customers_marketing.py` | New vs repeat, concentration, 90-day repeat, CAC, platform vs Shopify. | `outputs/phase4/` |
| 7. QA and sizing | `src/07_qa_and_sizing.py` | Independent recompute from raw files. 24 checks. Sizing of recommendations. | `outputs/phase5/` |

Folder names `phase2` … `phase5` in `outputs/` follow the original project phases. Phase 1 outputs are in the top level of `outputs/`.

Other documents:
- [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md): assumptions log and limits.
- [`docs/METHOD_NOTES.md`](docs/METHOD_NOTES.md): metric definitions and how matching, censoring and CAC work.
- `outputs/phase5/master_register_P1_P4.csv`: **Number Register**. Every number in the report has an ID, a numerator/denominator, a period and a source script.
- `outputs/phase5/qa_results.csv`: result of all 24 QA checks (all PASS).
- `docs/logs/`: console output of each run.

## How to run

```bash
pip install -r requirements.txt
./run_all.sh          # about 15 seconds
```

Scripts use relative paths: input is `data/raw/`, output is `outputs/`. Each script also accepts folder paths as arguments.

Tested with Python 3.13, pandas 3.0.5, numpy 2.5.3. Random seeds are fixed, so the bootstrap intervals are repeatable. Pandas 3 changed `groupby.apply`; older pandas may need small fixes.

## Data

`data/raw/` holds the 8 raw CSV files exactly as received (synthetic, not cleaned on purpose). See [`data/raw/README.md`](data/raw/README.md) for the data description. The files contain deliberate errors (voided and test orders, duplicate lines, orphan rows, messy labels). Every error and its decision is in `outputs/cleaning_log.csv`.

## What to keep in mind

- **Synthetic data.** Patterns come from a generator. Return rates by product, channel repeat rates and the November peak may be artifacts.
- **Observational data.** Differences between discounted and non-discounted orders do not show cause.
- **Right-censoring.** Recent orders had less time to be returned or repeated. Return rates use orders up to 2026-05-16. 90-day repeat uses first orders up to 2026-03-31.
- **Assumption A2.** The free gift (P13) has no cost in the data. I used $1.00 per unit as a placeholder. The effect is ±0.34% of contribution for $0–3.
- **No operating costs**, so no net profit.

## Author

Vasyl B. — eCommerce profitability analytics.
