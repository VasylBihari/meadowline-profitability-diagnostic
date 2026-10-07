"""Phase 1 - Cleaning + clean analytical tables. Meadowline Skin Co. (SYNTHETIC demo data).
Run: python3 02_clean.py [data_dir] [out_dir]
Every cleaning decision is explicit, counted, and verified after the fact.
"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "outputs")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)

# ---- ASSUMPTION PARAMETERS (change here, rerun) ----
GIFT_COGS_PER_UNIT = 1.0   # P13 gift sachet: cost unknown. $1.00 = PLACEHOLDER base case (sensitivity $0-3). Not founder data.
P12_COGS_RULE = "sum_of_components"  # P12 cogs missing -> sum of P01+P03+P07 cogs (no extra packaging)

load = lambda n: pd.read_csv(f"{D}/{n}.csv")
cu, o, oi, p = load("customers"), load("orders"), load("order_items"), load("products")
r, sh, dc, ms = load("returns"), load("shipping_costs"), load("discounts"), load("marketing_spend")

REG = []   # Number Register: (id, metric, value, numerator/denominator, period, filter, script)
def reg(i, metric, value, nd="", filt=""):
    REG.append((i, metric, value, nd, "2025-07-01..2026-06-30", filt, "02_clean.py"))

LOG = []   # cleaning decisions
def log(issue, n, decision):
    LOG.append((issue, n, decision)); print(f"[CLEAN] {issue}: {n} -> {decision}")

# ===== raw counts =====
raw = {"customers": len(cu), "orders": len(o), "order_items": len(oi), "products": len(p),
       "returns": len(r), "shipping_costs": len(sh), "discounts": len(dc), "marketing_spend": len(ms)}
for i, (k, v) in enumerate(raw.items(), 1):
    reg(f"DQ-R{i:02d}", f"raw rows: {k}", v, "", "none")

# ===== products =====
p["product_id"] = p.product_id.str.strip().str.upper()
comp = p.set_index("product_id").cogs
p12 = sum(comp[c] for c in p.loc[p.product_id == "P12", "bundle_components"].iloc[0].split("|"))
p["cogs_source"] = np.where(p.cogs.notna(), "founder_spreadsheet", "")
p.loc[p.product_id == "P12", "cogs"] = p12
p.loc[p.product_id == "P12", "cogs_source"] = "ASSUMPTION: sum of component cogs"
p = pd.concat([p, pd.DataFrame([{"product_id": "P13", "sku": "MS-GIFT-SACHET", "product_name": "Gift sachet (free gift)",
        "category": "Gift", "launch_date": None, "current_price": 0.0, "cogs": GIFT_COGS_PER_UNIT, "weight_g": np.nan,
        "bundle_components": None, "status": "not_in_products_file", "cogs_source": "ASSUMPTION: gift cost unknown (PENDING)"}])], ignore_index=True)
log("P12 cogs missing", 1, f"cogs = sum of components = {p12:.2f} (ASSUMPTION A1)")
log("P13 gift sachet absent from products", 1, f"added row; cogs = {GIFT_COGS_PER_UNIT} per unit (ASSUMPTION A2: placeholder, not founder data)")

# ===== orders =====
o["created_utc"] = pd.to_datetime(o.created_at, utc=True)
o["order_date"] = o.created_utc.dt.tz_localize(None).dt.normalize()
o["month"] = o.order_date.dt.to_period("M").astype(str)
o["is_voided"] = o.financial_status.eq("voided")
test_codes = set(dc.loc[dc.code.str.strip().str.upper().str.contains("TEST"), "order_id"])
o["is_test"] = o.payment_gateway.eq("bogus") | o.order_id.isin(test_codes)
o["is_valid"] = ~(o.is_voided | o.is_test)
log("voided orders", int(o.is_voided.sum()), "excluded (no shipment, no fee, no returns)")
log("test orders (bogus gateway / TEST100)", int(o.is_test.sum()), "excluded")
log("valid orders", int(o.is_valid.sum()), "base for all metrics")
reg("DQ-O01", "valid orders", int(o.is_valid.sum()), f"{int(o.is_valid.sum())} / {len(o)} raw orders", "not voided, not test")
reg("DQ-O02", "voided orders", int(o.is_voided.sum()), f"/ {len(o)} raw orders", "financial_status=voided")
reg("DQ-O03", "test orders", int(o.is_test.sum()), f"/ {len(o)} raw orders", "bogus gateway or TEST100")
reg("DQ-O04", "voided orders total_paid (never collected)", round(float(o.loc[o.is_voided, "total_paid"].sum()), 2), "", "voided")
valid_ids = set(o.loc[o.is_valid, "order_id"])

# ===== order_items =====
n0 = len(oi)
oi["product_id"] = oi.product_id.str.upper().str.replace("-", "", regex=False).str.strip()
_raw_pid = load("order_items").product_id
fixed = int((~_raw_pid.isin(p.product_id) & (_raw_pid != "P13")).sum())
log("product_id format variants (P-03, p03, p07)", fixed, "normalised to upper-case without hyphen")
inj = oi.order_item_id >= 600000
log("injected duplicate lines (order_item_id>=600000, same order+product as an existing line)", int(inj.sum()),
    "removed (order subtotal does not include them)")
chk = oi[inj].merge(oi[~inj], on=["order_id", "product_id"], how="left", suffixes=("", "_orig"))
assert chk.order_item_id_orig.notna().all(), "every injected line must duplicate an existing line"
oi = oi[~inj].copy()
oi["gross_line"] = oi.quantity * oi.unit_price
oi["net_line"] = oi.gross_line - oi.line_discount
oi["is_gift"] = oi.unit_price.eq(0)
oi = oi.merge(p[["product_id", "cogs"]].rename(columns={"cogs": "unit_cogs"}), on="product_id", how="left")
assert oi.unit_cogs.notna().all()
oi["cogs_line"] = oi.quantity * oi.unit_cogs
oi["is_valid"] = oi.order_id.isin(valid_ids)
reg("DQ-I01", "order lines after dedupe", len(oi), f"{n0} raw - {int(inj.sum())} duplicates", "")
reg("DQ-I02", "free gift lines (P13) on valid orders", int((oi.is_gift & oi.is_valid).sum()), "", "unit_price=0, valid orders")

# ===== reconciliation orders vs lines (after cleaning) =====
ag = oi.groupby("order_id").agg(items_gross=("gross_line", "sum"), items_disc=("line_discount", "sum"))
rc = o.set_index("order_id").join(ag)
bad_sub = int(((rc.subtotal - rc.items_gross).abs() > 0.02).sum())
bad_dis = int(((rc.discount_total - rc.items_disc).abs() > 0.02).sum())
bad_tot = int(((rc.total_paid - (rc.subtotal - rc.discount_total + rc.shipping_charged + rc.tax_total)).abs() > 0.02).sum())
print(f"[CHECK] subtotal vs lines mismatches after cleaning: {bad_sub} | discount_total vs lines: {bad_dis} | total_paid identity: {bad_tot}")
assert bad_sub == 0 and bad_dis == 0 and bad_tot == 0
reg("DQ-C01", "orders where subtotal != sum(qty*unit_price)", bad_sub, f"/ {len(o)}", "after cleaning")
reg("DQ-C02", "orders where discount_total != sum(line_discount)", bad_dis, f"/ {len(o)}", "after cleaning")
reg("DQ-C03", "orders where total_paid != subtotal-discount+shipping+tax", bad_tot, f"/ {len(o)}", "after cleaning")

# ===== discounts =====
dn0 = len(dc)
orph = ~dc.order_id.isin(o.order_id)
log("discount rows for non-existent orders (8201, 8202)", int(orph.sum()), "removed")
dc = dc[~orph].copy()
dc["code_clean"] = dc.code.str.strip().str.upper()
log("discount code variants ('LOYAL10 ', 'welcome15')", int((dc.code != dc.code_clean).sum()), "trimmed + upper-cased")
dc["is_valid"] = dc.order_id.isin(valid_ids)
reg("DQ-D01", "discount rows after orphan removal", len(dc), f"{dn0} raw - {int(orph.sum())}", "")

# ===== shipping =====
sn0 = len(sh)
dup = sh.duplicated(keep="first")
log("exact duplicate shipments (appended at end of file)", int(dup.sum()), "removed")
sh = sh[~dup].copy()
orph = ~sh.order_id.isin(o.order_id)
log("shipments for non-existent orders (8301-8303)", int(orph.sum()), "removed")
sh = sh[~orph].copy()
sh["postage_imputed"] = sh.postage_cost.isna()
# impute postage: linear fit on weight within service x zone (fallback: group median)
imp_lin, imp_med = [], []
for (svc, zn), g in sh.groupby(["service", "zone"]):
    known = g[g.postage_cost.notna()]
    miss = g.index[g.postage_cost.isna()]
    if len(miss) == 0: continue
    med = known.postage_cost.median()
    if known.weight_g.nunique() > 3 and len(known) >= 30:
        b, a = np.polyfit(known.weight_g, known.postage_cost, 1)
        pred = a + b * g.loc[miss, "weight_g"]
    else:
        pred = pd.Series(med, index=miss)
    sh.loc[miss, "postage_cost"] = pred.round(2)
    imp_lin.extend(pred.tolist()); imp_med.extend([med] * len(miss))
log("shipments with missing postage_cost", int(sh.postage_imputed.sum()),
    f"imputed (ASSUMPTION A3): weight-linear fit within service x zone; total imputed = {sum(imp_lin):.2f} (group-median alt = {sum(imp_med):.2f})")
reg("DQ-S01", "shipments after cleaning", len(sh), f"{sn0} raw - 4 dup - 3 orphan", "")
reg("DQ-S02", "imputed postage total", round(sum(imp_lin), 2), f"{int(sh.postage_imputed.sum())} shipments", "")
reg("DQ-S03", "imputed postage total, alt method (group median)", round(sum(imp_med), 2), "", "sensitivity")
sh["is_valid"] = sh.order_id.isin(valid_ids)
sh["shipment_cost"] = sh.postage_cost + sh.packaging_cost
ship_per_order = sh.groupby("order_id").size()
print("[CHECK] orders with 2 shipments after dedupe:", int((ship_per_order > 1).sum()))
print("[CHECK] valid orders without shipment:", int((~pd.Series(list(valid_ids)).isin(sh.order_id)).sum()))

# ===== returns =====
rn0 = len(r)
orph = ~r.order_id.isin(o.order_id) | ~r.order_item_id.isin(oi.order_item_id)
log("returns for non-existent orders (RET-X0001..4)", int(orph.sum()), "removed")
r = r[~orph].copy()
r["reason_clean"] = r.reason.str.strip().str.lower().str.replace(" ", "_", regex=False)
log("return reason label variants (Title Case)", int((r.reason != r.reason_clean).sum()), "normalised to snake_case")
r["return_dt"] = pd.to_datetime(r.return_date)
r = r.merge(o[["order_id", "order_date"]], on="order_id", how="left")
r["days_to_return"] = (r.return_dt - r.order_date).dt.days
r["date_invalid"] = r.days_to_return < 0
log("return date before order date (RET-00183, 00198, 00435)", int(r.date_invalid.sum()),
    "kept in economics (amounts valid, order is refunded); EXCLUDED from days-to-return stats (flag date_invalid)")
r["is_valid"] = r.order_id.isin(valid_ids)
reg("DQ-RT01", "returns after cleaning", len(r), f"{rn0} raw - 4 orphan", "")
reg("DQ-RT02", "returns with date before order date", int(r.date_invalid.sum()), f"/ {len(r)}", "")
reg("DQ-RT03", "refund_amount total, valid orders", round(float(r.loc[r.is_valid, "refund_amount"].sum()), 2), f"{int(r.is_valid.sum())} return rows", "valid orders")
reg("DQ-RT04", "return_shipping_cost total, valid orders", round(float(r.loc[r.is_valid, "return_shipping_cost"].sum()), 2), "", "valid orders")
print("[CHECK] shipped_back return shipping cost by reason (brand pays only where >0):")
print(r[r.return_method == "shipped_back"].groupby("reason_clean").return_shipping_cost.agg(["size", lambda s: (s > 0).sum(), "sum"]).rename(columns={"<lambda_0>": "n_cost>0"}).to_string())

# ===== customers -> persons =====
cu["created_utc"] = pd.to_datetime(cu.created_at, utc=True)
SRC = {"meta": "meta_ads", "facebook": "meta_ads", "facebook_ads": "meta_ads", "ig": "meta_ads",
       "google_ads": "google_ads", "google": "google_ads", "tiktok": "tiktok_ads",
       "influencer": "influencer", "direct": "direct", "organic_search": "organic_search", "referral": "referral"}
cu["source_clean"] = cu.acquisition_source.str.strip().str.lower().map(SRC).fillna("unknown")
log("acquisition_source label variants", int((cu.acquisition_source.notna() & (cu.acquisition_source != cu.acquisition_source.str.lower())).sum()),
    "mapped to 8 values; Facebook/facebook_ads/IG->meta_ads, Google->google_ads (ASSUMPTION A4: IG and 'Google' are ambiguous)")
amb = int(cu.acquisition_source.isin(["IG", "Google"]).sum())
reg("DQ-CU01", "customer records with ambiguous source label (IG, Google)", amb, f"/ {len(cu)} customer records", "")
cu = cu.sort_values(["email_hash", "created_utc"])
persons = cu.groupby("email_hash").agg(
    first_customer_id=("customer_id", "first"), n_customer_ids=("customer_id", "size"),
    created_utc=("created_utc", "min"), country=("country", "first"), region=("region", "first"),
    accepts_marketing=("accepts_marketing", "max"))
src = cu[cu.source_clean != "unknown"].groupby("email_hash").source_clean.first()
persons["acquisition_source"] = src.reindex(persons.index).fillna("unknown")
persons = persons.reset_index().rename(columns={"email_hash": "person_id"})
log("persons with >1 customer_id (guest checkout)", int((persons.n_customer_ids > 1).sum()),
    "person_id = email_hash; source = earliest non-blank source among the person's records")
cid2pid = cu.set_index("customer_id").email_hash
o["person_id"] = o.customer_id.map(cid2pid)
assert o.person_id.notna().all()
has_order = persons.person_id.isin(o.person_id)
reg("DQ-CU02", "distinct persons (email_hash)", len(persons), f"{len(cu)} customer records", "")
reg("DQ-CU03", "persons with >1 customer_id", int((persons.n_customer_ids > 1).sum()), f"/ {len(persons)} persons", "")
reg("DQ-CU04", "persons with no order at all (newsletter-only)", int((~has_order).sum()), f"/ {len(persons)} persons", "")
vo = o[o.is_valid]
reg("DQ-CU05", "distinct persons with >=1 valid order", int(vo.person_id.nunique()), f"/ {len(persons)} persons", "valid orders")
# first valid order per person within the window
vo_sorted = vo.sort_values(["person_id", "created_utc"])
first_oid = vo_sorted.groupby("person_id").order_id.first()
o["is_first_order_of_person"] = o.order_id.isin(first_oid.values)
seq = vo_sorted.assign(seq=vo_sorted.groupby("person_id").cumcount() + 1).set_index("order_id").seq
o["order_seq_person"] = o.order_id.map(seq)
pre = int((persons.created_utc < "2025-07-01").sum())
reg("DQ-CU06", "persons whose record was created before 2025-07-01", pre, f"/ {len(persons)}", "'new vs repeat' cannot see pre-window history")

# ===== returns on valid orders only for analytic tables; keep flags =====
# ===== row-count verification =====
print("\n[VERIFY] cleaned row counts")
cnt = {"orders (all, flagged)": len(o), "orders valid": int(o.is_valid.sum()), "lines (all)": len(oi),
       "lines on valid orders": int(oi.is_valid.sum()), "returns (all)": len(r), "returns on valid": int(r.is_valid.sum()),
       "shipments (all)": len(sh), "shipments on valid": int(sh.is_valid.sum()),
       "discount rows (all)": len(dc), "discount rows on valid": int(dc.is_valid.sum()), "persons": len(persons)}
for k, v in cnt.items(): print(f"  {k:28s} {v}")
assert int(sh.is_valid.sum()) == int(o.is_valid.sum()) + int(r[r.is_valid].refund_type.eq("replacement").sum()), "valid shipments = valid orders + replacement shipments"
assert r.is_valid.all(), "all returns should sit on valid orders"
print("[VERIFY] shipments on valid = valid orders + replacements: OK")

# ===== save =====
o.drop(columns=["created_utc"]).assign(created_utc=o.created_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(f"{OUT}/orders_clean.csv", index=False)
oi.to_csv(f"{OUT}/order_items_clean.csv", index=False)
p.to_csv(f"{OUT}/products_clean.csv", index=False)
r.drop(columns=["return_dt"]).to_csv(f"{OUT}/returns_clean.csv", index=False)
sh.to_csv(f"{OUT}/shipments_clean.csv", index=False)
dc.to_csv(f"{OUT}/discounts_clean.csv", index=False)
persons.assign(created_utc=persons.created_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(f"{OUT}/persons_clean.csv", index=False)
cu.assign(created_utc=cu.created_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_csv(f"{OUT}/customers_records_clean.csv", index=False)
ms.to_csv(f"{OUT}/marketing_spend_clean.csv", index=False)
pd.DataFrame(REG, columns=["id", "metric", "value", "numerator_denominator", "period", "filter", "source_script"]).to_csv(f"{OUT}/number_register.csv", index=False)
pd.DataFrame(LOG, columns=["issue", "rows", "decision"]).to_csv(f"{OUT}/cleaning_log.csv", index=False)

print("\n[GIFT SENSITIVITY] P13 gift units on valid orders:", int(oi.loc[oi.is_gift & oi.is_valid, "quantity"].sum()))
g = int(oi.loc[oi.is_gift & oi.is_valid, "quantity"].sum())
for c in [0, 1, 2, 3]: print(f"   COGS/unit {c}: total {g*c:.2f}")
print("[P12 SENSITIVITY] P12 units on valid orders:", int(oi.loc[(oi.product_id == 'P12') & oi.is_valid, 'quantity'].sum()),
      "| +/- $2 cogs/unit =", 2 * int(oi.loc[(oi.product_id == 'P12') & oi.is_valid, 'quantity'].sum()))
print("\nNUMBER REGISTER"); print(pd.DataFrame(REG, columns=["id", "metric", "value", "n/d", "period", "filter", "script"]).drop(columns=["period", "script"]).to_string(index=False))
