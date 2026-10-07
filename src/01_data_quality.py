"""Phase 1 - Data quality checks. Meadowline Skin Co. (SYNTHETIC demo data).
Run: python3 01_data_quality.py [data_dir]
Read-only: prints findings, changes nothing.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 200)

def load(n):
    return pd.read_csv(f"{D}/{n}.csv")

cu, o, oi, p = load("customers"), load("orders"), load("order_items"), load("products")
r, sh, dc, ms = load("returns"), load("shipping_costs"), load("discounts"), load("marketing_spend")

def H(t):
    print("\n" + "=" * 8, t, "=" * 8)

# ---------- 1. Row counts, keys, duplicates ----------
H("1. KEYS / DUPLICATES")
keys = {"customers": (cu, "customer_id"), "orders": (o, "order_id"), "order_items": (oi, "order_item_id"),
        "products": (p, "product_id"), "returns": (r, "return_id"), "shipping_costs": (sh, "shipment_id"),
        "discounts": (dc, "discount_application_id")}
for n, (df, k) in keys.items():
    print(f"{n:15s} rows={len(df):5d} key={k:24s} dup_keys={df[k].duplicated().sum():3d} "
          f"full_dup_rows={df.duplicated().sum():3d} null_key={df[k].isna().sum()}")
print("marketing_spend dup (month,channel):", ms.duplicated(["month", "channel"]).sum())
print("customers: distinct email_hash =", cu.email_hash.nunique(), "of", len(cu),
      "| persons with >1 customer_id:", (cu.groupby("email_hash").size() > 1).sum())
print("order_items dup (order_id, product_id):", oi.duplicated(["order_id", "product_id"]).sum())

# ---------- 2. Orphans ----------
H("2. ORPHANS (FK checks)")
def orph(child, col, parent, pcol, name):
    m = ~child[col].isin(parent[pcol])
    print(f"{name:45s} orphans={m.sum()}")
    return child[m]
orph(o, "customer_id", cu, "customer_id", "orders.customer_id -> customers")
orph(oi, "order_id", o, "order_id", "order_items.order_id -> orders")
x = orph(oi, "product_id", p, "product_id", "order_items.product_id -> products")
if len(x): print(x.groupby(["product_id", "sku"]).agg(lines=("quantity", "size"), qty=("quantity", "sum"), price=("unit_price", lambda s: sorted(s.unique())[:5])))
orph(sh, "order_id", o, "order_id", "shipping_costs.order_id -> orders")
orph(dc, "order_id", o, "order_id", "discounts.order_id -> orders")
x = orph(r, "order_id", o, "order_id", "returns.order_id -> orders")
if len(x): print(x.to_string())
x = orph(r, "order_item_id", oi, "order_item_id", "returns.order_item_id -> order_items")
print("orders with no items:", (~o.order_id.isin(oi.order_id)).sum())
print("orders with no shipment:", (~o.order_id.isin(sh.order_id)).sum())
print("customers with no orders:", (~cu.customer_id.isin(o.customer_id)).sum())
# returns consistency with item
rm = r.merge(oi, on="order_item_id", how="left", suffixes=("", "_i"))
print("returns where order_id != item's order_id:", ((rm.order_id != rm.order_id_i) & rm.order_id_i.notna()).sum())
print("returns where product_id != item's product_id:", ((rm["product_id"] != rm["product_id_i"]) & rm.product_id_i.notna()).sum())
print("returns where quantity_returned > item quantity:", (rm.quantity_returned > rm.quantity).sum())
print("returns with >1 row per order_item:", r.duplicated("order_item_id", keep=False).sum())

# ---------- 3. Missing values ----------
H("3. MISSING VALUES")
for n, df in [("customers", cu), ("orders", o), ("order_items", oi), ("products", p), ("returns", r),
              ("shipping_costs", sh), ("discounts", dc), ("marketing_spend", ms)]:
    na = df.isna().sum()
    na = na[na > 0]
    print(n, dict(na) if len(na) else "none")
print("products with null cogs:\n", p[p.cogs.isna()][["product_id", "product_name", "bundle_components"]].to_string())
print("shipping rows null postage:", sh.postage_cost.isna().sum(), "| their order statuses:")
print(sh[sh.postage_cost.isna()].merge(o[["order_id", "financial_status"]], on="order_id").financial_status.value_counts().to_string())
print("discounts null percentage by type:\n", dc[dc.percentage.isna()].discount_type.value_counts().to_string())

# ---------- 4. Dates ----------
H("4. DATES")
o["created"] = pd.to_datetime(o.created_at, utc=True)
cu["created"] = pd.to_datetime(cu.created_at, utc=True)
r["rdate"] = pd.to_datetime(r.return_date)
sh["sdate"] = pd.to_datetime(sh.ship_date)
print("orders created range:", o.created.min(), "->", o.created.max())
print("orders outside 2025-07-01..2026-06-30 (UTC):", ((o.created < "2025-07-01") | (o.created >= "2026-07-01")).sum())
print("customers created range:", cu.created.min(), "->", cu.created.max())
print("returns date range:", r.rdate.min(), "->", r.rdate.max(), "| after extract 2026-07-02:", (r.rdate > "2026-07-02").sum())
rr = r.merge(o[["order_id", "created", "financial_status"]], on="order_id", how="left")
rr["days"] = (rr.rdate - rr.created.dt.tz_localize(None).dt.normalize()).dt.days
print("return before order date:", (rr.days < 0).sum(), "| days-to-return describe:")
print(rr.days.describe().round(1).to_string())
print("returns > 90 days after order:", (rr.days > 90).sum())
print("returns on voided orders:", (rr.financial_status == "voided").sum())
ss = sh.merge(o[["order_id", "created"]], on="order_id", how="left")
ss["d"] = (ss.sdate - ss.created.dt.tz_localize(None).dt.normalize()).dt.days
print("ship before order:", (ss.d < 0).sum(), "| ship lag describe:"); print(ss.d.describe().round(1).to_string())
fo = o.groupby("customer_id").created.min().rename("first_order")
cc = cu.merge(fo, on="customer_id", how="inner")
print("customer record created AFTER first order:", (cc.created > cc.first_order + pd.Timedelta(days=1)).sum())
print("cancelled_at non-null:", o.cancelled_at.notna().sum(), "| by status:")
print(o[o.cancelled_at.notna()].financial_status.value_counts().to_string())
print("month sequence marketing:", sorted(ms.month.unique())[0], "->", sorted(ms.month.unique())[-1], "n_months", ms.month.nunique())

# ---------- 5. Test / voided ----------
H("5. TEST / VOIDED ORDERS")
print(o.groupby(["financial_status", "payment_gateway"]).size().unstack(fill_value=0).to_string())
test_codes = dc[dc.code.str.upper().str.contains("TEST")].order_id.unique()
bogus = o[o.payment_gateway == "bogus"]
print("bogus gateway orders:", len(bogus), "| TEST code orders:", len(test_codes))
print("overlap bogus & TEST code:", len(set(bogus.order_id) & set(test_codes)))
print(bogus[["order_id", "customer_id", "created_at", "financial_status", "subtotal", "discount_total", "discount_codes", "total_paid"]].to_string())
print("orders with discount code TEST not bogus:", sorted(set(test_codes) - set(bogus.order_id)))
print("voided orders: shipments:", sh.order_id.isin(o[o.financial_status == "voided"].order_id).sum(),
      "| returns:", r.order_id.isin(o[o.financial_status == "voided"].order_id).sum(),
      "| total_paid sum:", o[o.financial_status == "voided"].total_paid.sum().round(2),
      "| payment_fee sum:", o[o.financial_status == "voided"].payment_fee.sum().round(2))
print("unfulfilled vs status:"); print(pd.crosstab(o.fulfillment_status, o.financial_status).to_string())

# ---------- 6. Reconciliation ----------
H("6. RECONCILIATION")
oi["gross"] = oi.quantity * oi.unit_price
ag = oi.groupby("order_id").agg(items_gross=("gross", "sum"), items_disc=("line_discount", "sum"), units=("quantity", "sum"), n_lines=("quantity", "size"))
dagg = dc.groupby("order_id").agg(disc_amt_all=("amount", "sum"))
dli = dc[dc.applies_to == "line_items"].groupby("order_id").amount.sum().rename("disc_amt_items")
dsh = dc[dc.applies_to == "shipping"].groupby("order_id").amount.sum().rename("disc_amt_ship")
m = o.set_index("order_id").join([ag, dli, dsh]).fillna({"disc_amt_items": 0, "disc_amt_ship": 0})
m["d_subtotal"] = (m.subtotal - m.items_gross).round(2)
m["d_disc_lines"] = (m.discount_total - m.items_disc).round(2)
m["d_disc_codes"] = (m.discount_total - m.disc_amt_items).round(2)
m["calc_total"] = (m.subtotal - m.discount_total + m.shipping_charged + m.tax_total).round(2)
m["d_total"] = (m.total_paid - m.calc_total).round(2)
for c in ["d_subtotal", "d_disc_lines", "d_disc_codes", "d_total"]:
    print(f"{c:14s} nonzero(|x|>0.02)={int((m[c].abs() > 0.02).sum()):5d}  sum={m[c].sum():10.2f}  min={m[c].min():8.2f} max={m[c].max():8.2f}")
print("orders with discount_total>0 but no discount row:", ((m.discount_total > 0) & (m.disc_amt_items == 0)).sum())
print("orders with discount row but discount_total==0:", ((m.discount_total == 0) & (m.disc_amt_items > 0)).sum())
print("discount_codes text vs discounts rows (codes count):")
m["n_codes_text"] = o.set_index("order_id").discount_codes.fillna("").apply(lambda s: len([x for x in s.split(",") if x.strip()]))
m["n_codes_rows"] = dc.groupby("order_id").size()
m["n_codes_rows"] = m.n_codes_rows.fillna(0)
print("  mismatch:", (m.n_codes_text != m.n_codes_rows).sum())
bad = m[(m.d_subtotal.abs() > 0.02)].head(8)
print("sample subtotal mismatches:\n", bad[["subtotal", "items_gross", "d_subtotal", "n_lines"]].to_string())
bad = m[(m.d_total.abs() > 0.02)].head(8)
print("sample total mismatches:\n", bad[["subtotal", "discount_total", "shipping_charged", "tax_total", "total_paid", "calc_total", "d_total", "financial_status"]].to_string())
print("free-ship rows:", (dc.discount_type == "free_shipping").sum(), "| amounts describe:"); print(dc[dc.discount_type == "free_shipping"].amount.describe().round(2).to_string())
fs = dc[dc.discount_type == "free_shipping"].merge(o[["order_id", "shipping_charged"]], on="order_id")
print("free-ship orders shipping_charged describe:"); print(fs.shipping_charged.describe().round(2).to_string())

# ---------- 7. Prices / lines ----------
H("7. LINES / PRICES")
pm = oi.merge(p[["product_id", "current_price", "cogs"]], on="product_id", how="left")
pm["pdiff"] = (pm.unit_price - pm.current_price).round(2)
print("unit_price != current_price lines:", (pm.pdiff.abs() > 0.01).sum(), "of", len(pm))
print(pm[pm.pdiff.abs() > 0.01].groupby("product_id").pdiff.agg(["size", "min", "max"]).to_string())
print("zero-price (gift) lines:", (oi.unit_price == 0).sum())
print(oi[oi.unit_price == 0].product_id.value_counts().to_string())
print("qty describe:"); print(oi.quantity.describe().to_string())
print("negative values: items", (oi.line_discount < 0).sum(), "orders", (o[["subtotal", "discount_total", "shipping_charged", "tax_total", "total_paid", "payment_fee"]] < 0).sum().sum())
print("line_discount > gross lines:", (oi.line_discount > oi.gross + 0.01).sum())
print("payment_fee describe:"); print(o.payment_fee.describe().round(2).to_string())
print("P12 bundle lines:", (oi.product_id == "P12").sum(), "| components:", p.loc[p.product_id == "P12", "bundle_components"].values)
print("sku mismatch with products:", (~oi.set_index("product_id").sku.eq(p.set_index("product_id").sku.reindex(oi.product_id).values)).sum() if True else "")

# ---------- 8. Returns ----------
H("8. RETURNS")
rl = r.merge(oi[["order_item_id", "quantity", "unit_price", "line_discount", "gross"]], on="order_item_id", how="left")
rl["net_line"] = rl.gross - rl.line_discount
rl["refund_vs_net"] = (rl.refund_amount - rl.net_line * rl.quantity_returned / rl.quantity).round(2)
print("refund > net line value (+0.02):", (rl.refund_vs_net > 0.02).sum())
print("full refunds where refund != net*qty_ret/qty (|d|>0.02):", ((rl.refund_type == "full") & (rl.refund_vs_net.abs() > 0.02)).sum())
print("replacement rows refund>0:", ((rl.refund_type == "replacement") & (rl.refund_amount > 0)).sum(),
      "| replacement_sent flag != replacement type:", ((rl.refund_type == "replacement") != rl.replacement_sent).sum())
print("shipped_back with return_shipping_cost==0:", ((rl.return_method == "shipped_back") & (rl.return_shipping_cost == 0)).sum(),
      "of", (rl.return_method == "shipped_back").sum())
print("returnless with return_shipping_cost>0:", ((rl.return_method == "returnless_refund") & (rl.return_shipping_cost > 0)).sum())
print("restocked True but returnless:", ((rl.restocked) & (rl.return_method == "returnless_refund")).sum())
print("return_id prefix counts:"); print(r.return_id.str[:5].value_counts().to_string())
print("rows RET-X*:"); print(r[r.return_id.str.startswith("RET-X")].to_string())
print("returns with return date earlier than same order's other returns sequence (id order vs date): non-monotonic ids count:", (r.rdate.diff() < pd.Timedelta(0)).sum())
print("order financial_status vs returns present:");
st = o.set_index("order_id").financial_status
r["ostatus"] = r.order_id.map(st)
print(r.ostatus.value_counts(dropna=False).to_string())
print("refunded/partially_refunded orders without return row:", (o.financial_status.isin(["refunded", "partially_refunded"]) & ~o.order_id.isin(r.order_id)).sum())
print("paid orders with return rows:", ((o.financial_status == "paid") & o.order_id.isin(r.order_id)).sum())

# ---------- 9. Labels ----------
H("9. LABELS / NORMALISATION")
print("reason raw->normalised:", sorted(r.reason.unique()))
print("normalised reason:", r.reason.str.lower().str.replace(" ", "_").value_counts().to_dict())
print("codes upper-merge:", sorted(dc.code.unique()))
print("acq source raw:", cu.acquisition_source.fillna("<blank>").value_counts().to_dict())
print("accepts_marketing:", cu.accepts_marketing.value_counts().to_dict())
print("discount code vs text in orders.discount_codes - distinct:", sorted({c.strip() for s in o.discount_codes.dropna() for c in s.split(",")}))
print("country/ship_country mix customers vs order:")
oc = o.merge(cu[["customer_id", "country"]], on="customer_id", how="left")
print("  order ship_country != customer country:", (oc.ship_country != oc.country).sum())
print("shipping zone vs ship_country mismatch:", (sh.merge(o[["order_id", "ship_country"]], on="order_id").assign(z=lambda x: x.zone.replace({"US_domestic": "US"})).pipe(lambda x: (x.z != x.ship_country).sum())))
print("shipments per order:"); print(sh.groupby("order_id").size().value_counts().sort_index().to_string())

# ---------- 10. Customers ----------
H("10. CUSTOMERS")
d = cu.groupby("email_hash").customer_id.agg(["size", "min", "max"])
print("email_hash with multiple ids:", (d["size"] > 1).sum(), "| ids involved:", d[d["size"] > 1]["size"].sum())
print("multi-id persons: do the ids have orders?")
mi = cu[cu.email_hash.isin(d[d["size"] > 1].index)].merge(o.groupby("customer_id").size().rename("n_orders"), on="customer_id", how="left").fillna({"n_orders": 0})
print(mi.groupby("email_hash").n_orders.apply(lambda s: (s > 0).sum()).value_counts().to_string())
print("customers w/o orders (newsletter-only?):", (~cu.customer_id.isin(o.customer_id)).sum(), "| accepts_marketing among them:", cu[~cu.customer_id.isin(o.customer_id)].accepts_marketing.mean().round(3))
print("customer created before period start:", (cu.created < "2025-07-01").sum())
print("distinct persons ordering (valid statuses):", cu[cu.customer_id.isin(o.customer_id)].email_hash.nunique())

# ---------- 11. Marketing ----------
H("11. MARKETING")
print(ms.groupby("channel")[["spend", "platform_reported_orders", "platform_reported_revenue"]].sum().round(2).to_string())
print("rows:", len(ms), "| expected 48 | negative:", (ms.select_dtypes("number") < 0).sum().sum())
print("clicks>impressions:", (ms.clicks > ms.impressions).sum())
print("DONE")
