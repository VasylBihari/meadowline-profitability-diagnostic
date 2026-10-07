"""Phase 2 - Revenue baseline. Meadowline Skin Co. (SYNTHETIC demo data).
Input: clean tables from 02_clean.py. Valid orders only (not voided, not test).
Run: python3 03_revenue_baseline.py [clean_dir]
"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

C = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs")
OUT = f"{C}/phase2"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)

o = pd.read_csv(f"{C}/orders_clean.csv", parse_dates=["order_date"])
oi = pd.read_csv(f"{C}/order_items_clean.csv")
r = pd.read_csv(f"{C}/returns_clean.csv", parse_dates=["return_date"])
dc = pd.read_csv(f"{C}/discounts_clean.csv")

vo = o[o.is_valid].copy()
vi = oi[oi.is_valid].copy()
vr = r[r.is_valid].copy()
vd = dc[dc.is_valid].copy()
ids = set(vo.order_id)
assert set(vi.order_id) <= ids and set(vr.order_id) <= ids
assert vi.order_id.nunique() == len(vo), "every valid order has lines"

REG = []
def reg(i, metric, value, nd="", filt="valid orders"):
    REG.append((i, metric, value, nd, "2025-07-01..2026-06-30", filt, "03_revenue_baseline.py"))

# ---------- bridge ----------
gross = vi.gross_line.sum()
disc_items = vi.line_discount.sum()
refunds = vr.refund_amount.sum()
net = gross - disc_items - refunds
ship_charged = vo.shipping_charged.sum()
tax = vo.tax_total.sum()
free_ship_forgone = vd.loc[vd.discount_type == "free_shipping", "amount"].sum()
fs_orders = vd.loc[vd.discount_type == "free_shipping", "order_id"].nunique()
bridge = pd.DataFrame([
    ("Gross sales (sum qty x unit_price)", gross),
    ("- Item discounts", -disc_items),
    ("= Net sales before refunds", gross - disc_items),
    ("- Refunds (refund_amount)", -refunds),
    ("= NET REVENUE", net),
    ("memo: shipping charged to customers (not in net revenue)", ship_charged),
    ("memo: tax collected (excluded)", tax),
    ("memo: free-shipping discount value (not charged, not in item discounts)", free_ship_forgone),
], columns=["line", "usd"]).round(2)
print("\n=== REVENUE BRIDGE (valid orders) ==="); print(bridge.to_string(index=False))
reg("P2-01", "gross sales", round(gross, 2), f"{len(vi)} lines", "sum qty*unit_price")
reg("P2-02", "item discounts", round(disc_items, 2), f"= {disc_items/gross:.4f} of gross sales", "sum line_discount")
reg("P2-03", "refunds", round(refunds, 2), f"{len(vr)} return rows; = {refunds/(gross-disc_items):.4f} of net sales before refunds", "sum refund_amount")
reg("P2-04", "net revenue", round(net, 2), f"gross - discounts - refunds", "")
reg("P2-05", "shipping charged to customers", round(ship_charged, 2), "", "memo")
reg("P2-06", "free-shipping discount value (memo)", round(free_ship_forgone, 2), f"{fs_orders} orders", "memo")

# ---------- tie-out checks ----------
chk1 = abs(vo.subtotal.sum() - gross)
chk2 = abs(vo.discount_total.sum() - disc_items)
chk3 = abs(vo.total_paid.sum() - (vo.subtotal.sum() - vo.discount_total.sum() + ship_charged + tax))
chk4 = abs(vr.groupby("order_id").refund_amount.sum().sum() - refunds)
print(f"\n[TIE-OUT] orders.subtotal vs lines: {chk1:.2f} | orders.discount_total vs lines: {chk2:.2f} | total_paid identity: {chk3:.2f} | refunds by order vs rows: {chk4:.2f}")
assert max(chk1, chk2, chk3, chk4) < 0.05
# refunds by status coherence
st = vo.set_index("order_id").financial_status
print("[TIE-OUT] return rows by order status:\n", vr.assign(s=vr.order_id.map(st)).groupby(["s", "refund_type"]).size().to_string())

# ---------- KPIs ----------
paid_units = vi.loc[~vi.is_gift, "quantity"].sum()
gift_units = vi.loc[vi.is_gift, "quantity"].sum()
n_orders = len(vo)
persons = vo.person_id.nunique()
orders_with_ret = vr.order_id.nunique()
aov_gross = gross / n_orders
aov_after_disc = (gross - disc_items) / n_orders
aov_net = net / n_orders
kpi = pd.DataFrame([
    ("Valid orders", n_orders),
    ("Distinct customers (persons) with a valid order", persons),
    ("Paid units (excl. free gifts)", paid_units),
    ("Free gift units (P13)", gift_units),
    ("Orders containing a free gift", vi[vi.is_gift].order_id.nunique()),
    ("Units per order (paid units / orders)", paid_units / n_orders),
    ("AOV gross (gross / orders)", aov_gross),
    ("AOV after discounts, before refunds", aov_after_disc),
    ("AOV net of refunds (net revenue / orders)", aov_net),
    ("Net revenue per customer (net revenue / persons)", net / persons),
    ("Orders per customer (orders / persons)", n_orders / persons),
    ("Orders with >=1 discount code", vo.discount_codes.notna().sum()),
    ("Share of orders with a code (of valid orders)", vo.discount_codes.notna().mean()),
    ("Orders with >=1 return row", orders_with_ret),
    ("Share of orders with a return row (of valid orders, censored)", orders_with_ret / n_orders),
], columns=["kpi", "value"])
print("\n=== KPIs ==="); print(kpi.round(4).to_string(index=False))
reg("P2-07", "valid orders", n_orders, "", "")
reg("P2-08", "distinct customers (persons)", persons, "email_hash with >=1 valid order", "")
reg("P2-09", "paid units", int(paid_units), f"/ {n_orders} orders = {paid_units/n_orders:.4f} units/order", "excl. P13 gifts")
reg("P2-10", "free gift units", int(gift_units), f"on {vi[vi.is_gift].order_id.nunique()} orders", "P13")
reg("P2-11", "AOV gross", round(aov_gross, 2), f"{gross:.2f} / {n_orders}", "")
reg("P2-12", "AOV after discounts, before refunds", round(aov_after_disc, 2), f"{gross-disc_items:.2f} / {n_orders}", "")
reg("P2-13", "AOV net of refunds", round(aov_net, 2), f"{net:.2f} / {n_orders}", "")
reg("P2-14", "net revenue per customer", round(net / persons, 2), f"{net:.2f} / {persons}", "within window, not LTV")
reg("P2-15", "orders per customer", round(n_orders / persons, 3), f"{n_orders} / {persons}", "within window")
reg("P2-16", "share of orders with a discount code", round(vo.discount_codes.notna().mean(), 4), f"{int(vo.discount_codes.notna().sum())} / {n_orders} valid orders", "")
reg("P2-17", "share of orders with a return row", round(orders_with_ret / n_orders, 4), f"{orders_with_ret} / {n_orders} valid orders", "censored for recent orders")

# ---------- monthly (order-month basis) ----------
li = vi.merge(vo[["order_id", "month"]], on="order_id")
m = li.groupby("month").agg(gross=("gross_line", "sum"), discounts=("line_discount", "sum"),
                            paid_units=("quantity", lambda s: s[~li.loc[s.index, "is_gift"]].sum()))
m["orders"] = vo.groupby("month").size()
m["persons"] = vo.groupby("month").person_id.nunique()
m["first_time_orders"] = vo[vo.is_first_order_of_person].groupby("month").size()
rm = vr.merge(vo[["order_id", "month"]], on="order_id")
m["refunds_by_order_month"] = rm.groupby("month").refund_amount.sum()
m["refunds_by_return_month"] = vr.assign(rmo=vr.return_date.dt.to_period("M").astype(str)).groupby("rmo").refund_amount.sum()
m = m.fillna(0)
m["net_rev_order_basis"] = m.gross - m.discounts - m.refunds_by_order_month
m["net_rev_cash_basis"] = m.gross - m.discounts - m.refunds_by_return_month
m["disc_rate"] = m.discounts / m.gross
m["refund_rate_order_basis"] = m.refunds_by_order_month / (m.gross - m.discounts)
m["aov_after_disc"] = (m.gross - m.discounts) / m.orders
m["units_per_order"] = m.paid_units / m.orders
m["first_time_share"] = m.first_time_orders / m.orders
print("\n=== MONTHLY ==="); print(m.round(3).to_string())
tot = m[["gross", "discounts", "refunds_by_order_month", "refunds_by_return_month", "orders", "paid_units", "first_time_orders"]].sum()
print("\n[TIE-OUT] sum of months: gross %.2f (bridge %.2f) | disc %.2f (%.2f) | refunds order-basis %.2f / return-basis %.2f (bridge %.2f) | orders %d (%d)" % (
    tot.gross, gross, tot.discounts, disc_items, tot.refunds_by_order_month, tot.refunds_by_return_month, refunds, tot.orders, n_orders))
assert abs(tot.gross - gross) < .05 and abs(tot.discounts - disc_items) < .05
assert abs(tot.refunds_by_order_month - refunds) < .05 and abs(tot.refunds_by_return_month - refunds) < .05 and tot.orders == n_orders
m.round(4).to_csv(f"{OUT}/monthly_baseline.csv")

# half-year and peak comparison (descriptive only)
mm = m.reset_index()
h1, h2 = mm.iloc[:6], mm.iloc[6:]
print("\n[DESCRIPTIVE] Jul-Dec 2025 vs Jan-Jun 2026 (calendar halves, not a trend test)")
for nm, h in [("Jul-Dec 2025", h1), ("Jan-Jun 2026", h2)]:
    print(f"  {nm}: orders {int(h.orders.sum())}, gross {h.gross.sum():.2f}, net(order basis) {h.net_rev_order_basis.sum():.2f}")
reg("P2-18", "orders Jul-Dec 2025", int(h1.orders.sum()), "", "valid orders, calendar half")
reg("P2-19", "orders Jan-Jun 2026", int(h2.orders.sum()), "", "valid orders, calendar half")
reg("P2-20", "net revenue Jul-Dec 2025 (order basis)", round(h1.net_rev_order_basis.sum(), 2), "", "refunds attributed to order month")
reg("P2-21", "net revenue Jan-Jun 2026 (order basis)", round(h2.net_rev_order_basis.sum(), 2), "", "refunds attributed to order month; recent months have less return time")
best = mm.loc[mm.net_rev_order_basis.idxmax()]
reg("P2-22", f"peak month by net revenue (order basis): {best.month}", round(best.net_rev_order_basis, 2), f"{int(best.orders)} orders", "")
print(f"[DESCRIPTIVE] peak month: {best.month} net {best.net_rev_order_basis:.2f}, orders {int(best.orders)}")

# ---------- price-change effect on revenue (descriptive context) ----------
pc = vi[vi.product_id.isin(["P03", "P12"])].merge(vo[["order_id", "order_date"]], on="order_id")
pc["period"] = np.where(pc.order_date < "2026-01-15", "before 2026-01-15", "from 2026-01-15")
print("\n[CONTEXT] P03/P12 list-price levels used in the data:"); print(pc.groupby(["product_id", "period", "unit_price"]).size().to_string())

# ---------- sales channel, country (descriptive split of net sales before refunds) ----------
lo = vi.merge(vo[["order_id", "ship_country", "sales_channel"]], on="order_id")
for col in ["ship_country", "sales_channel"]:
    t = lo.groupby(col).agg(gross=("gross_line", "sum"), disc=("line_discount", "sum"))
    t["orders"] = vo.groupby(col).size()
    t["share_of_net_sales"] = (t.gross - t.disc) / (t.gross - t.disc).sum()
    t["aov_after_disc"] = (t.gross - t.disc) / t.orders
    print(f"\n=== by {col} ==="); print(t.round(3).to_string())
    t.round(4).to_csv(f"{OUT}/by_{col}.csv")
us_share = ((lo[lo.ship_country == 'US'].gross_line.sum() - lo[lo.ship_country == 'US'].line_discount.sum()) / (gross - disc_items))
reg("P2-23", "US share of net sales before refunds", round(us_share, 4), "US (gross-disc) / total (gross-disc)", "valid orders")

bridge.to_csv(f"{OUT}/bridge.csv", index=False)
kpi.round(4).to_csv(f"{OUT}/kpi.csv", index=False)
reg_df = pd.DataFrame(REG, columns=["id", "metric", "value", "numerator_denominator", "period", "filter", "source_script"])
reg_df.to_csv(f"{OUT}/number_register_phase2.csv", index=False)
print("\nNUMBER REGISTER (Phase 2)"); print(reg_df.drop(columns=["period", "source_script"]).to_string(index=False))
