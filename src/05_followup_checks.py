"""Phase 3b - follow-up checks: (1) return rate by discount status controlled for product AND first/repeat; (2) shipping recovery structure.
Run after 02_clean.py: python3 05_followup_checks.py [clean_dir]"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, pandas as pd
C = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
o = pd.read_csv(f"{C}/orders_clean.csv", parse_dates=["order_date"]); oi = pd.read_csv(f"{C}/order_items_clean.csv")
r = pd.read_csv(f"{C}/returns_clean.csv"); sh = pd.read_csv(f"{C}/shipments_clean.csv")
vo = o[o.is_valid]; L = oi[oi.is_valid & ~oi.is_gift].merge(vo[["order_id", "order_date", "is_first_order_of_person", "discount_total", "ship_country"]], on="order_id")
REG = []
def reg(m, v, nd=""): REG.append((f"P3b-{len(REG)+1:03d}", m, v, nd))

# ---- (1) CX3 refined ----
L = L.join(r[r.is_valid].groupby("order_item_id").quantity_returned.sum().rename("ret_units"), on="order_item_id").fillna({"ret_units": 0})
M = L[L.order_date <= "2026-05-16"].copy()
M["cust"] = np.where(M.is_first_order_of_person, "first", "repeat"); M["ds"] = np.where(M.discount_total > 0, "disc", "nodisc")
c = M.groupby(["product_id", "cust", "ds"]).agg(u=("quantity", "sum"), k=("ret_units", "sum")).unstack("ds"); c.columns = [f"{a}_{b}" for a, b in c.columns]; c = c.fillna(0)
c["ok"] = (c.u_disc >= 30) & (c.u_nodisc >= 30); c["rd"] = c.k_disc / c.u_disc; c["rn"] = c.k_nodisc / c.u_nodisc
print("=== CX3 refined: product x customer type cells (units>=30 on both sides) ==="); print(c.round(3).to_string())
ok = c[c.ok]; w = ok.u_disc / ok.u_disc.sum()
sd, sn = (w * ok.rd).sum(), (w * ok.rn).sum()
cov = ok.u_disc.sum() / c.u_disc.sum()
print(f"\nusable cells {len(ok)}/{len(c)}; coverage of discounted units {cov:.3f}")
print(f"STANDARDISED (product x first/repeat): discounted {sd:.4f} vs non-discounted {sn:.4f}; cells where discounted rate > non-discounted: {(ok.rd > ok.rn).sum()} of {len(ok)}")
rng = np.random.default_rng(11); bs = []
for _ in range(2000):
    bs.append(float((w.values * (rng.binomial(ok.u_disc.astype(int), ok.rd) / ok.u_disc.values - rng.binomial(ok.u_nodisc.astype(int), ok.rn) / ok.u_nodisc.values)).sum()))
lo, hi = np.percentile(bs, [2.5, 97.5]); print(f"standardised difference {sd-sn:+.4f}; parametric-bootstrap 95% interval {lo:+.4f}..{hi:+.4f}")
reg("CX3b product x first/repeat standardised mature return rate, discounted", round(sd, 4), f"{len(ok)} usable cells, coverage {cov:.3f} of discounted units")
reg("CX3b same, non-discounted", round(sn, 4), "same weights")
reg("CX3b difference 95% bootstrap interval low", round(lo, 4), ""); reg("CX3b difference 95% bootstrap interval high", round(hi, 4), "")
reg("CX3b cells where discounted return rate > non-discounted", f"{(ok.rd > ok.rn).sum()} of {len(ok)}", "")
fr = M.groupby("cust").agg(u=("quantity", "sum"), k=("ret_units", "sum"), disc_share=("ds", lambda s: (s == "disc").mean()))
print("\nshare of units on discounted orders by customer type (mature):"); print(fr.round(3).to_string())

# ---- (2) shipping recovery ----
S = sh[sh.is_valid].sort_values(["order_id", "ship_date", "shipment_id"]); S["rk"] = S.groupby("order_id").cumcount() + 1
first = S[S.rk == 1].set_index("order_id")
V = vo.set_index("order_id").join(first[["shipment_cost", "zone", "weight_g"]]).join(oi[oi.is_valid].groupby("order_id").apply(lambda g: (g.quantity * g.unit_price).sum()).rename("gross"))
V["net_ship"] = V.shipping_charged - V.shipment_cost
print("\n=== SHIPPING: charged vs original shipment cost (postage+packaging), valid orders; replacement shipments excluded ===")
print("distinct shipping_charged values (count):"); print(V.shipping_charged.round(2).value_counts().head(8).to_string())
fs = vo.order_id.isin(pd.read_csv(f"{C}/discounts_clean.csv").query("discount_type=='free_shipping'").order_id)
V["freeship_code"] = fs.values
V["bucket"] = pd.cut(V.gross, [0, 50, 75, 100, 1e9], labels=["<50", "50-75", "75-100", "100+"], right=False)
def sm(g): return pd.Series({"orders": len(g), "avg_charged": g.shipping_charged.mean(), "avg_cost": g.shipment_cost.mean(), "net_per_order": g.net_ship.mean(), "recovery": g.shipping_charged.sum() / g.shipment_cost.sum(), "share_charged_zero": (g.shipping_charged == 0).mean()})
print("-- by zone"); z = V.groupby("zone").apply(sm); print(z.round(3).to_string())
print("-- by gross basket bucket (US domestic only)"); u = V[V.zone == "US_domestic"].groupby("bucket", observed=True).apply(sm); print(u.round(3).to_string())
print("-- US domestic, orders with shipping_charged == 0 and NO FREESHIP code, by bucket"); zz = V[(V.zone == "US_domestic") & (V.shipping_charged == 0) & (~V.freeship_code)]
print(zz.groupby("bucket", observed=True).size().to_string(), "| total", len(zz), "| min gross", zz.gross.min() if len(zz) else None)
tot_c, tot_k = V.shipping_charged.sum(), V.shipment_cost.sum()
reg("shipping recovery, original shipments: charged / cost", round(tot_c / tot_k, 4), f"{tot_c:.2f} / {tot_k:.2f}")
reg("net shipping result, original shipments (charged - cost)", round(tot_c - tot_k, 2), "")
for zname, rw in z.iterrows(): reg(f"shipping recovery zone {zname}", round(rw.recovery, 4), f"{int(rw.orders)} orders; avg charged {rw.avg_charged:.2f}, avg cost {rw.avg_cost:.2f}")
for b_, rw in u.iterrows(): reg(f"US domestic recovery, basket {b_}", round(rw.recovery, 4), f"{int(rw.orders)} orders; share charged $0 {rw.share_charged_zero:.3f}")
reg("US orders with $0 shipping and no FREESHIP code", len(zz), f"/ {int((V.zone=='US_domestic').sum())} US orders")
pd.DataFrame(REG, columns=["id", "metric", "value", "numerator_denominator"]).to_csv(f"{C}/phase3/number_register_phase3b.csv", index=False)
print("\nREGISTER 3b"); print(pd.DataFrame(REG, columns=["id", "metric", "value", "numerator_denominator"]).to_string(index=False))
