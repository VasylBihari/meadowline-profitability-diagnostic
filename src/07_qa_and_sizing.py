"""Phase 5 - QA + sizing for recommendations. Meadowline Skin Co. (SYNTHETIC demo data).
Part A: independent recomputation from RAW files (different path from 02_clean.py).
Part B: automated QA of clean tables and the master Number Register.
Part C: sizing (bounds and scenarios, NOT forecasts) for the recommendations.
Run: python3 07_qa_and_sizing.py [raw_dir] [clean_dir]
"""
import sys, os, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "raw")
C = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "outputs")
OUT = f"{C}/phase5"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 100)
pd.options.display.float_format = "{:,.3f}".format
raw = lambda n: pd.read_csv(f"{RAW}/{n}.csv")
REG = []
def reg(metric, value, nd="", filt=""):
    REG.append((f"P5-{len(REG)+1:03d}", metric, value, nd, filt)); return REG[-1][0]
QA = []
def qa(name, ok, detail=""):
    QA.append((name, "PASS" if ok else "FAIL", detail)); print(f"[QA {'PASS' if ok else 'FAIL'}] {name} {detail}")

# =====================================================================
# PART A - independent recomputation from raw
# =====================================================================
print("=== PART A: independent recomputation from RAW files ===")
o, oi, r, sh, p, dcr = raw("orders"), raw("order_items"), raw("returns"), raw("shipping_costs"), raw("products"), raw("discounts")
valid = o[(o.financial_status != "voided") & (o.payment_gateway != "bogus")]
vid = set(valid.order_id)
gross_i = valid.subtotal.sum(); disc_i = valid.discount_total.sum()
rv = r[r.order_id.isin(vid) & r.order_item_id.isin(oi.order_item_id)]
ref_i = rv.refund_amount.sum()
net_i = gross_i - disc_i - ref_i
# cogs
oi2 = oi.copy(); oi2["pid"] = oi2.product_id.str.upper().str.replace("-", "", regex=False)
oi2 = oi2.drop_duplicates(["order_id", "pid", "quantity", "unit_price", "line_discount"], keep="first")
oi2 = oi2[oi2.order_id.isin(vid)]
cogs_map = p.assign(product_id=p.product_id.str.upper()).set_index("product_id").cogs.to_dict()
cogs_map["P12"] = cogs_map["P01"] + cogs_map["P03"] + cogs_map["P07"]; cogs_map["P13"] = 1.0
oi2["uc"] = oi2.pid.map(cogs_map); assert oi2.uc.notna().all()
cogs_i = (oi2.quantity * oi2.uc).sum()
rv = rv.merge(oi2[["order_item_id", "uc"]], on="order_item_id", how="left")
rec_i = (rv.restocked.astype(str).str.lower().eq("true") * rv.quantity_returned * rv.uc).sum()
repl_i = ((rv.refund_type == "replacement") * rv.quantity_returned * rv.uc).sum()
s2 = sh.drop_duplicates(); s2 = s2[s2.order_id.isin(vid)].copy()
s2["postage_cost"] = s2.postage_cost.fillna(s2.groupby(["service", "zone"]).postage_cost.transform("median"))
post_i = s2.postage_cost.sum(); pack_i = s2.packaging_cost.sum()
fee_i = valid.payment_fee.sum(); rship_i = rv.return_shipping_cost.sum(); shiprev_i = valid.shipping_charged.sum()
contrib_i = net_i + shiprev_i - (cogs_i - rec_i + repl_i) - post_i - pack_i - fee_i - rship_i
print(f"independent: gross {gross_i:.2f} disc {disc_i:.2f} refunds {ref_i:.2f} net {net_i:.2f} contribution {contrib_i:.2f}")
P2 = pd.read_csv(f"{C}/phase2/number_register_phase2.csv").set_index("id").value
P3 = pd.read_csv(f"{C}/phase3/number_register_phase3.csv").set_index("id").value
qa("gross sales raw vs register P2-01", abs(gross_i - float(P2["P2-01"])) < 0.05, f"{gross_i:.2f} vs {float(P2['P2-01']):.2f}")
qa("item discounts raw vs P2-02", abs(disc_i - float(P2["P2-02"])) < 0.05)
qa("refunds raw vs P2-03", abs(ref_i - float(P2["P2-03"])) < 0.05)
qa("net revenue raw vs P2-04", abs(net_i - float(P2["P2-04"])) < 0.05)
qa("valid orders raw vs P2-07", len(valid) == int(float(P2["P2-07"])), f"{len(valid)}")
d_c = contrib_i - float(P3["P3-002"])
qa("contribution raw-path vs P3-002 (tolerance $3.00: postage imputation method differs, known $2.91)", abs(d_c) < 3.0, f"difference {d_c:+.2f}")

# =====================================================================
# PART B - QA of clean tables and registers
# =====================================================================
print("\n=== PART B: automated QA ===")
cl = {n: pd.read_csv(f"{C}/{n}_clean.csv") for n in ["orders", "order_items", "returns", "shipments", "discounts", "persons", "products"]}
qa("clean orders: unique order_id", cl["orders"].order_id.is_unique)
qa("clean order_items: unique order_item_id", cl["order_items"].order_item_id.is_unique)
qa("clean returns: unique return_id and one row per order_item", cl["returns"].return_id.is_unique and cl["returns"].order_item_id.is_unique)
qa("clean shipments: unique shipment_id", cl["shipments"].shipment_id.is_unique)
qa("clean discounts: unique discount_application_id", cl["discounts"].discount_application_id.is_unique)
qa("clean persons: unique person_id", cl["persons"].person_id.is_unique)
for child, col, parent, pcol in [("order_items", "order_id", "orders", "order_id"), ("returns", "order_item_id", "order_items", "order_item_id"), ("shipments", "order_id", "orders", "order_id"), ("discounts", "order_id", "orders", "order_id")]:
    qa(f"no orphans: {child}.{col} -> {parent}", cl[child][col].isin(cl[parent][pcol]).all())
qa("every order has >=1 line", cl["orders"].order_id.isin(cl["order_items"].order_id).all())
qa("raw minus removed equals clean (lines 8835-11, shipments 5043-7, returns 473-4, discounts 2848-2)",
   (len(oi) - 11, len(sh) - 7, len(r) - 4, len(dcr) - 2) == (len(cl["order_items"]), len(cl["shipments"]), len(cl["returns"]), len(cl["discounts"])))
OF = pd.read_csv(f"{C}/phase3/order_frame.csv", parse_dates=["order_date"])
qa("order_frame: sum of order contribution equals bridge (P3-002)", abs(OF.contribution.sum() - float(P3["P3-002"])) < 0.05, f"{OF.contribution.sum():.2f}")
qa("order_frame: sum of net revenue equals P2-04", abs(OF.net_rev.sum() - float(P2["P2-04"])) < 0.05)
PT = pd.read_csv(f"{C}/phase4/person_table.csv", parse_dates=["first_date"])
qa("person_table: sum of contribution equals bridge", abs(PT.contribution.sum() - float(P3["P3-002"])) < 0.05, f"{PT.contribution.sum():.2f}")
qa("person_table: buyers = 3828 and orders sum = 5006", len(PT) == 3828 and int(PT.n_orders.sum()) == 5006)

# master register
regs = []
for ph, f in [("P1", f"{C}/number_register.csv"), ("P2", f"{C}/phase2/number_register_phase2.csv"), ("P3", f"{C}/phase3/number_register_phase3.csv"),
              ("P3b", f"{C}/phase3/number_register_phase3b.csv"), ("P4", f"{C}/phase4/number_register_phase4.csv")]:
    d = pd.read_csv(f); d["phase"] = ph; regs.append(d[["id", "metric", "value", "numerator_denominator", "phase"]] if "numerator_denominator" in d else d.rename(columns={"numerator / denominator": "numerator_denominator"})[["id", "metric", "value", "numerator_denominator", "phase"]])
M = pd.concat(regs, ignore_index=True)
qa("master register: unique IDs", M.id.is_unique, f"{len(M)} entries")
bad_word = M[M.metric.str.contains("profit", case=False)]
qa("register wording: no 'profit' in metric names", len(bad_word) == 0, str(bad_word.metric.tolist()))
numeric = pd.to_numeric(M.value, errors="coerce")
frac = M[(numeric > 0) & (numeric < 1) & (~M.metric.str.contains("margin|share|rate|ratio|CAC", case=False) == False) | ((numeric > 0) & (numeric < 1) & M.metric.str.contains("share|rate|/|margin", case=False))]
no_nd = frac[frac.numerator_denominator.isna() | (frac.numerator_denominator.astype(str).str.strip() == "")]
print("register rows with a proportion value and no numerator/denominator text (review):"); print(no_nd[["id", "metric", "value"]].to_string(index=False))
M.to_csv(f"{OUT}/master_register_P1_P4.csv", index=False)

# =====================================================================
# PART C - SIZING FOR RECOMMENDATIONS (bounds / scenarios only)
# =====================================================================
print("\n=== PART C: sizing ===")
OF["net_sales_pre_refund"] = OF.gross - OF.disc
dcn = cl["discounts"]; dcn = dcn[dcn.is_valid]
first_date = OF.groupby("person_id").order_date.min().rename("first_date")
OF = OF.join(first_date, on="person_id")
OF["d_from_first"] = (OF.order_date - OF.first_date).dt.days
c90 = OF[OF.d_from_first <= 90].groupby("person_id").contribution.sum().rename("c90")
PT = PT.join(c90, on="person_id"); PT["c90"] = PT.c90.fillna(0)
Mat = PT[PT.first_date <= "2026-03-31"].copy()

# ---- R1: WELCOME15 ----
w = dcn[(dcn.code_clean == "WELCOME15")]
w_total = w.amount.sum(); w_orders = w.order_id.nunique()
oth = dcn[(dcn.discount_type == "percentage") & (dcn.code_clean != "WELCOME15")].groupby("order_id").size()
stack_ids = set(w.order_id) & set(oth.index)
stack_amt = w[w.order_id.isin(stack_ids)].amount.sum()
print(f"WELCOME15: item-discount dollars {w_total:.2f} on {w_orders} orders (avg {w_total/w_orders:.2f}); stacked with another percentage code on {len(stack_ids)} orders, welcome portion {stack_amt:.2f}")
reg("WELCOME15 item-discount dollars (valid orders)", round(w_total, 2), f"{w_orders} orders; avg {w_total/w_orders:.2f}")
reg("WELCOME15 dollars / net revenue", round(w_total / float(P2["P2-04"]), 4), f"{w_total:.2f} / {float(P2['P2-04']):.2f}")
reg("orders stacking WELCOME15 with another percentage code", len(stack_ids), f"welcome portion {stack_amt:.2f}")
wb = Mat[Mat.first_welcome]
d_avg = OF[OF.order_id.isin(wb.first_order_id)].disc.mean()
c_avg = wb.c90.mean()
xstar = d_avg / (c_avg + d_avg)
print(f"break-even: welcome first-time buyers n={len(wb)}; avg discount on first order d={d_avg:.2f}; avg 90-day contribution c={c_avg:.2f}; break-even incremental share x*=d/(c+d)={xstar:.3f}")
reg("WELCOME15 break-even share of buyers that must be incremental", round(xstar, 4), f"d/(c+d) = {d_avg:.2f}/({c_avg:.2f}+{d_avg:.2f}); n={len(wb)} mature-cohort buyers", "assumes non-incremental buyer would have bought at full price; ignores repeat-rate difference")
reg("WELCOME15 avg discount on first order (mature cohort)", round(d_avg, 2), f"n={len(wb)}")
reg("WELCOME15 avg 90-day contribution per buyer (mature cohort)", round(c_avg, 2), f"n={len(wb)}")

# ---- R2: returns P06 / P12 ----
pe_ = pd.read_csv(f"{C}/phase3/product_economics.csv", index_col=0)
rows = []
for pid in ["P06", "P12"]:
    t = pe_.loc[pid]; rpu = t.refunds / t.ret_units
    rows.append((pid, t.units, t.ret_units, t.refunds, rpu, t.units * 0.01 * rpu))
R2 = pd.DataFrame(rows, columns=["product", "units_12m", "returned_units_12m", "refunds_12m", "refund_per_returned_unit", "value_of_1pp_lower_return_rate"]).set_index("product")
print("value of a 1 percentage-point lower unit return rate (refund dollars only, at 12-month volume):"); print(R2.round(2).to_string())
for pid, rw in R2.iterrows():
    reg(f"{pid}: refunds, 12 months", round(rw.refunds_12m, 2), f"{int(rw.returned_units_12m)} returned units / {int(rw.units_12m)} units")
    reg(f"{pid}: value of 1 pp lower unit return rate (refund dollars only)", round(rw.value_of_1pp_lower_return_rate, 2), f"{int(rw.units_12m)} units x 1% x {rw.refund_per_returned_unit:.2f}", "scaling, not a forecast")
L = cl["order_items"]; L = L[L.is_valid & ~L.is_gift & (L.product_id == "P06")].merge(OF[["order_id", "order_date", "is_first_order_of_person", "disc"]], on="order_id")
rr = cl["returns"].groupby("order_item_id").agg(ru=("quantity_returned", "sum"), rf=("refund_amount", "sum"))
L = L.join(rr, on="order_item_id").fillna({"ru": 0, "rf": 0}); L = L[L.order_date <= "2026-05-16"]
g = L.groupby([L.is_first_order_of_person, L.disc > 0]).agg(units=("quantity", "sum"), ret=("ru", "sum"), refund=("rf", "sum"))
print("P06 mature cohort by first order x discounted:"); print(g.to_string())
fd = g.loc[(True, True)]; fn = g.loc[(True, False)]
excess_units = fd.ret - fd.units * (fn.ret / fn.units); ex_refund = excess_units * (fd.refund / fd.ret)
print(f"P06 first-order discounted: {fd.ret:.0f} of {fd.units:.0f} units returned ({fd.ret/fd.units:.3f}) vs first-order not discounted {fn.ret:.0f} of {fn.units:.0f} ({fn.ret/fn.units:.3f}); excess units {excess_units:.1f}, refund dollars {ex_refund:.2f}")
reg("P06 first-order discounted: unit return rate (mature)", round(fd.ret / fd.units, 4), f"{int(fd.ret)} / {int(fd.units)} units")
reg("P06 first-order not discounted: unit return rate (mature)", round(fn.ret / fn.units, 4), f"{int(fn.ret)} / {int(fn.units)} units")
reg("P06 excess refund dollars if discounted first orders returned at the not-discounted rate (mature cohort)", round(ex_refund, 2), f"{excess_units:.1f} excess units x {fd.refund/fd.ret:.2f}", "associative upper bound, not cause")
irr = cl["returns"].merge(cl["order_items"][["order_item_id", "product_id"]], on="order_item_id", how="left", suffixes=("", "_i"))
irr = irr[(irr.is_valid) & (irr.product_id == "P06")]
reg("P06 returns with reason irritation_breakout", int((irr.reason_clean == "irritation_breakout").sum()), f"/ {len(irr)} P06 return rows")
reg("P06 refund dollars for irritation_breakout", round(irr[irr.reason_clean == "irritation_breakout"].refund_amount.sum(), 2), f"of {irr.refund_amount.sum():.2f} P06 refunds")

# ---- R3: channels ----
chn = pd.read_csv(f"{C}/phase4/channel_unit_economics_mature.csv", index_col=0)
tk, gg = chn.loc["tiktok_ads"], chn.loc["google_ads"]
tk_c90 = tk.c90 * tk.buyers; short = tk_c90 - tk.spend_Jul_Mar
print(f"TikTok Jul-Mar cohorts: spend {tk.spend_Jul_Mar:.2f}, 90-day contribution of its buyers {tk_c90:.2f}, shortfall {short:.2f}; contribution per $1 spend {tk['c90/CAC']:.3f}")
reg("TikTok: 90-day contribution of Jul-Mar buyers minus Jul-Mar spend", round(short, 2), f"{tk_c90:.2f} - {tk.spend_Jul_Mar:.2f}", "mature cohort; before opex")
for nm in ["meta_ads", "influencer", "google_ads"]:
    x = chn.loc[nm]; reg(f"{nm}: 90-day contribution of Jul-Mar buyers minus Jul-Mar spend", round(x.c90 * x.buyers - x.spend_Jul_Mar, 2), f"{x.c90*x.buyers:.2f} - {x.spend_Jul_Mar:.2f}", "mature cohort; before opex")
gap = gg["c90/CAC"] - tk["c90/CAC"]; shift = 0.25 * float(pd.read_csv(f"{C}/phase4/cac_12m.csv", index_col=0).loc["tiktok_ads", "spend"])
ceiling = shift * gap
print(f"gap in 90-day contribution per $1 spend, Google minus TikTok: {gap:.3f}; 25% of 12-month TikTok spend = {shift:.2f}; CEILING if Google kept its average return at that spend: {ceiling:.2f}")
reg("25% of 12-month TikTok spend", round(shift, 2), "")
reg("ceiling: shifting that amount at Google's average 90-day return (assumes no diminishing returns)", round(ceiling, 2), f"{shift:.2f} x ({gg['c90/CAC']:.3f} - {tk['c90/CAC']:.3f})", "UPPER BOUND; Google marginal CAC is unknown and likely higher")

# ---- R4: shipping ----
US = OF[OF.ship_country == "US"].copy()
sh1 = cl["shipments"][cl["shipments"].is_valid].sort_values(["order_id", "ship_date", "shipment_id"]).groupby("order_id").first()
US = US.join(sh1[["shipment_cost"]], on="order_id")
US["free_code"] = US.codes.fillna("").str.contains("FREESHIP")
nf = US[~US.free_code].copy(); nf["g50"] = nf.gross >= 50; nf["n50"] = (nf.gross - nf.disc) >= 50
print("US orders without FREESHIP: shipping charged $0 by [gross>=50, net-of-discount>=50]:")
print(nf.groupby(["g50", "n50"]).agg(orders=("order_id", "size"), share_free=("shipping_charged", lambda s: (s == 0).mean()), avg_charged=("shipping_charged", "mean")).to_string())
nf["bin"] = pd.cut(nf.gross, list(range(30, 85, 5)), right=False)
bn = nf.groupby("bin", observed=True).size(); print("US order count by list-value bin ($5):"); print(bn.to_string())
below = bn.loc[pd.Interval(45, 50, closed="left")]; above = bn.loc[pd.Interval(50, 55, closed="left")]
print(f"orders in [45,50): {below}; in [50,55): {above}; ratio above/below {above/below:.3f}")
reg("US orders with list value in [45,50)", int(below), "")
reg("US orders with list value in [50,55)", int(above), f"ratio {above/below:.3f}", "bunching check at free-shipping threshold")
free_50_75 = nf[(nf.gross >= 50) & (nf.gross < 75) & (nf.shipping_charged == 0)]; free_75 = nf[(nf.gross >= 75) & (nf.shipping_charged == 0)]
ceil_50_75 = len(free_50_75) * 5.95; ceil_75 = len(free_75) * 5.95
print(f"US free-shipped orders without code: 50-75: {len(free_50_75)} -> ceiling at $5.95 each {ceil_50_75:.2f}; 75+: {len(free_75)} -> {ceil_75:.2f}")
reg("US free-shipped orders, list value 50-75", len(free_50_75), f"x $5.95 = {ceil_50_75:.2f} ceiling before any behaviour change", "UPPER BOUND")
reg("US free-shipped orders, list value 75+", len(free_75), f"x $5.95 = {ceil_75:.2f} ceiling before any behaviour change", "UPPER BOUND")
reg("US orders: avg shipment cost, list value 50-75", round(free_50_75.shipment_cost.mean(), 2), f"n={len(free_50_75)}")

# ---- R5: BFCM / event ----
mo = pd.read_csv(f"{C}/phase4/monthly_marketing.csv", index_col=0)
oct_, nov, dec = mo.loc["2025-10"], mo.loc["2025-11"], mo.loc["2025-12"]
d_orders = nov.orders / oct_.orders - 1; d_cam = nov.contribution_after_marketing / oct_.contribution_after_marketing - 1; d_spend = nov.spend / oct_.spend - 1
print(f"Nov vs Oct 2025: orders {d_orders:+.3f}, spend {d_spend:+.3f}, contribution after marketing {d_cam:+.3f} ({nov.contribution_after_marketing:.2f} vs {oct_.contribution_after_marketing:.2f}); Dec CAM {dec.contribution_after_marketing:.2f}")
reg("Nov vs Oct 2025: orders change", round(d_orders, 4), f"{int(nov.orders)} / {int(oct_.orders)} - 1")
reg("Nov vs Oct 2025: marketing spend change", round(d_spend, 4), f"{nov.spend:.2f} / {oct_.spend:.2f} - 1")
reg("Nov vs Oct 2025: contribution after marketing change", round(d_cam, 4), f"{nov.contribution_after_marketing:.2f} / {oct_.contribution_after_marketing:.2f} - 1", "order-month basis; Oct/Nov refunds mostly complete")
nd = OF[OF.month.isin(["2025-11", "2025-12"])]
b = nd[nd.codes == "BFCM30"]; n0 = nd[nd.group == "no_code"]
bs = nd[nd.codes.fillna("").str.contains("BFCM30") & nd.codes.fillna("").str.contains("WELCOME15")]
print(f"Nov-Dec orders: BFCM30-only n={len(b)} contribution/order {b.contribution.mean():.2f}, margin {b.contribution.sum()/b.net_rev.sum():.3f}, gross/order {b.gross.mean():.2f}; no_code n={len(n0)} contribution/order {n0.contribution.mean():.2f}, margin {n0.contribution.sum()/n0.net_rev.sum():.3f}, gross/order {n0.gross.mean():.2f}")
print(f"BFCM30 + WELCOME15 stacked: n={len(bs)} depth {bs.disc.sum()/bs.gross.sum():.3f}, contribution/order {bs.contribution.mean():.2f}")
reg("Nov-Dec: BFCM30-only orders contribution per order", round(b.contribution.mean(), 2), f"n={len(b)}", "NOT mix-controlled")
reg("Nov-Dec: no_code orders contribution per order", round(n0.contribution.mean(), 2), f"n={len(n0)}", "NOT mix-controlled")
reg("Nov-Dec: BFCM30-only orders gross list value per order", round(b.gross.mean(), 2), f"n={len(b)}")
reg("Nov-Dec: no_code orders gross list value per order", round(n0.gross.mean(), 2), f"n={len(n0)}")
reg("Nov-Dec: BFCM30+WELCOME15 stacked orders: depth (discount / gross)", round(bs.disc.sum() / bs.gross.sum(), 4), f"n={len(bs)}")

# ---- healthy checks (do not manufacture leaks) ----
reg("contribution margin (context)", float(P3["P3-003"]), "P3-003")
P = pd.DataFrame(REG, columns=["id", "metric", "value", "numerator_denominator", "filter"]); P.to_csv(f"{OUT}/number_register_phase5.csv", index=False)
pd.DataFrame(QA, columns=["check", "status", "detail"]).to_csv(f"{OUT}/qa_results.csv", index=False)
print("\nQA SUMMARY:", pd.Series([q[1] for q in QA]).value_counts().to_dict())
print("\nNUMBER REGISTER (Phase 5)"); print(P[["id", "metric", "value", "numerator_denominator"]].to_string(index=False))
