"""Phase 3 - Product, discount and return economics. Meadowline Skin Co. (SYNTHETIC demo data).
Input: clean tables from 02_clean.py. Valid orders only.
Contribution = net revenue + shipping charged - COGS(net of restocked, plus replacements) - postage - packaging - payment fees - return shipping.
NOT net profit: no operating expenses, no marketing in this file.
Run: python3 04_product_discount_return_economics.py [clean_dir]
"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

C = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs")
OUT = f"{C}/phase3"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
pd.options.display.float_format = "{:,.3f}".format

o = pd.read_csv(f"{C}/orders_clean.csv", parse_dates=["order_date"])
oi = pd.read_csv(f"{C}/order_items_clean.csv")
r = pd.read_csv(f"{C}/returns_clean.csv", parse_dates=["return_date"])
sh = pd.read_csv(f"{C}/shipments_clean.csv")
dc = pd.read_csv(f"{C}/discounts_clean.csv")
p = pd.read_csv(f"{C}/products_clean.csv")
pname = p.set_index("product_id").product_name.to_dict()

vo = o[o.is_valid].copy(); ids = set(vo.order_id)
L = oi[oi.is_valid].copy(); vr = r[r.is_valid].copy(); vs = sh[sh.is_valid].copy(); vd = dc[dc.is_valid].copy()
MATURE_CUTOFF = pd.Timestamp("2026-05-16")   # = last return date (2026-06-30) minus max observed days-to-return (45)

REG = []
def reg(metric, value, nd="", filt="valid orders"):
    REG.append((f"P3-{len(REG)+1:03d}", metric, value, nd, filt))
    return REG[-1][0]
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; d = 1 + z*z/n
    c = (ph + z*z/(2*n)) / d; h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / d
    return (c-h, c+h)

# =====================================================================
# 1. LINE-LEVEL CONTRIBUTION
# =====================================================================
L = L.merge(vo[["order_id", "person_id", "order_date", "month", "is_first_order_of_person", "shipping_charged", "payment_fee", "discount_total"]], on="order_id")
og = L.groupby("order_id").gross_line.sum().rename("order_gross")
L = L.join(og, on="order_id")
assert (L.order_gross > 0).all()
L["w"] = L.gross_line / L.order_gross                         # default allocation: proportional to gross line value
pu = L[~L.is_gift].groupby("order_id").quantity.sum().rename("order_paid_units")
L = L.join(pu, on="order_id")
L["w_units"] = np.where(L.is_gift, 0.0, L.quantity / L.order_paid_units)   # alternative allocation (sensitivity)

# returns -> line
rl = vr.groupby("order_item_id").agg(refund=("refund_amount", "sum"), ret_units=("quantity_returned", "sum"),
                                       ret_ship=("return_shipping_cost", "sum"), n_ret=("return_id", "size"))
L = L.join(rl, on="order_item_id").fillna({"refund": 0, "ret_units": 0, "ret_ship": 0, "n_ret": 0})
vr = vr.merge(L[["order_item_id", "unit_cogs"]], on="order_item_id", how="left")
vr["cogs_recovered"] = np.where(vr.restocked, vr.quantity_returned * vr.unit_cogs, 0.0)
vr["repl_cogs"] = np.where(vr.refund_type == "replacement", vr.quantity_returned * vr.unit_cogs, 0.0)
x = vr.groupby("order_item_id")[["cogs_recovered", "repl_cogs"]].sum()
L = L.join(x, on="order_item_id").fillna({"cogs_recovered": 0, "repl_cogs": 0})

# shipments: first shipment per order = original; later = replacement (paired 1:1 with replacement return rows)
vs = vs.sort_values(["order_id", "ship_date", "shipment_id"])
vs["rk"] = vs.groupby("order_id").cumcount() + 1
orig = vs[vs.rk == 1].set_index("order_id").shipment_cost.rename("orig_ship")
assert len(orig) == len(vo)
rep = vs[vs.rk > 1][["order_id", "shipment_cost", "postage_cost", "packaging_cost"]]
rr = vr[vr.refund_type == "replacement"][["order_id", "order_item_id"]]
assert len(rep) == len(rr) and set(rep.order_id) == set(rr.order_id) and rr.order_id.is_unique and rep.order_id.is_unique
rmap = rep.merge(rr, on="order_id").set_index("order_item_id").shipment_cost.rename("repl_ship")
L = L.join(orig, on="order_id").join(rmap, on="order_item_id").fillna({"repl_ship": 0})

L["net_rev"] = L.gross_line - L.line_discount - L.refund
L["cogs_net"] = L.cogs_line - L.cogs_recovered + L.repl_cogs
def contrib(wcol):
    return (L.net_rev + L.shipping_charged * L[wcol] - L.cogs_net - L.orig_ship * L[wcol] - L.repl_ship
            - L.payment_fee * L[wcol] - L.ret_ship)
L["ship_alloc"] = L.orig_ship * L.w + L.repl_ship
L["fee_alloc"] = L.payment_fee * L.w
L["shiprev_alloc"] = L.shipping_charged * L.w
L["contribution"] = contrib("w")
L["contribution_alt"] = contrib("w_units")

# ---- aggregate tie-out (independent path) ----
gross = L.gross_line.sum(); disc = L.line_discount.sum(); refunds = vr.refund_amount.sum()
net = gross - disc - refunds
cogs_total = L.cogs_line.sum(); recov = vr.cogs_recovered.sum(); repl = vr.repl_cogs.sum()
postage = vs.postage_cost.sum(); pack = vs.packaging_cost.sum()
fees = vo.payment_fee.sum(); rship = vr.return_shipping_cost.sum(); shiprev = vo.shipping_charged.sum()
C_total = net + shiprev - (cogs_total - recov + repl) - postage - pack - fees - rship
print("=== CONTRIBUTION BRIDGE (valid orders) ===")
br = pd.DataFrame([("Net revenue", net), ("+ Shipping charged to customers", shiprev), ("- COGS of items sold (incl. gifts)", -cogs_total),
                   ("+ COGS recovered (restocked returns)", recov), ("- COGS of replacements sent", -repl),
                   ("- Postage (all shipments incl. replacements)", -postage), ("- Packaging", -pack), ("- Payment fees", -fees),
                   ("- Return shipping paid by brand", -rship), ("= CONTRIBUTION (before marketing, before opex)", C_total)], columns=["line", "usd"])
print(br.round(2).to_string(index=False))
assert abs(L.contribution.sum() - C_total) < 0.05, (L.contribution.sum(), C_total)
assert abs(L.contribution_alt.sum() - C_total) < 0.05
print(f"[TIE-OUT] sum of line contributions {L.contribution.sum():.2f} == aggregate {C_total:.2f}; alt allocation {L.contribution_alt.sum():.2f}")
i_net = reg("net revenue (recomputed)", round(net, 2), "gross - discounts - refunds")
i_c = reg("CONTRIBUTION total (gift cogs $1.00 placeholder)", round(C_total, 2), f"{C_total:.2f}", "before marketing and opex")
i_cm = reg("contribution margin", round(C_total / net, 4), f"{C_total:.2f} / {net:.2f} net revenue")
i_cogs = reg("COGS share of net revenue (net COGS / net revenue)", round((cogs_total - recov + repl) / net, 4), f"{cogs_total-recov+repl:.2f} / {net:.2f}")
i_ship = reg("postage+packaging / net revenue", round((postage + pack) / net, 4), f"{postage+pack:.2f} / {net:.2f}")
i_shr = reg("shipping charged / (postage+packaging)", round(shiprev / (postage + pack), 4), f"{shiprev:.2f} / {postage+pack:.2f}", "shipping recovery ratio")
i_fee = reg("payment fees / net revenue", round(fees / net, 4), f"{fees:.2f} / {net:.2f}")
br.round(2).to_csv(f"{OUT}/contribution_bridge.csv", index=False)

# sensitivities on the headline
gift_units = L.loc[L.is_gift, "quantity"].sum(); p12_units = L.loc[(L.product_id == "P12"), "quantity"].sum()
print("\n[SENSITIVITY] contribution under assumptions:")
sens = []
for g in [0, 1, 2, 3]:
    c = C_total + gift_units * (1.0 - g); sens.append((f"gift cogs ${g}/unit", c, c / net))
for dlt in [-2, 2]:
    c = C_total - dlt * p12_units; sens.append((f"P12 cogs {21.1+dlt:.2f} (delta {dlt:+d})", c, c / net))
sens.append(("postage imputation alt (-2.91)", C_total + 2.91, (C_total + 2.91) / net))
S = pd.DataFrame(sens, columns=["case", "contribution", "margin"]); print(S.round(4).to_string(index=False))
reg("contribution if gift cogs = $0", round(C_total + gift_units, 2), "", "sensitivity")
reg("contribution if gift cogs = $3", round(C_total - 2 * gift_units, 2), "", "sensitivity")
reg("contribution if P12 cogs +$2", round(C_total - 2 * p12_units, 2), "", "sensitivity")
S.to_csv(f"{OUT}/sensitivity.csv", index=False)

# =====================================================================
# 2. PRODUCT ECONOMICS
# =====================================================================
def prod_table(df, alt=False):
    g = df.groupby("product_id")
    t = pd.DataFrame({
        "units": g.quantity.sum(), "lines": g.size(), "gross": g.gross_line.sum(), "discounts": g.line_discount.sum(),
        "refunds": g.refund.sum(), "ret_units": g.ret_units.sum(), "net_rev": g.net_rev.sum(),
        "cogs_net": g.cogs_net.sum(), "ship_alloc": g.ship_alloc.sum(), "fee_alloc": g.fee_alloc.sum(),
        "shiprev": g.shiprev_alloc.sum(), "ret_ship": g.ret_ship.sum(),
        "contribution": g.contribution_alt.sum() if alt else g.contribution.sum()})
    return t
T = prod_table(L)
T["disc_rate"] = T.discounts / T.gross
T["refund_rate_value"] = T.refunds / (T.gross - T.discounts)
T["unit_return_rate"] = T.ret_units / T.units
T["margin"] = T.contribution / T.net_rev
T["contrib_per_unit"] = T.contribution / T.units
T["share_net_rev"] = T.net_rev / T.net_rev.sum()
T["share_contrib"] = T.contribution / T.contribution.sum()
T["rank_rev"] = T.net_rev.rank(ascending=False, method="min")
T["rank_contrib"] = T.contribution.rank(ascending=False, method="min")
T["avg_list_price"] = T.gross / T.units
T["cogs_per_unit"] = T.cogs_net / T.units
T.insert(0, "name", [pname[i] for i in T.index])
Talt = prod_table(L, alt=True)
T["rank_contrib_alt"] = Talt.contribution.rank(ascending=False, method="min")
T["margin_alt"] = Talt.contribution / Talt.net_rev
T.loc["P13", ["margin", "margin_alt", "refund_rate_value", "disc_rate", "unit_return_rate"]] = np.nan
print("\n=== PRODUCT ECONOMICS (alloc: proportional to gross line value; P13 = free gift, no revenue) ===")
show = ["name", "units", "gross", "disc_rate", "refund_rate_value", "unit_return_rate", "net_rev", "cogs_net", "ship_alloc", "contribution", "margin", "contrib_per_unit", "share_net_rev", "share_contrib", "rank_rev", "rank_contrib", "rank_contrib_alt", "margin_alt"]
T2 = T.sort_values("contribution", ascending=False)
T2["name"] = T2.name.str[:28]
print(T2[show].round(3).to_string())
print("[TIE-OUT] product contribution sum:", round(T.contribution.sum(), 2))
assert abs(T.contribution.sum() - C_total) < 0.05
T.round(4).to_csv(f"{OUT}/product_economics.csv")
top = T.drop("P13").sort_values("contribution", ascending=False)
reg("top product by contribution: " + top.index[0], round(top.contribution.iloc[0], 2), f"share of total contribution {top.share_contrib.iloc[0]:.4f}")
reg("P12 set margin", round(T.loc['P12', 'margin'], 4), f"{T.loc['P12','contribution']:.2f} / {T.loc['P12','net_rev']:.2f}")
reg("lowest-margin product (excl. gift): " + T.drop('P13').margin.idxmin(), round(T.drop('P13').margin.min(), 4), "contribution / net revenue of the product")
reg("highest-margin product (excl. gift): " + T.drop('P13').margin.idxmax(), round(T.drop('P13').margin.max(), 4), "contribution / net revenue of the product")
reg("gift program cost (277 units x $1.00 placeholder)", round(T.loc['P13', 'cogs_net'], 2), f"{int(gift_units)} units", "ASSUMPTION A2")
rank_moves = (T.drop("P13").rank_rev != T.drop("P13").rank_contrib).sum()
reg("products whose revenue rank differs from contribution rank (of 12)", int(rank_moves), "")
alt_moves = (T.drop("P13").rank_contrib != T.drop("P13").rank_contrib_alt).sum()
reg("products whose contribution rank changes with units-based allocation (of 12)", int(alt_moves), "allocation sensitivity")

# price-change effect for P03/P12 (descriptive)
L["price_period"] = np.where(L.order_date < "2026-01-15", "before", "from 2026-01-15")
pc = L[L.product_id.isin(["P03", "P12"])].groupby(["product_id", "price_period"]).agg(units=("quantity", "sum"), gross=("gross_line", "sum"), net_rev=("net_rev", "sum"), contribution=("contribution", "sum"))
pc["contrib_per_unit"] = pc.contribution / pc.units
print("\n[CONTEXT] P03/P12 before vs after list-price change:"); print(pc.round(2).to_string())
pc.round(3).to_csv(f"{OUT}/price_change_context.csv")

# =====================================================================
# 3. ORDER-LEVEL FRAME
# =====================================================================
ag = L.groupby("order_id").agg(gross=("gross_line", "sum"), disc=("line_discount", "sum"), refund=("refund", "sum"), net_rev=("net_rev", "sum"),
                               cogs_net=("cogs_net", "sum"), ship_alloc=("ship_alloc", "sum"), ret_ship=("ret_ship", "sum"),
                               contribution=("contribution", "sum"), ret_units=("ret_units", "sum"),
                               has_set=("product_id", lambda s: (s == "P12").any()),
                               paid_units=("quantity", lambda s: s[~L.loc[s.index, "is_gift"]].sum()))
O = vo.set_index("order_id").join(ag)
assert abs(O.contribution.sum() - C_total) < 0.05
codes = vd.groupby("order_id").code_clean.apply(lambda s: "+".join(sorted(s)))
O["codes"] = codes.reindex(O.index)
O["has_item_disc"] = O.discount_total > 0
O["has_freeship"] = O.codes.fillna("").str.contains("FREESHIP")
O["group"] = np.select([O.has_item_disc, O.codes.notna()], ["item_discount", "freeship_only"], "no_code")
O["units_b"] = pd.cut(O.paid_units, [0, 1, 2, 99], labels=["1", "2", "3+"])
O["cust"] = np.where(O.is_first_order_of_person, "first", "repeat")
print("\n=== ORDER GROUPS (valid orders) ==="); print(O.groupby("group").size().to_string())
O.reset_index().to_csv(f"{OUT}/order_frame.csv", index=False)

# =====================================================================
# 4. DISCOUNT ECONOMICS
# =====================================================================
def summ(g):
    return pd.Series({"orders": len(g), "gross": g.gross.sum(), "disc": g.disc.sum(), "net_rev": g.net_rev.sum(), "contribution": g.contribution.sum(),
                      "depth": g.disc.sum() / g.gross.sum(), "aov_gross": g.gross.mean(), "units_per_order": g.paid_units.mean(),
                      "contrib_per_order": g.contribution.mean(), "margin": g.contribution.sum() / g.net_rev.sum(),
                      "refund_rate_value": g.refund.sum() / (g.gross.sum() - g.disc.sum()), "first_share": g.is_first_order_of_person.mean()})
print("\n=== A. DISCOUNT GROUPS (pooled, NOT mix-controlled) ===")
GA = O.groupby("group").apply(summ); print(GA.round(3).to_string()); GA.round(4).to_csv(f"{OUT}/discount_groups_pooled.csv")
tot_orders = len(O)
share_disc_orders = (O.group == "item_discount").mean()
reg("share of orders with an item discount", round(share_disc_orders, 4), f"{int((O.group=='item_discount').sum())} / {tot_orders} valid orders")
reg("pooled margin: no_code orders", round(GA.loc['no_code', 'margin'], 4), f"{GA.loc['no_code','contribution']:.2f} / {GA.loc['no_code','net_rev']:.2f}")
reg("pooled margin: item_discount orders", round(GA.loc['item_discount', 'margin'], 4), f"{GA.loc['item_discount','contribution']:.2f} / {GA.loc['item_discount','net_rev']:.2f}")
reg("pooled contribution per order: no_code", round(GA.loc['no_code', 'contrib_per_order'], 2), f"/ {int(GA.loc['no_code','orders'])} orders")
reg("pooled contribution per order: item_discount", round(GA.loc['item_discount', 'contrib_per_order'], 2), f"/ {int(GA.loc['item_discount','orders'])} orders")
reg("total item-discount dollars", round(disc, 2), "")
reg("item-discount dollars / gross sales", round(disc / gross, 4), f"{disc:.2f} / {gross:.2f}")

# --- by code family ---
codes_m = vd[vd.discount_type == "percentage"].merge(vo[["order_id", "month"]], on="order_id")
cm = codes_m.groupby("code_clean").agg(rows=("order_id", "size"), months=("month", "nunique"), first_m=("month", "min"), last_m=("month", "max"), pct=("percentage", "median"), amt=("amount", "sum"))
def fam(c, months):
    if c == "WELCOME15": return "welcome"
    if c == "LOYAL10": return "loyalty"
    return "event" if months <= 2 else "creator_or_other"
cm["family"] = [fam(c, m) for c, m in zip(cm.index, cm.months)]
print("\n=== B. CODES (line-item discounts; amt = discount $ recorded in discounts.csv) ==="); print(cm.sort_values("amt", ascending=False).round(2).to_string())
cm.round(3).to_csv(f"{OUT}/codes.csv")
fam_map = cm.family.to_dict()
# per-order primary family: single-family orders only; stacked ones flagged
vdp = vd[vd.discount_type == "percentage"].copy(); vdp["family"] = vdp.code_clean.map(fam_map)
fo = vdp.groupby("order_id").family.apply(lambda s: "+".join(sorted(set(s))))
O["family"] = fo.reindex(O.index).fillna("none")
GF = O[O.has_item_disc].groupby("family").apply(summ)
print("\n=== C. FAMILY of item-discount orders (pooled) ==="); print(GF.round(3).to_string()); GF.round(4).to_csv(f"{OUT}/discount_families.csv")
share_by_fam = (GF.disc / GF.disc.sum()).round(4)
for f_, v_ in share_by_fam.items(): reg(f"share of item-discount dollars: family {f_}", float(v_), f"{GF.loc[f_,'disc']:.2f} / {GF.disc.sum():.2f}")

# --- CROSS-ANALYSIS 1: discounted vs non-discounted after mix control ---
print("\n=== CROSS-ANALYSIS 1: item_discount vs no_code, stratified by customer type x paid units x has set (cells need n>=30 on BOTH sides) ===")
S = O[O.group.isin(["item_discount", "no_code"])].copy()
S["cell"] = S.cust + "|u" + S.units_b.astype(str) + "|set" + S.has_set.astype(int).astype(str)
rows = []
for c, g in S.groupby("cell"):
    a, b = g[g.group == "item_discount"], g[g.group == "no_code"]
    rows.append({"cell": c, "n_disc": len(a), "n_nocode": len(b), "ok": len(a) >= 30 and len(b) >= 30,
                 "gross_disc": a.gross.mean(), "gross_no": b.gross.mean(), "cpo_disc": a.contribution.mean() if len(a) else np.nan,
                 "cpo_no": b.contribution.mean() if len(b) else np.nan, "margin_disc": a.contribution.sum() / a.net_rev.sum() if len(a) else np.nan,
                 "margin_no": b.contribution.sum() / b.net_rev.sum() if len(b) else np.nan,
                 "depth_disc": a.disc.sum() / a.gross.sum() if len(a) else np.nan})
CX = pd.DataFrame(rows); print(CX.round(3).to_string(index=False)); CX.round(4).to_csv(f"{OUT}/cross1_cells.csv", index=False)
ok = CX[CX.ok]
cov_disc = ok.n_disc.sum() / CX.n_disc.sum(); cov_no = ok.n_nocode.sum() / CX.n_nocode.sum()
w = ok.n_disc / ok.n_disc.sum()           # weights = mix of DISCOUNTED orders (ATT-style standardisation)
std_cpo_d = (w * ok.cpo_disc).sum(); std_cpo_n = (w * ok.cpo_no).sum()
std_gross_d = (w * ok.gross_disc).sum(); std_gross_n = (w * ok.gross_no).sum()
# margin standardised: weighted contribution / weighted net revenue need net rev per order
def cellmean(g, col): return g[col].mean()
nr_d = np.array([S[(S.cell == c) & (S.group == 'item_discount')].net_rev.mean() for c in ok.cell]); nr_n = np.array([S[(S.cell == c) & (S.group == 'no_code')].net_rev.mean() for c in ok.cell])
std_m_d = (w * ok.cpo_disc).sum() / (w * nr_d).sum(); std_m_n = (w * ok.cpo_no).sum() / (w * nr_n).sum()
pool_cpo_d = S[S.group == "item_discount"].contribution.mean(); pool_cpo_n = S[S.group == "no_code"].contribution.mean()
# bootstrap CI for standardised difference in contribution per order (resample orders within group)
rng = np.random.default_rng(42); diffs = []
arr_d = [S[(S.cell == c) & (S.group == "item_discount")].contribution.values for c in ok.cell]
arr_n = [S[(S.cell == c) & (S.group == "no_code")].contribution.values for c in ok.cell]
for _ in range(2000):
    dd = [rng.choice(a_, len(a_)).mean() - rng.choice(b_, len(b_)).mean() for a_, b_ in zip(arr_d, arr_n)]
    diffs.append(float((w.values * np.array(dd)).sum()))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\ncells usable: {len(ok)}/{len(CX)} | coverage: discounted orders {cov_disc:.3f}, no_code orders {cov_no:.3f}")
print(f"POOLED   contribution/order: discounted {pool_cpo_d:.2f} vs no_code {pool_cpo_n:.2f} (diff {pool_cpo_d-pool_cpo_n:+.2f})")
print(f"STANDARD contribution/order (mix of discounted orders): discounted {std_cpo_d:.2f} vs no_code {std_cpo_n:.2f} (diff {std_cpo_d-std_cpo_n:+.2f}; bootstrap 95% CI {lo:+.2f}..{hi:+.2f})")
print(f"STANDARD gross list value/order: discounted {std_gross_d:.2f} vs no_code {std_gross_n:.2f}")
print(f"STANDARD margin (contribution / net revenue): discounted {std_m_d:.4f} vs no_code {std_m_n:.4f}")
print("Simpson check: sign of pooled diff =", np.sign(pool_cpo_d - pool_cpo_n), "| sign of standardised diff =", np.sign(std_cpo_d - std_cpo_n),
      "| cells where discounted cpo > no_code cpo:", int((ok.cpo_disc > ok.cpo_no).sum()), "of", len(ok))
reg("CX1 cells usable (n>=30 both sides)", f"{len(ok)} of {len(CX)}", "")
reg("CX1 coverage of discounted orders by usable cells", round(cov_disc, 4), f"{int(ok.n_disc.sum())} / {int(CX.n_disc.sum())}")
reg("CX1 pooled contribution/order, discounted", round(pool_cpo_d, 2), "")
reg("CX1 pooled contribution/order, no_code", round(pool_cpo_n, 2), "")
reg("CX1 standardised contribution/order, discounted", round(std_cpo_d, 2), "weights = mix of discounted orders in usable cells")
reg("CX1 standardised contribution/order, no_code", round(std_cpo_n, 2), "same weights")
reg("CX1 standardised difference (disc - no_code), 95% bootstrap CI low", round(lo, 2), "2000 resamples within cell x group")
reg("CX1 standardised difference (disc - no_code), 95% bootstrap CI high", round(hi, 2), "")
reg("CX1 standardised gross list value/order, discounted", round(std_gross_d, 2), "")
reg("CX1 standardised gross list value/order, no_code", round(std_gross_n, 2), "")
reg("CX1 standardised margin, discounted", round(std_m_d, 4), "contribution / net revenue")
reg("CX1 standardised margin, no_code", round(std_m_n, 4), "contribution / net revenue")

# --- CROSS-ANALYSIS 2: first-order discount depth vs repeat within 90 days ---
print("\n=== CROSS-ANALYSIS 2: first-order discount depth vs 2nd order within 90 days (first orders 2025-07-01..2026-03-31) ===")
F = O[O.is_first_order_of_person & (O.order_date <= "2026-03-31")].copy()
F["depth"] = F.disc / F.gross
F["bucket"] = pd.cut(F.depth, [-1e-9, 1e-9, 0.13, 0.17, 0.26, 1.0], labels=["none", "<=12%", "~15%", "16-25%", ">=26%"])
nxt = vo.sort_values("created_utc" if "created_utc" in vo else "order_date").groupby("person_id").order_date.apply(list)
def repeat90(row):
    for d in nxt[row.person_id]:
        if d > row.order_date and (d - row.order_date).days <= 90: return True
    return False
F["rep90"] = F.apply(repeat90, axis=1)
F["peak"] = F.month.isin(["2025-11", "2025-12"])
def tab(df):
    g = df.groupby("bucket", observed=True).agg(n=("rep90", "size"), repeaters=("rep90", "sum"), gross=("gross", "mean"), contrib_first=("contribution", "mean"))
    g["rate"] = g.repeaters / g.n
    ci = [wilson(k, n) for k, n in zip(g.repeaters, g.n)]; g["ci_lo"] = [c[0] for c in ci]; g["ci_hi"] = [c[1] for c in ci]; g["usable"] = g.n >= 30
    return g
print("ALL first-order cohorts:"); TA = tab(F); print(TA.round(3).to_string())
print("EXCLUDING Nov-Dec 2025 cohorts (BFCM/holiday):"); TB = tab(F[~F.peak]); print(TB.round(3).to_string())
TA.round(4).to_csv(f"{OUT}/cross2_all.csv"); TB.round(4).to_csv(f"{OUT}/cross2_offpeak.csv")
print("By acquisition-month season only none-vs-any (off-peak):")
F["any"] = np.where(F.depth > 1e-9, "discounted", "none")
t2 = F[~F.peak].groupby("any").agg(n=("rep90", "size"), rate=("rep90", "mean")); print(t2.round(3).to_string())
for b_, rw in TA.iterrows(): reg(f"CX2 repeat<=90d rate, first-order depth {b_} (all cohorts)", round(rw.rate, 4), f"{int(rw.repeaters)} / {int(rw.n)} first orders")
for b_, rw in TB.iterrows(): reg(f"CX2 repeat<=90d rate, first-order depth {b_} (excl Nov-Dec)", round(rw.rate, 4), f"{int(rw.repeaters)} / {int(rw.n)} first orders")
reg("CX2 overall repeat<=90d rate (all cohorts)", round(F.rep90.mean(), 4), f"{int(F.rep90.sum())} / {len(F)} first orders")

# =====================================================================
# 5. RETURN ECONOMICS
# =====================================================================
print("\n=== RETURN ECONOMICS ===")
R = vr.merge(L[["order_item_id", "product_id", "order_date", "is_first_order_of_person", "discount_total", "gross_line", "line_discount", "unit_cogs", "quantity"]], on="order_item_id", how="left", suffixes=("", "_l"))
R["repl_ship"] = R.order_item_id.map(rmap).fillna(0)
R["impact"] = R.refund_amount + R.return_shipping_cost + R.repl_cogs + R.repl_ship - R.cogs_recovered
R["group"] = np.where(R.refund_type == "replacement", "replacement", np.where(R.return_method == "returnless_refund", "refund_returnless", "refund_shipped_back"))
ty = R.groupby("refund_type").agg(rows=("return_id", "size"), units=("quantity_returned", "sum"), refund=("refund_amount", "sum"), ret_ship=("return_shipping_cost", "sum"),
                                  repl_cost=("repl_cogs", "sum"), repl_ship=("repl_ship", "sum"), cogs_back=("cogs_recovered", "sum"), impact=("impact", "sum"))
print("-- by refund_type"); print(ty.round(2).to_string())
mt = R.groupby("group").agg(rows=("return_id", "size"), refund=("refund_amount", "sum"), restocked_rows=("restocked", "sum"), impact=("impact", "sum")); print("-- by handling"); print(mt.round(2).to_string())
tot_impact = R.impact.sum()
print(f"TOTAL economic impact of returns: {tot_impact:.2f} = refunds {R.refund_amount.sum():.2f} + return ship {R.return_shipping_cost.sum():.2f} + replacement COGS {R.repl_cogs.sum():.2f} + replacement shipping {R.repl_ship.sum():.2f} - COGS recovered {R.cogs_recovered.sum():.2f}")
contrib_before_returns = C_total + tot_impact
reg("returns: total economic impact on contribution", round(tot_impact, 2), "refunds + return ship + replacement COGS + replacement shipping - recovered COGS")
reg("returns impact / contribution before returns", round(tot_impact / contrib_before_returns, 4), f"{tot_impact:.2f} / {contrib_before_returns:.2f}")
reg("returns impact / net sales before refunds (gross-disc)", round(tot_impact / (gross - disc), 4), f"{tot_impact:.2f} / {gross-disc:.2f}")
reg("share of return rows paid as returnless refund", round((R.return_method == 'returnless_refund').mean(), 4), f"{int((R.return_method=='returnless_refund').sum())} / {len(R)} return rows")
rl_ref = R.loc[(R.return_method == 'returnless_refund') & (R.refund_type != 'replacement'), 'refund_amount'].sum()
reg("returnless refund dollars / all refund dollars", round(rl_ref / R.refund_amount.sum(), 4), f"{rl_ref:.2f} / {R.refund_amount.sum():.2f}")
reg("restocked units / returned units (refund rows only, excl replacements)", round(R[R.refund_type != 'replacement'].query('restocked').quantity_returned.sum() / R[R.refund_type != 'replacement'].quantity_returned.sum(), 4),
    f"{int(R[R.refund_type != 'replacement'].query('restocked').quantity_returned.sum())} / {int(R[R.refund_type != 'replacement'].quantity_returned.sum())}")
reg("COGS written off on refunded-not-restocked units", round((R[(R.refund_type != 'replacement') & (~R.restocked)].quantity_returned * R[(R.refund_type != 'replacement') & (~R.restocked)].unit_cogs).sum(), 2), "included in net COGS already")

# by reason
RR = R.groupby("reason_clean").agg(rows=("return_id", "size"), refund=("refund_amount", "sum"), impact=("impact", "sum"), shipped_back=("return_method", lambda s: (s == "shipped_back").mean()),
                                   replacement_share=("refund_type", lambda s: (s == "replacement").mean()))
RR["share_rows"] = RR.rows / RR.rows.sum(); RR["share_impact"] = RR.impact / RR.impact.sum(); RR["impact_per_return"] = RR.impact / RR.rows
print("-- by reason"); print(RR.sort_values("impact", ascending=False).round(3).to_string()); RR.round(4).to_csv(f"{OUT}/returns_by_reason.csv")
fulfil = RR.loc[["damaged_leaked", "wrong_item"], "rows"].sum()
reg("return rows with fulfilment reasons (damaged_leaked + wrong_item)", int(fulfil), f"{int(fulfil)} / {len(R)} return rows = {fulfil/len(R):.4f}")
reg("impact share of fulfilment reasons", round(RR.loc[['damaged_leaked', 'wrong_item'], 'impact'].sum() / RR.impact.sum(), 4), f"{RR.loc[['damaged_leaked','wrong_item'],'impact'].sum():.2f} / {RR.impact.sum():.2f}")

# unit return rates: all vs mature cohort
L["mature"] = L.order_date <= MATURE_CUTOFF
def rate_tab(df, by):
    g = df.groupby(by).agg(units=("quantity", "sum"), ret_units=("ret_units", "sum"))
    g["rate"] = g.ret_units / g.units
    ci = [wilson(k, n) for k, n in zip(g.ret_units, g.units)]; g["ci_lo"] = [c[0] for c in ci]; g["ci_hi"] = [c[1] for c in ci]; g["usable"] = g.units >= 30
    return g
Lp = L[~L.is_gift]
print("\n-- unit return rate: ALL vs MATURE (orders <= 2026-05-16) by product")
a = rate_tab(Lp, "product_id"); m_ = rate_tab(Lp[Lp.mature], "product_id")
PR = a[["units", "ret_units", "rate"]].join(m_[["units", "ret_units", "rate", "ci_lo", "ci_hi"]], rsuffix="_mature")
PR["name"] = [pname[i][:26] for i in PR.index]
print(PR.sort_values("rate_mature", ascending=False).round(3).to_string()); PR.round(4).to_csv(f"{OUT}/returns_by_product.csv")
tot_m = rate_tab(Lp[Lp.mature].assign(k=1), "k")
reg("unit return rate, mature cohort (orders <= 2026-05-16)", round(float(tot_m.rate.iloc[0]), 4), f"{int(tot_m.ret_units.iloc[0])} / {int(tot_m.units.iloc[0])} paid units", "returned units / paid units; all return types")
tot_a = rate_tab(Lp.assign(k=1), "k")
reg("unit return rate, all orders (censored)", round(float(tot_a.rate.iloc[0]), 4), f"{int(tot_a.ret_units.iloc[0])} / {int(tot_a.units.iloc[0])}")
Mt = Lp[Lp.mature]
hi_p = PR.sort_values("rate_mature", ascending=False)
for pid in list(hi_p.index[:3]) + list(hi_p.index[-2:]):
    reg(f"mature unit return rate {pid} {pname[pid][:22]}", round(hi_p.loc[pid, 'rate_mature'], 4), f"{int(hi_p.loc[pid,'ret_units_mature'])} / {int(hi_p.loc[pid,'units_mature'])} units; 95% CI {hi_p.loc[pid,'ci_lo']:.3f}-{hi_p.loc[pid,'ci_hi']:.3f}")
# chi-square-like check: is spread across products larger than chance? (permutation)
obs = (m_.rate.max() - m_.rate.min()); rng = np.random.default_rng(7)
pool_rate = m_.ret_units.sum() / m_.units.sum(); perm = []
for _ in range(2000):
    sim = rng.binomial(m_.units.values.astype(int), pool_rate) / m_.units.values
    perm.append(sim.max() - sim.min())
pval = (np.array(perm) >= obs).mean()
print(f"[TEST] max-min spread of product return rates (mature) = {obs:.4f}; share of binomial simulations with spread >= observed: {pval:.4f}")
reg("product return-rate spread (max-min), mature", round(obs, 4), f"simulation p={pval:.3f} under one common rate")

# by customer type, discount status (within product), reason by product
print("\n-- by customer type (mature, paid units)")
Mt = Mt.assign(cust=np.where(Mt.is_first_order_of_person, "first", "repeat"), disc_status=np.where(Mt.discount_total > 0, "item_discount", "no_discount"))
ct = rate_tab(Mt, "cust"); print(ct.round(4).to_string())
for k_, rw in ct.iterrows(): reg(f"mature unit return rate, {k_} orders", round(rw.rate, 4), f"{int(rw.ret_units)} / {int(rw.units)} units")
print("\n-- CROSS-ANALYSIS 3: discount status, standardised by product (mature, paid units, product cells need >=30 units both sides)")
cell = Mt.groupby(["product_id", "disc_status"]).agg(units=("quantity", "sum"), ret=("ret_units", "sum")).unstack()
cell.columns = [f"{a}_{b}" for a, b in cell.columns]
cell["rate_d"] = cell.ret_item_discount / cell.units_item_discount; cell["rate_n"] = cell.ret_no_discount / cell.units_no_discount
cell["ok"] = (cell.units_item_discount >= 30) & (cell.units_no_discount >= 30)
print(cell.round(3).to_string()); cell.round(4).to_csv(f"{OUT}/cross3_cells.csv")
ok3 = cell[cell.ok]; w3 = ok3.units_item_discount / ok3.units_item_discount.sum()
std_d = (w3 * ok3.rate_d).sum(); std_n = (w3 * ok3.rate_n).sum()
pool_d = Mt[Mt.disc_status == "item_discount"].ret_units.sum() / Mt[Mt.disc_status == "item_discount"].quantity.sum()
pool_n = Mt[Mt.disc_status == "no_discount"].ret_units.sum() / Mt[Mt.disc_status == "no_discount"].quantity.sum()
print(f"POOLED return rate: discounted {pool_d:.4f} vs no_discount {pool_n:.4f} | STANDARDISED by product: {std_d:.4f} vs {std_n:.4f} | cells {len(ok3)}/{len(cell)}")
reg("CX3 pooled mature return rate, discounted lines", round(pool_d, 4), "returned units / paid units")
reg("CX3 pooled mature return rate, non-discounted lines", round(pool_n, 4), "returned units / paid units, non-discounted lines, orders <= 2026-05-16")
reg("CX3 product-standardised return rate, discounted", round(std_d, 4), f"weights = discounted units, {len(ok3)} usable product cells")
reg("CX3 product-standardised return rate, non-discounted", round(std_n, 4), "same weights")

# monthly return rate by order month (shows censoring)
mo = Lp.groupby("month").agg(units=("quantity", "sum"), ret=("ret_units", "sum")); mo["rate"] = mo.ret / mo.units
print("\n-- unit return rate by ORDER month (recent months censored):"); print(mo.round(4).T.to_string())
mo.round(4).to_csv(f"{OUT}/returns_by_order_month.csv")

# concentration: top-3 products share of returned value vs share of net sales
cons = R.groupby("product_id").refund_amount.sum(); sal = (L.groupby("product_id").gross_line.sum() - L.groupby("product_id").line_discount.sum())
cdf = pd.DataFrame({"refund_share": cons / cons.sum(), "sales_share": sal / sal.sum()}).dropna(); cdf["index"] = cdf.refund_share / cdf.sales_share
print("\n-- refund share vs net-sales-before-refunds share by product"); print(cdf.sort_values("index", ascending=False).round(3).to_string())
cdf.round(4).to_csv(f"{OUT}/returns_concentration.csv")
t3 = cdf.sort_values("refund_share", ascending=False).head(3)
reg("top-3 products' share of refund dollars", round(t3.refund_share.sum(), 4), f"{t3.index.tolist()}")
reg("same 3 products' share of net sales before refunds", round(t3.sales_share.sum(), 4), "net sales before refunds of these 3 products / all products")
# reasons x product for top return product (n rule)
print("\n-- top return reasons within product (only reason cells with >=30 rows are interpretable)")
print(R.groupby(["product_id", "reason_clean"]).size().unstack(fill_value=0).to_string())

# =====================================================================
# 6. OUTPUT
# =====================================================================
P = pd.DataFrame(REG, columns=["id", "metric", "value", "numerator_denominator", "filter"]); P["period"] = "2025-07-01..2026-06-30"; P["source_script"] = "04_product_discount_return_economics.py"
P.to_csv(f"{OUT}/number_register_phase3.csv", index=False)
print("\nNUMBER REGISTER (Phase 3)"); print(P[["id", "metric", "value", "numerator_denominator"]].to_string(index=False))
