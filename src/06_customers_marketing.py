"""Phase 4 - Customer and marketing economics. Meadowline Skin Co. (SYNTHETIC demo data).
Inputs: clean tables (02_clean.py) and order_frame.csv (04_product_discount_return_economics.py; per-order contribution).
Contribution = before marketing and opex. NOT net profit. Customer value = observed in window, not LTV.
Run: python3 06_customers_marketing.py [clean_dir]
"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd

C = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs")
OUT = f"{C}/phase4"; os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
pd.options.display.float_format = "{:,.3f}".format

OF = pd.read_csv(f"{C}/phase3/order_frame.csv", parse_dates=["order_date"])
oi = pd.read_csv(f"{C}/order_items_clean.csv"); r = pd.read_csv(f"{C}/returns_clean.csv")
pe = pd.read_csv(f"{C}/persons_clean.csv"); cr = pd.read_csv(f"{C}/customers_records_clean.csv", parse_dates=["created_utc"])
ms = pd.read_csv(f"{C}/marketing_spend_clean.csv")
assert OF.is_valid.all()
REG = []
def reg(metric, value, nd="", filt="valid orders"):
    REG.append((f"P4-{len(REG)+1:03d}", metric, value, nd, filt)); return REG[-1][0]
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; d = 1 + z*z/n; c = (ph + z*z/(2*n)) / d; h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / d
    return (c-h, c+h)

# ---------------- person table ----------------
OF = OF.sort_values(["person_id", "order_date", "order_id"])
g = OF.groupby("person_id")
PT = pd.DataFrame({"n_orders": g.size(), "net_rev": g.net_rev.sum(), "contribution": g.contribution.sum(),
                   "first_date": g.order_date.min()})
PT["first_month"] = PT.first_date.dt.to_period("M").astype(str)
second = OF[OF.order_seq_person == 2].set_index("person_id").order_date
PT["second_date"] = second.reindex(PT.index)
PT["days_to_second"] = (PT.second_date - PT.first_date).dt.days
first_o = OF[OF.order_seq_person == 1].set_index("person_id")
PT["first_codes"] = first_o.codes.reindex(PT.index)
PT["first_welcome"] = PT.first_codes.fillna("").str.contains("WELCOME15")
PT["first_group"] = first_o.group.reindex(PT.index)
PT["first_net_sales"] = (first_o.gross - first_o.disc).reindex(PT.index)
PT["first_contribution"] = first_o.contribution.reindex(PT.index)
PT["first_order_id"] = first_o.order_id.reindex(PT.index)
assert len(PT) == 3828 and (PT.n_orders >= 1).all()
# source: raw label of the record that was used (earliest non-unknown record)
cr = cr.sort_values(["email_hash", "created_utc"])
known = cr[cr.source_clean != "unknown"].groupby("email_hash").first()[["acquisition_source", "source_clean"]]
known.columns = ["raw_label", "source"]
PT = PT.join(known, how="left"); PT["source"] = PT.source.fillna("unknown")
PT["ambiguous"] = PT.raw_label.isin(["IG", "Google"])
PT = PT.join(pe.set_index("person_id")[["accepts_marketing", "created_utc"]], how="left")
print("=== persons with a valid order:", len(PT), "| source distribution (clean):"); print(PT.source.value_counts().to_string())
print("ambiguous labels (IG, Google) among these persons:", int(PT.ambiguous.sum()))
reg("buyers with unknown source", int((PT.source == "unknown").sum()), f"/ {len(PT)} buyers = {(PT.source=='unknown').mean():.4f}")
reg("buyers with ambiguous label (IG/Google)", int(PT.ambiguous.sum()), f"/ {len(PT)} buyers = {PT.ambiguous.mean():.4f}")
PT.reset_index().to_csv(f"{OUT}/person_table.csv", index=False)

# ---------------- 1. new vs repeat, concentration ----------------
print("\n=== 1. NEW vs REPEAT (observed in window; NOT LTV) ===")
npers = len(PT); n_ord = len(OF)
rep_p = int((PT.n_orders >= 2).sum())
print("orders per person:"); print(PT.n_orders.value_counts().sort_index().to_string())
reg("buyers with >=2 orders in window (censored lower bound)", rep_p, f"/ {npers} buyers = {rep_p/npers:.4f}")
fo = OF[OF.order_seq_person == 1]; ro = OF[OF.order_seq_person >= 2]
tab = pd.DataFrame({"orders": [len(fo), len(ro)], "net_rev": [fo.net_rev.sum(), ro.net_rev.sum()], "contribution": [fo.contribution.sum(), ro.contribution.sum()],
                    "contrib_per_order": [fo.contribution.mean(), ro.contribution.mean()], "margin": [fo.contribution.sum()/fo.net_rev.sum(), ro.contribution.sum()/ro.net_rev.sum()],
                    "item_discount_share": [(fo.group == "item_discount").mean(), (ro.group == "item_discount").mean()]}, index=["first order", "repeat order"])
print(tab.round(3).to_string()); tab.round(4).to_csv(f"{OUT}/first_vs_repeat_orders.csv")
reg("share of orders that are repeat orders", round(len(ro)/n_ord, 4), f"{len(ro)} / {n_ord}")
reg("share of contribution from repeat orders", round(ro.contribution.sum()/OF.contribution.sum(), 4), f"{ro.contribution.sum():.2f} / {OF.contribution.sum():.2f}")
reg("contribution per first order", round(fo.contribution.mean(), 2), f"/ {len(fo)} first orders")
reg("contribution per repeat order", round(ro.contribution.mean(), 2), f"/ {len(ro)} repeat orders")
reg("item-discount share: first orders", round((fo.group == 'item_discount').mean(), 4), f"{int((fo.group=='item_discount').sum())} / {len(fo)}")
reg("item-discount share: repeat orders", round((ro.group == 'item_discount').mean(), 4), f"{int((ro.group=='item_discount').sum())} / {len(ro)}")

s = PT.contribution.sort_values(ascending=False); cs = s.cumsum() / s.sum()
conc = {}
for pct in [0.01, 0.05, 0.10, 0.20]:
    k = int(np.ceil(npers * pct)); conc[pct] = s.iloc[:k].sum() / s.sum(); reg(f"top {int(pct*100)}% of buyers: share of contribution", round(conc[pct], 4), f"{k} buyers; {s.iloc[:k].sum():.2f} / {s.sum():.2f}")
print("concentration of observed contribution:", {f"top {int(k*100)}%": round(v, 3) for k, v in conc.items()})
neg = int((PT.contribution <= 0).sum()); print("buyers with contribution <= 0 in window:", neg)
reg("buyers with total contribution <= 0", neg, f"/ {npers}")
pc_ = PT.contribution.describe(percentiles=[.1, .5, .9]); print(pc_.round(2).to_string())
reg("median contribution per buyer (window)", round(float(PT.contribution.median()), 2), f"mean {PT.contribution.mean():.2f}")

# time to second order
d2 = PT.days_to_second.dropna()
print("\ndays to second order (those who ordered twice): describe"); print(d2.describe(percentiles=[.25, .5, .75, .9]).round(1).to_string())
reg("median days to second order (among repeaters, censored)", float(d2.median()), f"n={len(d2)}")

# 90-day repeat by first-order month (mature cohorts: first order <= 2026-03-31)
M = PT[PT.first_date <= "2026-03-31"].copy()
M["rep90"] = (M.days_to_second <= 90)
cohort = M.groupby("first_month").agg(n=("rep90", "size"), rep=("rep90", "sum")); cohort["rate"] = cohort.rep / cohort.n
ci = [wilson(k, n) for k, n in zip(cohort.rep, cohort.n)]; cohort["lo"] = [c[0] for c in ci]; cohort["hi"] = [c[1] for c in ci]
print("\n=== 90-day repeat rate by first-order month (all cohorts have >=90 days of follow-up) ==="); print(cohort.round(3).to_string()); cohort.round(4).to_csv(f"{OUT}/repeat90_by_cohort.csv")
reg("90-day repeat rate, all mature cohorts", round(M.rep90.mean(), 4), f"{int(M.rep90.sum())} / {len(M)} first-time buyers (first order 2025-07..2026-03)")
reg("90-day repeat rate range across 9 cohort months", f"{cohort.rate.min():.4f} to {cohort.rate.max():.4f}", f"cohort sizes {int(cohort.n.min())}-{int(cohort.n.max())}")
# censoring illustration: ever-repeat rate by cohort month (all months)
ever = PT.groupby("first_month").agg(n=("n_orders", "size"), ever=("n_orders", lambda s: (s >= 2).sum())); ever["rate"] = ever.ever / ever.n
print("\n[CENSORING] share ordering again at any time in window, by first-order month:"); print(ever.round(3).T.to_string())

# ---------------- 2. MARKETING ----------------
print("\n=== 2. MARKETING: spend vs Shopify-side buyers ===")
CH = {"Meta Ads": "meta_ads", "Google Ads": "google_ads", "TikTok Ads": "tiktok_ads", "Influencer": "influencer"}
ms["source"] = ms.channel.map(CH); ms["month"] = pd.to_datetime(ms.month).dt.to_period("M").astype(str)
spend_tot = ms.groupby("source").spend.sum(); spend_all = ms.spend.sum()
reg("total marketing spend", round(spend_all, 2), "sum of 48 channel-months", "all channels, 12 months")
for k_, v_ in spend_tot.items(): reg(f"spend {k_}", round(v_, 2), f"= {v_/spend_all:.4f} of total spend")
nb = PT.groupby("source").size()
base_b = nb.reindex(list(CH.values())).fillna(0)
strict_b = PT[~PT.ambiguous].groupby("source").size().reindex(list(CH.values())).fillna(0)
unk = int((PT.source == "unknown").sum()); known_n = len(PT) - unk
gen_b = base_b + base_b / known_n * unk * 1.0                   # unknown allocated pro rata over all known-source buyers
t = pd.DataFrame({"spend": spend_tot, "buyers_strict": strict_b, "buyers_base": base_b, "buyers_generous": gen_b})
t["CAC_upper(strict)"] = t.spend / t.buyers_strict; t["CAC_base"] = t.spend / t.buyers_base; t["CAC_lower(generous)"] = t.spend / t.buyers_generous
print("CAC = spend / new buyers (first valid order in window) by source. strict = excl. ambiguous IG/Google; base = mapped labels; generous = base + unknown pro rata."); print(t.round(2).to_string())
t.round(3).to_csv(f"{OUT}/cac_12m.csv")
for k_, rw in t.iterrows():
    reg(f"CAC {k_} (base)", round(rw.CAC_base, 2), f"{rw.spend:.2f} / {int(rw.buyers_base)} new buyers", "12 months")
    reg(f"CAC {k_} range strict..generous", f"{rw['CAC_lower(generous)']:.2f} .. {rw['CAC_upper(strict)']:.2f}", f"buyers {int(rw.buyers_strict)} .. {rw.buyers_generous:.0f}", "12 months")
paid_b = int(base_b.sum()); reg("blended CAC, paid spend / ALL new buyers", round(spend_all/npers, 2), f"{spend_all:.2f} / {npers} new buyers", "12 months; includes organic buyers")
reg("share of new buyers attributed to the 4 paid sources (base)", round(paid_b/npers, 4), f"{paid_b} / {npers}")
print("new buyers by source (all):"); print(nb.sort_values(ascending=False).to_string())

# ---- first-order and 90-day economics by source (mature cohort) ----
M = M.join(OF.assign(first_date=OF.person_id.map(PT.first_date)).pipe(lambda d: d[(d.order_date - d.first_date).dt.days <= 90]).groupby("person_id").contribution.sum().rename("c90"), how="left")
M["c90"] = M.c90.fillna(0)
mat_spend = ms[ms.month <= "2026-03"].groupby("source").spend.sum()
mt = M.groupby("source").agg(buyers=("rep90", "size"), first_contrib=("first_contribution", "mean"), first_net_sales=("first_net_sales", "mean"), c90=("c90", "mean"), rep90=("rep90", "mean"), welcome_share=("first_welcome", "mean"))
mt["spend_Jul_Mar"] = mat_spend.reindex(mt.index)
mt["CAC_base"] = mt.spend_Jul_Mar / mt.buyers
mt["first_contrib/CAC"] = mt.first_contrib / mt.CAC_base; mt["c90/CAC"] = mt.c90 / mt.CAC_base
mt["usable(n>=30)"] = mt.buyers >= 30
print("\n=== CHANNEL UNIT ECONOMICS, mature cohort (first order 2025-07..2026-03; spend Jul-Mar; 90-day contribution per new buyer) ===")
print(mt.round(3).to_string()); mt.round(4).to_csv(f"{OUT}/channel_unit_economics_mature.csv")
for k_ in CH.values():
    rw = mt.loc[k_]
    reg(f"{k_}: 90-day contribution per new buyer / CAC (Jul-Mar cohort)", round(rw["c90/CAC"], 3), f"{rw.c90:.2f} / {rw.CAC_base:.2f}; {int(rw.buyers)} buyers", "mature cohort")
    reg(f"{k_}: first-order contribution per buyer / CAC", round(rw["first_contrib/CAC"], 3), f"{rw.first_contrib:.2f} / {rw.CAC_base:.2f}", "mature cohort")
    reg(f"{k_}: 90-day repeat rate", round(rw.rep90, 4), f"n={int(rw.buyers)}", "mature cohort")
free = M[M.source.isin(["direct", "organic_search", "referral"])]
print("\nunpaid sources (no spend in file): buyers", len(free), "| first-order contribution", round(free.first_contrib.mean() if 'first_contrib' in free else free.first_contribution.mean(), 2),
      "| 90-day contribution", round(free.c90.mean(), 2))
reg("unpaid sources (direct+organic+referral): 90-day contribution per new buyer", round(free.c90.mean(), 2), f"n={len(free)}", "mature cohort")
reg("paid sources: 90-day contribution per new buyer (all 4)", round(M[M.source.isin(CH.values())].c90.mean(), 2), f"n={int(M.source.isin(CH.values()).sum())}", "mature cohort")

# ---- platform-reported vs Shopify-side ----
print("\n=== PLATFORM-REPORTED vs SHOPIFY-SIDE (12 months) ===")
pr = ms.groupby("source").agg(spend=("spend", "sum"), plat_orders=("platform_reported_orders", "sum"), plat_revenue=("platform_reported_revenue", "sum"))
OF["src"] = OF.person_id.map(PT.source)
sh_all = OF.groupby("src").agg(shop_orders_all=("order_id", "size"), shop_net_sales_all=("gross", "sum"))
sh_all["shop_net_sales_all"] = OF.assign(ns=OF.gross - OF.disc).groupby("src").ns.sum()
sh_first = OF[OF.order_seq_person == 1].groupby("src").agg(shop_first_orders=("order_id", "size"), shop_first_net_sales=("gross", "sum"))
sh_first["shop_first_net_sales"] = OF[OF.order_seq_person == 1].assign(ns=OF.gross - OF.disc).groupby("src").ns.sum()
cmp_ = pr.join(sh_all).join(sh_first)
cmp_["plat_ROAS"] = cmp_.plat_revenue / cmp_.spend
cmp_["shop_first_ROAS(net sales/spend)"] = cmp_.shop_first_net_sales / cmp_.spend
cmp_["shop_all_ROAS(net sales/spend)"] = cmp_.shop_net_sales_all / cmp_.spend
cmp_["plat_orders/shop_first_orders"] = cmp_.plat_orders / cmp_.shop_first_orders
cmp_["plat_orders/shop_all_orders"] = cmp_.plat_orders / cmp_.shop_orders_all
print(cmp_.round(3).T.to_string()); cmp_.round(4).to_csv(f"{OUT}/platform_vs_shopify.csv")
tot_plat_orders = int(pr.plat_orders.sum()); tot_paid_all_orders = int(cmp_.shop_orders_all.sum()); tot_first_paid = int(cmp_.shop_first_orders.sum())
reg("platform-reported orders, 4 channels", tot_plat_orders, f"vs Shopify first orders from these 4 sources {tot_first_paid}; all orders by their buyers {tot_paid_all_orders}", "12 months")
reg("platform-reported orders / Shopify total valid orders", round(tot_plat_orders/n_ord, 4), f"{tot_plat_orders} / {n_ord}", "12 months")
reg("platform-reported revenue, 4 channels", round(pr.plat_revenue.sum(), 2), "", "12 months")
for k_, rw in cmp_.iterrows():
    reg(f"{k_}: platform ROAS (reported revenue / spend)", round(rw.plat_ROAS, 3), f"{rw.plat_revenue:.2f} / {rw.spend:.2f}", "12 months")
    reg(f"{k_}: Shopify-side first-order net sales / spend", round(rw["shop_first_ROAS(net sales/spend)"], 3), f"{rw.shop_first_net_sales:.2f} / {rw.spend:.2f}", "12 months; net sales before refunds")
    reg(f"{k_}: platform orders / Shopify first orders", round(rw["plat_orders/shop_first_orders"], 3), f"{int(rw.plat_orders)} / {int(rw.shop_first_orders)}", "12 months")

# ---- monthly blended, total-level contribution after marketing ----
print("\n=== MONTHLY: total contribution vs total marketing spend (blended; order-month basis) ===")
mo = OF.groupby("month").agg(orders=("order_id", "size"), contribution=("contribution", "sum"), net_rev=("net_rev", "sum"))
mo["new_buyers"] = OF[OF.order_seq_person == 1].groupby("month").size()
mo["spend"] = ms.groupby("month").spend.sum()
mo["blended_CAC"] = mo.spend / mo.new_buyers
mo["contribution_after_marketing"] = mo.contribution - mo.spend
mo["spend/net_rev"] = mo.spend / mo.net_rev
print(mo.round(2).to_string()); mo.round(3).to_csv(f"{OUT}/monthly_marketing.csv")
tc = OF.contribution.sum()
reg("total contribution (before marketing)", round(tc, 2), "", "valid orders")
reg("total contribution minus total marketing spend (all channels, before opex)", round(tc - spend_all, 2), f"{tc:.2f} - {spend_all:.2f}", "TOTAL level only; not net profit; recent-month refunds censored")
reg("marketing spend / net revenue", round(spend_all/OF.net_rev.sum(), 4), f"{spend_all:.2f} / {OF.net_rev.sum():.2f}")
reg("marketing spend / contribution", round(spend_all/tc, 4), f"{spend_all:.2f} / {tc:.2f}")
reg("blended CAC range across months", f"{mo.blended_CAC.min():.2f} to {mo.blended_CAC.max():.2f}", "12 monthly points", "descriptive only")
nov = mo.loc["2025-11"]; reg("Nov 2025 spend share of annual spend", round(nov.spend/spend_all, 4), f"{nov.spend:.2f} / {spend_all:.2f}")
reg("Nov 2025 contribution after marketing", round(nov.contribution_after_marketing, 2), f"{nov.contribution:.2f} - {nov.spend:.2f}")

# ---------------- 3. CX4: welcome code vs no code, within known unambiguous source ----------------
print("\n=== CX4: first order WELCOME15 vs no code, within source (unknown + ambiguous IG/Google EXCLUDED) ===")
lines = oi[oi.is_valid & ~oi.is_gift][["order_id", "quantity"]].merge(r[r.is_valid].groupby("order_id").quantity_returned.sum().rename("ret").reset_index(), on="order_id", how="left").fillna({"ret": 0})
lu = lines.groupby("order_id").agg(u=("quantity", "sum"), ret=("ret", "sum"))
F = PT[(PT.first_date <= "2026-03-31") & (PT.source != "unknown") & (~PT.ambiguous) & (PT.first_group.isin(["no_code", "item_discount"]))].copy()
F["arm"] = np.where(F.first_welcome, "welcome", np.where(F.first_group == "no_code", "none", "other_code"))
F = F[F.arm.isin(["welcome", "none"])].copy()
F = F.join(lu, on="first_order_id")
F["rep90"] = F.days_to_second <= 90
F["peak"] = F.first_month.isin(["2025-11", "2025-12"])
excl_n = int(((PT.first_date <= "2026-03-31") & ((PT.source == "unknown") | PT.ambiguous)).sum())
print(f"first-time buyers in mature cohort excluded as unknown/ambiguous: {excl_n} of {int((PT.first_date <= '2026-03-31').sum())}")
reg("CX4 mature-cohort buyers excluded (unknown or ambiguous label)", excl_n, f"/ {int((PT.first_date <= '2026-03-31').sum())}", "mature cohort")
def cx(Fx, tag):
    c = Fx.groupby(["source", "arm"]).agg(n=("rep90", "size"), rep=("rep90", "sum"), fc=("first_contribution", "mean"), u=("u", "sum"), ret=("ret", "sum"), acc=("accepts_marketing", "mean")).unstack("arm")
    c.columns = [f"{a}_{b}" for a, b in c.columns]; c = c.fillna(0)
    c["ok"] = (c.n_welcome >= 30) & (c.n_none >= 30)
    for m_ in ["welcome", "none"]:
        c[f"rep_{m_}"] = c[f"rep_{m_}"] / c[f"n_{m_}"].replace(0, np.nan); c[f"retrate_{m_}"] = c[f"ret_{m_}"] / c[f"u_{m_}"].replace(0, np.nan)
    print(f"\n-- {tag}"); print(c[["n_welcome", "n_none", "ok", "rep_welcome", "rep_none", "fc_welcome", "fc_none", "retrate_welcome", "retrate_none", "acc_welcome", "acc_none"]].round(3).to_string())
    ok = c[c.ok]
    if len(ok) == 0: print("no usable cells"); return c, None
    w = ok.n_welcome / ok.n_welcome.sum()
    res = {"cells": f"{len(ok)} of {len(c)}", "cov_welcome": ok.n_welcome.sum() / c.n_welcome.sum(),
           "rep_w": (w * ok.rep_welcome).sum(), "rep_n": (w * ok.rep_none).sum(),
           "fc_w": (w * ok.fc_welcome).sum(), "fc_n": (w * ok.fc_none).sum(),
           "ret_w": (w * ok.retrate_welcome).sum(), "ret_n": (w * ok.retrate_none).sum(),
           "acc_w": (w * ok.acc_welcome).sum(), "acc_n": (w * ok.acc_none).sum(),
           "cells_rep_lower": int((ok.rep_welcome < ok.rep_none).sum())}
    rng = np.random.default_rng(5); bs = []
    for _ in range(2000):
        d = (rng.binomial(ok.n_welcome.astype(int), ok.rep_welcome) / ok.n_welcome.values - rng.binomial(ok.n_none.astype(int), ok.rep_none) / ok.n_none.values)
        bs.append(float((w.values * d).sum()))
    res["rep_diff_lo"], res["rep_diff_hi"] = np.percentile(bs, [2.5, 97.5])
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    return c, res
c_all, res_all = cx(F, "ALL mature cohorts")
c_off, res_off = cx(F[~F.peak], "EXCLUDING Nov-Dec 2025 cohorts")
c_all.round(4).to_csv(f"{OUT}/cx4_cells_all.csv"); c_off.round(4).to_csv(f"{OUT}/cx4_cells_offpeak.csv")
for tag, rs in [("all cohorts", res_all), ("excl Nov-Dec", res_off)]:
    if rs is None: continue
    reg(f"CX4 ({tag}) usable source cells", rs["cells"], f"coverage of WELCOME15 buyers {rs['cov_welcome']:.3f}")
    reg(f"CX4 ({tag}) standardised 90-day repeat: welcome vs none", f"{rs['rep_w']:.4f} vs {rs['rep_n']:.4f}", f"diff 95% bootstrap {rs['rep_diff_lo']:+.4f}..{rs['rep_diff_hi']:+.4f}; cells with lower welcome rate {rs['cells_rep_lower']}")
    reg(f"CX4 ({tag}) standardised first-order contribution: welcome vs none", f"{rs['fc_w']:.2f} vs {rs['fc_n']:.2f}", "weights = WELCOME15 buyers per source")
    reg(f"CX4 ({tag}) standardised first-order return rate (units): welcome vs none", f"{rs['ret_w']:.4f} vs {rs['ret_n']:.4f}", "returned units / paid units on the first order")
    reg(f"CX4 ({tag}) accepts_marketing share: welcome vs none", f"{rs['acc_w']:.3f} vs {rs['acc_n']:.3f}", "selection evidence")
# selection descriptive without source restriction
sel = PT[PT.first_date <= "2026-03-31"].assign(arm=lambda d: np.where(d.first_welcome, "welcome", np.where(d.first_group == "no_code", "none", "other"))).groupby("arm").agg(n=("accepts_marketing", "size"), accepts=("accepts_marketing", "mean"))
print("\n[SELECTION] accepts_marketing share by first-order arm (all sources incl unknown):"); print(sel.round(3).to_string())

# platform vs Shopify, 4 paid channels in total (cited in the report: ROAS 2.67 vs 1.68 vs 2.21)
S_ = pr.spend.sum(); PR_ = pr.plat_revenue.sum(); F1_ = cmp_.shop_first_net_sales.sum(); A1_ = cmp_.shop_net_sales_all.sum()
reg("platform-reported revenue / platform spend (4 channels)", round(PR_ / S_, 4), f"{PR_:.2f} / {S_:.2f}", "12 months, 4 paid sources")
reg("Shopify-side first-order net sales / spend (4 channels)", round(F1_ / S_, 4), f"{F1_:.2f} / {S_:.2f}", "12 months, 4 paid sources")
reg("Shopify-side net sales, all orders of these buyers / spend", round(A1_ / S_, 4), f"{A1_:.2f} / {S_:.2f}", "12 months, 4 paid sources")
reg("platform-reported revenue / Shopify net sales of all orders of these buyers", round(PR_ / A1_, 4), f"{PR_:.2f} / {A1_:.2f}", "12 months, 4 paid sources")
reg("platform-reported revenue / Shopify first-order net sales", round(PR_ / F1_, 4), f"{PR_:.2f} / {F1_:.2f}", "12 months, 4 paid sources")
reg("platform-reported orders / Shopify first orders (4 channels)", round(tot_plat_orders / tot_first_paid, 4), f"{tot_plat_orders} / {tot_first_paid}", "12 months, 4 paid sources")
reg("platform-reported orders / Shopify all orders of these buyers", round(tot_plat_orders / tot_paid_all_orders, 4), f"{tot_plat_orders} / {tot_paid_all_orders}", "12 months, 4 paid sources")

P = pd.DataFrame(REG, columns=["id", "metric", "value", "numerator_denominator", "filter"]); P["period"] = "2025-07-01..2026-06-30"; P["source_script"] = "06_customers_marketing.py"
P.to_csv(f"{OUT}/number_register_phase4.csv", index=False)
print("\nNUMBER REGISTER (Phase 4)"); print(P[["id", "metric", "value", "numerator_denominator"]].to_string(index=False))
