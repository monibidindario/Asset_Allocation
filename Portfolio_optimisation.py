import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis as sp_kurtosis
from numpy.linalg import inv, LinAlgError
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
EXCEL_FILE = "dataexam may26.xlsx"
RF_ANNUAL  = 0.02
RF_DAILY   = (1 + RF_ANNUAL) ** (1 / 252) - 1
RF_MONTHLY = (1 + RF_ANNUAL) ** (1 / 12)  - 1
N_PLOT     = 200
N_FRONT    = 60
DELTA      = 3.0
TAU        = 0.05

NASDAQ_SELECTION = [
    "NVIDIA", "APPLE", "STEEL DYNAMICS", "META PLATFORMS A", "BROADCOM",
    "COSTCO WHOLESALE", "CME GROUP", "GILEAD SCIENCES", "PALO ALTO NETWORKS",
    "BAKER HUGHES A", "INTERACTIVE BROKERS GROUP A", "MARRIOTT INTL. 'A'",
]
NYSE_SELECTION = [
    "CARPENTER TECH.", "ELI LILLY", "CATERPILLAR", "JP MORGAN CHASE",
    "EXXON MOBIL", "COCA COLA", "WELLTOWER", "MASTERCARD",
    "MOTOROLA SOLUTIONS", "DYCOM INDS.", "GENERAL DYNAMICS", "MURPHY OIL",
]
NASDAQ_FINAL = [
    "NVIDIA", "APPLE", "STEEL DYNAMICS", "META PLATFORMS A", "BROADCOM",
    "COSTCO WHOLESALE", "CME GROUP", "GILEAD SCIENCES", "PALO ALTO NETWORKS",
    "BAKER HUGHES A", "INTERACTIVE BROKERS GROUP A", "MARRIOTT INTL.'A'",
]
NYSE_FINAL = [
    "CARPENTER TECH.", "ELI LILLY", "CATERPILLAR", "JP MORGAN CHASE & CO.",
    "EXXON MOBIL", "COCA COLA", "WELLTOWER", "MASTERCARD",
    "MOTOROLA SOLUTIONS", "DYCOM INDS.", "GENERAL DYNAMICS", "MURPHY OIL",
]
NASDAQ_META = {
    "NVIDIA":                      {"Sector": "Semiconductors",     "Mean_ann": 0.661, "Std_ann": 0.507, "Sharpe": 1.264},
    "APPLE":                       {"Sector": "Tech hardware",       "Mean_ann": 0.202, "Std_ann": 0.269, "Sharpe": 0.677},
    "STEEL DYNAMICS":              {"Sector": "Materials/Steel",     "Mean_ann": 0.316, "Std_ann": 0.372, "Sharpe": 0.796},
    "META PLATFORMS A":            {"Sector": "Social media",        "Mean_ann": 0.222, "Std_ann": 0.430, "Sharpe": 0.470},
    "BROADCOM":                    {"Sector": "Semiconductors",      "Mean_ann": 0.524, "Std_ann": 0.419, "Sharpe": 1.203},
    "COSTCO WHOLESALE":            {"Sector": "Consumer staples",    "Mean_ann": 0.219, "Std_ann": 0.221, "Sharpe": 0.901},
    "CME GROUP":                   {"Sector": "Finance/Derivatives", "Mean_ann": 0.081, "Std_ann": 0.195, "Sharpe": 0.313},
    "GILEAD SCIENCES":             {"Sector": "Healthcare/Pharma",   "Mean_ann": 0.151, "Std_ann": 0.235, "Sharpe": 0.557},
    "PALO ALTO NETWORKS":          {"Sector": "Cybersecurity",       "Mean_ann": 0.366, "Std_ann": 0.404, "Sharpe": 0.857},
    "BAKER HUGHES A":              {"Sector": "Energy",              "Mean_ann": 0.240, "Std_ann": 0.343, "Sharpe": 0.641},
    "INTERACTIVE BROKERS GROUP A": {"Sector": "Fintech/Brokerage",   "Mean_ann": 0.369, "Std_ann": 0.336, "Sharpe": 1.039},
    "MARRIOTT INTL. 'A'":          {"Sector": "Tourism/Hospitality", "Mean_ann": 0.214, "Std_ann": 0.282, "Sharpe": 0.688},
}
NYSE_META = {
    "CARPENTER TECH.":    {"Sector": "Materials/Metals",      "Mean_ann": 0.543,  "Std_ann": 0.460,  "Sharpe": 1.137},
    "ELI LILLY":          {"Sector": "Pharma/Biotech",        "Mean_ann": 0.367,  "Std_ann": 0.320,  "Sharpe": 1.084},
    "CATERPILLAR":        {"Sector": "Industrials/Machinery", "Mean_ann": 0.295,  "Std_ann": 0.299,  "Sharpe": 0.920},
    "JP MORGAN CHASE":    {"Sector": "Banking/Finance",       "Mean_ann": 0.144,  "Std_ann": 0.239,  "Sharpe": 0.519},
    "EXXON MOBIL":        {"Sector": "Energy/Oil & Gas",      "Mean_ann": 0.219,  "Std_ann": 0.263,  "Sharpe": 0.757},
    "COCA COLA":          {"Sector": "Consumer staples",      "Mean_ann": 0.088,  "Std_ann": 0.157,  "Sharpe": 0.433},
    "WELLTOWER":          {"Sector": "Real Estate/REIT",      "Mean_ann": 0.234,  "Std_ann": 0.232,  "Sharpe": 0.922},
    "MASTERCARD":         {"Sector": "Digital payments",      "Mean_ann": 0.087,  "Std_ann": 0.223,  "Sharpe": 0.300},
    "MOTOROLA SOLUTIONS": {"Sector": "Tech/Communications",   "Mean_ann": 0.154,  "Std_ann": 0.225,  "Sharpe": 0.596},
    "DYCOM INDS.":        {"Sector": "Infrastructure",        "Mean_ann": 0.387,  "Std_ann": 0.417,  "Sharpe": 0.880},
    "GENERAL DYNAMICS":   {"Sector": "Defense/Aerospace",     "Mean_ann": 0.1254, "Std_ann": 0.1686, "Sharpe": 0.625},
    "MURPHY OIL":         {"Sector": "Energy/Exploration",    "Mean_ann": 0.238,  "Std_ann": 0.454,  "Sharpe": 0.480},
}

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_nasdaq_sheet(sheet):
    raw   = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None)
    names = raw.iloc[3, 2:].tolist()
    dates = pd.to_datetime(raw.iloc[6:, 1], errors="coerce")
    data  = raw.iloc[6:, 2:].copy()
    data.columns = names; data.index = dates.values
    data = data.apply(pd.to_numeric, errors="coerce")
    return data[data.index.notna()].sort_index().loc[:, pd.notna(data.columns)]

def load_nyse_daily():
    raw   = pd.read_excel(EXCEL_FILE, sheet_name="nyse daily", header=None)
    names = raw.iloc[2, 2:].tolist()
    dates = pd.to_datetime(raw.iloc[5:, 1], errors="coerce")
    data  = raw.iloc[5:, 2:].copy()
    data.columns = names; data.index = dates.values
    data = data.apply(pd.to_numeric, errors="coerce")
    return data[data.index.notna()].sort_index().loc[:, pd.notna(data.columns)]

def load_nyse_monthly():
    raw   = pd.read_excel(EXCEL_FILE, sheet_name="nyse monthly", header=None)
    names = raw.iloc[1, 2:].tolist()
    dates = pd.to_datetime(raw.iloc[4:, 1], errors="coerce")
    data  = raw.iloc[4:, 2:].copy()
    data.columns = names; data.index = dates.values
    data = data.apply(pd.to_numeric, errors="coerce")
    return data[data.index.notna()].sort_index().loc[:, pd.notna(data.columns)]

def load_index_sheet(sheet):
    raw      = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None)
    dates_sp = pd.to_datetime(raw.iloc[5:, 1], errors="coerce")
    sp = pd.DataFrame({"SP500_PI": pd.to_numeric(raw.iloc[5:, 2], errors="coerce").values,
                        "SP500_RI": pd.to_numeric(raw.iloc[5:, 3], errors="coerce").values},
                       index=dates_sp.values)
    sp = sp[sp.index.notna()].sort_index()
    dates_nq = pd.to_datetime(raw.iloc[5:, 8], errors="coerce")
    nq = pd.DataFrame({"NASDAQ_PI": pd.to_numeric(raw.iloc[5:, 9],  errors="coerce").values,
                        "NASDAQ_RI": pd.to_numeric(raw.iloc[5:, 10], errors="coerce").values},
                       index=dates_nq.values)
    nq = nq[nq.index.notna()].sort_index()
    return sp, nq

# =============================================================================
# LOAD ALL DATA
# =============================================================================
print("\n[Loading data]")
px_nq_d_all = load_nasdaq_sheet("nasdaq daily")
px_nq_m_all = load_nasdaq_sheet("nasdaq monthly")
px_ny_d_all = load_nyse_daily()
px_ny_m_all = load_nyse_monthly()
sp500_d, nasdaq_idx_d = load_index_sheet("indexes daily")
sp500_m, nasdaq_idx_m = load_index_sheet("indexes monthly")

px_nq_d_all = px_nq_d_all.loc[:, px_nq_d_all.columns.notna()]
px_nq_m_all = px_nq_m_all.loc[:, px_nq_m_all.columns.notna()]
px_ny_d_all = px_ny_d_all.loc[:, px_ny_d_all.columns.notna()]
px_ny_m_all = px_ny_m_all.loc[:, px_ny_m_all.columns.notna()]

print(f"  Nasdaq daily   : {px_nq_d_all.shape[0]} obs, {px_nq_d_all.shape[1]} stocks")
print(f"  Nasdaq monthly : {px_nq_m_all.shape[0]} obs, {px_nq_m_all.shape[1]} stocks")
print(f"  NYSE   daily   : {px_ny_d_all.shape[0]} obs, {px_ny_d_all.shape[1]} stocks")
print(f"  NYSE   monthly : {px_ny_m_all.shape[0]} obs, {px_ny_m_all.shape[1]} stocks")

# =============================================================================
# SHARED HELPER FUNCTIONS
# =============================================================================

def desc_stats(ret, ann):
    """
    Works for both pd.Series (single asset) and pd.DataFrame (multiple assets).
    FIX: the Series branch avoids .apply() on scalar values.
    """
    if isinstance(ret, pd.Series):
        ret = ret.dropna()
        mu  = ret.mean()
        sd  = ret.std(ddof=1)
        return pd.Series({
            "Mean (period)": mu,
            "Var  (period)": ret.var(ddof=1),
            "Std  (period)": sd,
            "Skewness":      float(skew(ret)),
            "Exc. Kurtosis": float(sp_kurtosis(ret)),
            "Mean (ann.)":   mu * ann,
            "Std  (ann.)":   sd * np.sqrt(ann),
            "Sharpe (ann.)": (mu * ann - RF_ANNUAL) / (sd * np.sqrt(ann))
                             if sd > 0 else np.nan,
        })
    # DataFrame branch
    return pd.DataFrame({
        "Mean (period)": ret.mean(),
        "Var  (period)": ret.var(ddof=1),
        "Std  (period)": ret.std(ddof=1),
        "Skewness":      ret.apply(lambda x: float(skew(x.dropna()))),
        "Exc. Kurtosis": ret.apply(lambda x: float(sp_kurtosis(x.dropna()))),
        "Mean (ann.)":   ret.mean() * ann,
        "Std  (ann.)":   ret.std(ddof=1) * np.sqrt(ann),
        "Sharpe (ann.)": (ret.mean() * ann - RF_ANNUAL) / (ret.std(ddof=1) * np.sqrt(ann)),
    })

def port_stats(r, ann):
    mu = np.mean(r); sd = np.std(r, ddof=1)
    sr = (mu * ann - RF_ANNUAL) / (sd * np.sqrt(ann)) if sd > 0 else np.nan
    return {
        "Mean (period)": mu, "Var  (period)": np.var(r, ddof=1),
        "Std  (period)": sd, "Skewness": float(skew(r)),
        "Exc. Kurtosis": float(sp_kurtosis(r)),
        "Mean (ann.)": mu * ann, "Std  (ann.)": sd * np.sqrt(ann),
        "Sharpe (ann.)": sr,
    }

def best_match(target, candidates):
    t = target.upper().strip()
    for c in candidates:
        if str(c).upper().strip() == t: return c
    for c in candidates:
        if str(c).upper().strip().startswith(t[:8]): return c
    for c in candidates:
        if t[:8] in str(c).upper(): return c
    t_words = set(t.split()); best, best_score = None, 0
    for c in candidates:
        score = len(t_words & set(str(c).upper().split()))
        if score > best_score: best_score, best = score, c
    return best if best_score > 0 else None

def get_selected_returns(px_d_all, px_m_all, targets):
    def resolve(targets, px):
        cols = list(px.columns); res = {}
        for t in targets:
            m = best_match(t, cols)
            if m: res[t] = m
            else: print(f"  WARNING: '{t}' not found")
        return res
    map_d = resolve(targets, px_d_all); map_m = resolve(targets, px_m_all)
    common = [t for t in targets if t in map_d and t in map_m]
    px_d = px_d_all[[map_d[t] for t in common]].copy(); px_d.columns = common
    px_m = px_m_all[[map_m[t] for t in common]].copy(); px_m.columns = common
    ret_d = px_d.pct_change().iloc[1:]; ret_m = px_m.pct_change().iloc[1:]
    ret_d = ret_d.dropna(axis=1, thresh=int(0.8*len(ret_d))).fillna(ret_d.mean())
    ret_m = ret_m.dropna(axis=1, thresh=int(0.8*len(ret_m))).fillna(ret_m.mean())
    dropped = [s for s in common if s not in ret_d.columns or s not in ret_m.columns]
    if dropped: print(f"  Dropped (coverage <80%): {dropped}")
    sel = [s for s in common if s in ret_d and s in ret_m]
    return sel, ret_d[sel], ret_m[sel], px_d[sel], px_m[sel]

def get_returns_exact(px, sel):
    available = [s for s in sel if s in px.columns]
    missing   = [s for s in sel if s not in px.columns]
    if missing: print(f"  WARNING not found: {missing}")
    ret = px[available].pct_change().iloc[1:].fillna(px[available].pct_change().mean())
    return ret, available

def tangency_unconstrained(mu, Sigma, rf):
    try:
        Si = inv(Sigma); z = Si @ (mu - rf); den = np.ones(len(mu)) @ z
        return z / den if abs(den) > 1e-14 else np.ones(len(mu)) / len(mu)
    except LinAlgError:
        return np.ones(len(mu)) / len(mu)

def tangency_constrained(mu, Sigma, rf):
    n = len(mu); w0 = np.ones(n) / n
    def neg_sr(w): return -(w @ mu - rf) / np.sqrt(max(w @ Sigma @ w, 1e-14))
    res = minimize(neg_sr, w0, method="SLSQP", bounds=[(0, 1)]*n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 3000})
    return res.x if (res.success or res.fun < neg_sr(w0)) else w0

def compute_betas(ret_df, mkt):
    var_m = np.var(mkt, ddof=1)
    return pd.Series({s: np.cov(ret_df[s].values, mkt, ddof=1)[0, 1] / var_m
                      for s in ret_df.columns}, name="Beta")

def compute_betas_monthly(ret_df, bm):
    ret_p = ret_df.copy(); bm_p = bm.copy()
    ret_p.index = pd.to_datetime(ret_p.index).to_period("M")
    bm_p.index  = pd.to_datetime(bm_p.index).to_period("M")
    common = ret_p.index.intersection(bm_p.index)
    if len(common) == 0:
        return pd.Series({s: np.nan for s in ret_df.columns}, name="Beta")
    ret_c = ret_p.loc[common]; bm_c = bm_p.loc[common]
    var_m = np.var(bm_c.values, ddof=1)
    return pd.Series({s: np.cov(ret_c[s].values, bm_c.values, ddof=1)[0, 1] / var_m
                      for s in ret_df.columns}, name="Beta")

def plot_weights(w, names, title, fname, color="steelblue"):
    fig, ax = plt.subplots(figsize=(10, max(5, len(names)*0.45)))
    colors = ["tomato" if wi < 0 else color for wi in w]
    ax.barh([n[:24] for n in names], w*100, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Weight (%)"); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="x"); plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def compute_frontier(mu, Sigma, n_pts=N_FRONT, constrained=False):
    n = len(mu); w0 = np.ones(n)/n
    bds = [(0, 1)]*n if constrained else [(-10, 10)]*n
    res_gmv = minimize(lambda w: w@Sigma@w, w0, method="SLSQP", bounds=bds,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}],
                       options={"ftol": 1e-14, "maxiter": 2000})
    w_gmv = res_gmv.x if res_gmv.success else w0
    min_ret = float(w_gmv @ mu)
    max_ret = float(np.max(mu)) if constrained else float(np.max(mu)*1.5)
    f_std, f_ret = [], []
    for tr in np.linspace(min_ret, max_ret, n_pts):
        res = minimize(lambda w: w@Sigma@w, w0, method="SLSQP", bounds=bds,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum()-1},
                                    {"type": "eq", "fun": lambda w, tr=tr: w@mu-tr}],
                       options={"ftol": 1e-14, "maxiter": 2000})
        if res.success:
            f_std.append(np.sqrt(max(res.x@Sigma@res.x, 0))); f_ret.append(float(res.x@mu))
    return np.array(f_std), np.array(f_ret)

def plot_frontier(fs_unc, fr_unc, fs_con, fr_con, mu, Sg, w_unc, w_con, ann, title, fname):
    fig, ax = plt.subplots(figsize=(10, 7))
    def af(s, r): return s*np.sqrt(ann)*100, r*ann*100
    xu, yu = af(fs_unc, fr_unc); xc, yc = af(fs_con, fr_con)
    ax.plot(xu[np.argsort(xu)], yu[np.argsort(xu)], "b-", lw=2, label="Unconstrained frontier")
    ax.plot(xc[np.argsort(xc)], yc[np.argsort(xc)], "r-", lw=2, label="Constrained (w>=0) frontier")
    for w, col, mk, lbl in [(w_unc, "blue", "o", "Tangency-Unc."), (w_con, "red", "s", "Tangency-Con.")]:
        sr = np.sqrt(w@Sg@w)*np.sqrt(ann)*100; er = (w@mu)*ann*100
        ax.scatter([sr], [er], color=col, marker=mk, s=130, zorder=6, label=f"{lbl} ({er:.1f}%,{sr:.1f}%)")
    sr_u = np.sqrt(w_unc@Sg@w_unc)*np.sqrt(ann)*100; er_u = (w_unc@mu)*ann*100
    slope = (er_u - RF_ANNUAL*100) / sr_u if sr_u else 0
    x_cml = np.linspace(0, max(xu.max(), xc.max())*1.1, 200)
    ax.plot(x_cml, RF_ANNUAL*100 + slope*x_cml, "b--", lw=1, alpha=0.55, label="CML")
    ax.axhline(RF_ANNUAL*100, color="grey", ls=":", lw=1, label=f"RF={RF_ANNUAL*100:.0f}%")
    ax.set_xlabel("Annualised Std Dev (%)"); ax.set_ylabel("Annualised Mean Return (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def desc_stats_ann(ret, ann):
    return pd.DataFrame({"Mean_ann": ret.mean()*ann, "Std_ann": ret.std(ddof=1)*np.sqrt(ann)})

# =============================================================================
# PLOTTING HELPERS FOR FULL UNIVERSE (200 securities)
# =============================================================================

def plot_prices_all(px_df, title, fname, n=N_PLOT):
    cols = [c for c in px_df.columns[:n] if pd.notna(c)]
    fig, ax = plt.subplots(figsize=(16, 7))
    for c in cols:
        ts = px_df[c].dropna()
        if len(ts) == 0: continue
        norm = ts / ts.iloc[0] * 100
        ax.plot(norm.index, norm.values, lw=0.55, alpha=0.45)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Normalised Price (base=100)")
    ax.text(0.01, 0.98, f"{len(cols)} securities plotted",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_returns_all(ret_df, title, fname, n=N_PLOT):
    cols = [c for c in ret_df.columns[:n] if pd.notna(c)]
    fig, ax = plt.subplots(figsize=(16, 7))
    for c in cols:
        ts = ret_df[c].dropna()
        ax.plot(ts.index, ts.values, lw=0.45, alpha=0.35)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_ylabel("Return")
    ax.text(0.01, 0.98, f"{len(cols)} securities plotted",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_comparison_all(px_d, px_m, ret_d, ret_m, market, n=N_PLOT):
    cols_d = [c for c in px_d.columns[:n] if pd.notna(c)]
    cols_m = [c for c in px_m.columns[:n] if pd.notna(c)]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f"{market} - Prices and Returns Comparison ({min(n, len(cols_d))} securities)",
                 fontsize=14, fontweight="bold")
    ax = axes[0, 0]
    for c in cols_d:
        ts = px_d[c].dropna()
        if len(ts) == 0: continue
        norm = ts / ts.iloc[0] * 100
        ax.plot(norm.index, norm.values, lw=0.45, alpha=0.40)
    ax.set_title("Daily Prices (normalised to 100)"); ax.grid(True, alpha=0.25)
    ax = axes[0, 1]
    for c in cols_m:
        ts = px_m[c].dropna()
        if len(ts) == 0: continue
        norm = ts / ts.iloc[0] * 100
        ax.plot(norm.index, norm.values, lw=0.55, alpha=0.45)
    ax.set_title("Monthly Prices (normalised to 100)"); ax.grid(True, alpha=0.25)
    ax = axes[1, 0]
    for c in cols_d:
        if c in ret_d.columns:
            ts = ret_d[c].dropna()
            ax.plot(ts.index, ts.values, lw=0.35, alpha=0.30)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Daily Returns"); ax.grid(True, alpha=0.25)
    ax = axes[1, 1]
    for c in cols_m:
        if c in ret_m.columns:
            ts = ret_m[c].dropna()
            ax.plot(ts.index, ts.values, lw=0.50, alpha=0.35)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Monthly Returns"); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fname = f"comparison_{market.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_heatmap(matrix, title, fname, max_stocks=200, use_abs=False,
                 cmap="RdYlGn", vmin=-1, vmax=1):
    sub = matrix.iloc[:max_stocks, :max_stocks]
    vals = sub.abs().values if use_abs else sub.values
    if use_abs:
        vmin, vmax, cmap = 0, 1, "viridis_r"
    n = len(sub)
    fig, ax = plt.subplots(figsize=(max(12, n*0.09), max(10, n*0.08)))
    im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)
    if n <= 60:
        labels = [str(s)[:12] if pd.notna(s) else "N/A" for s in sub.columns]
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{n} securities"); ax.set_ylabel(f"{n} securities")
    ax.set_title(title + f" ({n} securities)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def print_weights_table(sel, w, label):
    print(f"\n  {label}:")
    print(f"    {'Stock':<38} {'Weight':>10}  {'%':>7}")
    print("    " + "-" * 58)
    for s, wi in zip(sel, w):
        print(f"    {s:<38} {wi:10.6f}  {wi*100:7.2f}%")
    print(f"    {'SUM':<38} {w.sum():10.6f}")
    neg = sum(1 for wi in w if wi < -1e-6)
    if neg: print(f"    [{neg} short position(s)]")

def plot_betas(sel, beta_d, beta_m, title, fname):
    x = np.arange(len(sel)); w = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x-w/2, [beta_d.get(s, 0) for s in sel], w, label="Daily",   color="steelblue")
    ax.bar(x+w/2, [beta_m.get(s, 0) for s in sel], w, label="Monthly", color="darkorange")
    ax.axhline(1, color="red", ls="--", lw=1, label="beta=1")
    ax.set_xticks(x); ax.set_xticklabels([s[:16] for s in sel], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Beta"); ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.25, axis="y"); plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_individual_prices(sel, px_df, meta, title, fname):
    n = len(sel); cols_plt = 4; rows_plt = (n + cols_plt - 1) // cols_plt
    fig, axes = plt.subplots(rows_plt, cols_plt, figsize=(18, rows_plt*3.2))
    axes = axes.flatten()
    for i, s in enumerate(sel):
        ax = axes[i]; ts = px_df[s].dropna()
        if len(ts) == 0: ax.set_visible(False); continue
        norm = ts / ts.iloc[0] * 100; ax.plot(norm.index, norm.values, lw=1.3, color=f"C{i}")
        m = meta.get(s, {}); sr_txt = f"  SR={m['Sharpe']:.2f}" if m.get("Sharpe") else ""
        ax.set_title(f"{s[:22]}{sr_txt}", fontsize=7, fontweight="bold")
        ax.tick_params(labelsize=5); ax.grid(True, alpha=0.25)
    for j in range(n, len(axes)): axes[j].set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold"); plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def plot_cumulative(port_r, idx_pi, idx_ri, title, fname):
    try:
        pr = port_r.copy(); pr.index = pd.to_datetime(pr.index, errors="coerce").to_period("M")
        pi = idx_pi.copy();  pi.index = pd.to_datetime(pi.index, errors="coerce").to_period("M")
        ri = idx_ri.copy();  ri.index = pd.to_datetime(ri.index, errors="coerce").to_period("M")
        common_p = pr.index.intersection(pi.index).intersection(ri.index)
        if len(common_p) > 0:
            pr = pr.loc[common_p]; pi = pi.loc[common_p]; ri = ri.loc[common_p]
            plot_idx = common_p.to_timestamp()
            pr.index = pi.index = ri.index = plot_idx
        else:
            common_p = port_r.index.intersection(idx_pi.index).intersection(idx_ri.index)
            if len(common_p) == 0:
                print(f"  Skipped (no overlapping dates): {fname}"); return
            pr = port_r.loc[common_p]; pi = idx_pi.loc[common_p]; ri = idx_ri.loc[common_p]
    except Exception:
        common_p = port_r.index.intersection(idx_pi.index).intersection(idx_ri.index)
        if len(common_p) == 0:
            print(f"  Skipped (no overlapping dates): {fname}"); return
        pr = port_r.loc[common_p]; pi = idx_pi.loc[common_p]; ri = idx_ri.loc[common_p]
    fig, ax = plt.subplots(figsize=(12, 6))
    for r, lbl, col, ls in [(pr, "EW Portfolio", "steelblue", "-"),
                              (pi, "Index PI",    "darkorange", "--"),
                              (ri, "Index RI",    "green",      ":")]:
        ax.plot(r.index, (1+r).cumprod().values, lw=2, label=lbl, color=col, ls=ls)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Return (base=1)"); ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def sml_table(sel, stats, beta, w_unc, w_con, mkt_mean_ann, mrp, chosen_two, freq):
    pbeta_unc = float(beta.reindex(sel).fillna(1.0) @ w_unc)
    pbeta_con = float(beta.reindex(sel).fillna(1.0) @ w_con)
    er_unc = (w_unc @ np.array([stats.loc[s, "Mean_ann"] for s in sel])) * 100
    er_con = (w_con @ np.array([stats.loc[s, "Mean_ann"] for s in sel])) * 100
    print(f"\n  SML - {freq}   E[R] = Rf + beta*MRP   Rf={RF_ANNUAL*100:.1f}%  MRP={mrp*100:.2f}%")
    print(f"  {'Security':<38} {'beta':>6} {'Actual%':>9} {'SML%':>8} {'alpha%':>8}")
    print("  " + "-" * 73)
    for s in chosen_two:
        if s not in beta.index or s not in stats.index: continue
        b = beta[s]; act = stats.loc[s, "Mean_ann"]*100; sml_r = (RF_ANNUAL + b*mrp)*100
        print(f"  {s:<38} {b:6.4f} {act:9.3f}% {sml_r:8.3f}% {act-sml_r:8.3f}%")
    print(f"  {'─'*73}")
    for lbl, bp, er in [("Portfolio (unc.)", pbeta_unc, er_unc), ("Portfolio (con.)", pbeta_con, er_con)]:
        sml_r = (RF_ANNUAL + bp*mrp)*100
        print(f"  {lbl:<38} {bp:6.4f} {er:9.3f}% {sml_r:8.3f}% {er-sml_r:8.3f}%")
    return pbeta_unc, pbeta_con

def plot_sml(sel, stats, beta, w_unc, w_con, mkt_mean_ann, mrp, chosen_two, pb_unc, pb_con, title, fname):
    fig, ax = plt.subplots(figsize=(10, 7))
    b_range = np.linspace(min(0, beta.min()*0.9), beta.max()*1.15, 200)
    ax.plot(b_range, (RF_ANNUAL + b_range*mrp)*100, "k-", lw=2, label="SML")
    for s in sel:
        if s not in beta.index or s not in stats.index: continue
        b = beta[s]; r = stats.loc[s, "Mean_ann"]*100
        col = "crimson" if s in chosen_two else "steelblue"
        sz  = 130 if s in chosen_two else 45
        ax.scatter([b], [r], color=col, s=sz, zorder=5)
        if s in chosen_two:
            ax.annotate(s[:16], (b, r), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color="crimson", fontweight="bold")
    ax.scatter([1], [mkt_mean_ann*100], color="green", s=160, marker="D", zorder=6,
               label="Market (beta=1)")
    er_unc = (w_unc @ np.array([stats.loc[s, "Mean_ann"] for s in sel])) * 100
    er_con = (w_con @ np.array([stats.loc[s, "Mean_ann"] for s in sel])) * 100
    for bp, er, lbl, mk in [(pb_unc, er_unc, "Port.Unc.", "*"), (pb_con, er_con, "Port.Con.", "s")]:
        ax.scatter([bp], [er], color="purple", s=180, marker=mk, zorder=7,
                   label=f"{lbl} (b={bp:.2f})")
    ax.axhline(RF_ANNUAL*100, color="grey", ls=":", lw=1)
    ax.set_xlabel("Beta"); ax.set_ylabel("Ann. Mean Return (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}")

def bm_monthly_aligned(bm, ret_df):
    bm_p = bm.copy(); bm_p.index = pd.to_datetime(bm_p.index).to_period("M")
    ret_p = ret_df.copy(); ret_p.index = pd.to_datetime(ret_p.index).to_period("M")
    common = bm_p.index.intersection(ret_p.index)
    return bm_p.loc[common] if len(common) > 0 else bm_p

def print_beta_table(sel, beta_d, beta_m, w_unc_d, w_con_d, w_unc_m, w_con_m, label):
    print(f"\n  {label}  (beta = Cov(Ri,Rm)/Var(Rm))")
    print(f"  {'Stock':<38} {'Beta Daily':>11} {'Beta Monthly':>13}")
    print("  " + "-" * 62)
    for s in sel:
        bd = beta_d.get(s, np.nan); bm_ = beta_m.get(s, np.nan)
        print(f"  {s:<38} {bd:11.4f} {bm_:13.4f}")
    pb_unc_d = float(beta_d.reindex(sel).fillna(1.0) @ w_unc_d)
    pb_con_d = float(beta_d.reindex(sel).fillna(1.0) @ w_con_d)
    pb_unc_m = float(beta_m.reindex(sel).fillna(1.0) @ w_unc_m)
    pb_con_m = float(beta_m.reindex(sel).fillna(1.0) @ w_con_m)
    print(f"  {'─'*62}")
    print(f"  {'Portfolio beta daily   (unc.)':<38} {pb_unc_d:11.4f}")
    print(f"  {'Portfolio beta daily   (con.)':<38} {pb_con_d:11.4f}")
    print(f"  {'Portfolio beta monthly (unc.)':<38} {' ':>11} {pb_unc_m:13.4f}")
    print(f"  {'Portfolio beta monthly (con.)':<38} {' ':>11} {pb_con_m:13.4f}")

# =============================================================================
# Q1 - NASDAQ: RETURNS AND DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*72)
print("  Q1 - NASDAQ: Returns and Descriptive Statistics")
print("="*72)

ret_nq_all_d = px_nq_d_all.pct_change().iloc[1:]
ret_nq_all_m = px_nq_m_all.pct_change().iloc[1:]
_before_nq_d = set(ret_nq_all_d.columns)
_before_nq_m = set(ret_nq_all_m.columns)
ret_nq_all_d = ret_nq_all_d.dropna(axis=1, thresh=int(0.8*len(ret_nq_all_d)))
ret_nq_all_m = ret_nq_all_m.dropna(axis=1, thresh=int(0.8*len(ret_nq_all_m)))
ret_nq_all_d = ret_nq_all_d.loc[:, ret_nq_all_d.columns.notna()].fillna(ret_nq_all_d.mean())
ret_nq_all_m = ret_nq_all_m.loc[:, ret_nq_all_m.columns.notna()].fillna(ret_nq_all_m.mean())
_drop_nq_d = [c for c in _before_nq_d if c not in ret_nq_all_d.columns]
_drop_nq_m = [c for c in _before_nq_m if c not in ret_nq_all_m.columns]
if _drop_nq_d: print(f"  Dropped Nasdaq daily  (coverage <80%): {_drop_nq_d}")
if _drop_nq_m: print(f"  Dropped Nasdaq monthly(coverage <80%): {_drop_nq_m}")

stats_nq_all_d = desc_stats(ret_nq_all_d, 252)
stats_nq_all_m = desc_stats(ret_nq_all_m, 12)

print(f"\n  Nasdaq daily  : {ret_nq_all_d.shape[1]} stocks, {ret_nq_all_d.shape[0]} obs")
print(f"  Nasdaq monthly: {ret_nq_all_m.shape[1]} stocks, {ret_nq_all_m.shape[0]} obs")
print("\n--- Nasdaq Daily: Descriptive Statistics ---")
print(stats_nq_all_d.round(6).to_string())
print("\n--- Nasdaq Monthly: Descriptive Statistics ---")
print(stats_nq_all_m.round(6).to_string())

plot_prices_all(px_nq_d_all, f"Nasdaq Daily Prices ({px_nq_d_all.shape[1]} securities)",
                "nasdaq_prices_daily.png")
plot_prices_all(px_nq_m_all, f"Nasdaq Monthly Prices ({px_nq_m_all.shape[1]} securities)",
                "nasdaq_prices_monthly.png")
plot_returns_all(ret_nq_all_d, f"Nasdaq Daily Returns ({ret_nq_all_d.shape[1]} securities)",
                 "nasdaq_returns_daily.png")
plot_returns_all(ret_nq_all_m, f"Nasdaq Monthly Returns ({ret_nq_all_m.shape[1]} securities)",
                 "nasdaq_returns_monthly.png")
plot_comparison_all(px_nq_d_all, px_nq_m_all, ret_nq_all_d, ret_nq_all_m, "Nasdaq")

# =============================================================================
# Q2 - NYSE: RETURNS AND DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*72)
print("  Q2 - NYSE: Returns and Descriptive Statistics (full universe)")
print("="*72)

ret_ny_all_d = px_ny_d_all.pct_change().iloc[1:]
ret_ny_all_m = px_ny_m_all.pct_change().iloc[1:]
_before_ny_d = set(ret_ny_all_d.columns)
_before_ny_m = set(ret_ny_all_m.columns)
ret_ny_all_d = ret_ny_all_d.dropna(axis=1, thresh=int(0.8*len(ret_ny_all_d)))
ret_ny_all_m = ret_ny_all_m.dropna(axis=1, thresh=int(0.8*len(ret_ny_all_m)))
ret_ny_all_d = ret_ny_all_d.loc[:, ret_ny_all_d.columns.notna()].fillna(ret_ny_all_d.mean())
ret_ny_all_m = ret_ny_all_m.loc[:, ret_ny_all_m.columns.notna()].fillna(ret_ny_all_m.mean())
_drop_ny_d = [c for c in _before_ny_d if c not in ret_ny_all_d.columns]
_drop_ny_m = [c for c in _before_ny_m if c not in ret_ny_all_m.columns]
if _drop_ny_d: print(f"  Dropped NYSE daily  (coverage <80%): {_drop_ny_d}")
if _drop_ny_m: print(f"  Dropped NYSE monthly(coverage <80%): {_drop_ny_m}")

stats_ny_all_d = desc_stats(ret_ny_all_d, 252)
stats_ny_all_m = desc_stats(ret_ny_all_m, 12)

print(f"\n  NYSE daily  : {ret_ny_all_d.shape[1]} stocks, {ret_ny_all_d.shape[0]} obs")
print(f"  NYSE monthly: {ret_ny_all_m.shape[1]} stocks, {ret_ny_all_m.shape[0]} obs")
print("\n--- NYSE Daily: Descriptive Statistics ---")
print(stats_ny_all_d.round(6).to_string())
print("\n--- NYSE Monthly: Descriptive Statistics ---")
print(stats_ny_all_m.round(6).to_string())

plot_prices_all(px_ny_d_all, f"NYSE Daily Prices ({px_ny_d_all.shape[1]} securities)",
                "nyse_prices_daily_all.png")
plot_prices_all(px_ny_m_all, f"NYSE Monthly Prices ({px_ny_m_all.shape[1]} securities)",
                "nyse_prices_monthly_all.png")
plot_returns_all(ret_ny_all_d, f"NYSE Daily Returns ({ret_ny_all_d.shape[1]} securities)",
                 "nyse_returns_daily.png")
plot_returns_all(ret_ny_all_m, f"NYSE Monthly Returns ({ret_ny_all_m.shape[1]} securities)",
                 "nyse_returns_monthly.png")
plot_comparison_all(px_ny_d_all, px_ny_m_all, ret_ny_all_d, ret_ny_all_m, "NYSE")

# =============================================================================
# Q3 - VARIANCE-COVARIANCE AND CORRELATION MATRICES (full universe)
# =============================================================================
print("\n" + "="*72)
print("  Q3 - Variance-Covariance and Correlation Matrices")
print("="*72)

cov_nq_all_d  = ret_nq_all_d.cov();  corr_nq_all_d = ret_nq_all_d.corr()
cov_nq_all_m  = ret_nq_all_m.cov();  corr_nq_all_m = ret_nq_all_m.corr()
cov_ny_all_d  = ret_ny_all_d.cov();  corr_ny_all_d = ret_ny_all_d.corr()
cov_ny_all_m  = ret_ny_all_m.cov();  corr_ny_all_m = ret_ny_all_m.corr()

print(f"\n  Nasdaq daily  cov/corr: {cov_nq_all_d.shape}")
print(f"  Nasdaq monthly cov/corr: {cov_nq_all_m.shape}")
print(f"  NYSE   daily  cov/corr: {cov_ny_all_d.shape}")
print(f"  NYSE   monthly cov/corr: {cov_ny_all_m.shape}")
print("\n--- Nasdaq Daily Corr (first 10) ---")
print(corr_nq_all_d.iloc[:10, :10].round(4).to_string())
print("\n--- NYSE Daily Corr (first 10) ---")
print(corr_ny_all_d.iloc[:10, :10].round(4).to_string())

# Covariance heatmaps
plot_heatmap(cov_nq_all_d,  "Nasdaq Daily - Covariance Matrix",
             "cov_nasdaq_daily.png",   use_abs=False, cmap="viridis",
             vmin=cov_nq_all_d.values.min(), vmax=cov_nq_all_d.values.max())
plot_heatmap(cov_nq_all_m,  "Nasdaq Monthly - Covariance Matrix",
             "cov_nasdaq_monthly.png", use_abs=False, cmap="viridis",
             vmin=cov_nq_all_m.values.min(), vmax=cov_nq_all_m.values.max())
plot_heatmap(cov_ny_all_d,  "NYSE Daily - Covariance Matrix",
             "cov_nyse_daily.png",     use_abs=False, cmap="viridis",
             vmin=cov_ny_all_d.values.min(), vmax=cov_ny_all_d.values.max())
plot_heatmap(cov_ny_all_m,  "NYSE Monthly - Covariance Matrix",
             "cov_nyse_monthly.png",   use_abs=False, cmap="viridis",
             vmin=cov_ny_all_m.values.min(), vmax=cov_ny_all_m.values.max())

# Absolute correlation heatmaps
plot_heatmap(corr_nq_all_d, "Nasdaq Daily - Correlation |rho|",
             "corr_abs_nasdaq_daily.png",   use_abs=True)
plot_heatmap(corr_nq_all_m, "Nasdaq Monthly - Correlation |rho|",
             "corr_abs_nasdaq_monthly.png", use_abs=True)
plot_heatmap(corr_ny_all_d, "NYSE Daily - Correlation |rho|",
             "corr_abs_nyse_daily.png",     use_abs=True)
plot_heatmap(corr_ny_all_m, "NYSE Monthly - Correlation |rho|",
             "corr_abs_nyse_monthly.png",   use_abs=True)

# Signed correlation heatmaps
plot_heatmap(corr_nq_all_d, "Nasdaq Daily - Correlation (signed)",
             "corr_nasdaq_daily.png",   use_abs=False)
plot_heatmap(corr_nq_all_m, "Nasdaq Monthly - Correlation (signed)",
             "corr_nasdaq_monthly.png", use_abs=False)
plot_heatmap(corr_ny_all_d, "NYSE Daily - Correlation (signed)",
             "corr_nyse_daily.png",     use_abs=False)
plot_heatmap(corr_ny_all_m, "NYSE Monthly - Correlation (signed)",
             "corr_nyse_monthly.png",   use_abs=False)

# =============================================================================
# Q4 / Q5 - SECURITY SELECTION
# =============================================================================
print("\n" + "="*72)
print("  Q4 - Nasdaq Security Selection")
print("="*72)

sel_nq, ret_nq_d, ret_nq_m, ppx_nq_d, ppx_nq_m = \
    get_selected_returns(px_nq_d_all, px_nq_m_all, NASDAQ_SELECTION)
sel_ny, ret_ny_d, ret_ny_m, ppx_ny_d, ppx_ny_m = \
    get_selected_returns(px_ny_d_all, px_ny_m_all, NYSE_SELECTION)

stats_nq_d = desc_stats(ret_nq_d, 252); stats_nq_m = desc_stats(ret_nq_m, 12)
stats_ny_d = desc_stats(ret_ny_d, 252); stats_ny_m = desc_stats(ret_ny_m, 12)
corr_nq_d  = ret_nq_d.corr();           corr_ny_d  = ret_ny_d.corr()

print(f"\n  {'#':<4} {'Name':<38} {'Mean(ann)%':>10} {'Std(ann)%':>9} {'Sharpe':>7}  Sector")
print("  " + "-"*84)
for i, s in enumerate(sel_nq, 1):
    m = NASDAQ_META.get(s, {})
    mn = f"{m['Mean_ann']*100:.1f}%" if m.get("Mean_ann") is not None else "N/A"
    sd = f"{m['Std_ann']*100:.1f}%"  if m.get("Std_ann")  is not None else "N/A"
    sr = f"{m['Sharpe']:.2f}"         if m.get("Sharpe")  is not None else "N/A"
    print(f"  {i:<4} {s:<38} {mn:>10} {sd:>9} {sr:>7}  {m.get('Sector','')}")

print("\n" + "="*72)
print("  Q5 - NYSE Security Selection")
print("="*72)
print(f"\n  {'#':<4} {'Name':<38} {'Mean(ann)%':>10} {'Std(ann)%':>9} {'Sharpe':>7}  Sector")
print("  " + "-"*84)
for i, s in enumerate(sel_ny, 1):
    m = NYSE_META.get(s, {})
    mn = f"{m['Mean_ann']*100:.1f}%" if m.get("Mean_ann") is not None else "N/A"
    sd = f"{m['Std_ann']*100:.1f}%"  if m.get("Std_ann")  is not None else "N/A"
    sr = f"{m['Sharpe']:.2f}"         if m.get("Sharpe")  is not None else "N/A"
    print(f"  {i:<4} {s:<38} {mn:>10} {sd:>9} {sr:>7}  {m.get('Sector','')}")

# =============================================================================
# Q6 - PRICE PLOTS: SELECTED SECURITIES
# =============================================================================
print("\n" + "="*72)
print("  Q6 - Price Plots: Selected Securities")
print("="*72)

plot_individual_prices(sel_nq, ppx_nq_d, NASDAQ_META, "Q6 - Nasdaq Daily Prices",   "nasdaq_sel_prices_daily.png")
plot_individual_prices(sel_nq, ppx_nq_m, NASDAQ_META, "Q6 - Nasdaq Monthly Prices", "nasdaq_sel_prices_monthly.png")
plot_individual_prices(sel_ny, ppx_ny_d, NYSE_META,   "Q6 - NYSE Daily Prices",     "nyse_sel_prices_daily.png")
plot_individual_prices(sel_ny, ppx_ny_m, NYSE_META,   "Q6 - NYSE Monthly Prices",   "nyse_sel_prices_monthly.png")

# =============================================================================
# Q7 / Q8 - MV OPTIMAL PORTFOLIO (unconstrained)
# =============================================================================
print("\n" + "="*72)
print("  Q7 - NYSE: MV Optimal Portfolio")
print("="*72)

cov_nq_d = ret_nq_d.cov(); mu_nq_d = ret_nq_d.mean().values
cov_nq_m = ret_nq_m.cov(); mu_nq_m = ret_nq_m.mean().values
cov_ny_d = ret_ny_d.cov(); mu_ny_d = ret_ny_d.mean().values
cov_ny_m = ret_ny_m.cov(); mu_ny_m = ret_ny_m.mean().values

w_ny_unc_d = tangency_unconstrained(mu_ny_d, cov_ny_d.values, RF_DAILY)
w_ny_unc_m = tangency_unconstrained(mu_ny_m, cov_ny_m.values, RF_MONTHLY)

print_weights_table(sel_ny, w_ny_unc_d, "NYSE Daily")
print_weights_table(sel_ny, w_ny_unc_m, "NYSE Monthly")
for w, ann, lbl in [(w_ny_unc_d, 252, "Daily"), (w_ny_unc_m, 12, "Monthly")]:
    cov = cov_ny_d if ann == 252 else cov_ny_m; mu = mu_ny_d if ann == 252 else mu_ny_m
    er = w@mu*ann*100; sd = np.sqrt(w@cov.values@w)*np.sqrt(ann)*100
    print(f"  {lbl}: E[R]={er:.3f}%  sigma={sd:.3f}%  SR={(er-RF_ANNUAL*100)/sd:.4f}")

print("\n" + "="*72)
print("  Q8 - Nasdaq: MV Optimal Portfolio")
print("="*72)

w_nq_unc_d = tangency_unconstrained(mu_nq_d, cov_nq_d.values, RF_DAILY)
w_nq_unc_m = tangency_unconstrained(mu_nq_m, cov_nq_m.values, RF_MONTHLY)

print_weights_table(sel_nq, w_nq_unc_d, "Nasdaq Daily")
print_weights_table(sel_nq, w_nq_unc_m, "Nasdaq Monthly")
for w, ann, lbl in [(w_nq_unc_d, 252, "Daily"), (w_nq_unc_m, 12, "Monthly")]:
    cov = cov_nq_d if ann == 252 else cov_nq_m; mu = mu_nq_d if ann == 252 else mu_nq_m
    er = w@mu*ann*100; sd = np.sqrt(w@cov.values@w)*np.sqrt(ann)*100
    print(f"  {lbl}: E[R]={er:.3f}%  sigma={sd:.3f}%  SR={(er-RF_ANNUAL*100)/sd:.4f}")

# =============================================================================
# Q9 / Q10 - MV OPTIMAL PORTFOLIO (constrained, w >= 0)
# =============================================================================
print("\n" + "="*72)
print("  Q9 - Nasdaq: MV Optimal Portfolio (constrained, w >= 0)")
print("="*72)

w_nq_con_d = tangency_constrained(mu_nq_d, cov_nq_d.values, RF_DAILY)
w_nq_con_m = tangency_constrained(mu_nq_m, cov_nq_m.values, RF_MONTHLY)

print_weights_table(sel_nq, w_nq_con_d, "Nasdaq Daily (constrained)")
print_weights_table(sel_nq, w_nq_con_m, "Nasdaq Monthly (constrained)")
for w, ann, lbl in [(w_nq_con_d, 252, "Daily"), (w_nq_con_m, 12, "Monthly")]:
    cov = cov_nq_d if ann == 252 else cov_nq_m; mu = mu_nq_d if ann == 252 else mu_nq_m
    er = w@mu*ann*100; sd = np.sqrt(w@cov.values@w)*np.sqrt(ann)*100
    print(f"  {lbl}: E[R]={er:.3f}%  sigma={sd:.3f}%  SR={(er-RF_ANNUAL*100)/sd:.4f}")

print("\n" + "="*72)
print("  Q10 - NYSE: MV Optimal Portfolio (constrained, w >= 0)")
print("="*72)

w_ny_con_d = tangency_constrained(mu_ny_d, cov_ny_d.values, RF_DAILY)
w_ny_con_m = tangency_constrained(mu_ny_m, cov_ny_m.values, RF_MONTHLY)

print_weights_table(sel_ny, w_ny_con_d, "NYSE Daily (constrained)")
print_weights_table(sel_ny, w_ny_con_m, "NYSE Monthly (constrained)")
for w, ann, lbl in [(w_ny_con_d, 252, "Daily"), (w_ny_con_m, 12, "Monthly")]:
    cov = cov_ny_d if ann == 252 else cov_ny_m; mu = mu_ny_d if ann == 252 else mu_ny_m
    er = w@mu*ann*100; sd = np.sqrt(w@cov.values@w)*np.sqrt(ann)*100
    print(f"  {lbl}: E[R]={er:.3f}%  sigma={sd:.3f}%  SR={(er-RF_ANNUAL*100)/sd:.4f}")

# Portfolio return series
pr_ny_unc_d = ret_ny_d.values @ w_ny_unc_d; pr_ny_con_d = ret_ny_d.values @ w_ny_con_d
pr_ny_unc_m = ret_ny_m.values @ w_ny_unc_m; pr_ny_con_m = ret_ny_m.values @ w_ny_con_m
pr_nq_unc_d = ret_nq_d.values @ w_nq_unc_d; pr_nq_con_d = ret_nq_d.values @ w_nq_con_d
pr_nq_unc_m = ret_nq_m.values @ w_nq_unc_m; pr_nq_con_m = ret_nq_m.values @ w_nq_con_m

# =============================================================================
# Q11 / Q12 - PORTFOLIO STATISTICS
# =============================================================================
print("\n" + "="*72)
print("  Q11 - NYSE: Portfolio Statistics")
print("="*72)

ps_ny = pd.DataFrame({
    "Unc_Daily":   port_stats(pr_ny_unc_d, 252), "Con_Daily":   port_stats(pr_ny_con_d, 252),
    "Unc_Monthly": port_stats(pr_ny_unc_m, 12),  "Con_Monthly": port_stats(pr_ny_con_m, 12),
})
print(ps_ny.round(6).to_string())

print("\n" + "="*72)
print("  Q12 - Nasdaq: Portfolio Statistics")
print("="*72)

ps_nq = pd.DataFrame({
    "Unc_Daily":   port_stats(pr_nq_unc_d, 252), "Con_Daily":   port_stats(pr_nq_con_d, 252),
    "Unc_Monthly": port_stats(pr_nq_unc_m, 12),  "Con_Monthly": port_stats(pr_nq_con_m, 12),
})
print(ps_nq.round(6).to_string())

# =============================================================================
# Q13 / Q14 - EFFICIENT FRONTIER
# =============================================================================
print("\n" + "="*72)
print("  Q13 - Nasdaq: Efficient Frontier")
print("="*72)
print("  Computing frontiers ...")

fs_nq_unc_d, fr_nq_unc_d = compute_frontier(mu_nq_d, cov_nq_d.values, constrained=False)
fs_nq_con_d, fr_nq_con_d = compute_frontier(mu_nq_d, cov_nq_d.values, constrained=True)
fs_nq_unc_m, fr_nq_unc_m = compute_frontier(mu_nq_m, cov_nq_m.values, constrained=False)
fs_nq_con_m, fr_nq_con_m = compute_frontier(mu_nq_m, cov_nq_m.values, constrained=True)

plot_frontier(fs_nq_unc_d, fr_nq_unc_d, fs_nq_con_d, fr_nq_con_d,
              mu_nq_d, cov_nq_d.values, w_nq_unc_d, w_nq_con_d, 252,
              "Q13 - Nasdaq: Efficient Frontier (Daily)", "nasdaq_frontier_daily.png")
plot_frontier(fs_nq_unc_m, fr_nq_unc_m, fs_nq_con_m, fr_nq_con_m,
              mu_nq_m, cov_nq_m.values, w_nq_unc_m, w_nq_con_m, 12,
              "Q13 - Nasdaq: Efficient Frontier (Monthly)", "nasdaq_frontier_monthly.png")

print("\n" + "="*72)
print("  Q14 - NYSE: Efficient Frontier")
print("="*72)
print("  Computing frontiers ...")

fs_ny_unc_d, fr_ny_unc_d = compute_frontier(mu_ny_d, cov_ny_d.values, constrained=False)
fs_ny_con_d, fr_ny_con_d = compute_frontier(mu_ny_d, cov_ny_d.values, constrained=True)
fs_ny_unc_m, fr_ny_unc_m = compute_frontier(mu_ny_m, cov_ny_m.values, constrained=False)
fs_ny_con_m, fr_ny_con_m = compute_frontier(mu_ny_m, cov_ny_m.values, constrained=True)

plot_frontier(fs_ny_unc_d, fr_ny_unc_d, fs_ny_con_d, fr_ny_con_d,
              mu_ny_d, cov_ny_d.values, w_ny_unc_d, w_ny_con_d, 252,
              "Q14 - NYSE: Efficient Frontier (Daily)", "nyse_frontier_daily.png")
plot_frontier(fs_ny_unc_m, fr_ny_unc_m, fs_ny_con_m, fr_ny_con_m,
              mu_ny_m, cov_ny_m.values, w_ny_unc_m, w_ny_con_m, 12,
              "Q14 - NYSE: Efficient Frontier (Monthly)", "nyse_frontier_monthly.png")

# =============================================================================
# Q15 - INDEX STATISTICS AND COMPARISON WITH PORTFOLIOS
# =============================================================================
print("\n" + "="*72)
print("  Q15 - Index Statistics and Comparison with Portfolios")
print("="*72)

ri_nq_d_PI = nasdaq_idx_d["NASDAQ_PI"].pct_change().dropna()
ri_nq_d_RI = nasdaq_idx_d["NASDAQ_RI"].pct_change().dropna()
ri_nq_m_PI = nasdaq_idx_m["NASDAQ_PI"].pct_change().dropna()
ri_nq_m_RI = nasdaq_idx_m["NASDAQ_RI"].pct_change().dropna()
ri_sp_d_PI  = sp500_d["SP500_PI"].pct_change().dropna()
ri_sp_d_RI  = sp500_d["SP500_RI"].pct_change().dropna()
ri_sp_m_PI  = sp500_m["SP500_PI"].pct_change().dropna()
ri_sp_m_RI  = sp500_m["SP500_RI"].pct_change().dropna()

# desc_stats handles pd.Series correctly (isinstance branch)
idx_stats = pd.DataFrame({
    "NASDAQ_PI_Daily": desc_stats(ri_nq_d_PI, 252),
    "NASDAQ_RI_Daily": desc_stats(ri_nq_d_RI, 252),
    "NASDAQ_PI_Mon":   desc_stats(ri_nq_m_PI, 12),
    "NASDAQ_RI_Mon":   desc_stats(ri_nq_m_RI, 12),
    "SP500_PI_Daily":  desc_stats(ri_sp_d_PI,  252),
    "SP500_RI_Daily":  desc_stats(ri_sp_d_RI,  252),
    "SP500_PI_Mon":    desc_stats(ri_sp_m_PI,  12),
    "SP500_RI_Mon":    desc_stats(ri_sp_m_RI,  12),
})
print("\n--- Index Statistics ---")
print(idx_stats.round(6).to_string())

ew_nq_d = ret_nq_d.mean(axis=1); ew_nq_m = ret_nq_m.mean(axis=1)
ew_ny_d = ret_ny_d.mean(axis=1); ew_ny_m = ret_ny_m.mean(axis=1)

print("\n--- NASDAQ EW portfolio vs NASDAQ Composite (Daily) ---")
cmp_nd = pd.DataFrame({
    "EW_Port":  desc_stats(ew_nq_d, 252),
    "NASDAQ_PI": desc_stats(ri_nq_d_PI, 252),
    "NASDAQ_RI": desc_stats(ri_nq_d_RI, 252),
    "Diff":      desc_stats(ew_nq_d, 252) - desc_stats(ri_nq_d_PI, 252),
})
print(cmp_nd.round(6).to_string())

print("\n--- NYSE EW portfolio vs S&P500 (Daily) ---")
cmp_yd = pd.DataFrame({
    "EW_Port":  desc_stats(ew_ny_d, 252),
    "SP500_PI": desc_stats(ri_sp_d_PI, 252),
    "SP500_RI": desc_stats(ri_sp_d_RI, 252),
    "Diff":     desc_stats(ew_ny_d, 252) - desc_stats(ri_sp_d_PI, 252),
})
print(cmp_yd.round(6).to_string())

plot_cumulative(ew_nq_d, ri_nq_d_PI, ri_nq_d_RI,
                "Q15 - Nasdaq vs NASDAQ Composite (Daily)",   "cumulative_nasdaq_daily.png")
plot_cumulative(ew_nq_m, ri_nq_m_PI, ri_nq_m_RI,
                "Q15 - Nasdaq vs NASDAQ Composite (Monthly)", "cumulative_nasdaq_monthly.png")
plot_cumulative(ew_ny_d, ri_sp_d_PI, ri_sp_d_RI,
                "Q15 - NYSE vs S&P500 (Daily)",               "cumulative_nyse_daily.png")
plot_cumulative(ew_ny_m, ri_sp_m_PI, ri_sp_m_RI,
                "Q15 - NYSE vs S&P500 (Monthly)",             "cumulative_nyse_monthly.png")

# =============================================================================
# Q16 / Q17 - BETA
# =============================================================================
print("\n" + "="*72)
print("  Q16 - NYSE: Beta per Security and Portfolio Beta")
print("="*72)

bm_ny_d = sp500_d["SP500_RI"].pct_change().dropna()
bm_ny_m = sp500_m["SP500_RI"].pct_change().dropna()
bm_nq_d = nasdaq_idx_d["NASDAQ_RI"].pct_change().dropna()
bm_nq_m = nasdaq_idx_m["NASDAQ_RI"].pct_change().dropna()

beta_ny_d = compute_betas(ret_ny_d.reindex(bm_ny_d.index).dropna(how="all"),
                           bm_ny_d.reindex(ret_ny_d.index).dropna().values)
beta_ny_m = compute_betas_monthly(ret_ny_m, bm_ny_m)
beta_nq_d = compute_betas(ret_nq_d.reindex(bm_nq_d.index).dropna(how="all"),
                           bm_nq_d.reindex(ret_nq_d.index).dropna().values)
beta_nq_m = compute_betas_monthly(ret_nq_m, bm_nq_m)

print_beta_table(sel_ny, beta_ny_d, beta_ny_m, w_ny_unc_d, w_ny_con_d,
                 w_ny_unc_m, w_ny_con_m, "NYSE  |  Market: SP500 RI")
plot_betas(sel_ny, beta_ny_d, beta_ny_m, "Q16 - NYSE: Security Betas", "nyse_betas.png")

print("\n" + "="*72)
print("  Q17 - Nasdaq: Beta per Security and Portfolio Beta")
print("="*72)

print_beta_table(sel_nq, beta_nq_d, beta_nq_m, w_nq_unc_d, w_nq_con_d,
                 w_nq_unc_m, w_nq_con_m, "Nasdaq  |  Market: NASDAQ RI")
plot_betas(sel_nq, beta_nq_d, beta_nq_m, "Q17 - Nasdaq: Security Betas", "nasdaq_betas.png")

# =============================================================================
# Q18 / Q19 - SECURITY MARKET LINE
# =============================================================================
print("\n" + "="*72)
print("  Q18 - NYSE: Security Market Line")
print("="*72)

stats_nq_d_ann = desc_stats_ann(ret_nq_d, 252); stats_nq_m_ann = desc_stats_ann(ret_nq_m, 12)
stats_ny_d_ann = desc_stats_ann(ret_ny_d, 252); stats_ny_m_ann = desc_stats_ann(ret_ny_m, 12)

ny_chosen = [beta_ny_d.reindex(sel_ny).idxmin(), beta_ny_d.reindex(sel_ny).idxmax()]
nq_chosen = [beta_nq_d.reindex(sel_nq).idxmin(), beta_nq_d.reindex(sel_nq).idxmax()]

mkt_ny_d = np.mean(bm_ny_d.values)*252; mrp_ny_d = mkt_ny_d - RF_ANNUAL
bm_ny_m_al = bm_monthly_aligned(bm_ny_m, ret_ny_m)
bm_nq_m_al = bm_monthly_aligned(bm_nq_m, ret_nq_m)
mkt_ny_m = np.mean(bm_ny_m_al.values)*12; mrp_ny_m = mkt_ny_m - RF_ANNUAL

pb_ny_unc_d, pb_ny_con_d = sml_table(sel_ny, stats_ny_d_ann, beta_ny_d,
                                       w_ny_unc_d, w_ny_con_d, mkt_ny_d, mrp_ny_d, ny_chosen, "Daily")
pb_ny_unc_m, pb_ny_con_m = sml_table(sel_ny, stats_ny_m_ann, beta_ny_m,
                                       w_ny_unc_m, w_ny_con_m, mkt_ny_m, mrp_ny_m, ny_chosen, "Monthly")

plot_sml(sel_ny, stats_ny_d_ann, beta_ny_d, w_ny_unc_d, w_ny_con_d,
         mkt_ny_d, mrp_ny_d, ny_chosen, pb_ny_unc_d, pb_ny_con_d,
         "Q18 - NYSE: SML (Daily)", "nyse_sml_daily.png")
plot_sml(sel_ny, stats_ny_m_ann, beta_ny_m, w_ny_unc_m, w_ny_con_m,
         mkt_ny_m, mrp_ny_m, ny_chosen, pb_ny_unc_m, pb_ny_con_m,
         "Q18 - NYSE: SML (Monthly)", "nyse_sml_monthly.png")

print("\n" + "="*72)
print("  Q19 - Nasdaq: Security Market Line")
print("="*72)

mkt_nq_d = np.mean(bm_nq_d.values)*252; mrp_nq_d = mkt_nq_d - RF_ANNUAL
mkt_nq_m = np.mean(bm_nq_m_al.values)*12; mrp_nq_m = mkt_nq_m - RF_ANNUAL

pb_nq_unc_d, pb_nq_con_d = sml_table(sel_nq, stats_nq_d_ann, beta_nq_d,
                                       w_nq_unc_d, w_nq_con_d, mkt_nq_d, mrp_nq_d, nq_chosen, "Daily")
pb_nq_unc_m, pb_nq_con_m = sml_table(sel_nq, stats_nq_m_ann, beta_nq_m,
                                       w_nq_unc_m, w_nq_con_m, mkt_nq_m, mrp_nq_m, nq_chosen, "Monthly")

plot_sml(sel_nq, stats_nq_d_ann, beta_nq_d, w_nq_unc_d, w_nq_con_d,
         mkt_nq_d, mrp_nq_d, nq_chosen, pb_nq_unc_d, pb_nq_con_d,
         "Q19 - Nasdaq: SML (Daily)", "nasdaq_sml_daily.png")
plot_sml(sel_nq, stats_nq_m_ann, beta_nq_m, w_nq_unc_m, w_nq_con_m,
         mkt_nq_m, mrp_nq_m, nq_chosen, pb_nq_unc_m, pb_nq_con_m,
         "Q19 - Nasdaq: SML (Monthly)", "nasdaq_sml_monthly.png")

# =============================================================================
# SWITCH TO FINAL SELECTIONS (Q20-Q24)
# =============================================================================
print("\n[Switching to final security selection for Q20-Q24]")
ret_nd, sel_nd = get_returns_exact(px_nq_d_all, NASDAQ_FINAL)
ret_nm, sel_nm = get_returns_exact(px_nq_m_all, NASDAQ_FINAL)
ret_yd, sel_yd = get_returns_exact(px_ny_d_all, NYSE_FINAL)
ret_ym, sel_ym = get_returns_exact(px_ny_m_all, NYSE_FINAL)
print(f"  Nasdaq daily  : {len(sel_nd)} stocks  |  Nasdaq monthly: {len(sel_nm)} stocks")
print(f"  NYSE   daily  : {len(sel_yd)} stocks  |  NYSE   monthly: {len(sel_ym)} stocks")

# =============================================================================
# BAYESIAN ALLOCATION (Standard model - known Sigma, unknown mean)
# =============================================================================
print("\n" + "="*72)
print("  BAYESIAN ALLOCATION (Standard model - known Sigma, unknown mean)")
print("="*72)

def bayesian_standard(ret_df, sel, ann, rf, label, color="steelblue"):
    N = len(sel); T = len(ret_df)
    mu_hat = ret_df.mean().values
    Sigma  = ret_df.cov().values
    mu_0   = np.full(N, np.mean(mu_hat)); Sigma_0 = 2.0 * Sigma
    Si0 = inv(Sigma_0); Si = inv(Sigma)
    Sigma_post = inv(Si0 + T*Si)
    mu_post    = Sigma_post @ (Si0 @ mu_0 + T*Si @ mu_hat)
    print(f"\n  [{label}]  T={T}, N={N}")
    print(f"  {'Stock':<42} {'mu_hat%':>8} {'mu_0%':>8} {'mu_post%':>9} {'Shrink':>8}")
    print("  " + "-"*79)
    for s, mh, m0, mp in zip(sel, mu_hat, mu_0, mu_post):
        shrink = (mp - mh) / (m0 - mh) if abs(m0 - mh) > 1e-10 else 0
        print(f"  {s:<42} {mh*ann*100:8.3f} {m0*ann*100:8.3f} {mp*ann*100:9.3f} {shrink:8.4f}")
    exc_post = mu_post - rf * np.ones(N)
    z = inv(Sigma) @ exc_post; denom = np.ones(N) @ z
    w_unc = z / denom if abs(denom) > 1e-12 else np.ones(N) / N
    def neg_sr(w):
        r = w @ mu_post; s = np.sqrt(max(w @ Sigma @ w, 1e-14)); return -(r - rf) / s
    res = minimize(neg_sr, np.ones(N)/N, method="SLSQP", bounds=[(0, 1)]*N,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 3000})
    w_con = res.x if res.success else np.ones(N) / N
    r_unc = ret_df.values @ w_unc; r_con = ret_df.values @ w_con
    print(f"\n  Statistics - {label}:")
    print(pd.DataFrame({"Bayes_Unc": port_stats(r_unc, ann),
                         "Bayes_Con": port_stats(r_con, ann)}).round(6).to_string())
    plot_weights(w_con, sel, f"Bayesian Constrained - {label}",
                 f"bayes_con_{label.lower().replace(' ', '_')}.png", color)
    return w_unc, w_con, r_unc, r_con, mu_post

w_Bayes_nd_unc, w_Bayes_nd_con, r_Bayes_nd_unc, r_Bayes_nd_con, mu_post_nd = \
    bayesian_standard(ret_nd, sel_nd, 252, RF_DAILY,   "Nasdaq Daily",   "darkorange")
w_Bayes_nm_unc, w_Bayes_nm_con, r_Bayes_nm_unc, r_Bayes_nm_con, mu_post_nm = \
    bayesian_standard(ret_nm, sel_nm, 12,  RF_MONTHLY, "Nasdaq Monthly", "darkorange")
w_Bayes_yd_unc, w_Bayes_yd_con, r_Bayes_yd_unc, r_Bayes_yd_con, mu_post_yd = \
    bayesian_standard(ret_yd, sel_yd, 252, RF_DAILY,   "NYSE Daily",     "steelblue")
w_Bayes_ym_unc, w_Bayes_ym_con, r_Bayes_ym_unc, r_Bayes_ym_con, mu_post_ym = \
    bayesian_standard(ret_ym, sel_ym, 12,  RF_MONTHLY, "NYSE Monthly",   "steelblue")

# =============================================================================
# Q20 / Q21 - BLACK-LITTERMAN
# =============================================================================

def views_nasdaq(sel, mu, Sigma, ann):
    N = len(sel); P, q, lbl = [], [], []
    def ix(n): return sel.index(n) if n in sel else None
    for name, pct, desc in [("NVIDIA", 0.80, "AI demand"),
                              ("GILEAD SCIENCES", 0.10, "defensive pharma")]:
        i = ix(name)
        if i is not None:
            r = np.zeros(N); r[i] = 1.0; P.append(r); q.append(pct/ann)
            lbl.append(f"ABS: {name} -> {pct*100:.0f}% [{desc}]")
    for hi, lo, sp, desc in [("BROADCOM", "CME GROUP", 0.20, "semi>derivatives"),
                               ("PALO ALTO NETWORKS", "BAKER HUGHES A", 0.10, "cyber>energy")]:
        ih = ix(hi); il = ix(lo)
        if ih is not None and il is not None:
            r = np.zeros(N); r[ih] = 1.0; r[il] = -1.0; P.append(r); q.append(sp/ann)
            lbl.append(f"REL: {hi}-{lo}={sp*100:.0f}% [{desc}]")
    return np.array(P), np.array(q), lbl

def views_nyse(sel, mu, Sigma, ann):
    N = len(sel); P, q, lbl = [], [], []
    def ix(n): return sel.index(n) if n in sel else None
    for name, pct, desc in [("ELI LILLY", 0.35, "GLP-1 drugs"),
                              ("COCA COLA", 0.06, "defensive")]:
        i = ix(name)
        if i is not None:
            r = np.zeros(N); r[i] = 1.0; P.append(r); q.append(pct/ann)
            lbl.append(f"ABS: {name} -> {pct*100:.0f}% [{desc}]")
    for hi, lo, sp, desc in [
            ("CARPENTER TECH.", "MURPHY OIL", 0.15, "specialty steel > oil exploration"),
            ("GENERAL DYNAMICS", "MASTERCARD", 0.05, "defense spending > digital payments")]:
        ih = ix(hi); il = ix(lo)
        if ih is not None and il is not None:
            r = np.zeros(N); r[ih] = 1.0; r[il] = -1.0; P.append(r); q.append(sp/ann)
            lbl.append(f"REL: {hi}-{lo}={sp*100:.0f}% [{desc}]")
    return np.array(P), np.array(q), lbl

def black_litterman(ret_df, sel, ann, rf, label, views_fn):
    mu = ret_df.mean().values; Sigma = ret_df.cov().values; N = len(sel)
    w_eq = np.ones(N) / N; pi_eq = DELTA * Sigma @ w_eq
    P, q, vlabels = views_fn(sel, mu, Sigma, ann)
    Omega = np.diag(np.diag(TAU * P @ Sigma @ P.T))
    print(f"\n  [{label}] Views:"); [print(f"    {l}") for l in vlabels]
    tSi = inv(TAU * Sigma); OmInv = inv(Omega)
    Sigma_BL = inv(tSi + P.T @ OmInv @ P)
    mu_BL    = Sigma_BL @ (tSi @ pi_eq + P.T @ OmInv @ q)
    print(f"\n  {'Stock':<42} {'pi_eq%':>8} {'mu_BL%':>8}")
    for s, pe, mb in zip(sel, pi_eq, mu_BL):
        print(f"  {s:<42} {pe*ann*100:8.3f} {mb*ann*100:8.3f}")
    w_raw = inv(Sigma) @ mu_BL / DELTA; denom = w_raw.sum()
    w_unc = w_raw / denom if abs(denom) > 1e-12 else w_raw
    def neg_sr(w):
        r = w @ mu_BL; s = np.sqrt(max(w @ Sigma @ w, 1e-14)); return -(r - rf) / s
    res = minimize(neg_sr, np.ones(N)/N, method="SLSQP", bounds=[(0, 1)]*N,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 3000})
    w_con = res.x if res.success else np.ones(N) / N
    r_unc = ret_df.values @ w_unc; r_con = ret_df.values @ w_con
    print(f"\n  Statistics [{label}]:")
    print(pd.DataFrame({"BL_Unc": port_stats(r_unc, ann),
                         "BL_Con": port_stats(r_con, ann)}).round(6).to_string())
    return w_unc, w_con, r_unc, r_con

print("\n" + "="*72)
print("  Q20 - Black-Litterman: NYSE (daily & monthly)")
print("="*72)
print("  delta=3.0, tau=0.05  |  2 absolute + 2 relative views")

w_BL_yd_unc, w_BL_yd_con, r_BL_yd_unc, r_BL_yd_con = \
    black_litterman(ret_yd, sel_yd, 252, RF_DAILY,   "NYSE Daily",   views_nyse)
w_BL_ym_unc, w_BL_ym_con, r_BL_ym_unc, r_BL_ym_con = \
    black_litterman(ret_ym, sel_ym, 12,  RF_MONTHLY, "NYSE Monthly", views_nyse)
plot_weights(w_BL_yd_con, sel_yd, "BL Constrained - NYSE Daily",    "bl_nyse_daily.png",   "steelblue")
plot_weights(w_BL_ym_con, sel_ym, "BL Constrained - NYSE Monthly",  "bl_nyse_monthly.png", "steelblue")

print("\n" + "="*72)
print("  Q21 - Black-Litterman: Nasdaq (daily & monthly)")
print("="*72)

w_BL_nd_unc, w_BL_nd_con, r_BL_nd_unc, r_BL_nd_con = \
    black_litterman(ret_nd, sel_nd, 252, RF_DAILY,   "Nasdaq Daily",   views_nasdaq)
w_BL_nm_unc, w_BL_nm_con, r_BL_nm_unc, r_BL_nm_con = \
    black_litterman(ret_nm, sel_nm, 12,  RF_MONTHLY, "Nasdaq Monthly", views_nasdaq)
plot_weights(w_BL_nd_con, sel_nd, "BL Constrained - Nasdaq Daily",   "bl_nasdaq_daily.png",   "darkorange")
plot_weights(w_BL_nm_con, sel_nm, "BL Constrained - Nasdaq Monthly", "bl_nasdaq_monthly.png", "darkorange")

# =============================================================================
# Q22 / Q23 - GLOBAL MINIMUM VARIANCE PORTFOLIO
# =============================================================================

def gmvp(ret_df, sel, ann, label):
    mu = ret_df.mean().values; Sigma = ret_df.cov().values; N = len(sel); ones = np.ones(N)
    try:
        Si = inv(Sigma); c = float(ones@Si@ones); b = float(ones@Si@mu)
        w_unc = (Si@ones)/c; mu_mv = b/c; var_mv = 1.0/c
    except LinAlgError:
        w_unc = ones/N; mu_mv = float(mu.mean()); var_mv = float(np.diag(Sigma).mean())
        c = np.nan; b = np.nan
    res = minimize(lambda w: w@Sigma@w, ones/N, method="SLSQP", bounds=[(0, 1)]*N,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}],
                   options={"ftol": 1e-14, "maxiter": 3000})
    w_con = res.x if res.success else ones/N
    r_unc = ret_df.values @ w_unc; r_con = ret_df.values @ w_con
    print(f"\n  [{label}]  c={c:.6f}  b={b:.6f}  "
          f"sigma_mv(ann)={np.sqrt(var_mv)*np.sqrt(ann)*100:.3f}%  "
          f"mu_mv(ann)={mu_mv*ann*100:.3f}%")
    print(f"  Statistics:")
    print(pd.DataFrame({"GMVP_Unc": port_stats(r_unc, ann),
                         "GMVP_Con": port_stats(r_con, ann)}).round(6).to_string())
    return w_unc, w_con, r_unc, r_con

print("\n" + "="*72)
print("  Q22 - GMVP: NYSE")
print("="*72)
print("  w_GMVP = Sigma^-1*1 / (1'*Sigma^-1*1)   (analytical unconstrained)")

w_GMVP_yd_unc, w_GMVP_yd_con, r_GMVP_yd_unc, r_GMVP_yd_con = gmvp(ret_yd, sel_yd, 252, "NYSE Daily")
w_GMVP_ym_unc, w_GMVP_ym_con, r_GMVP_ym_unc, r_GMVP_ym_con = gmvp(ret_ym, sel_ym, 12,  "NYSE Monthly")
plot_weights(w_GMVP_yd_con, sel_yd, "GMVP Constrained - NYSE Daily",    "gmvp_nyse_daily.png",   "steelblue")
plot_weights(w_GMVP_ym_con, sel_ym, "GMVP Constrained - NYSE Monthly",  "gmvp_nyse_monthly.png", "steelblue")

print("\n" + "="*72)
print("  Q23 - GMVP: Nasdaq")
print("="*72)

w_GMVP_nd_unc, w_GMVP_nd_con, r_GMVP_nd_unc, r_GMVP_nd_con = gmvp(ret_nd, sel_nd, 252, "Nasdaq Daily")
w_GMVP_nm_unc, w_GMVP_nm_con, r_GMVP_nm_unc, r_GMVP_nm_con = gmvp(ret_nm, sel_nm, 12,  "Nasdaq Monthly")
plot_weights(w_GMVP_nd_con, sel_nd, "GMVP Constrained - Nasdaq Daily",   "gmvp_nasdaq_daily.png",   "darkorange")
plot_weights(w_GMVP_nm_con, sel_nm, "GMVP Constrained - Nasdaq Monthly", "gmvp_nasdaq_monthly.png", "darkorange")

# =============================================================================
# Q24 - LINEAR COMBINATIONS OF PORTFOLIOS
# =============================================================================
print("\n" + "=" * 72)
print("  Q24 - Linear Combinations of Portfolios")
print("=" * 72)
print("  Methods: MV (constrained), BL (constrained), Bayesian (constrained), GMVP (constrained)")

COMBOS = {
    "Equal (25/25/25/25)":          (0.25, 0.25, 0.25, 0.25),
    "Risk-focused (10/20/30/40)":   (0.10, 0.20, 0.30, 0.40),
    "Return-focused (40/40/10/10)": (0.40, 0.40, 0.10, 0.10),
    "Bayes-dom. (15/35/35/15)":     (0.15, 0.35, 0.35, 0.15),
}

def mv_tangency_con(ret_df, rf):
    mu=ret_df.mean().values; Sigma=ret_df.cov().values; N=len(mu); w0=np.ones(N)/N
    def neg_sr(w): r=w@mu; s=np.sqrt(max(w@Sigma@w,1e-14)); return -(r-rf)/s
    res=minimize(neg_sr,w0,method="SLSQP",bounds=[(0,1)]*N,
                 constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"ftol":1e-14,"maxiter":3000})
    return res.x if res.success else w0

datasets_q24 = {
    "Nasdaq_Daily":   (ret_nd, sel_nd, 252, RF_DAILY,   views_nasdaq, "darkorange",
                       w_BL_nd_con, w_Bayes_nd_con, w_GMVP_nd_con),
    "Nasdaq_Monthly": (ret_nm, sel_nm, 12,  RF_MONTHLY, views_nasdaq, "darkorange",
                       w_BL_nm_con, w_Bayes_nm_con, w_GMVP_nm_con),
    "NYSE_Daily":     (ret_yd, sel_yd, 252, RF_DAILY,   views_nyse,   "steelblue",
                       w_BL_yd_con, w_Bayes_yd_con, w_GMVP_yd_con),
    "NYSE_Monthly":   (ret_ym, sel_ym, 12,  RF_MONTHLY, views_nyse,   "steelblue",
                       w_BL_ym_con, w_Bayes_ym_con, w_GMVP_ym_con),
}

all_combo_results = {}; all_combo_weights = {}

for label, (ret_df, sel, ann, rf, vfn, color, w_bl, w_bayes, w_gmvp) in datasets_q24.items():
    w_mv = mv_tangency_con(ret_df, rf)
    print(f"\n  {'─'*70}")
    print(f"  {label}")
    print(f"  {'─'*70}")
    print(f"\n  Pure portfolio statistics:")
    pure = pd.DataFrame({m: port_stats(ret_df.values@w, ann)
                          for m,w in [("MV",w_mv),("BL",w_bl),("Bayes",w_bayes),("GMVP",w_gmvp)]})
    print(pure.round(6).to_string())
    print(f"\n  {'Strategy':<38} {'Mean%':>8} {'Std%':>8} {'Var':>10} {'Skew':>7} {'Kurt':>7} {'Sharpe':>8}")
    print("  " + "-" * 90)
    combo_results = {}; combo_weights = {}
    for cname,(lMV,lBL,lBayes,lGMVP) in COMBOS.items():
        w_c = lMV*w_mv + lBL*w_bl + lBayes*w_bayes + lGMVP*w_gmvp
        r_c = ret_df.values @ w_c; ps = port_stats(r_c, ann)
        print(f"  {cname:<38} {ps['Mean (ann.)']*100:8.3f} {ps['Std  (ann.)']*100:8.3f} "
              f"{ps['Var  (period)']:10.6f} {ps['Skewness']:7.4f} {ps['Exc. Kurtosis']:7.4f} "
              f"{ps['Sharpe (ann.)']:8.4f}")
        combo_results[cname] = ps; combo_weights[cname] = w_c
    all_combo_results[label] = combo_results; all_combo_weights[label] = combo_weights

    # Cumulative return plot
    fig, ax = plt.subplots(figsize=(12,6))
    colors_ = ["steelblue","darkorange","green","crimson"]
    for i,(cname,w_c) in enumerate(combo_weights.items()):
        r_c = ret_df.values @ w_c; cum = (1+r_c).cumprod()
        ax.plot(ret_df.index[:len(cum)], cum, lw=1.6, label=cname[:28], color=colors_[i])
    ax.set_title(f"Q24 - Cumulative Returns ({label})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Return"); ax.legend(fontsize=8); ax.grid(True,alpha=0.25)
    plt.tight_layout(); fname_c=f"combo_cumulative_{label.lower().replace(' ','_')}.png"
    plt.savefig(fname_c, dpi=150, bbox_inches="tight"); plt.close(); print(f"  Saved: {fname_c}")

# =============================================================================
# SAVE ALL RESULTS TO EXCEL
# =============================================================================
print("\n[Saving all results to Excel]")

with pd.ExcelWriter("results_Q1_Q3.xlsx", engine="openpyxl") as xl:
    stats_nq_all_d.round(8).to_excel(xl, "Q1_Nasdaq_Daily")
    stats_nq_all_m.round(8).to_excel(xl, "Q1_Nasdaq_Monthly")
    stats_ny_all_d.round(8).to_excel(xl, "Q2_NYSE_Daily")
    stats_ny_all_m.round(8).to_excel(xl, "Q2_NYSE_Monthly")
    corr_nq_all_d.round(6).to_excel(xl,  "Q3_NQ_Corr_Daily")
    corr_nq_all_m.round(6).to_excel(xl,  "Q3_NQ_Corr_Monthly")
    cov_nq_all_d.round(10).to_excel(xl,  "Q3_NQ_Cov_Daily")
    corr_ny_all_d.round(6).to_excel(xl,  "Q3_NY_Corr_Daily")
    corr_ny_all_m.round(6).to_excel(xl,  "Q3_NY_Corr_Monthly")
    cov_ny_all_d.round(10).to_excel(xl,  "Q3_NY_Cov_Daily")
print("  Saved: results_Q1_Q3.xlsx")

with pd.ExcelWriter("results_Q7_Q14.xlsx", engine="openpyxl") as xl:
    pd.DataFrame({"Stock":sel_ny,"w_unc_d":w_ny_unc_d,"w_con_d":w_ny_con_d,
                  "w_unc_m":w_ny_unc_m,"w_con_m":w_ny_con_m}).to_excel(xl,"Q7_Q10_NYSE",index=False)
    pd.DataFrame({"Stock":sel_nq,"w_unc_d":w_nq_unc_d,"w_con_d":w_nq_con_d,
                  "w_unc_m":w_nq_unc_m,"w_con_m":w_nq_con_m}).to_excel(xl,"Q8_Q9_Nasdaq",index=False)
    ps_ny.round(6).to_excel(xl,"Q11_NYSE_Stats"); ps_nq.round(6).to_excel(xl,"Q12_Nasdaq_Stats")
print("  Saved: results_Q7_Q14.xlsx")

with pd.ExcelWriter("results_Q15_Q19.xlsx", engine="openpyxl") as xl:
    idx_stats.round(8).to_excel(xl,"Q15_Index_Stats")
    cmp_nd.round(8).to_excel(xl,"Q15_NQ_vs_Index_Daily")
    cmp_yd.round(8).to_excel(xl,"Q15_NY_vs_Index_Daily")
    pd.DataFrame({"Stock":sel_ny,"Beta_d":beta_ny_d.reindex(sel_ny).values,
                  "Beta_m":beta_ny_m.reindex(sel_ny).values}).to_excel(xl,"Q16_NYSE_Betas",index=False)
    pd.DataFrame({"Stock":sel_nq,"Beta_d":beta_nq_d.reindex(sel_nq).values,
                  "Beta_m":beta_nq_m.reindex(sel_nq).values}).to_excel(xl,"Q17_Nasdaq_Betas",index=False)
print("  Saved: results_Q15_Q19.xlsx")

with pd.ExcelWriter("results_Q20_Q24.xlsx", engine="openpyxl") as xl:
    for label in datasets_q24:
        tag = label.replace(" ","_")
        combo_df = pd.DataFrame(all_combo_results[label]).T
        combo_df.round(8).to_excel(xl, f"Combo_{tag}")
        cw_df = pd.DataFrame(all_combo_weights[label]).T
        cw_df.columns = datasets_q24[label][1]
        cw_df.round(8).to_excel(xl, f"ComboW_{tag}")
print("  Saved: results_Q20_Q24.xlsx")

print("\n" + "=" * 72)
print("  DONE - All Q1-Q24 complete")
print("=" * 72)
