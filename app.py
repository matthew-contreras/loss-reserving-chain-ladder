"""
Loss Reserving & Chain Ladder Analysis  |  Streamlit portfolio app
Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Loss Reserving | Chain Ladder", page_icon="📊", layout="wide")

REQUIRED = ["AccidentYear", "DevelopmentPeriod", "IncrementalPaid"]
INFL_GRID = [-5, 0, 5, 10, 15]  # scenarios (%) shown in the sensitivity table
money = lambda x: f"${x / 1e6:,.1f}M"


# --------------------------------------------------------------------------- #
# 1. DATA INPUT
# --------------------------------------------------------------------------- #
@st.cache_data
def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Synthetic P&C incremental paid claims, AY 2015-2024, dev periods 1-10.

    Incremental paid(AY, d) = Ultimate(AY) x Pattern(d) x CalendarInflation x Noise
      - Ultimate grows ~6%/yr (exposure growth) with +/-5% randomness
      - Pattern = % of ultimate paid in each development period (sums to 100%)
      - Calendar-year inflation = 3% per year, applied on calendar year AY + d - 1
      - Noise = lognormal(0, 6%) process variance
    Only cells with calendar year <= 2024 are observed (upper-left triangle).
    """
    rng = np.random.default_rng(seed)
    pattern = np.array([.28, .22, .16, .11, .08, .06, .04, .03, .015, .005])
    pattern /= pattern.sum()
    rows = []
    for i, ay in enumerate(range(2015, 2025)):
        ultimate = 8_000_000 * 1.06**i * rng.normal(1, 0.05)
        for d in range(1, 11):
            cy = ay + d - 1
            if cy <= 2024:
                amt = ultimate * pattern[d - 1] * 1.03 ** (cy - 2015) * rng.lognormal(0, 0.06)
                rows.append((ay, d, round(amt)))
    return pd.DataFrame(rows, columns=REQUIRED)


def validate_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns / types of an uploaded file; raise ValueError with a clear message."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Expected: {', '.join(REQUIRED)}.")
    df = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    bad = int(df.isna().any(axis=1).sum())
    if bad == len(df):
        raise ValueError("No valid numeric rows found in the file.")
    if bad:
        st.warning(f"Dropped {bad} row(s) with missing or non-numeric values.")
    df = df.dropna()
    if ((df[["AccidentYear", "DevelopmentPeriod"]] % 1) != 0).any().any() or (df.DevelopmentPeriod < 1).any():
        raise ValueError("AccidentYear must be whole years and DevelopmentPeriod whole numbers >= 1.")
    if (df.IncrementalPaid < 0).any():
        st.warning("Negative incremental payments found (salvage/subrogation?). They are kept as-is.")
    return df.astype({"AccidentYear": int, "DevelopmentPeriod": int})


# --------------------------------------------------------------------------- #
# 2. ACTUARIAL ENGINE
# --------------------------------------------------------------------------- #
def build_triangles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Incremental -> cumulative paid triangle.

    Cum(AY, d) = sum_{k<=d} Incremental(AY, k).  Unobserved (future) cells stay NaN.
    """
    inc = df.pivot_table(index="AccidentYear", columns="DevelopmentPeriod",
                         values="IncrementalPaid", aggfunc="sum").sort_index()
    inc = inc.reindex(columns=range(1, int(inc.columns.max()) + 1))
    if inc.shape[0] < 3 or inc.shape[1] < 3:
        raise ValueError("Need at least 3 accident years and 3 development periods.")
    obs = inc.notna().values
    if not np.all(obs[:, :-1] >= obs[:, 1:]):
        raise ValueError("Triangle has gaps: each accident year must have consecutive development "
                         "periods starting at 1, with no missing periods in between.")
    cum = inc.cumsum(axis=1).where(inc.notna())
    return inc, cum


def link_ratios(cum: pd.DataFrame) -> pd.DataFrame:
    """Age-to-age (link) ratios: f(AY, j) = Cum(AY, j+1) / Cum(AY, j)."""
    v = cum.values
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(v[:, :-1] > 0, v[:, 1:] / v[:, :-1], np.nan)
    return pd.DataFrame(r, index=cum.index, columns=[f"{j}-{j + 1}" for j in cum.columns[:-1]])


def average_factors(cum: pd.DataFrame) -> pd.DataFrame:
    """Development factors per age-to-age column.

    Volume-weighted:  f_j = sum_AY Cum(AY, j+1) / sum_AY Cum(AY, j)   (only AYs with both ages)
        -> weights each year by its loss volume, so large years dominate.
    Simple average:   f_j = mean_AY [ Cum(AY, j+1) / Cum(AY, j) ].
    Columns without usable data default to 1.0.
    """
    v, n = cum.values, cum.shape[1]
    vol, simple = np.ones(n - 1), np.ones(n - 1)
    for j in range(n - 1):
        m = ~np.isnan(v[:, j]) & ~np.isnan(v[:, j + 1]) & (v[:, j] > 0)
        if m.any():
            vol[j] = v[m, j + 1].sum() / v[m, j].sum()
            simple[j] = np.mean(v[m, j + 1] / v[m, j])
    cols = [f"{j}-{j + 1}" for j in cum.columns[:-1]]
    return pd.DataFrame([vol, simple], index=["Volume-weighted", "Simple average"], columns=cols)


def project(cum: pd.DataFrame, f: np.ndarray, tail: float, inflation: float) -> dict:
    """Chain-ladder projection with a tail factor and a calendar-year inflation scenario.

    1. Project future cumulative losses:  Cum(AY, d+1) = Cum(AY, d) x f_d
    2. Future incremental = Cum(AY, d+1) - Cum(AY, d)   (the unadjusted projection)
    3. Inflation scenario: each future incremental is scaled by (1 + i)^t, where t is the number of
       calendar years beyond the latest diagonal in which the payment is made. Historical factors
       already embed past inflation, so i is an *additional* shift (0% = pure chain ladder).
    4. Tail: extra development beyond the last period, Ult = Cum(AY, n) x tail  (tail increment is
       also inflated).
    5. IBNR / unpaid reserve = Ultimate - Paid to date (the latest diagonal).
    """
    v, (m, n) = cum.values.astype(float), cum.shape
    ays = cum.index.values
    k = (~np.isnan(v)).sum(axis=1)                      # observed ages per AY
    latest_cy = int((ays + k - 1).max())                # latest calendar year in the data
    raw = v.copy()
    for i in range(m):                                  # step 1: unadjusted projection
        for d in range(k[i], n):
            raw[i, d] = raw[i, d - 1] * f[d - 1]
    adj, ult = v.copy(), np.zeros(m)
    for i in range(m):
        run = v[i, k[i] - 1]
        for d in range(k[i], n):                        # steps 2-3
            run += (raw[i, d] - raw[i, d - 1]) * (1 + inflation) ** (ays[i] + d - latest_cy)
            adj[i, d] = run
        ult[i] = adj[i, -1] + adj[i, -1] * (tail - 1) * (1 + inflation) ** (ays[i] + n - latest_cy)  # step 4
    tri = pd.DataFrame(adj, index=cum.index, columns=cum.columns)
    paid = pd.Series([v[i, k[i] - 1] for i in range(m)], index=cum.index)
    return {"triangle": tri, "ultimate": pd.Series(ult, index=cum.index), "paid": paid, "obs": ~np.isnan(v)}


def build_summary(base: dict, scen: dict) -> pd.DataFrame:
    """Reserve summary by accident year (baseline vs scenario)."""
    paid = base["paid"]
    s = pd.DataFrame({
        "Paid to Date": paid,
        "CDF to Ultimate": base["ultimate"] / paid.replace(0, np.nan),
        "Ultimate (Baseline)": base["ultimate"],
        "Ultimate (Scenario)": scen["ultimate"],
        "IBNR (Baseline)": base["ultimate"] - paid,
        "IBNR (Scenario)": scen["ultimate"] - paid,
    })
    s["IBNR % of Ultimate"] = s["IBNR (Scenario)"] / s["Ultimate (Scenario)"]
    s.index.name = "AccidentYear"
    return s


def with_total(s: pd.DataFrame) -> pd.DataFrame:
    """Append a Total row and return a flat table suitable for CSV export."""
    t = s.sum(numeric_only=True)
    t["CDF to Ultimate"] = t["Ultimate (Baseline)"] / t["Paid to Date"]
    t["IBNR % of Ultimate"] = t["IBNR (Scenario)"] / t["Ultimate (Scenario)"]
    return pd.concat([s, pd.DataFrame([t], index=["Total"])]).reset_index(names="AccidentYear")


# --------------------------------------------------------------------------- #
# 3. VISUALIZATIONS & STYLING
# --------------------------------------------------------------------------- #
def heatmap(cum: pd.DataFrame) -> go.Figure:
    fig = px.imshow(cum, text_auto=".3s", aspect="auto", color_continuous_scale="YlGnBu",
                    labels=dict(x="Development Period", y="Accident Year", color="Cum. Paid"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    fig.update_layout(title="Cumulative Paid Loss Triangle")
    return fig


def dev_curves(tri: pd.DataFrame, obs: np.ndarray) -> go.Figure:
    fig, colors, x = go.Figure(), px.colors.qualitative.Plotly, list(tri.columns)
    for i, ay in enumerate(tri.index):
        k, y, c = int(obs[i].sum()), tri.loc[ay].values, colors[i % len(colors)]
        fig.add_trace(go.Scatter(x=x[:k], y=y[:k], mode="lines+markers", name=str(ay),
                                 legendgroup=str(ay), line=dict(color=c)))
        if k < len(x):
            fig.add_trace(go.Scatter(x=x[k - 1:], y=y[k - 1:], mode="lines", legendgroup=str(ay),
                                     showlegend=False, line=dict(color=c, dash="dot"),
                                     name=f"{ay} (projected)"))
    fig.update_layout(title="Loss Development Curves (solid = actual, dotted = projected)",
                      xaxis_title="Development Period", yaxis_title="Cumulative Paid Loss",
                      legend_title="Accident Year")
    return fig


def paid_vs_ibnr(s: pd.DataFrame) -> go.Figure:
    d = s[["Paid to Date", "IBNR (Scenario)"]].reset_index().melt(
        "AccidentYear", var_name="Measure", value_name="Amount")
    fig = px.bar(d, x="AccidentYear", y="Amount", color="Measure", barmode="group",
                 title="Paid Losses vs. Estimated IBNR by Accident Year")
    fig.update_xaxes(type="category")
    return fig


def style_ratios(ratios: pd.DataFrame, avg: np.ndarray, thr: float):
    """Highlight link ratios deviating more than `thr` (relative) from the selected factor."""
    with np.errstate(invalid="ignore"):
        flag = np.abs(ratios.values / avg - 1) > thr
    css = pd.DataFrame(np.where(flag, "background-color:#ffb3b3;font-weight:bold", ""),
                       index=ratios.index, columns=ratios.columns)
    return ratios.style.apply(lambda _: css, axis=None).format("{:.3f}", na_rep=""), int(flag.sum())


# --------------------------------------------------------------------------- #
# 4. APP
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("📊 Loss Reserving & Chain Ladder Analysis")
    st.caption("Paid chain-ladder reserving with tail factor and inflation scenario analysis.")

    sb = st.sidebar
    sb.header("Data")
    use_syn = sb.toggle("Use synthetic P&C dataset", value=True)
    up = None
    if use_syn:
        sb.download_button("Download sample CSV", generate_synthetic_data().to_csv(index=False),
                           "sample_claims.csv", "text/csv")
    else:
        up = sb.file_uploader("Upload CSV", type="csv", help="Columns: " + ", ".join(REQUIRED))
        if up is None:
            st.info("Upload a CSV with columns **AccidentYear, DevelopmentPeriod, IncrementalPaid**, "
                    "or switch on the synthetic dataset in the sidebar.")
            st.stop()
    try:
        raw = generate_synthetic_data() if use_syn else validate_claims(pd.read_csv(up))
        inc, cum = build_triangles(raw)
    except Exception as e:  # malformed upload -> friendly message instead of a stack trace
        st.error(f"Could not process the data: {e}")
        st.stop()

    sb.header("Assumptions")
    method = sb.selectbox("Development factor average", ["Volume-weighted", "Simple average"])
    tail = sb.slider("Tail factor (beyond last period)", 1.000, 1.200, 1.000, 0.005, format="%.3f")
    sb.header("Scenarios")
    infl = sb.slider("Inflation Adjustment Scenario (%)", -5.0, 15.0, 0.0, 0.5,
                     help="Extra annual inflation applied to future incremental payments.")
    thr = sb.slider("High-variance LDF threshold (% from selected)", 1, 30, 5)

    ratios, avgs = link_ratios(cum), average_factors(cum)
    f = avgs.loc[method].values
    base, scen = project(cum, f, tail, 0.0), project(cum, f, tail, infl / 100)
    summary = build_summary(base, scen)
    total_paid = summary["Paid to Date"].sum()

    t1, t2, t3, t4 = st.tabs(["Data & Triangles", "Development Factors",
                              "Reserving & Scenarios", "Executive Summary"])

    with t1:
        c = st.columns(4)
        c[0].metric("Accident years", cum.shape[0])
        c[1].metric("Development periods", cum.shape[1])
        c[2].metric("Claim records", len(raw))
        c[3].metric("Total paid to date", money(total_paid))
        st.plotly_chart(heatmap(cum))
        a, b = st.columns(2)
        a.subheader("Incremental paid triangle")
        a.dataframe(inc.style.format("{:,.0f}", na_rep=""))
        b.subheader("Cumulative paid triangle")
        b.dataframe(cum.style.format("{:,.0f}", na_rep=""))

    with t2:
        st.subheader("Age-to-age link ratios")
        styled, n_flag = style_ratios(ratios, f, thr / 100)
        st.caption(f"Red cells deviate more than {thr}% from the selected factor: **{n_flag}** flagged.")
        st.dataframe(styled)
        st.subheader("Average development factors")
        sel = pd.DataFrame([f], index=["Selected ATA factor"], columns=avgs.columns)
        sel["Tail"] = tail
        cdf = np.cumprod(np.append(f, tail)[::-1])[::-1]  # CDF_j = f_j x f_{j+1} x ... x tail
        sel.loc["CDF to ultimate", :] = cdf
        st.dataframe(pd.concat([avgs, sel]).style.format("{:.4f}", na_rep=""))
        fig = px.box(ratios.melt(var_name="Age-to-Age", value_name="Link ratio").dropna(),
                     x="Age-to-Age", y="Link ratio", points="all", title="Link ratio dispersion by age")
        st.plotly_chart(fig)

    with t3:
        st.subheader(f"Scenario: {infl:+.1f}% additional annual inflation on future payments")
        c = st.columns(3)
        c[0].metric("Total IBNR (Baseline)", money(summary["IBNR (Baseline)"].sum()))
        c[1].metric("Total IBNR (Scenario)", money(summary["IBNR (Scenario)"].sum()),
                    delta=money(summary["IBNR (Scenario)"].sum() - summary["IBNR (Baseline)"].sum()),
                    delta_color="inverse")
        c[2].metric("Total Ultimate (Scenario)", money(summary["Ultimate (Scenario)"].sum()))
        st.plotly_chart(dev_curves(scen["triangle"], scen["obs"]))
        st.plotly_chart(paid_vs_ibnr(summary))
        st.subheader("Projected (completed) cumulative triangle")
        st.dataframe(scen["triangle"].style.format("{:,.0f}"))
        st.subheader("Inflation sensitivity")
        sens = pd.DataFrame({"Inflation (%)": INFL_GRID, "Total IBNR": [
            project(cum, f, tail, r / 100)["ultimate"].sum() - total_paid for r in INFL_GRID]})
        sens["Change vs 0%"] = sens["Total IBNR"] / sens.loc[sens["Inflation (%)"] == 0, "Total IBNR"].iloc[0] - 1
        a, b = st.columns(2)
        a.dataframe(sens.style.format({"Total IBNR": "{:,.0f}", "Change vs 0%": "{:+.1%}"}), hide_index=True)
        b.plotly_chart(px.line(sens, x="Inflation (%)", y="Total IBNR", markers=True,
                               title="Total IBNR vs. inflation scenario"))

    with t4:
        ibnr = summary["IBNR (Scenario)"]
        top = ibnr.idxmax()
        recent = ibnr.tail(3).sum() / ibnr.sum() if ibnr.sum() else 0
        c = st.columns(4)
        c[0].metric("Paid to date", money(total_paid))
        c[1].metric("Ultimate (Scenario)", money(summary["Ultimate (Scenario)"].sum()))
        c[2].metric("Total IBNR", money(ibnr.sum()))
        c[3].metric("IBNR % of ultimate", f"{ibnr.sum() / summary['Ultimate (Scenario)'].sum():.1%}")
        st.markdown(
            f"- **Method:** {method} chain ladder, tail factor **{tail:.3f}**, inflation scenario **{infl:+.1f}%**.\n"
            f"- Accident year **{top}** carries the largest reserve ({money(ibnr.max())}); the 3 most recent "
            f"years hold **{recent:.0%}** of total IBNR.\n"
            f"- Versus the 0% inflation baseline, the scenario changes total IBNR by "
            f"**{money(ibnr.sum() - summary['IBNR (Baseline)'].sum())}**.\n"
            f"- **{n_flag}** link ratios exceed the {thr}% variance threshold; review before finalising selections.")
        export = with_total(summary)
        st.dataframe(export.style.format({
            **{c: "{:,.0f}" for c in export.columns if c not in ("AccidentYear", "CDF to Ultimate", "IBNR % of Ultimate")},
            "CDF to Ultimate": "{:.3f}", "IBNR % of Ultimate": "{:.1%}"}), hide_index=True)
        st.download_button("⬇️ Download Results (CSV)", export.to_csv(index=False),
                           "reserving_summary.csv", "text/csv")
        st.caption("IBNR here = Ultimate − Paid to date (total unpaid reserve on a paid basis). "
                   "Illustrative portfolio project; not actuarial advice.")


if __name__ == "__main__":
    main()
