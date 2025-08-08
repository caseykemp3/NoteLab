import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="NoteLab — SPX Structured Note Backtester", page_icon="💹", layout="wide")

st.markdown(
    """
    <style>
    .small { font-size: 0.85rem; color: #666; }
    .metric-row { display: flex; gap: 1rem; }
    .metric-card { padding: 0.8rem 1rem; border: 1px solid #EEE; border-radius: 8px; background: #FAFAFA; }
    .sidebar .sidebar-content { width: 360px; }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data(show_spinner=False)
def load_spy():
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="max", auto_adjust=False, progress=False)
        spy = spy.rename(columns=str.title)
        spy = spy[['Close', 'Adj Close']].dropna()
        spy.index.name = "Date"
        spy = spy.sort_index()
        return spy
    except Exception:
        return None

def _to_series(obj) -> pd.Series:
    """Coerce a Series/DataFrame/scalar to a 1-D pd.Series."""
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:,0]
        return obj.mean(axis=1)  # fallback
    # scalar/ndarray
    return pd.Series(obj)

def monthly_series(prices) -> pd.Series:
    """Month-end series from daily prices (last valid obs per month). Ensures 1-D Series."""
    s = _to_series(prices).resample("ME").last().dropna()
    s.name = getattr(prices, "name", "Price")
    return s

def to_total_return(prices) -> pd.Series:
    s = _to_series(prices).astype(float)
    rets = s.pct_change().fillna(0.0)
    tri = (1 + rets).cumprod()
    tri.iloc[0] = 1.0
    tri.name = "TRI"
    return tri.astype(float)

def calc_drawdown(path: pd.Series) -> pd.Series:
    path = _to_series(path).astype(float)
    peak = path.cummax()
    dd = path / peak - 1.0
    return dd

def stats_from_returns(returns, periods_per_year=12):
    r = _to_series(returns).dropna().astype(float)
    if len(r) == 0:
        return dict(CAGR=np.nan, Vol=np.nan, MaxDD=np.nan, Worst12m=np.nan, Sharpe=np.nan)
    cum = float((1+r).prod())
    years = len(r) / periods_per_year
    cagr = cum**(1/years) - 1 if years > 0 else np.nan
    vol = float(r.std(ddof=0)) * np.sqrt(periods_per_year)
    path = (1+r).cumprod()
    maxdd = float(calc_drawdown(path).min())
    if len(r) >= 12:
        roll12 = (1+r).rolling(12).apply(lambda x: np.prod(1+x)-1, raw=True).dropna()
        worst12 = float(roll12.min())
    else:
        worst12 = np.nan
    sharpe = (r.mean() * periods_per_year) / vol if (np.isfinite(vol) and vol > 0.0) else np.nan
    return dict(CAGR=cagr, Vol=vol, MaxDD=maxdd, Worst12m=worst12, Sharpe=sharpe)

def note_payoff_total_return(R, buffer: float, participation: float, coupon_annual: float,
                             term_months: int, guaranteed_coupon: bool, contingent_hits: int = 0):
    R = np.asarray(R, dtype=float)
    price_leg = np.where(
        R >= 0.0, participation * R,
        np.where(R >= -buffer, 0.0, R + buffer)
    )
    if guaranteed_coupon:
        coupon = coupon_annual * (term_months / 12.0)
    else:
        coupon = (coupon_annual / 12.0) * max(int(contingent_hits), 0)
    total = price_leg + coupon
    return float(total) if total.size == 1 else total

def contingent_hits_monthly(prices: pd.Series, start_idx: int, end_idx: int, barrier: float = 0.0):
    tri = _to_series(prices).astype(float)
    tri = tri / float(tri.iloc[start_idx])
    obs = tri.iloc[start_idx+1 : end_idx+1]
    Rm = obs - 1.0
    return int((Rm >= barrier).sum())

def ladder_returns_monthly(prices_tr, start_date, end_date, term_months: int,
                           buffer: float, participation: float, coupon_annual: float,
                           guaranteed_coupon: bool, contingent: bool, barrier: float, cap: float|None):
    m_prices = monthly_series(prices_tr).astype(float)
    idx = m_prices.index

    start_date = max(pd.to_datetime(start_date), idx[0])
    end_date   = min(pd.to_datetime(end_date), idx[-1])
    last_start = end_date - pd.DateOffset(months=term_months)
    cohorts = pd.date_range(start=start_date, end=last_start, freq="ME")
    cohorts = pd.DatetimeIndex([d for d in cohorts if d in idx])

    N = term_months
    out = []

    for s in cohorts:
        si = idx.get_loc(s)
        ei = si + N
        if ei >= len(idx):
            continue
        S0 = float(np.asarray(m_prices.iloc[si]).item() if hasattr(m_prices.iloc[si], "item") else m_prices.iloc[si])
        ST = float(np.asarray(m_prices.iloc[ei]).item() if hasattr(m_prices.iloc[ei], "item") else m_prices.iloc[ei])
        R  = ST / S0 - 1.0
        if cap is not None:
            R = float(np.minimum(R, cap))
        hits = 0
        if (not guaranteed_coupon) and contingent:
            hits = contingent_hits_monthly(m_prices, si, ei, barrier=barrier)
        note_R = float(note_payoff_total_return(
            R=R, buffer=buffer, participation=participation,
            coupon_annual=coupon_annual, term_months=N,
            guaranteed_coupon=guaranteed_coupon, contingent_hits=hits
        ))
        out.append((idx[ei], note_R))

    out = pd.DataFrame(out, columns=["Maturity", "NoteReturn"]).set_index("Maturity")
    mret = (out["NoteReturn"] / N).reindex(idx, fill_value=0.0).astype(float)
    cum = (1.0 + mret).cumprod(); cum.iloc[0] = 1.0
    return mret, cum

# ---- UI ----
st.sidebar.title("NoteLab Controls")
data_source = st.sidebar.radio("Data source", ["Fetch SPY (yfinance)", "Upload CSV"], index=0)
spy = load_spy() if data_source == "Fetch SPY (yfinance)" else None
uploaded = None
if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

price_mode = st.sidebar.selectbox("Index Mode", ["Total Return (Adj Close)", "Price only (Close)"], index=0)

min_date = pd.Timestamp("2000-01-03")
max_date = pd.Timestamp.today().normalize()
if spy is not None:
    min_date = max(min_date, spy.index[0])
    max_date = min(max_date, spy.index[-1])

default_start = pd.Timestamp("2010-01-01")
default_end   = max_date

date_range = st.sidebar.date_input("Date range", value=(default_start, default_end),
                                   min_value=min_date.to_pydatetime().date(),
                                   max_value=max_date.to_pydatetime().date())
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = default_start.date(), default_end.date()

st.sidebar.markdown("**Template:** Buffered Growth Note (v1)")
buffer_pct = st.sidebar.slider("Protection (buffer %)", 0, 40, 10, 1) / 100.0
participation = st.sidebar.slider("Upside participation (×)", 0.0, 2.0, 1.0, 0.05)
coupon_pct = st.sidebar.slider("Coupon (annual %)", 0.0, 10.0, 3.0, 0.25) / 100.0
term_months = st.sidebar.slider("Term (months)", 6, 60, 12, 1)

coupon_mode = st.sidebar.radio("Coupon mode", ["Guaranteed", "Contingent"], index=0)
guaranteed = (coupon_mode == "Guaranteed")
barrier_pct = 0.0
contingent = not guaranteed
if contingent:
    barrier_pct = st.sidebar.slider("Coupon barrier (%) — monthly obs vs start", -50, 50, 0, 5) / 100.0

cap_toggle = st.sidebar.toggle("Cap upside", value=False)
cap_value = None
if cap_toggle:
    cap_value = st.sidebar.slider("Cap level (% underlying return)", 5, 200, 80, 5) / 100.0

mode = st.sidebar.radio("Simulation mode", ["Rolling monthly ladder", "Single issuance"], index=0)
st.sidebar.markdown("---"); st.sidebar.caption("NoteLab v0.2 — demo build.")

# Data load
if data_source == "Fetch SPY (yfinance)":
    if spy is None:
        st.error("Could not fetch SPY from yfinance. Try uploading a CSV."); st.stop()
    prices = spy.copy()
else:
    if uploaded is None:
        st.info("Upload a CSV to proceed."); st.stop()
    df = pd.read_csv(uploaded)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date")
    else:
        df = df.rename(columns={df.columns[0]:"Date"}); df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date")
    df = df.rename(columns=lambda c: c.strip().title())
    prices = df.copy()

col = "Adj Close" if price_mode.startswith("Total Return") and "Adj Close" in prices.columns else "Close"
if col not in prices.columns:
    st.error(f"Selected column '{col}' not found. Found: {list(prices.columns)}"); st.stop()

prices = prices[[col]].dropna().rename(columns={col:"Price"})
prices = prices.loc[(prices.index >= pd.to_datetime(start_date)) & (prices.index <= pd.to_datetime(end_date))]
if len(prices) < 260:
    st.warning("Selected window is quite short. Consider a longer range.")

tri = to_total_return(prices["Price"])

st.title("NoteLab — SPX Structured Note Backtester")
st.write(
    f"**Window:** {prices.index[0].date()} → {prices.index[-1].date()}  ·  "
    f"**Index mode:** {'Adj Close (total return proxy)' if col=='Adj Close' else 'Close (price only)'}  ·  "
    f"**Mode:** {mode}"
)

# Price chart
fig_top = go.Figure()
fig_top.add_trace(go.Scatter(x=prices.index, y=_to_series(prices['Price']), name="SPY Price", mode="lines"))
fig_top.update_layout(title="SPY Price (zoom / pan)", xaxis=dict(rangeslider=dict(visible=True)), yaxis_title="Price")
st.plotly_chart(fig_top, use_container_width=True)

if mode == "Rolling monthly ladder":
    mret_ladder, cum_ladder = ladder_returns_monthly(
        prices_tr=tri, start_date=prices.index[0], end_date=prices.index[-1],
        term_months=term_months, buffer=buffer_pct, participation=participation,
        coupon_annual=coupon_pct, guaranteed_coupon=guaranteed,
        contingent=contingent, barrier=barrier_pct, cap=cap_value
    )
    m_prices = monthly_series(prices["Price"])
    mret_spy = m_prices.pct_change().dropna()
    # Align to common months
    common_idx = mret_ladder.index.intersection(mret_spy.index)
    mret_ladder = _to_series(mret_ladder.reindex(common_idx).fillna(0.0)).astype(float)
    mret_spy    = _to_series(mret_spy.reindex(common_idx).fillna(0.0)).astype(float)

    cum_spy = (1.0 + mret_spy).cumprod(); cum_spy.iloc[0] = 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_ladder.index, y=_to_series(cum_ladder), name="Note Ladder", mode="lines"))
    fig.add_trace(go.Scatter(x=cum_spy.index, y=_to_series(cum_spy), name="SPY (monthly)", mode="lines"))
    fig.update_layout(title="Cumulative Returns — Note Ladder vs SPY (monthly)", xaxis_title="Date", yaxis_title="Growth of $1")
    st.plotly_chart(fig, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Performance stats (monthly)")
        s_note = stats_from_returns(mret_ladder); s_spy = stats_from_returns(mret_spy)
        fmt = lambda x: "—" if (x is None or (isinstance(x,float) and (np.isnan(x) or np.isinf(x)))) else f"{x:.2%}"
        st.markdown(
            "<div class='metric-row'>"
            f"<div class='metric-card'><b>Note CAGR</b><br>{fmt(s_note['CAGR'])}</div>"
            f"<div class='metric-card'><b>Note Vol</b><br>{fmt(s_note['Vol'])}</div>"
            f"<div class='metric-card'><b>Note MaxDD</b><br>{fmt(s_note['MaxDD'])}</div>"
            f"<div class='metric-card'><b>Note Sharpe</b><br>{'—' if np.isnan(s_note['Sharpe']) else f'{s_note['Sharpe']:.2f}'}</div>"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='metric-row' style='margin-top:0.5rem;'>"
            f"<div class='metric-card'><b>SPY CAGR</b><br>{fmt(s_spy['CAGR'])}</div>"
            f"<div class='metric-card'><b>SPY Vol</b><br>{fmt(s_spy['Vol'])}</div>"
            f"<div class='metric-card'><b>SPY MaxDD</b><br>{fmt(s_spy['MaxDD'])}</div>"
            f"<div class='metric-card'><b>SPY Sharpe</b><br>{'—' if np.isnan(s_spy['Sharpe']) else f'{s_spy['Sharpe']:.2f}'}</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with colB:
        st.subheader("Cohort outcomes (last 12 matured)")
        m_tri = monthly_series(tri); idx = m_tri.index
        last_start = pd.to_datetime(prices.index[-1]) - pd.DateOffset(months=term_months)
        cohorts = pd.date_range(start=prices.index[0], end=last_start, freq="ME")
        cohorts = pd.DatetimeIndex([d for d in cohorts if d in idx])
        N = term_months; records = []
        for s in cohorts:
            si = idx.get_loc(s); ei = si + N
            if ei >= len(idx): continue
            S0 = float(np.asarray(m_tri.iloc[si]).item() if hasattr(m_tri.iloc[si], "item") else m_tri.iloc[si])
            ST = float(np.asarray(m_tri.iloc[ei]).item() if hasattr(m_tri.iloc[ei], "item") else m_tri.iloc[ei])
            R = ST / S0 - 1.0
            if cap_value is not None: R = float(np.minimum(R, cap_value))
            hits = 0
            if contingent and (not guaranteed):
                hits = contingent_hits_monthly(m_tri, si, ei, barrier=barrier_pct)
            note_R = float(note_payoff_total_return(R, buffer_pct, participation, coupon_pct, N, guaranteed, hits))
            records.append({"Start": idx[si].date(), "Maturity": idx[ei].date(), "Underlying R": R, "Note R": note_R})
        cohort_df = pd.DataFrame(records)
        if len(cohort_df): st.dataframe(cohort_df.tail(12))
        else: st.info("No full-term cohorts in the selected range/term.")

    exp_df = pd.DataFrame({
        "Monthly Return — Note": _to_series(mret_ladder),
        "Monthly Return — SPY": _to_series(mret_spy),
        "Cumulative — Note": (1+_to_series(mret_ladder)).cumprod(),
        "Cumulative — SPY": (1+_to_series(mret_spy)).cumprod(),
    })
    st.download_button("Download results (CSV)", exp_df.to_csv().encode("utf-8"), "notelab_results.csv", "text/csv")

else:
    st.subheader("Single issuance cohort")
    all_days = prices.index
    default_issue = all_days[0]
    issue_date = st.date_input("Issuance date", value=default_issue.date(),
                               min_value=all_days[0].date(), max_value=all_days[-1].date())
    issue_dt = pd.to_datetime(issue_date)
    issue_idx = all_days.searchsorted(issue_dt)
    if issue_idx >= len(all_days): st.error("Issuance date outside data range."); st.stop()
    issue_date = all_days[issue_idx]
    maturity_date = issue_date + relativedelta(months=term_months)
    mat_idx = prices.index.searchsorted(maturity_date)
    if mat_idx >= len(prices.index):
        st.warning("Maturity date is beyond available data; adjust term or end date."); st.stop()
    maturity_date = prices.index[mat_idx]

    S0 = float(np.asarray(to_total_return(prices['Price']).loc[issue_date]).item())
    ST = float(np.asarray(to_total_return(prices['Price']).loc[maturity_date]).item())
    R  = ST / S0 - 1.0
    if cap_value is not None: R = float(np.minimum(R, cap_value))

    hits = 0
    if (not guaranteed) and contingent:
        m_tri = monthly_series(to_total_return(prices['Price']))
        m_idx = m_tri.index
        s_m = m_idx.searchsorted(issue_date); e_m = m_idx.searchsorted(maturity_date)
        if e_m > s_m: hits = contingent_hits_monthly(m_tri, s_m, e_m, barrier=barrier_pct)

    note_R = float(note_payoff_total_return(R, buffer_pct, participation, coupon_pct, term_months, guaranteed, hits))

    st.write(f"**Issuance:** {issue_date.date()}  ·  **Maturity:** {maturity_date.date()}  ·  "
             f"**Underlying term return:** {R:+.2%}  ·  **Note total return:** {note_R:+.2%}")

    seg = prices.loc[issue_date:maturity_date]
    fig_path = go.Figure()
    fig_path.add_trace(go.Scatter(x=seg.index, y=_to_series(seg["Price"]), name="SPY Price", mode="lines"))
    fig_path.update_layout(title="SPY Path over the Note Term", xaxis_title="Date", yaxis_title="Price",
                           xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_path, use_container_width=True)

    xR = np.linspace(-0.60, 0.80, 300)
    xR_cap = np.minimum(xR, cap_value) if cap_value is not None else xR
    y = np.where(xR_cap >= 0, participation * xR_cap,
                 np.where(xR_cap >= -buffer_pct, 0.0, xR_cap + buffer_pct))
    coupon_total = coupon_pct * (term_months/12) if guaranteed else (coupon_pct/12) * hits
    y_total = y + coupon_total
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=xR, y=y_total, name="Note total return", mode="lines"))
    fig_payoff.add_hline(y=0, line_width=1); fig_payoff.add_vline(x=0, line_width=1)
    fig_payoff.update_layout(title="Payoff Diagram (total return vs underlying term return)",
                             xaxis_title="Underlying return over term", yaxis_title="Note total return")
    st.plotly_chart(fig_payoff, use_container_width=True)

with st.expander("Assumptions & Notes"):
    st.markdown("""
    **Simplifications in this demo**  
    • Uses SPY as a proxy for the S&P 500; *Adj Close* approximates total return.  
    • Ladder path assumes only matured vintages realize P&L each month (no MTM).  
    • Contingent coupons check monthly observations vs a barrier relative to the start level.  
    • Cap (if enabled) is applied to the underlying return before participation.  
    • No fees, taxes, or issuance frictions modeled.  
    • Results are hypothetical, for illustration only.
    """)