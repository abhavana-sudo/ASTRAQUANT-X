"""ASTRAQUANT X: Random Forest stock signal demo.
Run: pip install streamlit yfinance pandas numpy scikit-learn plotly
     streamlit run astraquant_random_forest.py
Educational predictions only; not investment advice.
"""
from time import perf_counter
from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

FEATURES = ["RSI14", "MACD", "MACD_signal", "SMA20_ratio", "SMA50_ratio", "Volume_change", "Daily_return"]
CLASSES = ["SELL", "HOLD", "BUY"]

st.set_page_config(page_title="ASTRAQUANT X | Random Forest", page_icon="📈", layout="wide")
st.title("📈 ASTRAQUANT X — Random Forest Trading Signals")
st.caption("Educational prototype · Daily closing-price predictions · Not financial advice")

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError("No market data found. Check the ticker and date range.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.reset_index()

# Common company-name shortcuts; other names use Yahoo Finance search.
COMPANY_ALIASES = {
    "APPLE": "AAPL", "APPLE INC": "AAPL", "MICROSOFT": "MSFT",
    "MICROSOFT CORPORATION": "MSFT", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL",
    "ALPHABET INC": "GOOGL", "TESLA": "TSLA", "AMAZON": "AMZN",
    "NVIDIA": "NVDA", "META": "META", "FACEBOOK": "META",
    "RELIANCE": "RELIANCE.NS", "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS", "TATA CONSULTANCY SERVICES": "TCS.NS",
    "INFOSYS": "INFY.NS", "HDFC BANK": "HDFCBANK.NS",
}

@st.cache_data(ttl=3600, show_spinner=False)
def search_companies(name):
    # Yahoo Finance search results are not guaranteed to be complete.
    results = yf.Search(name, max_results=12).quotes
    candidates = []
    for item in results:
        symbol = item.get("symbol")
        if not symbol or item.get("quoteType", "EQUITY") not in ("EQUITY", "ETF", "ADR"):
            continue
        label = item.get("longname") or item.get("shortname") or symbol
        exchange = item.get("exchange") or ""
        if symbol not in [x[0] for x in candidates]:
            candidates.append((symbol, f"{label} — {symbol} ({exchange})"))
    return candidates

def clean_prices(raw):
    df = raw.copy()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(df.columns):
        raise ValueError("Downloaded data is missing required OHLCV columns.")
    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["Date"]).drop_duplicates(subset="Date").sort_values("Date")
    for c in needed[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df

def add_indicators(df):
    d = df.copy()
    close = d["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI14"] = (100 - 100 / (1 + rs)).fillna(50)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["SMA20"] = close.rolling(20).mean()
    d["SMA50"] = close.rolling(50).mean()
    d["SMA20_ratio"] = d["SMA20"] / close - 1
    d["SMA50_ratio"] = d["SMA50"] / close - 1
    d["Volume_change"] = d["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    d["Daily_return"] = close.pct_change()
    d["Future_return"] = close.shift(-1) / close - 1
    return d

def make_model():
    return RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=5,
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose data", ["Live Yahoo Finance", "Example dataset (offline)", "Upload OHLCV CSV"], index=1)
    uploaded_csv = st.file_uploader("Upload CSV (Date, Open, High, Low, Close, Volume)", type="csv") if source == "Upload OHLCV CSV" else None
    st.header("Stock selection")
    company_input = st.text_input("Search any listed company or ticker", "Apple").strip()
    ticker = ""
    if source == "Live Yahoo Finance":
        st.caption("Searches Yahoo Finance globally. Choose the correct company and exchange below.")
        exact_ticker = st.checkbox("Use input as exact ticker symbol (skip company search)", value=False)
        if company_input:
            if exact_ticker:
                ticker = company_input.upper()
            else:
                try:
                    matches = search_companies(company_input)
                    if matches:
                        # Prefer an exact symbol match; otherwise require a visible selection.
                        labels = [label for _, label in matches]
                        exact = next((i for i, (symbol, _) in enumerate(matches)
                                      if symbol.upper() == company_input.upper()), 0)
                        selected_label = st.selectbox("Matching companies — select the correct listing", labels, index=exact)
                        ticker = dict((label, symbol) for symbol, label in matches)[selected_label]
                    else:
                        st.warning("No matching publicly listed company found. Try the full name or its exchange ticker.")
                except Exception as exc:
                    st.warning(f"Yahoo Finance search is unavailable: {exc}. Check internet access or use exact ticker mode.")
    else:
        ticker = company_input.upper() or "SAMPLE"
        st.caption("Company search applies to Live Yahoo Finance; offline/uploaded CSV uses the dataset's prices.")
    st.caption(f"Selected ticker: **{ticker or 'Not selected'}**")
    start = st.date_input("Start date", value=pd.Timestamp("2024-01-01").date() if source != "Live Yahoo Finance" else pd.Timestamp("2026-01-01").date())
    end = st.date_input("End date (exclusive)", value=pd.Timestamp("2025-01-01").date() if source != "Live Yahoo Finance" else pd.Timestamp.today().date())
    st.caption("Same-year dates are supported. Earlier years are downloaded automatically for model training.")
    threshold = st.slider("Next-day movement threshold (%)", 0.1, 3.0, 0.5, 0.1)
    st.divider()
    st.subheader("Review 3 simulation settings")
    initial_capital = st.number_input("Simulated starting capital ($)", min_value=100.0, value=10000.0, step=1000.0)
    fee_pct = st.number_input("Fee per trade side (%)", min_value=0.0, max_value=2.0, value=0.10, step=0.05)
    slippage_pct = st.number_input("Slippage per trade side (%)", min_value=0.0, max_value=2.0, value=0.05, step=0.05)
    tune_model = st.checkbox("Tune Random Forest (time-series CV)", value=False, help="Uses only pre-test training data; takes longer.")
    run = st.button("Load Data & Train Model", type="primary", use_container_width=True)

if not run:
    st.info("Choose a stock in the sidebar, then click **Load Data & Train Model**.")
    st.stop()
if not ticker or start >= end:
    st.error("Enter a ticker and choose an end date after the start date.")
    st.stop()

try:
    with st.spinner("Downloading prices and training Random Forest..."):
        # Always download extra history for training, even when the selected
        # display/test period is only a few months in the same calendar year.
        # Up to five years of earlier data provides indicator warm-up and
        # chronological training observations.
        history_start = (pd.Timestamp(start) - pd.DateOffset(years=5)).date()
        if source == "Example dataset (offline)":
            example_path = Path(__file__).with_name("astraquant_example_stock_data_2014_2026.csv")
            if not example_path.exists():
                raise ValueError("Put astraquant_example_stock_data_2014_2026.csv in the same folder as this Python file.")
            raw = pd.read_csv(example_path)
            st.info("Example mode uses synthetic stock prices, not actual market data. Dataset dates: 2014-01-01 to 2026-11-13 (weekdays; synthetic data).")
        elif source == "Upload OHLCV CSV":
            if uploaded_csv is None:
                raise ValueError("Upload a CSV file containing Date, Open, High, Low, Close and Volume.")
            raw = pd.read_csv(uploaded_csv)
        else:
            raw = download_prices(ticker, str(history_start), str(end))
        clean = clean_prices(raw)
        # Do not use data later than the requested analysis period.
        clean = clean[clean["Date"] < pd.Timestamp(end)].copy()
        if clean.empty:
            raise ValueError("No data before your selected end date. Check that the selected dataset includes earlier dates.")
        data = add_indicators(clean)
        valid = data.dropna(subset=FEATURES).copy()
        if len(valid) < 180:
            raise ValueError("Insufficient historical data, even after automatic training-history download. Try a later year or another stock.")
        # Restrict evaluation to dates the user selected, but keep all earlier
        # rows for training. A test label requires the next trading close to
        # be available within the requested interval.
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        selected = valid[(valid["Date"] >= start_ts) & (valid["Date"] < end_ts)].copy()
        if len(selected) < 22:
            raise ValueError("The selected interval needs about 22 trading days for meaningful testing. A full year is not required.")
        latest = selected.iloc[[-1]].copy()
        t = threshold / 100
        labelled = valid.loc[valid["Future_return"].notna()].copy()
        labelled["Target"] = np.select(
            [labelled["Future_return"] > t, labelled["Future_return"] < -t],
            ["BUY", "SELL"], default="HOLD"
        )
        # The last pre-period row's label uses the first selected-period close.
        # Exclude it to keep all training labels strictly before the test period.
        train = labelled[labelled["Date"] < start_ts].iloc[:-1].copy()
        test = labelled[(labelled["Date"] >= start_ts) &
                        (labelled["Date"] < end_ts)].copy()
        # Do not score the last selected bar if its future close is outside the
        # requested date window (end date is exclusive).
        test = test[test["Date"] < selected["Date"].max()].copy()
        validation_mode = "Train before selected start; test within selected interval"
        # The sample starts in 2014, so an analysis beginning in 2014 has no
        # earlier history. In that case, split the selected period chronologically
        # instead of failing or training on future test observations.
        if len(train) < 100 or train["Target"].nunique() < 2:
            if len(test) < 125:
                raise ValueError(
                    "There is not enough history before your start date. For a 2014 start, "
                    "choose at least 6-9 months of data (e.g. 2014-01-01 to 2015-01-01). "
                    "For shorter analysis periods, choose a start date in 2015 or later."
                )
            split = max(101, int(len(test) * 0.70))
            if len(test) - split < 20:
                split = len(test) - 20
            period_train = test.iloc[:split-1].copy()  # purge one boundary label
            period_test = test.iloc[split:].copy()
            if len(period_train) < 100 or period_train["Target"].nunique() < 2:
                raise ValueError(
                    "The available training observations do not contain enough distinct BUY/SELL/HOLD "
                    "labels. Try a different movement threshold or a longer date interval."
                )
            train, test = period_train, period_test
            validation_mode = "Chronological 70/30 split within selected interval (no earlier training history)"
        if len(test) < 20:
            raise ValueError("At least 20 test trading days are needed. Extend your end date.")
        model = make_model()
        tuning_summary = "Default parameters (no tuning)"
        if tune_model:
            if len(train) < 240:
                raise ValueError("At least 240 training observations are required for time-series tuning.")
            grid = GridSearchCV(
                RandomForestClassifier(random_state=42, class_weight="balanced_subsample", n_jobs=-1),
                {"n_estimators": [100, 200], "max_depth": [5, 9], "min_samples_leaf": [3, 7]},
                cv=TimeSeriesSplit(n_splits=3, gap=1), scoring="f1_macro", n_jobs=1,
                error_score=np.nan,
            )
            grid.fit(train[FEATURES], train["Target"])
            model = grid.best_estimator_
            tuning_summary = f"Time-series CV best macro-F1: {grid.best_score_:.3f}; parameters: {grid.best_params_}"
        training_start = perf_counter()
        model.fit(train[FEATURES], train["Target"])
        training_seconds = perf_counter() - training_start
        batch_start = perf_counter()
        predicted = model.predict(test[FEATURES])
        batch_seconds = perf_counter() - batch_start
        # Repeated single-row timing measures model prediction computation only.
        # This is not real exchange order execution latency.
        timing_rows = test[FEATURES].head(min(50, len(test)))
        inference_start = perf_counter()
        for i in range(len(timing_rows)):
            model.predict(timing_rows.iloc[[i]])
        single_row_ms = (perf_counter() - inference_start) * 1000 / len(timing_rows)
        test = test.copy()
        test["Prediction"] = predicted
        latest_signal = model.predict(latest[FEATURES])[0]
        probabilities = dict(zip(model.classes_, model.predict_proba(latest[FEATURES])[0]))
except Exception as exc:
    st.error(f"Could not complete the run: {exc}")
    st.stop()

st.subheader(f"Market overview — {ticker}")
a, b, c = st.columns(3)
a.metric("Latest selected close", f"{selected['Close'].iloc[-1]:,.2f}")
b.metric("Selected trading rows", f"{len(selected):,}")
c.metric("Latest model signal", latest_signal)
st.caption(f"Signal refers to the trading period following {pd.Timestamp(latest['Date'].iloc[0]).date()}, based on the latest available daily bar. The market may already have moved.")

with st.expander("Original and cleaned data", expanded=False):
    st.write("Original data")
    st.dataframe(raw.head(12), use_container_width=True)
    st.write("Cleaned data")
    st.dataframe(clean.head(12), use_container_width=True)

st.subheader("Price chart and predicted test-period signals")
fig = go.Figure()
fig.add_trace(go.Scatter(x=selected["Date"], y=selected["Close"], name="Close", mode="lines"))
for signal, symbol, color in [("BUY", "triangle-up", "green"), ("SELL", "triangle-down", "red")]:
    rows = test[test["Prediction"] == signal]
    fig.add_trace(go.Scatter(x=rows["Date"], y=rows["Close"], mode="markers", name=signal,
                             marker=dict(symbol=symbol, size=10, color=color)))
fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Adjusted close")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Technical indicators")
c1, c2 = st.columns(2)
with c1:
    rsi_fig = go.Figure(go.Scatter(x=data["Date"], y=data["RSI14"], name="RSI (14)"))
    rsi_fig.add_hline(y=70, line_dash="dash")
    rsi_fig.add_hline(y=30, line_dash="dash")
    rsi_fig.update_layout(title="RSI", template="plotly_dark")
    st.plotly_chart(rsi_fig, use_container_width=True)
with c2:
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD_signal"], name="Signal line"))
    macd_fig.update_layout(title="MACD", template="plotly_dark")
    st.plotly_chart(macd_fig, use_container_width=True)
ma_fig = go.Figure()
for column in ["Close", "SMA20", "SMA50"]:
    ma_fig.add_trace(go.Scatter(x=data["Date"], y=data[column], name=column))
ma_fig.update_layout(title="Moving averages", template="plotly_dark")
st.plotly_chart(ma_fig, use_container_width=True)

st.subheader("Random Forest evaluation — held-out chronological test set")
true = test["Target"]
cols = st.columns(4)
for col, name, score in zip(cols, ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"],
                            [accuracy_score(true, predicted),
                             precision_score(true, predicted, labels=CLASSES, average="macro", zero_division=0),
                             recall_score(true, predicted, labels=CLASSES, average="macro", zero_division=0),
                             f1_score(true, predicted, labels=CLASSES, average="macro", zero_division=0)]):
    col.metric(name, f"{score:.1%}")
st.caption(f"Training rows: {len(train)} (before selected start) · Test rows: {len(test)} (selected dates) · Chronological split with boundary purge · No shuffled data")
cm = confusion_matrix(true, predicted, labels=CLASSES)
st.dataframe(pd.DataFrame(cm, index=[f"Actual {x}" for x in CLASSES], columns=[f"Predicted {x}" for x in CLASSES]), use_container_width=True)
st.text(classification_report(true, predicted, labels=CLASSES, zero_division=0))
majority = train["Target"].mode().iloc[0]
st.write(f"Training-majority baseline ({majority} on every test row): {accuracy_score(true, [majority]*len(test)):.1%} accuracy")

st.subheader("Latest prediction probabilities")
st.bar_chart(pd.DataFrame({"Probability": [probabilities.get(k, 0.0) for k in CLASSES]}, index=CLASSES))
st.caption("Random Forest probabilities are fractions of votes across trees and may be poorly calibrated.")


# --- RISK ANALYSIS (historical, educational; not a probability of loss) ---
st.subheader("Risk Analysis & Trader Guidance")
st.caption("Risk figures use historical daily prices. The risk score is a transparent "
           "0–100 heuristic, NOT the probability of losing money.")

risk_df = clean[["Date", "High", "Low", "Close"]].copy()
risk_df["Return"] = risk_df["Close"].pct_change()
risk_df["Volatility_20d_pct"] = risk_df["Return"].rolling(20).std() * np.sqrt(252) * 100
previous_close = risk_df["Close"].shift(1)
true_range = pd.concat([
    risk_df["High"] - risk_df["Low"],
    (risk_df["High"] - previous_close).abs(),
    (risk_df["Low"] - previous_close).abs()
], axis=1).max(axis=1)
risk_df["ATR14"] = true_range.rolling(14).mean()
risk_df["ATR_pct"] = risk_df["ATR14"] / risk_df["Close"] * 100
risk_df["Drawdown_pct"] = (risk_df["Close"] / risk_df["Close"].cummax() - 1) * 100

recent = risk_df["Return"].dropna().tail(252)
volatility = float(risk_df["Volatility_20d_pct"].iloc[-1])
atr_pct = float(risk_df["ATR_pct"].iloc[-1])
max_drawdown = float(risk_df["Drawdown_pct"].min())
# Historical 5th percentile of daily returns: approximately 5% of observed
# daily returns were worse than this figure. Not a forecast or worst-case bound.
historical_var_95 = max(0.0, -float(recent.quantile(0.05)) * 100)
confidence = float(max(probabilities.values()) * 100)

# Simple, explicitly defined demo risk index, not a trained risk model.
# Volatility contributes up to 40 points, ATR up to 30, and the
# observed daily-loss tail up to 30.
risk_score = round(min(40, volatility / 60 * 40) +
                   min(30, atr_pct / 5 * 30) +
                   min(30, historical_var_95 / 5 * 30))
risk_level = "LOW" if risk_score < 35 else "MODERATE" if risk_score < 65 else "HIGH"

r1, r2, r3 = st.columns(3)
r1.metric("Risk index (0–100)", f"{risk_score}/100", risk_level)
r2.metric("20-day annualized volatility", f"{volatility:.2f}%")
r3.metric("ATR as % of price", f"{atr_pct:.2f}%")
r4, r5, r6 = st.columns(3)
r4.metric("Historical 95% one-day VaR", f"{historical_var_95:.2f}%")
r5.metric("Maximum historical drawdown", f"{abs(max_drawdown):.2f}%")
r6.metric("Top prediction vote share", f"{confidence:.1f}%")

st.progress(risk_score / 100)
st.write(f"**Risk category: {risk_level}**")
st.caption("VaR is estimated from up to 252 observed daily returns. For example, "
           "a 2% VaR means about 5% of historical daily returns were worse than "
           "-2%; larger losses remain possible. Drawdown is measured over the "
           "selected history, not a prediction. The vote share is not a calibrated "
           "probability of a correct forecast.")

st.markdown("#### Investment insights — why this stock may or may not be attractive")
st.caption("These insights explain the model's technical signals using the selected stock's "
           "historical data. They are not a personalized recommendation or a "
           "fundamental valuation of the company.")

latest_row = latest.iloc[0]
insights = []
if latest_signal == "BUY":
    insights.append(f"**Potential upside signal:** Random Forest predicts a next-day "
                    f"rise greater than {threshold:.2f}%. This is a model forecast, "
                    "not a guarantee of a gain.")
elif latest_signal == "SELL":
    insights.append(f"**Potential downside signal:** Random Forest predicts a next-day "
                    f"decline greater than {threshold:.2f}%. This does not support "
                    "a short-term bullish case on its own.")
else:
    insights.append(f"**Neutral signal:** Random Forest expects the next-day move "
                    f"to remain within ±{threshold:.2f}%; the model does not show "
                    "a strong directional opportunity.")

if pd.notna(latest_row["SMA50"]) and pd.notna(latest_row["SMA20"]):
    if latest_row["Close"] > latest_row["SMA20"] > latest_row["SMA50"]:
        insights.append("**Positive price trend:** The latest close is above both "
                        "20-day and 50-day moving averages, with the shorter "
                        "average above the longer one. This supports a bullish "
                        "technical interpretation.")
    elif latest_row["Close"] < latest_row["SMA20"] < latest_row["SMA50"]:
        insights.append("**Weak price trend:** The latest close is below both "
                        "moving averages, which weakens the bullish case.")
    else:
        insights.append("**Mixed price trend:** The price and moving averages "
                        "are not aligned, so trend evidence is inconclusive.")

rsi_value = float(latest_row["RSI14"])
if rsi_value > 70:
    insights.append(f"**RSI ({rsi_value:.1f}):** Strong recent upward momentum, "
                    "but an elevated RSI can also indicate reversal risk.")
elif rsi_value < 30:
    insights.append(f"**RSI ({rsi_value:.1f}):** Recent downward momentum is "
                    "strong; an oversold reading alone does not prove a rebound.")
else:
    insights.append(f"**RSI ({rsi_value:.1f}):** Momentum is not in the "
                    "conventional overbought or oversold zone.")

if latest_row["MACD"] > latest_row["MACD_signal"]:
    insights.append("**MACD:** Above its signal line, suggesting positive "
                    "short-term momentum.")
else:
    insights.append("**MACD:** At or below its signal line, suggesting weaker "
                    "short-term momentum.")

insights.append(f"**Risk versus opportunity:** Historical risk index is "
                f"{risk_score}/100 ({risk_level.lower()}); recent annualized "
                f"volatility is {volatility:.2f}%. Higher risk can offset "
                "an otherwise positive technical signal.")
insights.append(f"**Model confidence indicator:** The most-voted prediction "
                f"received {confidence:.1f}% of tree votes. This is not "
                "the probability that an investment will succeed.")

for insight in insights:
    st.markdown("- " + insight)

if latest_signal == "BUY" and risk_level != "HIGH":
    st.info("**Overall technical interpretation:** Some indicators may support "
            "a bullish case, but the company’s valuation, earnings, news, "
            "liquidity and trading costs are not assessed by this model.")
elif latest_signal == "BUY":
    st.warning("**Overall technical interpretation:** The model signals "
               "potential upside, but elevated historical risk makes that "
               "signal less convincing.")
else:
    st.info("**Overall technical interpretation:** The current model signal "
            "does not provide a clear short-term bullish reason to invest. "
            "Fundamental analysis could lead to a different long-term view.")

st.markdown("#### Historical risk chart")
risk_fig = go.Figure()
risk_fig.add_trace(go.Scatter(x=risk_df["Date"],
                              y=risk_df["Volatility_20d_pct"],
                              name="20-day annualized volatility (%)"))
risk_fig.update_layout(template="plotly_dark", xaxis_title="Date",
                       yaxis_title="Volatility (%)")
st.plotly_chart(risk_fig, use_container_width=True)

risk_export = risk_df[["Date", "Close", "Volatility_20d_pct",
                       "ATR_pct", "Drawdown_pct"]].copy()
st.download_button("Download historical risk data (CSV)",
                   risk_export.to_csv(index=False),
                   "astraquant_risk_metrics.csv", "text/csv")
st.caption("Educational demonstration only. Risk index thresholds are illustrative "
           "and not validated or tailored to any investor.")

st.subheader("Feature importance")
importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
st.bar_chart(importance.set_index("Feature"))
st.caption("Feature importance indicates association with the fitted model, not causation or a guaranteed reason to trade.")

st.subheader("Actual vs predicted signals")
st.dataframe(test[["Date", "Close", "Future_return", "Target", "Prediction"]].tail(60), use_container_width=True)
st.download_button("Download test predictions (CSV)", test[["Date", "Close", "Future_return", "Target", "Prediction"]].to_csv(index=False), "astraquant_predictions.csv", "text/csv")
st.info("Review 3: see the historical backtest below. It is an illustrative simulation, not live execution or evidence of future profitability.")


# ===========================================================================
# REVIEW 3 — OUT-OF-SAMPLE TRADING BACKTEST AND EFFICIENCY EVALUATION
# ===========================================================================
st.divider()
st.header("Review 3 — Trading Backtest & Final Evaluation")
st.caption("Out-of-sample historical simulation. The signal is produced after the "
           "day's closing price and is applied to the NEXT trading day's open-to-close move. "
           "No same-day look-ahead trading is allowed. This is NOT an HFT execution system.")

# The source labels classify close(t) -> close(t+1), but the backtest can only
# execute after signal formation. Therefore, returns are from open(t+1) to
# close(t+1). These are distinct outcomes; do not equate classification
# accuracy with strategy profitability.
next_bars = clean[["Date", "Open", "Close"]].copy()
next_bars["Next_open"] = next_bars["Open"].shift(-1)
next_bars["Next_close"] = next_bars["Close"].shift(-1)
next_bars["Next_date"] = next_bars["Date"].shift(-1)
trades = test[["Date", "Close", "Target", "Prediction"]].merge(
    next_bars[["Date", "Next_date", "Next_open", "Next_close"]], on="Date", how="left"
).dropna(subset=["Next_open", "Next_close"]).copy()
trades = trades[(trades["Next_open"] > 0) & (trades["Next_close"] > 0)].reset_index(drop=True)

if len(trades) < 10:
    st.error("Not enough next-day prices for a meaningful backtest. Try a longer history.")
    st.stop()

# Long for BUY, short for SELL, flat for HOLD; one-day exposure, no overnight
# holdings. Fixed capital fraction of 100% per active day, no leverage beyond
# 1x notional; short trades are hypothetical and ignore borrow availability,
# borrow fees, margin calls and financing. Results may overstate performance.
trades["Position"] = trades["Prediction"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).astype(int)
trades["Intraday_return"] = trades["Next_close"] / trades["Next_open"] - 1
trades["Gross_strategy_return"] = trades["Position"] * trades["Intraday_return"]
round_trip_cost = 2 * (fee_pct + slippage_pct) / 100
trades["Trading_cost_pct"] = np.where(trades["Position"] != 0, round_trip_cost * 100, 0.0)
trades["Net_strategy_return"] = trades["Gross_strategy_return"] - np.where(
    trades["Position"] != 0, round_trip_cost, 0.0)
trades["Equity"] = initial_capital * (1 + trades["Net_strategy_return"]).cumprod()
trades["Gross_equity"] = initial_capital * (1 + trades["Gross_strategy_return"]).cumprod()
# Passive adjusted-close buy-and-hold comparison over the same test dates.
trades["Benchmark_return"] = trades["Next_close"] / trades["Close"] - 1
trades["Benchmark_equity"] = initial_capital * (1 + trades["Benchmark_return"]).cumprod()
# The benchmark incurs a single entry and exit cost, not one round trip daily.
trades["Benchmark_equity_after_cost"] = trades["Benchmark_equity"] * (1 - round_trip_cost)

net_returns = trades["Net_strategy_return"]
active = trades[trades["Position"] != 0]
wins = active[active["Net_strategy_return"] > 0]
losses = active[active["Net_strategy_return"] < 0]
net_profit = float(trades["Equity"].iloc[-1] - initial_capital)
roi = (float(trades["Equity"].iloc[-1]) / initial_capital - 1) * 100
peak = pd.concat([pd.Series([initial_capital]), trades["Equity"]], ignore_index=True).cummax().iloc[1:].to_numpy()
drawdown = trades["Equity"].to_numpy() / peak - 1
max_drawdown_pct = float(-np.min(drawdown) * 100)
vol_pct = float(net_returns.std(ddof=1) * np.sqrt(252) * 100) if len(net_returns) > 1 else 0.0
sharpe = (float(net_returns.mean() / net_returns.std(ddof=1)) * np.sqrt(252)
          if len(net_returns) > 1 and net_returns.std(ddof=1) > 0 else float('nan'))
win_rate = len(wins) / len(active) * 100 if len(active) else float('nan')
gross_wins = float(wins["Net_strategy_return"].sum())
gross_losses = float(-losses["Net_strategy_return"].sum())
profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('nan')

st.subheader("Simulated profitability (test period only)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Net profit / loss ($)", f"{net_profit:,.2f}")
m2.metric("ROI after costs", f"{roi:.2f}%")
m3.metric("Win rate (active days)", f"{win_rate:.1f}%" if np.isfinite(win_rate) else "N/A")
m4.metric("Profit factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "N/A")
m5, m6, m7, m8 = st.columns(4)
m5.metric("Maximum drawdown", f"{max_drawdown_pct:.2f}%")
m6.metric("Annualized strategy volatility", f"{vol_pct:.2f}%")
m7.metric("Sharpe (risk-free = 0)", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
m8.metric("Simulated active trades", str(len(active)))

st.write(f"**Assumptions:** initial simulated capital ${initial_capital:,.0f}; "
         f"fees {fee_pct:.2f}% per side; slippage {slippage_pct:.2f}% per side; "
         "one hypothetical round trip per BUY or SELL signal; HOLD remains in cash. "
         "Returns compound; fractional positions are assumed.")
st.warning("The SELL backtest assumes short selling. Real short selling may be unavailable, "
           "costly, restricted or require margin. No borrowing fees, taxes, market impact, "
           "financing, corporate-action execution effects or order rejections are simulated. "
           "Daily OHLCV data cannot establish actual fill quality or HFT profitability.")

curve = go.Figure()
curve.add_trace(go.Scatter(x=trades["Next_date"], y=trades["Equity"],
                           name="Random Forest strategy (after costs)"))
curve.add_trace(go.Scatter(x=trades["Next_date"], y=trades["Gross_equity"],
                           name="Strategy (before costs)", line=dict(dash="dot")))
curve.add_trace(go.Scatter(x=trades["Next_date"], y=trades["Benchmark_equity_after_cost"],
                           name="Buy & hold benchmark (after entry/exit costs)"))
curve.update_layout(title="Simulated equity curves — held-out period", template="plotly_dark",
                    xaxis_title="Trading date", yaxis_title="Simulated equity ($)")
st.plotly_chart(curve, use_container_width=True)

st.subheader("Risk and transaction-cost analysis")
fee_total = float((trades["Trading_cost_pct"] / 100).sum() * initial_capital)
st.write(f"Approximate sum of round-trip costs as a fraction of initial capital: "
         f"**${fee_total:,.2f}** (non-compounded illustration). "
         "The equity curve applies costs to each day's compounded return.")
st.write(f"Test-period benchmark ROI after a single entry/exit cost: "
         f"**{(trades['Benchmark_equity_after_cost'].iloc[-1] / initial_capital - 1) * 100:.2f}%**")

st.subheader("Model efficiency and latency")
e1, e2, e3 = st.columns(3)
e1.metric("Training time", f"{training_seconds:.3f} s")
e2.metric("Batch inference", f"{batch_seconds * 1000:.2f} ms for {len(test)} rows")
e3.metric("Single-row prediction avg", f"{single_row_ms:.2f} ms")
throughput = len(test) / batch_seconds if batch_seconds > 0 else float('nan')
st.write(f"**Batch throughput:** {throughput:,.0f} predictions/second" if np.isfinite(throughput)
         else "Batch throughput unavailable")
st.caption("These are local/server Python computation timings. They exclude data download, "
           "exchange connectivity, brokerage API delays, order routing and actual trade execution. "
           "For fair comparisons, measure on the same machine and repeat runs.")

st.subheader("Backtesting validation and limitations")
st.write(f"Validation method: **{validation_mode}**. "
         f"Chronological training set: **{len(train)} rows**; "
         f"untouched test set: **{len(test)} rows**; "
         "one boundary training row purged to avoid target overlap. "
         "No model fitting uses test labels. Features are calculated from data "
         "available on or before the signal date.")
st.write("The original classification labels measure next-close versus signal-day close, "
         "while simulated trades measure next-open to next-close returns. This conservative "
         "execution timing avoids pretending that a closing-price signal could have been "
         "traded at that same closing price.")
st.write("This is a single held-out historical period, **not** a walk-forward retraining "
         "study. Test-period strategy settings should not be repeatedly tuned to maximize "
         "reported results. Daily-bar data is not enough to validate high-frequency trading.")

st.subheader("Trader research checklist")
if roi <= 0:
    st.warning("The historical simulation lost money after assumed costs. Investigate "
               "whether trading costs or weak predictions explain the result.")
elif max_drawdown_pct > 20:
    st.warning("The simulation shows a positive return but a substantial historical "
               "drawdown. Consider whether that downside would be acceptable in a "
               "hypothetical strategy study.")
else:
    st.info("The simulation is positive under these assumptions. Check stability "
            "across different date ranges, stocks and costs before drawing conclusions.")
st.write("Check earnings announcements, liquidity, changes in volatility, short-selling "
         "constraints, model uncertainty and the difference between historical simulation "
         "and live execution. No trading action is guaranteed or recommended.")

st.subheader("Download Review 3 results")
st.dataframe(trades[["Date", "Next_date", "Prediction", "Position", "Next_open",
                     "Next_close", "Gross_strategy_return", "Trading_cost_pct",
                     "Net_strategy_return", "Equity"]].tail(30), use_container_width=True)
st.download_button("Download full backtest CSV", trades.to_csv(index=False),
                   "astraquant_review3_backtest.csv", "text/csv")
summary = pd.DataFrame([{
    "ticker": ticker, "train_rows": len(train), "test_rows": len(test),
    "accuracy": accuracy_score(true, predicted),
    "precision_macro": precision_score(true, predicted, labels=CLASSES, average="macro", zero_division=0),
    "recall_macro": recall_score(true, predicted, labels=CLASSES, average="macro", zero_division=0),
    "f1_macro": f1_score(true, predicted, labels=CLASSES, average="macro", zero_division=0),
    "initial_capital": initial_capital, "net_profit": net_profit, "roi_pct": roi,
    "win_rate_pct": win_rate, "profit_factor": profit_factor,
    "max_drawdown_pct": max_drawdown_pct, "annual_volatility_pct": vol_pct,
    "sharpe_risk_free_zero": sharpe, "active_trades": len(active),
    "fee_pct_per_side": fee_pct, "slippage_pct_per_side": slippage_pct,
    "training_seconds": training_seconds, "batch_inference_seconds": batch_seconds,
    "single_row_inference_ms": single_row_ms, "batch_predictions_per_second": throughput
}])
st.download_button("Download Review 3 metrics CSV", summary.to_csv(index=False),
                   "astraquant_review3_metrics.csv", "text/csv")
