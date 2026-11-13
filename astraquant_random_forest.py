"""ASTRAQUANT X: Random Forest stock signal demo.
Run: pip install streamlit yfinance pandas numpy scikit-learn plotly
     streamlit run astraquant_random_forest.py
Educational predictions only; not investment advice.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

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

def clean_prices(raw):
    df = raw.copy()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(df.columns):
        raise ValueError("Downloaded data is missing required OHLCV columns.")
    df = df[needed].drop_duplicates(subset="Date").sort_values("Date")
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
    st.header("Stock selection")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    start = st.date_input("Start date", value=pd.Timestamp("2020-01-01").date())
    end = st.date_input("End date (exclusive)", value=pd.Timestamp.today().date())
    threshold = st.slider("Next-day movement threshold (%)", 0.1, 3.0, 0.5, 0.1)
    run = st.button("Load Data & Train Model", type="primary", use_container_width=True)

if not run:
    st.info("Choose a stock in the sidebar, then click **Load Data & Train Model**.")
    st.stop()
if not ticker or start >= end:
    st.error("Enter a ticker and choose an end date after the start date.")
    st.stop()

try:
    with st.spinner("Downloading prices and training Random Forest..."):
        raw = download_prices(ticker, str(start), str(end))
        clean = clean_prices(raw)
        data = add_indicators(clean)
        valid = data.dropna(subset=FEATURES).copy()
        if len(valid) < 180:
            raise ValueError("Insufficient data. Choose at least about one year of trading history.")
        # Final date has unknown future return: reserve it only for the latest prediction.
        latest = valid.iloc[[-1]].copy()
        labelled = valid.loc[valid["Future_return"].notna()].copy()
        t = threshold / 100
        labelled["Target"] = np.select([labelled["Future_return"] > t,
                                          labelled["Future_return"] < -t],
                                         ["BUY", "SELL"], default="HOLD")
        split = int(len(labelled) * 0.8)
        # Purge the last training sample: its next-day target falls in the test period.
        train, test = labelled.iloc[:split-1], labelled.iloc[split:]
        if len(train) < 100 or len(test) < 20 or train["Target"].nunique() < 2:
            raise ValueError("Not enough training/test rows or distinct training labels. Try a longer date range or a different threshold.")
        model = make_model()
        model.fit(train[FEATURES], train["Target"])
        predicted = model.predict(test[FEATURES])
        test = test.copy()
        test["Prediction"] = predicted
        latest_signal = model.predict(latest[FEATURES])[0]
        probabilities = dict(zip(model.classes_, model.predict_proba(latest[FEATURES])[0]))
except Exception as exc:
    st.error(f"Could not complete the run: {exc}")
    st.stop()

st.subheader(f"Market overview — {ticker}")
a, b, c = st.columns(3)
a.metric("Latest close", f"{clean['Close'].iloc[-1]:,.2f}")
b.metric("Rows after cleaning", f"{len(clean):,}")
c.metric("Latest model signal", latest_signal)
st.caption(f"Signal refers to the trading period following {pd.Timestamp(latest['Date'].iloc[0]).date()}, based on the latest available daily bar. The market may already have moved.")

with st.expander("Original and cleaned data", expanded=False):
    st.write("Original data")
    st.dataframe(raw.head(12), use_container_width=True)
    st.write("Cleaned data")
    st.dataframe(clean.head(12), use_container_width=True)

st.subheader("Price chart and predicted test-period signals")
fig = go.Figure()
fig.add_trace(go.Scatter(x=clean["Date"], y=clean["Close"], name="Close", mode="lines"))
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
st.caption(f"Training rows: {len(train)} · Test rows: {len(test)} · Chronological 80/20 split with one-row purge · No shuffled data")
cm = confusion_matrix(true, predicted, labels=CLASSES)
st.dataframe(pd.DataFrame(cm, index=[f"Actual {x}" for x in CLASSES], columns=[f"Predicted {x}" for x in CLASSES]), use_container_width=True)
st.text(classification_report(true, predicted, labels=CLASSES, zero_division=0))
majority = train["Target"].mode().iloc[0]
st.write(f"Training-majority baseline ({majority} on every test row): {accuracy_score(true, [majority]*len(test)):.1%} accuracy")

st.subheader("Latest prediction probabilities")
st.bar_chart(pd.DataFrame({"Probability": [probabilities.get(k, 0.0) for k in CLASSES]}, index=CLASSES))
st.caption("Random Forest probabilities are fractions of votes across trees and may be poorly calibrated.")

st.subheader("Feature importance")
importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
st.bar_chart(importance.set_index("Feature"))
st.caption("Feature importance indicates association with the fitted model, not causation or a guaranteed reason to trade.")

st.subheader("Actual vs predicted signals")
st.dataframe(test[["Date", "Close", "Future_return", "Target", "Prediction"]].tail(60), use_container_width=True)
st.download_button("Download test predictions (CSV)", test[["Date", "Close", "Future_return", "Target", "Prediction"]].to_csv(index=False), "astraquant_predictions.csv", "text/csv")
st.warning("Review 2 prototype: this is next-day classification, not a tested profitable strategy. No trading costs, slippage, or execution constraints are included.")
