
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Market Prediction AI", page_icon="📈")
st.title("📈 Market Prediction AI")
st.caption("Experimental next-day price estimates")

assets = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Apple": "AAPL",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Motors": "TATAMOTORS.NS",
}

choice = st.selectbox("Choose an asset", list(assets))
ticker = assets[choice]

@st.cache_data(ttl=900)
def load_data(symbol):
    d = yf.download(
        symbol, period="5y", interval="1d",
        auto_adjust=True, progress=False,
        multi_level_index=False
    )
    if d.empty:
        raise ValueError("No market data returned.")
    return d[["Open", "High", "Low", "Close", "Volume"]].dropna()

def make_features(d):
    d = d.copy()
    d["return_1d"] = d["Close"].pct_change()
    d["return_5d"] = d["Close"].pct_change(5)
    d["return_10d"] = d["Close"].pct_change(10)
    d["ma_5"] = d["Close"].rolling(5).mean()
    d["ma_20"] = d["Close"].rolling(20).mean()
    d["ma_50"] = d["Close"].rolling(50).mean()

    cols = [
        "Open", "High", "Low", "Close", "Volume",
        "return_1d", "return_5d", "return_10d",
        "ma_5", "ma_20", "ma_50"
    ]
    d[cols] = d[cols].replace([np.inf, -np.inf], np.nan)
    return d, cols

try:
    with st.spinner("Downloading data and training AI..."):
        data = load_data(ticker)
        df, features = make_features(data)
        df["target"] = df["Close"].shift(-1)
        ready = df.dropna(subset=features + ["target"])

        if len(ready) < 200:
            raise ValueError("Not enough historical data.")

        X = ready[features]
        y = ready["target"]
        split = int(len(ready) * 0.8)

        if split == 0 or split >= len(ready):
            raise ValueError("Unable to split the data.")

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X.iloc[:split], y.iloc[:split])

        test_pred = model.predict(X.iloc[split:])
        test_mae = mean_absolute_error(y.iloc[split:], test_pred)
        baseline_mae = mean_absolute_error(
            y.iloc[split:], X.iloc[split:]["Close"]
        )

        actual_direction = (
            y.iloc[split:].to_numpy()
            > X.iloc[split:]["Close"].to_numpy()
        )

        predicted_direction = (
            test_pred
            > X.iloc[split:]["Close"].to_numpy()
        )

        directional_accuracy = (
            actual_direction == predicted_direction
        ).mean() * 100
        
        model.fit(X, y)
        latest = df[features].dropna().iloc[[-1]]
        last_price = float(latest["Close"].iloc[0])
        predicted = float(model.predict(latest)[0])
        change = (predicted / last_price - 1) * 100

    st.subheader(choice)
    st.metric("Latest available close", f"{last_price:,.4f}")
    st.metric(
        "Estimated next close",
        f"{predicted:,.4f}",
        delta=f"{change:+.2f}%"
    )

    st.write("Historical test MAE:", round(test_mae, 4))
st.write("Baseline MAE:", round(baseline_mae, 4))
st.write(
    "Directional accuracy:",
    f"{directional_accuracy:.2f}%"
)

    st.subheader("Recent historical prices")
    st.line_chart(data["Close"].tail(180))

    st.subheader("Historical test: actual vs predicted")
    result = pd.DataFrame({
        "Actual next close": y.iloc[split:],
        "AI prediction": test_pred
    })
    st.line_chart(result)

    st.warning(
        "For education only, not financial advice. "
        "The model can be wrong. Do not trade based on this estimate alone." 
  except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Try again later if the market data provider is unavailable.") 
    
