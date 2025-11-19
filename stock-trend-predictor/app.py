from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.io as pio
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Feature Engineering
def create_features(df):
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["Volatility"] = df["Close"].rolling(20, min_periods=1).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(5)
    df.dropna(inplace=True)
    return df

# ML Models
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
        "Ridge Regression": Ridge(alpha=1.0),
        "ElasticNet Regression": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
        "Polynomial Regression": None,  # handled separately
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Support Vector Machine": SVR(kernel="rbf", C=10, gamma="scale"),
    }

# Train + Forecast
def train_and_forecast(df):
    df = create_features(df)

    if df.shape[0] < 60:
        raise Exception("Not enough data to train model.")

    X = df[["MA20", "MA50", "Volatility", "Momentum"]]
    y = df["Close"]

    # Train/test split with min 30 rows test
    test_size_rows = max(30, int(len(X) * 0.10))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_rows, shuffle=False
    )

    scaler = MinMaxScaler()
    X_train_scaled, X_test_scaled = (
        scaler.fit_transform(X_train),
        scaler.transform(X_test),
    )

    models = get_models()
    scores = {}
    preds = {}

    for name, model in models.items():
        try:
            if name == "Polynomial Regression":
                poly = PolynomialFeatures(2, include_bias=False)
                Xtr = poly.fit_transform(X_train_scaled)
                Xts = poly.transform(X_test_scaled)
                lin = LinearRegression()
                lin.fit(Xtr, y_train)
                y_pred = lin.predict(Xts)
                models[name] = (lin, poly)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            scores[name] = {"RMSE": round(rmse, 4), "R2": round(r2, 4)}
            preds[name] = y_pred

        except Exception as e:
            print(f"{name} failed: {e}")
            scores[name] = {"RMSE": np.inf, "R2": -np.inf}

    best = min(scores, key=lambda k: scores[k]["RMSE"])
    best_pred = preds[best]

    # Forecast next 30 days
    if best == "Polynomial Regression":
        model, poly = models[best]
    else:
        model = models[best]

    future_data = []
    last_features = X.iloc[-1].values
    last_date = df.index[-1]
    last_close_prices = y.iloc[-5:].values

    for i in range(30):
        scaled = scaler.transform(last_features.reshape(1, -1))

        if best == "Polynomial Regression":
            scaled = poly.transform(scaled)

        next_price = float(model.predict(scaled)[0])
        next_date = last_date + timedelta(days=1)

        future_data.append({"Date": next_date, "Predicted_Close": next_price})

        # Update features
        last_features[0] = (last_features[0] * 19 + next_price) / 20  # MA20
        last_features[1] = (last_features[1] * 49 + next_price) / 50  # MA50
        last_features[2] = X.iloc[-1]["Volatility"]  # keep simple
        last_features[3] = next_price - last_close_prices[0]  # momentum

        last_close_prices = np.append(last_close_prices[1:], next_price)
        last_date = next_date

    forecast_df = pd.DataFrame(future_data)

    compare_df = pd.DataFrame(
        {"Date": X_test.index, "Actual": y_test.values, "Predicted": best_pred}
    )

    return scores, best, compare_df, forecast_df, df

# Plotly Graphs
def create_graphs(df, compare_df, forecast_df):
    candle = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )
    candle.update_layout(template="plotly_dark", title="Historical Candlestick Chart")
    candle_html = pio.to_html(candle, full_html=False)

    comp = go.Figure()
    comp.add_trace(go.Scatter(x=compare_df["Date"], y=compare_df["Actual"], name="Actual"))
    comp.add_trace(
        go.Scatter(x=compare_df["Date"], y=compare_df["Predicted"], name="Predicted")
    )
    comp_html = pio.to_html(comp, full_html=False)

    forecast = go.Figure()
    forecast.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical"))
    forecast.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted_Close"],
            name="30-Day Forecast",
        )
    )
    forecast_html = pio.to_html(forecast, full_html=False)

    return candle_html, comp_html, forecast_html

# Web Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        try:
            if file and file.filename:
                df = pd.read_csv(file, index_col=0)
                df.index = pd.to_datetime(df.index, dayfirst=True, errors="raise")

                source = f"Loaded from file: {file.filename}"
            else:
                return render_template("index.html", error="Please upload a CSV file.")

            df.index = pd.to_datetime(df.index)

            scores, best, compare_df, forecast_df, processed_df = train_and_forecast(df)

            candle_html, comp_html, forecast_html = create_graphs(
                processed_df, compare_df, forecast_df
            )

            return render_template(
                "results.html",
                scores=scores,
                best_model=best,
                source_info=source,
                candle_html=candle_html,
                comp_html=comp_html,
                forecast_html=forecast_html,
            )

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
