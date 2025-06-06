import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Data Preparation ---

# Load NSE Tata data
df_nse = pd.read_csv(r"C:\Users\bhagyashree.s\Desktop\Stock price prediction using LSTM\NSE-Tata-Global-Beverages-Limited.csv")
df_nse["Date"] = pd.to_datetime(df_nse["Date"], format="%Y-%m-%d")
df_nse = df_nse.sort_values("Date")

new_data = pd.DataFrame({
    "Date": df_nse["Date"],
    "Close": df_nse["Close"]
})
new_data.set_index("Date", inplace=True)

dataset = new_data.values
train = dataset[0:987, :]
valid = dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

model_path = r"C:\Users\bhagyashree.s\Desktop\Stock price prediction using LSTM\lstm_stock_model.h5"
model = load_model(model_path)

inputs = new_data[-len(valid)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train_df = new_data[:987]
valid_df = new_data[987:].copy()
valid_df["Predictions"] = closing_price

# Load other stock data
csv_path = r"C:\Users\bhagyashree.s\Desktop\Stock price prediction using LSTM\stock_data.csv"
df = pd.read_csv(csv_path)

df["Date"] = pd.to_datetime(df["Date"])

# --- Streamlit UI ---

st.title("Stock Price Analysis Dashboard")

tab = st.tabs(["NSE-TATAGLOBAL Stock Data", "Facebook Stock Data"])

# --- Tab 1: NSE-TATAGLOBAL ---
with tab[0]:
    st.header("Actual Closing Price")
    st.plotly_chart(go.Figure(
        data=[go.Scatter(x=valid_df.index, y=valid_df["Close"], mode="markers")],
        layout=go.Layout(
            title="Actual Closing Price",
            xaxis_title="Date",
            yaxis_title="Closing Rate"
        )
    ), use_container_width=True)

    st.header("LSTM Predicted Closing Price")
    st.plotly_chart(go.Figure(
        data=[go.Scatter(x=valid_df.index, y=valid_df["Predictions"], mode="markers")],
        layout=go.Layout(
            title="Predicted Closing Price",
            xaxis_title="Date",
            yaxis_title="Closing Rate"
        )
    ), use_container_width=True)

# --- Tab 2: Facebook Stock Data ---
with tab[1]:
    st.header("Stocks High vs Lows")

    stocks = df["Stock"].unique().tolist()
    selected_stocks_highlow = st.multiselect(
        "Select Stocks for High and Low Prices", options=stocks, default=["FB"]
    )

    if selected_stocks_highlow:
        fig_highlow = go.Figure()
        for stock in selected_stocks_highlow:
            stock_df = df[df["Stock"] == stock]
            fig_highlow.add_trace(go.Scatter(
                x=stock_df["Date"],
                y=stock_df["High"],
                mode='lines',
                name=f"High {stock}"
            ))
            fig_highlow.add_trace(go.Scatter(
                x=stock_df["Date"],
                y=stock_df["Low"],
                mode='lines',
                name=f"Low {stock}"
            ))
        fig_highlow.update_layout(
            title=f"High and Low Prices Over Time",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1M', step='month', stepmode='backward'),
                        dict(count=6, label='6M', step='month', stepmode='backward'),
                        dict(step='all')
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        st.plotly_chart(fig_highlow, use_container_width=True)

    st.header("Market Volume")

    selected_stocks_volume = st.multiselect(
        "Select Stocks for Volume", options=stocks, default=["FB"]
    )

    if selected_stocks_volume:
        fig_volume = go.Figure()
        for stock in selected_stocks_volume:
            stock_df = df[df["Stock"] == stock]
            fig_volume.add_trace(go.Scatter(
                x=stock_df["Date"],
                y=stock_df["Volume"],
                mode='lines',
                name=f"Volume {stock}"
            ))
        fig_volume.update_layout(
            title=f"Market Volume Over Time",
            xaxis_title="Date",
            yaxis_title="Transactions Volume",
            height=600,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label='1M', step='month', stepmode='backward'),
                        dict(count=6, label='6M', step='month', stepmode='backward'),
                        dict(step='all')
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        st.plotly_chart(fig_volume, use_container_width=True)







#streamlit run app.py