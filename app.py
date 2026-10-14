import random
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Dự báo giá cổ phiếu LSTM", page_icon="📈", layout="wide")
st.title("📈 Dự báo giá cổ phiếu bằng LSTM")
st.caption("Chỉ phục vụ học tập/tham khảo, không phải khuyến nghị đầu tư.")

# ---------------- Sidebar ----------------
st.sidebar.header("Thiết lập")
ticker = st.sidebar.text_input("Mã cổ phiếu (Yahoo Finance)", "VNM.VN")
start = st.sidebar.date_input("Từ ngày", date(2018, 1, 1))
window = st.sidebar.slider("Window size (số ngày nhìn lại)", 20, 120, 60, 5)
epochs = st.sidebar.slider("Epochs tối đa", 5, 50, 10)
n_future = st.sidebar.slider("Số ngày dự báo", 1, 30, 10)
seed = st.sidebar.number_input("Seed", value=42, step=1)
run = st.sidebar.button("🚀 Huấn luyện & dự báo", type="primary")


@st.cache_data(show_spinner="Đang tải dữ liệu...", ttl=3600)
def load_data(ticker, start):
    df = yf.download(ticker, start=str(start), end=str(date.today()), auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


def create_sequences(arr, w):
    X, y = [], []
    for i in range(w, len(arr)):
        X.append(arr[i - w:i])
        y.append(arr[i])
    return np.array(X), np.array(y)


def train_and_predict(close, window, epochs, n_future, seed):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential

    np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)

    data = close.values.reshape(-1, 1)
    train_size = int(0.8 * len(data))
    train_data, test_data = data[:train_size], data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled, window)
    test_combined = np.vstack((train_scaled[-window:], test_scaled))
    X_test, y_test = create_sequences(test_combined, window)

    val_size = int(0.1 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]

    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    history = model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=32, shuffle=False, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_real, test_pred))),
        "MAE": float(mean_absolute_error(y_real, test_pred)),
        "R2": float(r2_score(y_real, test_pred)),
    }

    last = np.vstack((train_scaled, test_scaled))[-window:].reshape(1, window, 1)
    fut = []
    for _ in range(n_future):
        p = model.predict(last, verbose=0)
        fut.append(p[0, 0])
        last = np.append(last[:, 1:, :], p.reshape(1, 1, 1), axis=1)
    fut = scaler.inverse_transform(np.array(fut).reshape(-1, 1)).flatten()

    test_index = close.index[train_size:]
    return history.history, y_real.flatten(), test_pred.flatten(), fut, metrics, test_index


# ---------------- Main ----------------
try:
    df = load_data(ticker, start)
except Exception as e:
    st.error(f"Không tải được dữ liệu: {e}")
    st.stop()

if df.empty:
    st.error("Không có dữ liệu cho mã này. Thử lại mã khác (vd: VNM.VN, FPT.VN, VCB.VN).")
    st.stop()

close = df["Close"]
c1, c2, c3 = st.columns(3)
c1.metric("Giá đóng cửa gần nhất", f"{close.iloc[-1]:,.0f}")
c2.metric("Số phiên dữ liệu", f"{len(close):,}")
c3.metric("Ngày cuối", close.index[-1].strftime("%d/%m/%Y"))

st.subheader(f"Lịch sử giá {ticker}")
st.line_chart(close)

if not run:
    st.info("Chọn tham số ở thanh bên trái rồi bấm **Huấn luyện & dự báo**.")
    st.stop()

if len(close) < window * 5:
    st.error("Dữ liệu quá ít so với window size. Hãy lùi ngày bắt đầu hoặc giảm window.")
    st.stop()

with st.spinner("Đang huấn luyện mô hình LSTM (có thể mất 1–3 phút)..."):
    hist, y_real, pred, fut, m, test_index = train_and_predict(close, window, epochs, n_future, int(seed))

st.subheader("Độ chính xác trên tập test")
a, b, c = st.columns(3)
a.metric("RMSE", f"{m['RMSE']:,.2f}")
b.metric("MAE", f"{m['MAE']:,.2f}")
c.metric("R²", f"{m['R2']:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Loss: Train vs Validation**")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["loss"], label="Train Loss")
    ax.plot(hist["val_loss"], label="Validation Loss")
    ax.legend(); ax.set_xlabel("Epoch")
    st.pyplot(fig)
with right:
    st.markdown("**Giá thật vs Dự báo (tập test)**")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(test_index, y_real, label="Giá thật")
    ax.plot(test_index, pred, label="Dự báo")
    ax.legend(); fig.autofmt_xdate()
    st.pyplot(fig)

st.subheader(f"Dự báo {n_future} phiên tới")
future_dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=n_future)
fdf = pd.DataFrame({"Ngày": future_dates.strftime("%d/%m/%Y"), "Giá dự báo": fut.round(0)})
col_t, col_g = st.columns([1, 2])
col_t.dataframe(fdf, hide_index=True, use_container_width=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(test_index[-60:], y_real[-60:], label="Giá thật")
ax.plot(test_index[-60:], pred[-60:], label="Dự báo test")
ax.plot(future_dates, fut, marker="o", label=f"{n_future} ngày tới")
ax.legend(); fig.autofmt_xdate()
col_g.pyplot(fig)
st.caption("Dự báo nhiều bước dùng kết quả của chính mô hình làm đầu vào nên sai số tích lũy, chỉ mang tính tham khảo.")
