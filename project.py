import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from scipy.stats import norm

# ============================================================
# 1. Black-Scholes Model
# ============================================================
def black_scholes(S0, K, T, r, sigma, option_type):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return price

# ============================================================
# 2. Monte Carlo Simulation
# ============================================================
def monte_carlo(S0, K, T, r, sigma, N, option_type):
    # สุ่มค่าจาก Normal Distribution
    Z = np.random.normal(0, 1, N)

    # จำลองราคาสินทรัพย์ ณ วันหมดอายุ
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # คำนวณ Payoff
    if option_type == "Call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    # ราคา Option
    option_price = np.exp(-r * T) * np.mean(payoff)

    # Standard Error
    standard_error = np.exp(-r * T) * np.std(payoff, ddof=1) / np.sqrt(N)

    return option_price, standard_error, ST, payoff

# ============================================================
# 3. ดึงข้อมูลตลาด (Optimization 1: ใส่ Cache เพื่อลดภาระ API)
# ============================================================
@st.cache_data(ttl=3600)  # เก็บข้อมูลไว้ 1 ชั่วโมง
def get_market_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="1y")

    if data.empty:
        return None, None, None

    # ราคาปัจจุบัน
    S0 = data["Close"].iloc[-1]

    # Optimization 2: ใช้ dropna() เพื่อจัดการค่า NaN ใน Log Return
    log_returns = np.log(data["Close"] / data["Close"].shift(1)).dropna()

    # Annualized Volatility
    sigma = log_returns.std() * np.sqrt(252)

    return data, S0, sigma

# ============================================================
# 4. ตั้งค่าหน้า Web Application
# ============================================================
st.set_page_config(page_title="Option Pricing Simulator", layout="wide")

# ============================================================
# 5. หัวข้อ Project
# ============================================================
st.title("Mathematical Option Pricing Simulator")
st.subheader("Black-Scholes Model และ Monte Carlo Simulation")
st.write(
    """
    โปรแกรมนี้ใช้หลักการทางคณิตศาสตร์ในการประเมินราคาเหมาะสมทางทฤษฎีของสัญญา Option
    โดยเปรียบเทียบผลลัพธ์จากวิธี Black-Scholes และ Monte Carlo Simulation
    """
)

# ============================================================
# 6. Input Parameters (Sidebar)
# ============================================================
st.sidebar.header("Input Parameters")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
K = st.sidebar.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Expiration (T)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", min_value=0.0, max_value=0.30, value=0.05, step=0.01, format="%.2f")
N = st.sidebar.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
run_button = st.sidebar.button("Run Simulation")

# ============================================================
# 7. เมื่อกด Run
# ============================================================
if run_button:
    # --------------------------------------------------------
    # ดึงข้อมูลตลาด
    # --------------------------------------------------------
    data, S0, sigma = get_market_data(ticker_symbol)

    if data is None:
        st.error("ไม่พบข้อมูล กรุณาตรวจสอบ Ticker Symbol ว่าถูกต้องและมีอยู่ในตลาด")
        st.stop()
        
    # Optimization 4: ดักจับกรณีความผันผวนเป็น 0 หรือไม่สามารถคำนวณได้
    if pd.isna(sigma) or sigma <= 0:
        st.error("ไม่สามารถคำนวณความผันผวน (Volatility) ได้ ข้อมูลประวัติอาจไม่เพียงพอ")
        st.stop()

    # --------------------------------------------------------
    # คำนวณ Models
    # --------------------------------------------------------
    bs_price = black_scholes(S0, K, T, r, sigma, option_type)
    mc_price, standard_error, ST, payoff = monte_carlo(S0, K, T, r, sigma, N, option_type)

    error = abs(bs_price - mc_price)
    lower = mc_price - 1.96 * standard_error
    upper = mc_price + 1.96 * standard_error

    if option_type == "Call":
        probability_itm = np.mean(ST > K)
    else:
        probability_itm = np.mean(ST < K)

    # ========================================================
    # 8. แสดงข้อมูลตลาด
    # ========================================================
    st.header(f"Market Data: {ticker_symbol}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price (S₀)", f"{S0:.2f}")
    with col2:
        st.metric("Volatility (σ)", f"{sigma:.2%}")
    with col3:
        st.metric("Option Type", option_type)

    # ========================================================
    # 9. ผลการคำนวณ
    # ========================================================
    st.header("Option Pricing Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Black-Scholes", f"{bs_price:.4f}")
    with col2:
        st.metric("Monte Carlo", f"{mc_price:.4f}")
    with col3:
        st.metric("Absolute Error", f"{error:.4f}")
    with col4:
        st.metric("Probability ITM", f"{probability_itm:.2%}")

    st.write(f"**95% Confidence Interval ของ Monte Carlo:** [{lower:.4f}, {upper:.4f}]")

    # ========================================================
    # 10. Graph 1: Distribution
    # ========================================================
    st.header("1. Distribution of Asset Price at Expiration")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.hist(ST, bins=50, alpha=0.7, color='teal', edgecolor='black')
    ax1.axvline(K, linestyle="--", linewidth=2, color='red', label="Strike Price K")
    ax1.axvline(S0, linestyle=":", linewidth=2, color='orange', label="Current Price S₀")
    ax1.set_xlabel("Asset Price at Expiration")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Monte Carlo Distribution")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1)

    # ========================================================
    # 11. Graph 2: Simulated Price Paths (Optimization 3: Vectorization)
    # ========================================================
    st.header("2. Monte Carlo Simulated Price Paths")
    steps = 100
    dt = T / steps
    number_of_paths = 100

    # คำนวณแบบ Vectorization เพื่อความรวดเร็วโดยไม่ต้องใช้ For Loop
    Z_paths = np.random.normal(0, 1, (steps, number_of_paths))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z_paths
    daily_returns = np.exp(drift + diffusion)

    # รวมผลลัพธ์สะสม (Cumulative Product) และจัดเตรียม Array
    price_paths = np.zeros((steps + 1, number_of_paths))
    price_paths[0] = S0
    price_paths[1:] = S0 * np.cumprod(daily_returns, axis=0)
    
    time = np.linspace(0, T, steps + 1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(time, price_paths, alpha=0.15)
    ax2.axhline(K, linestyle="--", linewidth=2, color='red', label="Strike Price K")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Asset Price")
    ax2.set_title("Monte Carlo Simulated Price Paths")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)

    # ========================================================
    # 12. Graph 3: Historical Price
    # ========================================================
    st.header("3. Historical Closing Price")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(data.index, data["Close"], color='blue')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.set_title(f"{ticker_symbol} Historical Closing Price")
    ax3.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ========================================================
    # 13. สรุปผล
    # ========================================================
    st.header("Summary")
    st.write(
        f"""
        จากการคำนวณ **{option_type} Option** ของ **{ticker_symbol}** พบว่า
        - ราคาปัจจุบันของสินทรัพย์ = **{S0:.4f}**
        - Volatility = **{sigma:.2%}**
        - ราคาโดย Black-Scholes = **{bs_price:.4f}**
        - ราคาโดย Monte Carlo = **{mc_price:.4f}**
        - ค่าความคลาดเคลื่อน = **{error:.4f}**

        ผลลัพธ์จาก Monte Carlo ควรมีค่าใกล้เคียงกับราคาที่คำนวณจาก Black-Scholes 
        โดยค่าความคลาดเคลื่อนสามารถเปลี่ยนแปลงได้ตามจำนวนครั้งของการจำลอง (Simulations)
        """
    )

else:
    st.info("กรุณากรอกข้อมูลทางด้านซ้าย แล้วกด Run Simulation")