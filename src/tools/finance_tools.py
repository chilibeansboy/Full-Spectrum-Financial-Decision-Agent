# src/tools/finance_tools.py (Modified to be a Helper Module)
import yfinance as yf
import pandas as pd
import talib # 導入 ta-lib
from typing import Tuple
# 移除 from langchain_core.tools import tool

# 注意: 此函式不再是 LangChain Tool，而是被 Agent 直接調用的 Helper 函式。
def fetch_fundamental_and_kline_data(ticker: str) -> Tuple[str, pd.DataFrame]:
    """
    Retrieves stock data for a given ticker, returning a formatted fundamental/performance string 
    for the LLM (Fundamental Analyst) and the raw K-line DataFrame for the 
    Technical Specialist.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # --- 1. 獲取並準備 K-line Data (DataFrame) ---
        kline_df = stock.history(period="1y")[['Open', 'High', 'Low', 'Close', 'Volume']]
        kline_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if kline_df.empty:
            return f"No price data found for {ticker}.", pd.DataFrame()
            
        # --- 2. 獲取並準備 Fundamental Data (Text Summary) ---
        info = stock.info
        
        valuation = {
            "Market Cap": info.get("marketCap"),
            "Enterprise Value": info.get("enterpriseValue"),
            "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "PEG Ratio": info.get("pegRatio"),
            "Price/Book": info.get("priceToBook"),
            "Price/Sales": info.get("priceToSalesTrailing12Months"),
            "EV/EBITDA": info.get("enterpriseToEbitda"),
        }
        
        financials = {
            "Revenue Growth (YoY)": info.get("revenueGrowth"),
            "Earnings Growth (YoY)": info.get("earningsGrowth"),
            "Gross Margins": info.get("grossMargins"),
            "Operating Margins": info.get("operatingMargins"),
            "Return on Equity (ROE)": info.get("returnOnEquity"),
            "Total Cash": info.get("totalCash"),
            "Total Debt": info.get("totalDebt"),
            "Free Cash Flow": info.get("freeCashflow"),
        }
        
        estimates = {
            "Target Mean Price": info.get("targetMeanPrice"),
            "Target High": info.get("targetHighPrice"),
            "Target Low": info.get("targetLowPrice"),
            "Recommendation": info.get("recommendationKey"),
            "Number of Analyst Opinions": info.get("numberOfAnalystOpinions")
        }
        
        # Price Performance Summary (使用 K-line data)
        current_price = kline_df.iloc[-1]["Close"]
        price_1mo_ago = kline_df.iloc[-22]["Close"] if len(kline_df) > 22 else kline_df.iloc[0]["Close"]
        price_6mo_ago = kline_df.iloc[-126]["Close"] if len(kline_df) > 126 else kline_df.iloc[0]["Close"]
        
        performance = {
            "Current Price": f"{current_price:.2f}",
            "52 Week High": info.get("fiftyTwoWeekHigh"),
            "52 Week Low": info.get("fiftyTwoWeekLow"),
            "1 Month Return": f"{((current_price - price_1mo_ago) / price_1mo_ago) * 100:.2f}%",
            "6 Month Return": f"{((current_price - price_6mo_ago) / price_6mo_ago) * 100:.2f}%",
            "YTD Return": f"{((current_price - kline_df.iloc[0]['Close']) / kline_df.iloc[0]['Close']) * 100:.2f}%"
        }
        
        # 組合為 LLM 的報告文本
        fundamental_text_summary = f"""
Ticker: {ticker}
--- VALUATION ---
{valuation}

--- FINANCIALS ---
{financials}

--- ANALYST ESTIMATES ---
{estimates}

--- PRICE PERFORMANCE ---
{performance}

--- RECENT PRICE DATA (Last 5 Days) ---
{kline_df.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']].to_string()}
"""
        
        return fundamental_text_summary, kline_df

    except Exception as e:
        error_msg = f"Error fetching data for {ticker}: {str(e)}"
        return error_msg, pd.DataFrame()


def calculate_technical_indicators(kline_data: pd.DataFrame) -> str:
    """
    Calculates core technical indicators (RSI, MACD, Stochastic) and returns a summary string.
    """
    if kline_data.empty or len(kline_data) < 30:
        return "Technical analysis is not possible: insufficient K-line data (requires >= 30 data points)."

    close = kline_data['Close'].values
    high = kline_data['High'].values
    low = kline_data['Low'].values
    
    rsi = talib.RSI(close, timeperiod=14)
    latest_rsi = rsi[-1] if not pd.isna(rsi[-1]) else float('nan')
    
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    latest_macd_hist = macdhist[-1] if not pd.isna(macdhist[-1]) else float('nan')
    
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    latest_slowk = slowk[-1] if not pd.isna(slowk[-1]) else float('nan')
    latest_slowd = slowd[-1] if not pd.isna(slowd[-1]) else float('nan')

    report = (
        "***Technical Indicator Calculation Results (Latest Data Point)***\n"
        f"- **RSI (14)**: {latest_rsi:.2f}\n"
        f"- **MACD Histogram**: {latest_macd_hist:.2f}\n"
        f"- **Stochastic Oscillator (%K/%D)**: {latest_slowk:.2f} / {latest_slowd:.2f}\n"
    )
    
    return report