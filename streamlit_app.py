import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
import warnings
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller

import plotly.graph_objects as go
import plotly.express as px
import base64
from io import StringIO
from openai import OpenAI
import io
import os
import cvxpy as cp
from scipy.optimize import minimize


from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
# Set page configuration
st.set_page_config(page_title="BlackRock Advanced Stock Screener", layout="wide")

# Custom CSS to improve the interface
st.markdown("""
<style>
    /* Hide Streamlit branding */
    .css-1d391kg {
        display: none;
    }
    /* Overall background */
    body {
        background-color: #f0f2f5 !important;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 5rem;
        padding-right: 2rem;
        padding-left: 2rem;
        margin: auto;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #333744;
        color: #ffffff;
    }
    /* Links color */
    a {
        color: #0083B8;
    }
    /* Widget labels */
    .Widget>label {
        color: #4f4f4f;
        font-weight: 400;
    }
    /* Buttons */
    .stButton>button {
        border: 2px solid #007bff; /* Updated to blue */
        line-height: 2.5;
        border-radius: 20px;
        color: #ffffff;
        background-color: #007bff; /* Updated to blue */
        transition: all 0.3s;
        box-shadow: none;
    }
    .stButton>button:hover {
        background-color: #006f9a; /* Darker blue for hover effect */
    }
    /* Form input aesthetics */
    .stTextInput>div>div>input, .stSelectbox>select, .stTextArea>div>div>textarea {
        border-radius: 20px;
        border: 1px solid #ced4da;
    }
    /* Markdown text adjustments */
    .markdown-text-container {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    /* Table styling */
    .stTable>div>div>div>div {
        background-color: #fff;
        color: #333;
    }
    /* Radio buttons */
    .stRadio>div>div {
        background-color: #fafafa;
        border-radius: 20px;
    }
    /* Slider styling */
    .stSlider>div>div>div>div {
        background-color: #007bff; /* Updated to blue */
        border-radius: 20px;
    }
    /* File uploader */
    .stFileUploader>div>div>div>button {
        border-radius: 20px;
        border: 2px dashed #007bff; /* Updated to blue */
    }
    /* Checkboxes */
    .stCheckbox>div>div>label>span {
        background-color: #0083B8; /* Updated to blue */
        border-radius: 3px;
    }
    /* Select boxes */
    .stSelectbox>select {
        border-radius: 20px;
        border: 1px solid #ced4da;
    }
    /* Progress bars */
    .stProgress>div>div>div>div {
        background-color: #0083B8; /* Updated to blue */
    }
    /* Hide hamburger menu and Streamlit footer */
    header {visibility: hidden;}
    .css-1kyxreq {visibility: hidden;}
    .css-1v3fvcr {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def get_sp500_tickers(index_name):
    if index_name == 'S&P 500':
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
    elif index_name == 'S&P 100':
        url = 'https://en.wikipedia.org/wiki/S%26P_100'
        tables = pd.read_html(url)
        tickers = tables[2]['Symbol'].tolist()
    elif index_name == 'Dow Jones':
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        tables = pd.read_html(url)
        tickers = tables[1]['Symbol'].tolist()
    elif index_name == 'Nasdaq-100':
        url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        tables = pd.read_html(url)
        tickers = tables[4]['Ticker'].tolist()
    elif index_name == 'S&P 600':
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
    elif index_name == 'Russell 1000':
        url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
        tables = pd.read_html(url)
        tickers = tables[2]['Symbol'].tolist()  # This table index might change

    else:
        tickers = []
    return tickers

def get_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

def calculate_metrics(df):
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(1 + df['Daily_Return'])
    df['Volatility'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal_Line'] = calculate_macd(df['Close'])
    df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
    df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df['Close'])
    df['ATR'] = calculate_atr(df)
    df['OBV'] = calculate_obv(df)
    df['EPS_Growth'] = df['Close'].pct_change(periods=252)  # Assuming 252 trading days in a year
    df['Price_to_Sales'] = df['Close'] / (df['Volume'] * df['Close'].mean())  # Rough estimate
    return df

def calculate_sma(data, window):
    return data.rolling(window=window).mean()
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < 0]
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_calmar_ratio(returns, window=36):
    total_return = (1 + returns).prod() - 1
    max_drawdown = (returns.rolling(window=window, min_periods=1).max() - returns).max()
    return total_return / max_drawdown

def calculate_beta(stock_returns, market_returns):
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

def calculate_alpha(stock_returns, market_returns, risk_free_rate=0.05):
    beta = calculate_beta(stock_returns, market_returns)
    # Corrected formula for alpha calculation
    alpha = (np.mean(stock_returns) - risk_free_rate) - (beta * (np.mean(market_returns) - risk_free_rate))
    return alpha * 252  # Annualized alpha

def calculate_information_ratio(stock_returns, benchmark_returns):
    active_return = stock_returns - benchmark_returns
    tracking_error = np.std(active_return) * np.sqrt(252)
    return np.mean(active_return) * 252 / tracking_error

def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, 100 * (1 - confidence_level))

def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_hurst_exponent(time_series, max_lag=100):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def perform_adf_test(time_series):
    result = adfuller(time_series)
    return result[1]  # p-value
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        return {
            'P/E': info.get('trailingPE', np.nan),
            'P/B': info.get('priceToBook', np.nan),
            'Debt/Equity': info.get('debtToEquity', np.nan),
            'ROE': info.get('returnOnEquity', np.nan),
            'Profit Margin': info.get('profitMargins', np.nan),
            'Revenue Growth': info.get('revenueGrowth', np.nan),
            'Earnings Growth': info.get('earningsGrowth', np.nan),
            'Free Cash Flow': info.get('freeCashflow', np.nan),
            'Dividend Yield': info.get('dividendYield', np.nan),
            'Market Cap': info.get('marketCap', np.nan),
            'EV/EBITDA': info.get('enterpriseToEbitda', np.nan),
            'Price to Sales': info.get('priceToSalesTrailing12Months', np.nan),
            'Operating Margin': info.get('operatingMargins', np.nan),
            'Quick Ratio': info.get('quickRatio', np.nan)
        }
    except:
        return {k: np.nan for k in ['P/E', 'P/B', 'Debt/Equity', 'ROE', 'Profit Margin', 'Revenue Growth', 'Earnings Growth', 'Free Cash Flow', 'Dividend Yield', 'Market Cap', 'EV/EBITDA', 'Price to Sales', 'Operating Margin', 'Quick Ratio']}

def backtest_strategy(df, window_short=50, window_long=200):
    df['SMA_Short'] = df['Close'].rolling(window=window_short).mean()
    df['SMA_Long'] = df['Close'].rolling(window=window_long).mean()
    
    df['Signal'] = np.where(df['SMA_Short'] > df['SMA_Long'], 1, 0)
    df['Signal'] = df['Signal'].shift(1)
    
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'] * df['Returns']
    
    cumulative_returns = (1 + df['Strategy_Returns']).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = calculate_sharpe_ratio(df['Strategy_Returns'])
    max_drawdown = calculate_max_drawdown(df['Strategy_Returns'])
    
    return {
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }
def convert_df_to_csv(df):
    # Create a bytes buffer
    output = io.BytesIO()
    # Write the dataframe to the buffer as an Excel file
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    # Seek to the start of the stream
    output.seek(0)
    # Return the binary content of the Excel file
    return output.getvalue()

def screen_stock(ticker):
    try:
        df = get_stock_data(ticker)
        if df.empty:
            return None
        
        df = calculate_metrics(df)
        latest = df.iloc[-1]
        
        # Get market data (S&P 500 as proxy)
        market_data = get_stock_data('^GSPC')
        market_returns = market_data['Close'].pct_change().dropna()
        
        stock_returns = df['Daily_Return'].dropna()
        
        # Calculate advanced metrics
        sharpe_ratio = calculate_sharpe_ratio(stock_returns)
        sortino_ratio = calculate_sortino_ratio(stock_returns)
        calmar_ratio = calculate_calmar_ratio(stock_returns)
        beta = calculate_beta(stock_returns, market_returns)
        alpha = calculate_alpha(stock_returns, market_returns)
        information_ratio = calculate_information_ratio(stock_returns, market_returns)
        hurst_exponent = calculate_hurst_exponent(df['Close'].values)
        var_95 = calculate_var(stock_returns)
        cvar_95 = calculate_cvar(stock_returns)
        max_drawdown = calculate_max_drawdown(stock_returns)
        adf_p_value = perform_adf_test(df['Close'])
        
        fundamental_data = get_fundamental_data(ticker)
        
        # Machine Learning: Linear Regression for trend prediction
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df['Close'].values
        model = LinearRegression().fit(X, y)
        trend_slope = model.coef_[0]
        trend_score = model.score(X, y)
        
        # Backtesting
        backtest_results = backtest_strategy(df)
        
        # Scoring system
        technical_score = sum([
            np.interp(latest['Close'], [latest['SMA_200'], latest['SMA_50']], [0, 15]),
            15 if 30 < latest['RSI'] < 70 else 0,
            np.interp(latest['MACD'] - latest['Signal_Line'], [-0.5, 0.5], [0, 15]),
            15 if latest['Momentum'] > 0 else 0,
            15 if trend_slope > 0 and trend_score > 0.6 else 0,
            np.interp(latest['Close'], [latest['Bollinger_Lower'], latest['Bollinger_Upper']], [0, 15]),
            10 if latest['OBV'] > df['OBV'].mean() else 0
        ])
        
        fundamental_score = sum([
            np.interp(fundamental_data['P/E'], [60, 10], [0, 10]),
            np.interp(fundamental_data['P/B'], [5, 0], [0, 10]),
            np.interp(fundamental_data['Debt/Equity'], [1.5, 0.5], [0, 10]),
            np.interp(fundamental_data['ROE'], [0.15, 0.2], [5, 10]),
            np.interp(fundamental_data['Profit Margin'], [0.1, 0.2], [5, 10]),
            np.interp(fundamental_data['Revenue Growth'], [0.1, 0.2], [5, 10]),
            np.interp(fundamental_data['Earnings Growth'], [0.1, 0.2], [5, 10]),
            np.interp(fundamental_data['Free Cash Flow'], [0.3, 0.5], [5, 10]),
            np.interp(fundamental_data['EV/EBITDA'], [15, 5], [0, 10]),  # New
            np.interp(fundamental_data['Price to Sales'], [10, 1], [0, 10]),  # New
            np.interp(fundamental_data['Operating Margin'], [0.1, 0.3], [0, 10]),  # New
            np.interp(fundamental_data['Quick Ratio'], [0.5, 2], [0, 10])  # New
        ])
        
        performance_score = sum([
            np.interp(sharpe_ratio, [1, 1.2], [0, 15]),
            np.interp(sortino_ratio, [0.8, 1], [0, 15]),
            np.interp(calmar_ratio, [0.3, 0.5], [0, 15]),
            np.interp(information_ratio, [0.3, 0.5], [0, 15]),
            10 if var_95 > -0.02 else 0,
            10 if cvar_95 > -0.03 else 0,
            10 if max_drawdown > -0.2 else 0,
            np.interp(backtest_results['Total Return'], [0.3, 0.5], [0, 10])
        ])
        
        total_score = (0.25 * technical_score + 0.35 * fundamental_score + 0.4 * performance_score)
        
        # Determine rating based on total score
        if total_score >= 80:
            rating = "Strong Buy"
        elif total_score >= 60:
            rating = "Buy"
        elif total_score >= 40:
            rating = "Hold"
        elif total_score >= 20:
            rating = "Sell"
        else:
            rating = "Strong Sell"
        
        return {
            'Ticker': ticker,
            'Price': latest['Close'],
            'Total Score': total_score,
            'Rating': rating,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Beta': beta,
            'Alpha': alpha,
            'Information Ratio': information_ratio,
            'Hurst Exponent': hurst_exponent,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Max Drawdown': max_drawdown,
            'ADF p-value': adf_p_value,
            'Trend Slope': trend_slope,
            'Trend R2': trend_score,
            'Volatility': latest['Volatility'],
            'RSI': latest['RSI'],
            'P/E': fundamental_data['P/E'],
            'Revenue Growth': fundamental_data['Revenue Growth'],
            'Earnings Growth': fundamental_data['Earnings Growth'],
            'Backtest Total Return': backtest_results['Total Return'],
            'Backtest Sharpe Ratio': backtest_results['Sharpe Ratio'],
            'Backtest Max Drawdown': backtest_results['Max Drawdown'],
            'Operating Margin': fundamental_data['Operating Margin'],
            'Quick Ratio': fundamental_data['Quick Ratio'],
        }
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
    
    return None
@st.cache_data
def LLM_feedback(data, user_question=None):
    if 'Ticker' in data and data['Ticker'] != 'Portfolio':
        # This is for individual stock analysis
        prompt = f"""
        Analyze the following stock data and provide insights:
        Ticker: {data['Ticker']}
        Price: ${data['Price']:.2f}
        Total Score: {data['Total Score']:.2f}
        Rating: {data['Rating']}
        Sharpe Ratio: {data['Sharpe Ratio']:.2f}
        Alpha: {data['Alpha']:.4f}
        Beta: {data['Beta']:.2f}
        VaR (95%): {data['VaR (95%)']:.2%}
        Max Drawdown: {data['Max Drawdown']:.2%}
        P/E Ratio: {data['P/E']:.2f}
        Revenue Growth: {data['Revenue Growth']:.2%}
        Earnings Growth: {data['Earnings Growth']:.2%}
        Backtest Total Return: {data['Backtest Total Return']:.2%}
        Backtest Sharpe Ratio: {data['Backtest Sharpe Ratio']:.2f}

        Provide a brief analysis of the stock's performance, potential risks, and opportunities. 
        Also, suggest whether this stock might be a good fit for different types of investment strategies.
        """
    else:
        # This is for portfolio analysis
        prompt = f"""
        Analyze the following portfolio data and provide insights:
        Investor Age: {data['Age']}
        Investment Goal: {data['Goal']}
        Risk Tolerance: {data['Risk Level']}
        Stocks in Portfolio: {', '.join(data['Stocks'])}
        Stock Allocations: {', '.join([f'{stock}: {allocation:.2%}' for stock, allocation in zip(data['Stocks'], data['Allocations'])])}

        Provide a brief analysis of the portfolio composition, its alignment with the investor's profile (age, goal, and risk tolerance), 
        potential risks, and opportunities. Suggest any improvements or adjustments that might be beneficial. answer user questions
        """

    if user_question:
        prompt += f"\n\nUser's Question: {user_question}\n\nPlease provide a detailed answer to the user's question based on the given data."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst providing concise on stock and portfolio data. Also answer user questiion on stocks and finance"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
def screen_stocks(index_name):
    tickers = get_sp500_tickers(index_name)
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        for result in executor.map(screen_stock, tickers):
            if result:
                results.append(result)
    
    return pd.DataFrame(results).sort_values('Total Score', ascending=False)

def fetch_data(tickers, period="5y"):
    data = yf.download(tickers, period=period, progress=False)['Adj Close']
    return data

def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

def portfolio_performance(weights, returns):
    annual_return = np.sum(returns.mean() * weights) * 252
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return annual_return, annual_volatility

def neg_sharpe_ratio(weights, returns, risk_free_rate=0.02):
    p_return, p_volatility = portfolio_performance(weights, returns)
    sharpe_ratio = (p_return - risk_free_rate) / p_volatility
    return -sharpe_ratio

def max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown
def portfolio_performance(weights, mean_returns, covariance):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
    return portfolio_return, portfolio_volatility
def build_portfolio(age, goal, risk_level, screened_stocks):
    # Define risk tolerance based on risk level
    risk_tolerance = {
        'Low': 0.05,
        'Medium': 0.1,
        'High': 0.15
    }
    target_volatility = risk_tolerance[risk_level]

    # Adjust number of stocks based on risk level
    num_stocks = {
        'Low': 25,
        'Medium': 20,
        'High': 15
    }[risk_level]

    # Filter stocks based on Total Score
    top_stocks = screened_stocks.nlargest(num_stocks, 'Total Score')

    # Get historical price data
    tickers = top_stocks['Ticker'].tolist()
    price_data = fetch_data(tickers)

    # Calculate returns and covariance matrix
    returns = calculate_returns(price_data)
    mean_returns = returns.mean() * 252
    covariance = returns.cov() * 252

    # Monte Carlo simulation to find optimal portfolio
    num_portfolios = 10000
    results = np.zeros((4, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return, portfolio_volatility = portfolio_performance(weights, mean_returns, covariance)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = sharpe_ratio

    # Find the portfolio with volatility closest to the target volatility
    min_diff_index = np.argmin(np.abs(results[1] - target_volatility))
    optimal_weights = weights_record[min_diff_index]

    # Create the portfolio dataframe
    portfolio = pd.DataFrame({
        'Ticker': tickers,
        'Allocation': optimal_weights,
        'Price': top_stocks["Price"].values,
        'Total Score': top_stocks['Total Score'].values,
        'Rating': top_stocks['Rating'].values
    })

    # Sort by allocation (descending) and reset index
    portfolio = portfolio.sort_values('Allocation', ascending=False).reset_index(drop=True)

    # Adjust allocations based on age and goal
    bond_allocation = min(age / 110, 0.6)  # Increase bond allocation with age, max 60%
    if goal == "Short-term Savings":
        bond_allocation = max(bond_allocation, 0.4)  # At least 40% bonds for short-term goals
    elif goal == "Retirement":
        bond_allocation = max(bond_allocation, 0.2)  # At least 20% bonds for retirement
    elif goal == "Wealth Accumulation":
        bond_allocation = max(bond_allocation * 0.5, 0.1)  # Reduce bond allocation for wealth accumulation
    # If the age is less than 38 and the goal is not retirement, exclude bonds
    if age < 38 and goal.lower() != 'retirement':
        bond_allocation = 0

    # Add bond ETF to the portfolio
    bond_etf = pd.DataFrame({
        'Ticker': ['AGG'],
        'Allocation': [bond_allocation],
        'Price': [yf.Ticker('AGG').history(period="1d")['Close'].iloc[-1]],
        'Total Score': [np.nan],
        'Rating': ['N/A']
    })

    # Adjust stock allocations based on risk level and age
    stock_allocation_sum = portfolio['Allocation'].sum()
    age_factor = 1 - (age / 100)  # Younger investors can take more risk
    risk_adjustment_factor = risk_tolerance[risk_level] * age_factor
    portfolio['Allocation'] = portfolio['Allocation'] * (1 - bond_allocation) / stock_allocation_sum
    portfolio['Allocation'] *= risk_adjustment_factor

    # Normalize allocations to sum to 1 after adjustment
    portfolio['Allocation'] /= portfolio['Allocation'].sum()

    # Combine stocks and bond ETF
    portfolio = pd.concat([portfolio, bond_etf], ignore_index=True)

    # Calculate portfolio performance metrics
    portfolio_return, portfolio_volatility = portfolio_performance(optimal_weights, mean_returns, covariance)
    sharpe_ratio = (portfolio_return - 0.025) / portfolio_volatility
    max_dd = max_drawdown(returns.dot(optimal_weights))
    cumulative_returns = (1 + returns.dot(optimal_weights)).cumprod() - 1
    annual_return = cumulative_returns.iloc[-1]
    risk_free_rate = 0.025
    sortino_ratio = (portfolio_return - risk_free_rate) / np.std(returns[returns < 0].dot(optimal_weights))

    portfolio_metrics = {
        'Return': portfolio_return,
        'Volatility': portfolio_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_dd,
        'Cumulative Return': annual_return,
        'Sortino Ratio': sortino_ratio
    }

    return portfolio, portfolio_metrics

def plot_stock_data(stock_data, ticker):
    stock_data['SMA_50'] = calculate_sma(stock_data['Close'], 50)
    stock_data['SMA_200'] = calculate_sma(stock_data['Close'], 200)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Price'
    ))
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['SMA_50'],
        name='50-day SMA',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['SMA_200'],
        name='200-day SMA',
        line=dict(color='green')
    ))
    fig.update_layout(
        title=f'{ticker} Stock Price and Moving Averages',
        yaxis_title='Price',
        xaxis_title='Date'
    )
    return fig

import plotly.express as px

def plot_portfolio_allocation(portfolio):
    fig = px.pie(portfolio, 
                 values='Allocation', 
                 names='Ticker', 
                 title='Portfolio Allocation',
                 color_discrete_sequence=px.colors.sequential.Viridis)
    
    fig.update_traces(textinfo='percent+label', 
                      hoverinfo='label+percent+value',
                      marker=dict(line=dict(color='#000000', width=2)))
    
    fig.update_layout(title_text='Portfolio Allocation by Ticker',
                      title_x=0.5,  # Center the title
                      showlegend=True,
                      legend_title_text='Tickers',
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=-0.2,
                          xanchor="center",
                          x=0.5
                      ))

    return fig

from finvizfinance.quote import finvizfinance
def get_news_sentiment(ticker):
    try:
        stock = finvizfinance(ticker)  # Create an instance for the given ticker
        outer_ratings_df = stock.ticker_outer_ratings()
        news_df = stock.ticker_news()
        # Get only the title and link of the top 10 news items
        top_news_df = news_df[['Title', 'Link']].head(10)
        top_outer = outer_ratings_df[['Date', 'Status', 'Outer', 'Rating', 'Price']].head(6)
        # Initialize the Sentiment Intensity Analyzer
        sia = SentimentIntensityAnalyzer()   
        # Apply sentiment analysis to each title and create a new column 'Sentiment'
        top_news_df['Sentiment'] = top_news_df['Title'].apply(lambda title: sia.polarity_scores(title)['compound'])
        # Filter out news items with a sentiment score of 0
        filtered_df = top_news_df[top_news_df['Sentiment'] != 0]
        # Calculate the average sentiment score of the filtered DataFrame
        if not filtered_df.empty:
            average_sentiment = filtered_df['Sentiment'].mean()
        else:
            average_sentiment = 0  # Return 0 if all sentiments are 0 or if there are no news items
        # Filter `top_news_df` to include only rows where 'Sentiment' is not 0 for `top_news`
        top_news = filtered_df[['Title', 'Link']]
        return top_outer, average_sentiment, top_news
    except Exception as e:
        # Handle exceptions by returning an error message or logging the error
        print(f"An error occurred: {e}")
        # Return None or default values to indicate failure
        return None, None, None
# Setting a better visual style
import seaborn as sns
sns.set(style="whitegrid")

def plot_efficient_frontier(risk_levels, returns_efficient):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(risk_levels, returns_efficient, marker='o', linestyle='-', color='b')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Risk (Standard Deviation)')
    ax.set_ylabel('Return')
    ax.grid(True)
    return fig
def calculate_risk_metrics(returns, market_returns):
    # Volatility
    vol = returns.std() * np.sqrt(252)
    
    # Value at Risk (VaR) - 95% and 99%
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # Conditional Value at Risk (CVaR) - 95% and 99%
    cvar_95 = returns[returns <= var_95].mean()
    cvar_99 = returns[returns <= var_99].mean()
    
    # Beta
    covariance = np.cov(returns, market_returns)[0, 1]
    beta = covariance / market_returns.var()
    
    # Sharpe Ratio
    sharpe_ratio = (returns.mean() * 252 - 0.02) / vol
    
    return vol, var_95, cvar_95, var_99, cvar_99, beta, sharpe_ratio

def calculate_additional_risk_metrics(returns, market_returns):
    # Drawdown metrics
    cumulative_returns = (1 + returns).cumprod()
    drawdowns = (cumulative_returns / cumulative_returns.cummax() - 1)
    max_drawdown = drawdowns.min()
    recovery_time = (drawdowns < 0).astype(int).groupby((drawdowns >= 0).astype(int).cumsum()).cumsum().max()
    
    # Risk-adjusted returns
    sortino_ratio = returns.mean() / returns[returns < 0].std() * np.sqrt(252)
    treynor_ratio = (returns.mean() * 252 - 0.02) / market_returns.mean()
    
    # Correlation and covariance
    correlation = returns.corr(market_returns)
    covariance = returns.cov(market_returns)
    
    # VaR and CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    var_99 = np.percentile(returns, 1)
    cvar_99 = returns[returns <= var_99].mean()
    
    return {
        'Max Drawdown': max_drawdown,
        'Recovery Time': recovery_time,
        'Sortino Ratio': sortino_ratio,
        'Treynor Ratio': treynor_ratio,
        'Correlation with Market': correlation,
        'Covariance with Market': covariance,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'VaR (99%)': var_99,
        'CVaR (99%)': cvar_99
    }


def plot_returns_distribution(returns, label):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name=f'{label} Returns', opacity=0.75))
    fig.add_trace(go.Scatter(x=np.linspace(returns.min(), returns.max(), 100),
                             y=norm.pdf(np.linspace(returns.min(), returns.max(), 100), returns.mean(), returns.std()),
                             mode='lines', name='KDE', line=dict(color='red')))
    fig.update_layout(title=f'{label} Returns Distribution', xaxis_title='Returns', yaxis_title='Frequency')
    return fig

def plot_drawdown(portfolio_data):
    drawdowns = (portfolio_data / portfolio_data.cummax() - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns.min(axis=1), mode='lines', name='Drawdown'))
    fig.update_layout(title='Portfolio Drawdown', xaxis_title='Date', yaxis_title='Drawdown')
    return fig
import plotly.figure_factory as ff
def plot_returns_distribution(returns, ticker):
    # Calculate statistics
    mean = returns.mean()
    std_dev = returns.std()
    
    # Create histogram
    hist_data = [returns]
    group_labels = [ticker]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=returns.std() * 2, show_rug=False, colors=['#636EFA'])
    
    # Add KDE line
    kde_x = np.linspace(returns.min(), returns.max(), 1000)
    kde_y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((kde_x - mean) / std_dev) ** 2)
    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='KDE', line=dict(color='red', width=2, dash='dot')))

    # Add mean and standard deviation lines
    fig.add_vline(x=mean, line_dash="dash", line_color="green", annotation_text="Mean", annotation_position="top left")
    fig.add_vline(x=mean + std_dev, line_dash="dash", line_color="blue", annotation_text="+1 Std Dev", annotation_position="top right")
    fig.add_vline(x=mean - std_dev, line_dash="dash", line_color="blue", annotation_text="-1 Std Dev", annotation_position="bottom right")
    
    # Update layout for better readability and aesthetics
    fig.update_layout(
        title=f"{ticker} Returns Distribution",
        xaxis_title="Returns",
        yaxis_title="Density",
        xaxis=dict(title_font_size=14),
        yaxis=dict(title_font_size=14),
        title_x=0.5,
        legend=dict(x=0.01, y=0.99),
        paper_bgcolor="black",
        plot_bgcolor="black"
    )
    
    return fig
def plot_drawdown_curve(returns, title):
    cumulative_returns = (1 + returns).cumprod()
    drawdowns = cumulative_returns / cumulative_returns.cummax() - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns, mode='lines', name='Drawdown'))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Drawdown',
        xaxis=dict(title_font_size=14),
        yaxis=dict(title_font_size=14),
        title_x=0.5,
        paper_bgcolor="black",
        plot_bgcolor="black"
    )
    
    return fig
def plot_correlation_heatmap(returns_data, title):
    correlation_matrix = returns_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(title)
    plt.show()
def plot_return_boxplot(returns, ticker):
    fig = go.Figure()
    fig.add_trace(go.Box(y=returns, name=ticker))
    fig.update_layout(
        title=f'{ticker} Return Distribution (Box Plot)',
        yaxis_title='Returns',
        xaxis=dict(title_font_size=14),
        yaxis=dict(title_font_size=14),
        title_x=0.5,
        paper_bgcolor="black",
        plot_bgcolor="black"
    )
    
    return fig
def plot_efficient_frontier(risk_levels, returns_efficient):
    fig = go.Figure()
    
    # Add the efficient frontier trace
    fig.add_trace(go.Scatter(
        x=risk_levels,
        y=returns_efficient,
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='royalblue', width=3),
        marker=dict(color='royalblue', size=8, symbol='circle')
    ))
    
    # Add a title and labels
    fig.update_layout(
        title={
            'text': 'Efficient Frontier',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Risk (Standard Deviation)',
        yaxis_title='Expected Return',
        xaxis=dict(
            showline=True,
            showgrid=True,
            zeroline=True,
            linecolor='black',
            linewidth=2,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            zeroline=True,
            linecolor='black',
            linewidth=2,
            gridcolor='lightgray'
        ),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=30, t=60, b=40)
    )
    
    # Add a color bar (if needed)
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Value",
            tickvals=[0, 1],
            ticktext=["Low", "High"]
        )
    )
    
    return fig

def run_monte_carlo(weights, returns, n_simulations=10000, time_horizon=252*3):
    # Ensure weights are numpy array
    weights = np.array(weights)
    
    # Portfolio returns and volatility
    portfolio_return = returns.mean().dot(weights) * time_horizon
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * time_horizon, weights)))
    
    # Simulate returns
    simulated_returns = np.random.normal(portfolio_return, portfolio_volatility, n_simulations)
    
    # Set up the visualization style
    sns.set(style='whitegrid', palette='dark', rc={'figure.figsize':(16, 8)})
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    # Histogram of simulated returns with gradient color
    sns.histplot(simulated_returns, bins=50, kde=True, color='teal', edgecolor='black', ax=ax[0])
    ax[0].set_title('Histogram of Simulated Returns', fontsize=18, fontweight='bold', color='darkblue')
    ax[0].set_xlabel('Return', fontsize=16, fontweight='bold')
    ax[0].set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    ax[0].grid(True, linestyle='--', alpha=0.7)
    
    # Density plot of simulated returns with vibrant color
    sns.kdeplot(simulated_returns, shade=True, color='orangered', ax=ax[1], linewidth=2)
    ax[1].set_title('Density Plot of Simulated Returns', fontsize=18, fontweight='bold', color='darkblue')
    ax[1].set_xlabel('Return', fontsize=16, fontweight='bold')
    ax[1].set_ylabel('Density', fontsize=16, fontweight='bold')
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    return simulated_returns


def calculate_portfolio_metrics(weights, returns, benchmark_returns):
    portfolio_return = returns.mean().dot(weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    # Calculate VaR and CVaR
    confidence_level = 0.95
    simulated_returns = run_monte_carlo(weights, returns)
    var = np.percentile(simulated_returns, 100 * (1 - confidence_level))
    cvar = simulated_returns[simulated_returns <= var].mean()

    # Calculate additional metrics
    downside_returns = returns[returns < 0]
    downside_volatility = np.sqrt(np.mean(downside_returns ** 2))
    sortino_ratio = portfolio_return / downside_volatility
    
    # Calculate Treynor Ratio
    benchmark_beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
    treynor_ratio = portfolio_return / benchmark_beta
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calculate Tracking Error and Information Ratio
    benchmark_returns = benchmark_returns.values
    tracking_error = np.std(returns - benchmark_returns)
    information_ratio = portfolio_return / tracking_error
    
    return {
        "Expected Return": portfolio_return,
        "Volatility": portfolio_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "VaR (95%)": -var,
        "CVaR (95%)": -cvar,
        "Sortino Ratio": sortino_ratio,
        "Treynor Ratio": treynor_ratio,
        "Maximum Drawdown": max_drawdown,
        "Tracking Error": tracking_error,
        "Information Ratio": information_ratio
        
    }   
    
def optimize_portfolio(returns, target_volatility, min_allocation=0.05):
    n = len(returns.columns)
    mu = returns.mean() * 252
    S = returns.cov() * 252

    # Make sure S is symmetric
    S = (S + S.T) / 2

    w = cp.Variable(n)
    ret = mu.values @ w
    risk = cp.quad_form(w, S.values)

    # Constraints: sum of weights = 1, weights >= minimum allocation, risk <= target volatility squared
    constraints = [
        cp.sum(w) == 1,
        w >= min_allocation,  # Ensure minimum allocation
        cp.sum(w) == 1,      # Ensure weights sum to 1
        risk <= target_volatility**2
    ]

    prob = cp.Problem(cp.Maximize(ret), constraints)
    prob.solve()

    # Check if the problem was solved
    if prob.status not in ["infeasible", "unbounded"]:
        return w.value
    else:
        print("Optimization problem was not solved.")
        return None


def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Stock Screener", "Portfolio Builder","Risk analysis","Portfolio Optimization",'News/ Sentiment Analysis'])

    if app_mode == "Stock Screener":
        st.title("Advanced Stock Screener")

        def rerun():
            st.session_state.screened_stocks = None

        
        index_selection = st.selectbox("Select the index for analysis", ["S&P 500", "S&P 100","S&P 600",'Dow Jones',"Russell 1000",'Nasdaq-100'], on_change=rerun)
        
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("Welcome to the Advanced Stock Screener. This tool allows you to analyze stocks from index and get AI-powered insights.")
        with col2:
            run_screener = st.button("Run Stock Screener", key="run_screener")

        if run_screener or st.session_state.get('screened_stocks') is not None:
            if st.session_state.get('screened_stocks') is None:
                with st.spinner("Running stock screener..."):
                    st.session_state.screened_stocks = screen_stocks(index_selection)
                st.success("Stock screening complete!")
            
            screened_stocks = st.session_state.screened_stocks

            st.subheader("Top 10 Stocks")
            st.dataframe(screened_stocks.head(10)[['Ticker', 'Price', 'Total Score', 'Rating', 'Sharpe Ratio', 'Alpha', 'Beta', 'VaR (95%)', 'Max Drawdown', 'Backtest Total Return']])

            csv = convert_df_to_csv(screened_stocks)
            st.download_button(
                label="Download full results as excel",
                data=csv,
                file_name="stock_screening_results.xlsx",
                mime="text/csv",
            )

            st.subheader("Top 10 Stocks Performance")
            fig = go.Figure()
            for i, stock in screened_stocks.head(10).iterrows():
                ticker = stock['Ticker']
                df = get_stock_data(ticker)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=ticker))
            fig.update_layout(title="Price Movement of Top 5 Stocks", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)

            st.subheader("Detailed Stock Analysis")
            selected_stock = st.selectbox("Select a stock for detailed analysis:", screened_stocks['Ticker'].tolist())

            if selected_stock:
                stock_data = screened_stocks[screened_stocks['Ticker'] == selected_stock].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"### {selected_stock} Analysis")
                    st.write(f"Price: ${stock_data['Price']:.2f}")
                    st.write(f"Total Score: {stock_data['Total Score']:.2f}")
                    st.write(f"Rating: {stock_data['Rating']}")
                    
                    st.write("#### Technical Indicators")
                    st.write(f"Sharpe Ratio: {stock_data['Sharpe Ratio']:.2f}")
                    st.write(f"Alpha: {stock_data['Alpha']:.4f}")
                    st.write(f"Beta: {stock_data['Beta']:.2f}")
                    st.write(f"VaR (95%): {stock_data['VaR (95%)']:.2%}")
                    st.write(f"Max Drawdown: {stock_data['Max Drawdown']:.2%}")
                    st.write(f"Trend Slope: {stock_data['Trend Slope']:.4f}")
                    st.write(f"Hurst Exponent: {stock_data['Hurst Exponent']:.4f}")
                    st.write(f"Volatility: {stock_data['Volatility']:.2%}")

                with col2:
                    st.write("#### Fundamental Indicators")
                    st.write(f"P/E Ratio: {stock_data['P/E']:.2f}")
                    st.write(f"Revenue Growth: {stock_data['Revenue Growth']:.2%}")
                    st.write(f"Earnings Growth: {stock_data['Earnings Growth']:.2%}")
                    st.write(f"Operating Margin: {stock_data['Operating Margin']:.2%}")
                    st.write(f"Quick Ratio: {stock_data['Quick Ratio']:.2f}")
                    st.write(f"Backtest Total Return: {stock_data['Backtest Total Return']:.2%}")
                    st.write(f"Backtest Sharpe Ratio: {stock_data['Backtest Sharpe Ratio']:.2f}")

                st.subheader("Stock Price Chart")
                stock_df = get_stock_data(selected_stock)
                st.plotly_chart(plot_stock_data(stock_df, selected_stock))

                st.subheader("AI-Powered Analysis")
                with st.spinner("Generating AI analysis..."):
                    chatgpt_analysis =LLM_feedback(stock_data)
                st.write(chatgpt_analysis)

                st.subheader("Ask a Question")
                user_question = st.text_input("Ask a question about this stock:")
                if user_question:
                    with st.spinner("Generating answer..."):
                        answer =LLM_feedback(stock_data, user_question)
                    st.write("Answer:", answer)

    elif app_mode == "Portfolio Builder":
        st.title("Portfolio Builder")
        def rerun():
            st.session_state.screened_stocks = None

        st.write("Welcome to the Portfolio Builder. This tool helps you create a personalized investment portfolio based on your age, goals, and risk tolerance.")
        index_selection = st.selectbox("Select the index for analysis", ["S&P 500", "S&P 100","S&P 600",'Dow Jones',"Russell 1000",'Nasdaq-100'], on_change=rerun)
        
        tickers = get_sp500_tickers(index_selection)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("What is your age?", 18, 100, 30)
        with col2:
            goal = st.selectbox("What is your investment goal?", ["Retirement", "Short-term Savings", "Wealth Accumulation"])
        with col3:
            risk_level = st.selectbox("What is your risk tolerance?", ["Low", "Medium", "High"])

        if st.button("Build Portfolio"):
            with st.spinner("Building your portfolio..."):
                if 'screened_stocks' not in st.session_state:
                    st.session_state.screened_stocks = screen_stocks(index_selection)
                portfolio, portfolio_metrics = build_portfolio(age, goal, risk_level, st.session_state.screened_stocks)

            st.success("Portfolio built successfully!")

            st.subheader("Your Personalized Portfolio")
            st.dataframe(portfolio[['Ticker', 'Allocation', 'Price', 'Total Score', 'Rating']])

            csv = convert_df_to_csv(portfolio)
            st.download_button(
                label="Download portfolio as Excel",
                data=csv,
                file_name="personalized_portfolio.xlsx",
                mime="text/csv",
            )

            st.subheader("Portfolio Metrics")
            st.write(f"**Annual Return:** {portfolio_metrics['Return']:.2%}")
            st.write(f"**Volatility:** {portfolio_metrics['Volatility']:.2%}")
            st.write(f"**Sharpe Ratio:** {portfolio_metrics['Sharpe Ratio']:.2f}")
            st.write(f"**Sortino Ratio:** {portfolio_metrics['Sortino Ratio']:.2f}")
            st.write(f"**Cumulative Return:** {portfolio_metrics['Cumulative Return']:.2%}")
            st.write(f"**Maximum Drawdown:** {portfolio_metrics['Max Drawdown']:.2%}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Portfolio Allocation")
                st.plotly_chart(plot_portfolio_allocation(portfolio))
            
            with col2:
                st.subheader("Portfolio Analysis")
                portfolio_analysis =LLM_feedback({
                    'Ticker': 'Portfolio',
                    'Age': age,
                    'Goal': goal,
                    'Risk Level': risk_level,
                    'Stocks': portfolio['Ticker'].tolist(),
                    'Allocations': portfolio['Allocation'].tolist()
                })
                st.write(portfolio_analysis)
    elif app_mode == "Risk analysis":
        st.title("Risk analysis")
        st.write("Welcome to the Risk Analysis tool. This tool helps you analyze the risk of individual stocks or your entire portfolio.")

        analysis_type = st.radio("Choose analysis type:", ["Single Stock", "Portfolio"])

        if analysis_type == "Single Stock":
            ticker = st.text_input("Enter stock ticker:")
            if ticker:
                stock_data = get_stock_data(ticker)
                returns = stock_data['Close'].pct_change().dropna()

                # Market data for beta calculation
                market_data = get_stock_data('SPY')
                market_returns = market_data['Close'].pct_change().dropna()

                vol, var_95, cvar_95, var_99, cvar_99, beta, sharpe_ratio = calculate_risk_metrics(returns, market_returns)
                
                # Additional metrics
                additional_metrics = calculate_additional_risk_metrics(returns, market_returns)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader(f"Risk Metrics for {ticker}")
                    st.write(f"**Volatility:** {vol:.2%}")
                    st.write(f"**Value at Risk (95%):** {var_95:.2%}")
                    st.write(f"**Conditional Value at Risk (95%):** {cvar_95:.2%}")
                    st.write(f"**Value at Risk (99%):** {var_99:.2%}")
                    st.write(f"**Conditional Value at Risk (99%):** {cvar_99:.2%}")
                    st.write(f"**Beta:** {beta:.2f}")
                    st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
                    st.subheader(f"{ticker} Drawdown Curve")
                    drawdown_fig = plot_drawdown_curve(returns, f'{ticker} Drawdown Curve')
                    st.plotly_chart(drawdown_fig)
                with col2:
                    
                    st.write(f"**Maximum Drawdown:** {additional_metrics['Max Drawdown']:.2%}")
                    st.write(f"**Recovery Time (days):** {additional_metrics['Recovery Time']}")
                    st.write(f"**Sortino Ratio:** {additional_metrics['Sortino Ratio']:.2f}")
                    st.write(f"**Treynor Ratio:** {additional_metrics['Treynor Ratio']:.2f}")
                    st.write(f"**Correlation with Market:** {additional_metrics['Correlation with Market']:.2f}")
                    
                    st.subheader(f"{ticker} Returns Distribution")
                    fig = plot_returns_distribution(returns, ticker)
                    st.plotly_chart(fig)
                
                
                
                st.subheader(f"{ticker} Return Distribution (Box Plot)")
                boxplot_fig = plot_return_boxplot(returns, ticker)
                st.plotly_chart(boxplot_fig)
                

        elif analysis_type == "Portfolio":
            uploaded_file = st.file_uploader("Upload your portfolio CSV file", type="csv")
            if uploaded_file is not None:
                portfolio = pd.read_csv(uploaded_file)
                st.write("Uploaded Portfolio:")
                st.dataframe(portfolio)

                tickers = portfolio['Ticker'].tolist()
                weights = portfolio['Allocation'].values

                portfolio_data = yf.download(tickers, period='2y')['Adj Close']
                portfolio_returns = portfolio_data.pct_change().dropna()

                # Calculate weighted returns
                weighted_returns = (portfolio_returns * weights).sum(axis=1)

                # Market data for beta calculation
                market_data = get_stock_data('SPY')
                market_returns = market_data['Close'].pct_change().dropna()

                vol, var_95, cvar_95, var_99, cvar_99, beta, sharpe_ratio = calculate_risk_metrics(weighted_returns, market_returns)
                max_drawdown = (portfolio_data / portfolio_data.cummax() - 1).min().min()

                st.subheader("Portfolio Risk Metrics")
                st.write(f"**Annual Return:** {weighted_returns.mean() * 252:.2%}")
                st.write(f"**Volatility:** {vol:.2%}")
                st.write(f"**Value at Risk (95%):** {var_95:.2%}")
                st.write(f"**Conditional Value at Risk (95%):** {cvar_95:.2%}")
                st.write(f"**Value at Risk (99%):** {var_99:.2%}")
                st.write(f"**Conditional Value at Risk (99%):** {cvar_99:.2%}")
                st.write(f"**Maximum Drawdown:** {max_drawdown:.2%}")
                st.write(f"**Beta:** {beta:.2f}")
                st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

                st.subheader("Portfolio Allocation")
                st.dataframe(portfolio)

                st.subheader("Portfolio Returns Distribution")
                fig = plot_returns_distribution(weighted_returns, 'Portfolio')
                st.plotly_chart(fig)

                st.subheader("Portfolio Drawdown")
                drawdown_fig = plot_drawdown(portfolio_data)
                st.plotly_chart(drawdown_fig)
    elif app_mode == "Portfolio Optimization":
        st.title("Portfolio Optimization")
        st.write("Welcome to the Portfolio Optimization tool. This tool helps you optimize your portfolio for maximum return given a risk constraint.")

            # Input method selection
        input_method = st.radio("Select input method:", ("Manual Entry", "CSV Upload"))

        if input_method == "Manual Entry":
            tickers = st.text_input("Enter stock tickers (comma-separated):")
            if tickers:
                tickers = [t.strip().upper() for t in tickers.split(',')]
        else:
            uploaded_file = st.file_uploader("Upload CSV file with tickers", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                tickers = df['Ticker'].tolist()  # Assuming the CSV has a 'Ticker' column

        if 'tickers' in locals() and tickers:
            st.write(f"Selected Tickers: {', '.join(tickers)}")

            # Download data
            try:
                data = yf.download(tickers, period="3y")['Adj Close']
                returns = data.pct_change().dropna()
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return

            # User input for target volatility
            target_volatility = st.slider("Select target annual volatility:", 
                                            min_value=0.05, max_value=0.50, 
                                            value=0.20, step=0.01)

            # Optimize portfolio
            optimal_weights = optimize_portfolio(returns, target_volatility)

            # Display optimal portfolio allocation
            case1, case2 = st.columns(2)
            
            with case1:
                st.write("Optimal Portfolio Allocation:")
                for ticker, weight in zip(tickers, optimal_weights):
                    st.write(f"{ticker}: {weight:.2%}")

            # Calculate and display portfolio metrics
            metrics = calculate_portfolio_metrics(optimal_weights, returns, returns.mean())
            with case2:
                st.write("\nPortfolio Metrics:")
                for metric, value in metrics.items():
                    if metric in ['Expected Return', 'Volatility']:
                        st.write(f"{metric}: {value:.2%}")
                    elif metric in ['Beta', 'VaR (95%)', 'CVaR (95%)']:
                        st.write(f"{metric}: {value:.2f}")
                    else:
                        
                        st.write(f"{metric}: {value}")

            risk_levels = np.linspace(0.05, 0.50, 100)
            returns_efficient = []
            for risk_level in risk_levels:
                weights = optimize_portfolio(returns, risk_level)
                if weights is not None:
                    if weights.shape[0] == returns.shape[1]:
                        portfolio_return = returns.mean().dot(weights) * 252
                        returns_efficient.append(portfolio_return)
                    

            # Efficient Frontier
            st.plotly_chart(plot_efficient_frontier(risk_levels, returns_efficient))

            # Monte Carlo Simulation
            st.write("\nMonte Carlo Simulation:")
            simulated_returns = run_monte_carlo(optimal_weights, returns)
            fig, ax = plt.subplots()
            ax.hist(simulated_returns, bins=50, alpha=0.7, color='blue')
            ax.set_xlabel('Portfolio Return')
            ax.set_ylabel('Frequency')
            ax.set_title('Monte Carlo Simulation of Portfolio Returns')
            st.pyplot(fig)

    elif app_mode == "News/ Sentiment Analysis":
        st.title("News/ Sentiment Analysis")
        st.write("Welcome to the News/ Sentiment Analysis tool. This tool helps you get sentiment score of a stock and the news sorrouding it.")
        
        ticker = st.text_input("Enter the stock ticker:").upper()
        
        if ticker and not ticker.isalpha():
            st.error("Please enter a valid stock ticker consisting only of letters.")
        elif st.button("Sentiment score") and ticker:
            with st.spinner('sentiment score...'):
                try:
                    # Assuming df and fundamental_data are defined earlier in the code or fetched within this block
                    top_outer, average_sentiment, top_news = get_news_sentiment( ticker)
                    st.success(f"The sentiment score for {ticker} is: {average_sentiment}")
                    # Formatting top_news for display
                    news_display = "\n".join([f"{index + 1}. {row['Title']} - {row['Link']}" for index, row in top_news.iterrows()])
                    st.success(f"The news and links for {ticker} are:  \n{news_display}")
                    # Formatting top_outer for display
                    outer_display = "\n".join([f"{index + 1}. Date: {row['Date']}, Status: {row['Status']}, Outer: {row['Outer']}, Rating: {row['Rating']}, Price: {row['Price']}" for index, row in top_outer.iterrows()])
                    st.success(f"Top Outer Ratings for {ticker} are: \n{outer_display}")
                except Exception as e:
                    st.error(f"Failed to estimate fair value due to: {e}")
if __name__ == "__main__":
    main()
