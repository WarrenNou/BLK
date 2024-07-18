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
import os

warnings.filterwarnings('ignore')
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
# Set page configuration
st.set_page_config(page_title="BlackRock Advanced Stock Screener", layout="wide")

# Custom CSS to improve the interface
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0056b3;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #003d82;
    }
</style>
""", unsafe_allow_html=True)



def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    tables = pd.read_html(url)
    sp100 = tables[2]
    return sp100['Symbol'].tolist()

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

def calculate_alpha(stock_returns, market_returns, risk_free_rate=0.02):
    beta = calculate_beta(stock_returns, market_returns)
    alpha = np.mean(stock_returns) - risk_free_rate - beta * np.mean(market_returns)
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
            'Market Cap': info.get('marketCap', np.nan)
        }
    except:
        return {k: np.nan for k in ['P/E', 'P/B', 'Debt/Equity', 'ROE', 'Profit Margin', 'Revenue Growth', 'Earnings Growth', 'Free Cash Flow', 'Dividend Yield', 'Market Cap']}
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
    return df.to_csv(index=False).encode('utf-8')

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
            15 if latest['Close'] > latest['SMA_50'] > latest['SMA_200'] else 0,
            15 if 30 < latest['RSI'] < 70 else 0,
            15 if latest['MACD'] > latest['Signal_Line'] else 0,
            15 if latest['Momentum'] > 0 else 0,
            15 if trend_slope > 0 and trend_score > 0.6 else 0,
            15 if latest['Close'] > latest['Bollinger_Upper'] else 0,
            10 if latest['OBV'] > df['OBV'].mean() else 0
        ])
        
        fundamental_score = sum([
            10 if 10 < fundamental_data['P/E'] < 30 else 5 if 30 <= fundamental_data['P/E'] < 60 else 0,
            10 if 0 < fundamental_data['P/B'] < 2 else 5 if 2 <= fundamental_data['P/B'] < 5 else 0,
            10 if fundamental_data['Debt/Equity'] < 0.5 else 5 if 0.5 <= fundamental_data['Debt/Equity'] < 1.5 else 0,
            10 if fundamental_data['ROE'] > 0.2 else 5 if 0.15 <= fundamental_data['ROE'] <= 0.2 else 0,
            10 if fundamental_data['Profit Margin'] > 0.2 else 5 if 0.1 <= fundamental_data['Profit Margin'] <= 0.2 else 0,
            10 if fundamental_data['Revenue Growth'] > 0.2 else 5 if 0.1 <= fundamental_data['Revenue Growth'] <= 0.2 else 0,
            10 if fundamental_data['Earnings Growth'] > 0.2 else 5 if 0.1 <= fundamental_data['Earnings Growth'] <= 0.2 else 0,
            10 if fundamental_data['Free Cash Flow'] > 0.5 else 5 if 0.3 <= fundamental_data['Free Cash Flow'] <= 0.5 else 0,
            10 if fundamental_data['Dividend Yield'] > 0.02 else 5 if 0.01 <= fundamental_data['Dividend Yield'] <= 0.02 else 0,
            10 if fundamental_data['Market Cap'] > 1e10 else 5 if 1e9 <= fundamental_data['Market Cap'] <= 1e10 else 0
        ])
        
        performance_score = sum([
            15 if sharpe_ratio > 1.2 else 0,
            15 if sortino_ratio > 1 else 0,
            15 if calmar_ratio > 0.5 else 0,
            15 if information_ratio > 0.5 else 0,
            10 if var_95 > -0.02 else 0,
            10 if cvar_95 > -0.03 else 0,
            10 if max_drawdown > -0.2 else 0,
            10 if backtest_results['Total Return'] > 0.5 else 0
        ])
        
        total_score = (0.35 * technical_score + 0.25 * fundamental_score + 0.4 * performance_score)
        
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
            'Backtest Max Drawdown': backtest_results['Max Drawdown']
        }
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
    
    return None
@st.cache_data
def get_chatgpt_analysis(data, user_question=None):
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
        potential risks, and opportunities. Suggest any improvements or adjustments that might be beneficial.
        """

    if user_question:
        prompt += f"\n\nUser's Question: {user_question}\n\nPlease provide a detailed answer to the user's question based on the given data."

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a financial analyst providing concise on stock and portfolio data."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
def screen_stocks():
    tickers = get_sp500_tickers()
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        for result in executor.map(screen_stock, tickers):
            if result:
                results.append(result)
    
    return pd.DataFrame(results).sort_values('Total Score', ascending=False)

import cvxpy as cp

def build_portfolio(age, goal, risk_level, screened_stocks):
    # Define bond allocation based on age
    if age < 35:
        bond_allocation = 0.1
    elif 35 <= age < 50:
        bond_allocation = 0.2
    else:
        bond_allocation = 0.3

    # Define risk tolerance based on risk level
    risk_tolerance = {
        'Low': 0.05,
        'Medium': 0.1,
        'High': 0.2
    }
    
    if risk_level not in risk_tolerance:
        raise ValueError("Invalid risk level provided. Choose from 'Conservative', 'Moderate', 'Aggressive'.")
    
    target_volatility = risk_tolerance[risk_level]

    # Ensure 'Market Cap' column exists and is numeric
    if 'Market Cap' not in screened_stocks.columns:
        screened_stocks['Market Cap'] = np.nan
    screened_stocks['Market Cap'] = pd.to_numeric(screened_stocks['Market Cap'], errors='coerce')

    # Filter stocks based on market cap
    large_cap = screened_stocks[screened_stocks['Market Cap'] > 10e9]
    mid_cap = screened_stocks[(screened_stocks['Market Cap'] > 2e9) & (screened_stocks['Market Cap'] <= 10e9)]
    small_cap = screened_stocks[screened_stocks['Market Cap'] <= 2e9]

    # If any category is empty, fill it with the top stocks from the screened list
    if large_cap.empty:
        large_cap = screened_stocks.nlargest(5, 'Total Score')
    if mid_cap.empty:
        mid_cap = screened_stocks.nlargest(5, 'Total Score').iloc[5:10]
    if small_cap.empty:
        small_cap = screened_stocks.nlargest(5, 'Total Score').iloc[10:15]

    # Combine the stocks
    portfolio_stocks = pd.concat([large_cap, mid_cap, small_cap])

    # Expand the portfolio to 15 stocks based on 'Total Score'
    portfolio_stocks = screened_stocks.nlargest(15, 'Total Score')

    # Get historical price data
    tickers = portfolio_stocks['Ticker'].tolist()
    price_data = yf.download(tickers, period="5y")['Adj Close']

    # Calculate returns and covariance matrix
    returns = price_data.pct_change().mean() * 252
    covariance = price_data.pct_change().cov() * 252

    # Set up the optimization problem
    num_stocks = len(tickers)
    weights = cp.Variable(num_stocks)
    portfolio_return = returns.values @ weights
    portfolio_risk = cp.quad_form(weights, covariance.values)

    # Objective: Maximize returns while controlling risk
    objective = cp.Maximize(portfolio_return - target_volatility * portfolio_risk)
    constraints = [cp.sum(weights) == (1 - bond_allocation), weights >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Get the optimal weights
    optimal_weights = weights.value

    # Create the portfolio dataframe
    portfolio = portfolio_stocks.copy()
    portfolio['Allocation'] = optimal_weights

    # Add bond ETF to the portfolio
    bond_df = pd.DataFrame({
        'Ticker': ['AGG'],
        'Allocation': [bond_allocation]
    })

    portfolio = pd.concat([portfolio, bond_df], ignore_index=True)

    # Monte Carlo simulation
    num_simulations = 100
    num_days = 252
    simulation_results = np.zeros((num_simulations, num_days))

    for i in range(num_simulations):
        simulated_prices = pd.DataFrame(index=np.arange(num_days), columns=tickers)
        for ticker in tickers:
            simulated_prices[ticker] = simulated_prices.index.to_series().apply(
                lambda x: price_data[ticker].iloc[-1] * np.exp(
                    (returns[ticker] / 252) * x - 0.5 * (covariance.loc[ticker, ticker] / 252) * x +
                    np.sqrt(covariance.loc[ticker, ticker] / 252) * np.random.normal()
                )
            )
        portfolio_values = simulated_prices.dot(optimal_weights)
        simulation_results[i, :] = portfolio_values

    # Plot the results of the Monte Carlo simulation
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_results.T)
    plt.title('Monte Carlo Simulation of Portfolio Value Over Time')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value')
    plt.show()

    return portfolio

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

def plot_portfolio_allocation(portfolio):
    fig = px.pie(portfolio, values='Allocation', names='Ticker', title='Portfolio Allocation')
    return fig

def main():
    st.sidebar.image("https://www.blackrock.com/blk-logo-white-rgb.svg", use_column_width=True)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Stock Screener", "Portfolio Builder"])

    if app_mode == "Stock Screener":
        st.title("Stock Screener")
        
        col1, col2 = st.columns([2,1])
        with col1:
            st.write("Welcome to the Advanced Stock Screener. This tool allows you to analyze stocks from the S&P 100 index and get AI-powered insights.")
        with col2:
            run_screener = st.button("Run Stock Screener", key="run_screener")

        if run_screener or 'screened_stocks' in st.session_state:
            if 'screened_stocks' not in st.session_state:
                with st.spinner("Running stock screener..."):
                    st.session_state.screened_stocks = screen_stocks()
                st.success("Stock screening complete!")
            
            screened_stocks = st.session_state.screened_stocks

            st.subheader("Top 10 Stocks")
            st.dataframe(screened_stocks.head(10)[['Ticker', 'Price', 'Total Score', 'Rating', 'Sharpe Ratio', 'Alpha', 'Beta', 'VaR (95%)', 'Max Drawdown', 'Backtest Total Return']])

            csv = convert_df_to_csv(screened_stocks)
            st.download_button(
                label="Download full results as CSV",
                data=csv,
                file_name="stock_screening_results.csv",
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

                with col2:
                    st.write("#### Fundamental Indicators")
                    st.write(f"P/E Ratio: {stock_data['P/E']:.2f}")
                    st.write(f"Revenue Growth: {stock_data['Revenue Growth']:.2%}")
                    st.write(f"Earnings Growth: {stock_data['Earnings Growth']:.2%}")
                    st.write(f"Backtest Total Return: {stock_data['Backtest Total Return']:.2%}")
                    st.write(f"Backtest Sharpe Ratio: {stock_data['Backtest Sharpe Ratio']:.2f}")

                st.subheader("Stock Price Chart")
                stock_df = get_stock_data(selected_stock)
                st.plotly_chart(plot_stock_data(stock_df, selected_stock))

                st.subheader("AI-Powered Analysis")
                with st.spinner("Generating AI analysis..."):
                    chatgpt_analysis = get_chatgpt_analysis(stock_data)
                st.write(chatgpt_analysis)

                st.subheader("Ask a Question")
                user_question = st.text_input("Ask a question about this stock:")
                if user_question:
                    with st.spinner("Generating answer..."):
                        answer = get_chatgpt_analysis(stock_data, user_question)
                    st.write("Answer:", answer)

    elif app_mode == "Portfolio Builder":
        st.title("BlackRock Portfolio Builder")
        
        st.write("Welcome to the BlackRock Portfolio Builder. This tool helps you create a personalized investment portfolio based on your age, goals, and risk tolerance.")

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
                    st.session_state.screened_stocks = screen_stocks()
                portfolio = build_portfolio(age, goal, risk_level, st.session_state.screened_stocks)

            st.success("Portfolio built successfully!")

            st.subheader("Your Personalized Portfolio")
            st.dataframe(portfolio[['Ticker', 'Allocation', 'Price', 'Total Score', 'Rating']])

            csv = convert_df_to_csv(portfolio)
            st.download_button(
                label="Download portfolio as CSV",
                data=csv,
                file_name="personalized_portfolio.csv",
                mime="text/csv",
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Portfolio Allocation")
                st.plotly_chart(plot_portfolio_allocation(portfolio))
            
            with col2:
                st.subheader("Portfolio Analysis")
                portfolio_analysis = get_chatgpt_analysis({
                    'Ticker': 'Portfolio',
                    'Age': age,
                    'Goal': goal,
                    'Risk Level': risk_level,
                    'Stocks': portfolio['Ticker'].tolist(),
                    'Allocations': portfolio['Allocation'].tolist()
                })
                st.write(portfolio_analysis)

if __name__ == "__main__":
    main()