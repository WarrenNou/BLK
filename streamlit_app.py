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
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp100 = tables[0]
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
def screen_stocks():
    tickers = get_sp500_tickers()
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        for result in executor.map(screen_stock, tickers):
            if result:
                results.append(result)
    
    return pd.DataFrame(results).sort_values('Total Score', ascending=False)

def build_portfolio(age, goal, risk_level, screened_stocks):
    # Define risk tolerance based on risk level
    risk_tolerance = {
        'Low': 0.05,
        'Medium': 0.1,
        'High': 0.25
    }
    target_volatility = risk_tolerance[risk_level]

    # Filter stocks based on Total Score
    top_stocks = screened_stocks.nlargest(20, 'Total Score')

    # Get historical price data
    tickers = top_stocks['Ticker'].tolist()
    price_data = yf.download(tickers, period="2y", progress=False)['Adj Close']

    # Calculate returns and covariance matrix
    returns = price_data.pct_change().mean() * 252
    covariance = price_data.pct_change().cov() * 252

    # Define the objective function (negative Sharpe Ratio)
    def objective(weights):
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
        return -sharpe_ratio

    # Define constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    bounds = tuple((0, 0.25) for _ in range(len(tickers)))  # No short selling, max 25% in one stock

    # Perform the optimization
    initial_weights = np.array([1/len(tickers)] * len(tickers))
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Get the optimal weights
    optimal_weights = result.x

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
    bond_allocation = min(age / 110, 0.4)  # Increase bond allocation with age, max 40%
    if goal == "Short-term Savings":
        bond_allocation = max(bond_allocation, 0.3)  # At least 30% bonds for short-term goals

    # Add bond ETF to the portfolio
    bond_etf = pd.DataFrame({
        'Ticker': ['AGG'],
        'Allocation': [bond_allocation],
        'Price': [yf.Ticker('AGG').history(period="1d")['Close'].iloc[-1]],
        'Total Score': [np.nan],
        'Rating': ['N/A']
    })

    # Adjust stock allocations
    stock_allocation_sum = portfolio['Allocation'].sum()
    portfolio['Allocation'] = portfolio['Allocation'] * (1 - bond_allocation) / stock_allocation_sum

    # Combine stocks and bond ETF
    portfolio = pd.concat([portfolio, bond_etf], ignore_index=True)

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

def main():
    st.sidebar.image("https://www.bing.com/images/search?view=detailV2&ccid=bq9jclpI&id=6E1D986277EA13212A92D2DE8F7A8016C2006AFE&thid=OIP.bq9jclpIyuRRoUdtHP5pZwHaHa&mediaurl=https%3a%2f%2fcdn.pulse2.com%2fcdn%2f2020%2f04%2fblackrock_logo.png&cdnurl=https%3a%2f%2fth.bing.com%2fth%2fid%2fR.6eaf63725a48cae451a1476d1cfe6967%3frik%3d%252fmoAwhaAeo%252fe0g%26pid%3dImgRaw%26r%3d0&exph=1200&expw=1200&q=blackrock&simid=608020842785494443&FORM=IRPRST&ck=CCFBFD8F8A23810E8C7BFB5441144329&selectedIndex=1&itb=0&ajaxhist=0&ajaxserp=0", use_column_width=True)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Stock Screener", "Portfolio Builder",'News/ Sentiment Analysis'])

    if app_mode == "Stock Screener":
        st.title(" Advanced Stock Screener")
        
        col1, col2 = st.columns([2,1])
        with col1:
            st.write("Welcome to the Advanced Stock Screener. This tool allows you to analyze stocks from the S&P 500 index and get AI-powered insights.")
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
                    chatgpt_analysis = get_chatgpt_analysis(stock_data)
                st.write(chatgpt_analysis)

                st.subheader("Ask a Question")
                user_question = st.text_input("Ask a question about this stock:")
                if user_question:
                    with st.spinner("Generating answer..."):
                        answer = get_chatgpt_analysis(stock_data, user_question)
                    st.write("Answer:", answer)

    elif app_mode == "Portfolio Builder":
        st.title(" Portfolio Builder")
        
        st.write("Welcome to the  Portfolio Builder. This tool helps you create a personalized investment portfolio based on your age, goals, and risk tolerance.")

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
                label="Download portfolio as xcel",
                data=csv,
                file_name="personalized_portfolio.xlsx",
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
