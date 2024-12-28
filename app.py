import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(page_title="Trading Hours Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    aapl = pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Project-File-Return-Differences-between-Trading-and-Non-trading-Hours/main/Datasets/AAPL_data%20(1).csv')
    amzn = pd.read_csv('https://raw.githubusercontent.com/RishikeshMahadevan/Project-File-Return-Differences-between-Trading-and-Non-trading-Hours/main/Datasets/AMZN_data.csv')
    
    # Convert timestamp to datetime
    aapl['timestamp'] = pd.to_datetime(aapl['timestamp'])
    amzn['timestamp'] = pd.to_datetime(amzn['timestamp'])
    
    return aapl, amzn

def prepare_trading_data(df):
    """Prepare data for trading analysis"""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create time column
    df['time'] = df['timestamp'].dt.time
    df['date'] = df['timestamp'].dt.date
    
    # Filter for regular trading hours (9:30 AM to 4:00 PM)
    market_open = pd.to_datetime('09:30:00').time()
    market_close = pd.to_datetime('16:00:00').time()
    
    trading_hours_df = df[
        (df['time'] >= market_open) & 
        (df['time'] <= market_close)
    ].copy()
    
    return trading_hours_df

def get_previous_trading_close(df, current_date):
    """Find the previous trading day's closing price"""
    current_date = pd.to_datetime(current_date)
    
    # Look back up to 5 days to find the previous trading day
    for i in range(1, 5):
        prev_date = current_date - pd.Timedelta(days=i)
        prev_day_data = df[df['date'] == prev_date.date()]
        
        if not prev_day_data.empty:
            return prev_day_data['c'].iloc[-1]  # Return the closing price
    
    return None

def calculate_returns(df):
    """Calculate returns for different time periods"""
    df = prepare_trading_data(df)
    returns_dict = {}
    
    # Sort dataframe by timestamp to ensure chronological order
    df = df.sort_values('timestamp')
    
    for date in sorted(df['date'].unique()):
        day_data = df[df['date'] == date]
        
        # Night Return (Previous trading day close to 9:30 open)
        try:
            prev_close = get_previous_trading_close(df, date)
            if prev_close is not None:
                today_open = day_data['o'].iloc[0]   # Today's open at 9:30 AM
                night_return = (today_open - prev_close) / prev_close
            else:
                night_return = np.nan
        except:
            night_return = np.nan
    
        # AM Return (9:30 to 10:30)
        try:
            open_price = day_data['o'].iloc[0]
            am_cutoff = pd.to_datetime('10:30:00').time()
            am_data = day_data[day_data['time'] <= am_cutoff]
            if not am_data.empty:
                am_close = am_data['c'].iloc[-1]
                am_return = (am_close - open_price) / open_price
            else:
                am_return = np.nan
        except:
            am_return = np.nan
            
        # Mid Return (10:30 to 15:00)
        try:
            mid_start = pd.to_datetime('10:30:00').time()
            mid_end = pd.to_datetime('15:00:00').time()
            mid_start_price = day_data[day_data['time'] <= mid_start]['c'].iloc[-1]
            mid_end_price = day_data[day_data['time'] <= mid_end]['c'].iloc[-1]
            mid_return = (mid_end_price - mid_start_price) / mid_start_price
        except:
            mid_return = np.nan
            
        # PM Return (15:00 to 16:00)
        try:
            pm_start = pd.to_datetime('15:00:00').time()
            pm_start_price = day_data[day_data['time'] <= pm_start]['c'].iloc[-1]
            pm_close = day_data['c'].iloc[-1]
            pm_return = (pm_close - pm_start_price) / pm_start_price
        except:
            pm_return = np.nan
            
        returns_dict[date] = {
            'Night_Return': night_return,
            'AM_Return': am_return,
            'Mid_Return': mid_return,
            'PM_Return': pm_return
        }
    
    # Convert to DataFrame
    returns_df = pd.DataFrame.from_dict(returns_dict, orient='index')
    
    return returns_df

def calculate_period_strategy_returns(aapl_returns, amzn_returns, aapl_weight=0.5, amzn_weight=0.5, initial_capital=100000):
    """
    Calculate strategy returns using the same methodology as the notebook
    """
    capital_aapl = initial_capital * aapl_weight
    capital_amzn = initial_capital * amzn_weight
    periods = ['Night_Return', 'AM_Return', 'Mid_Return', 'PM_Return']
    
    # Initialize portfolio DataFrame
    portfolio = pd.DataFrame(index=aapl_returns.index)
    
    # Vectorized calculations for each period
    for period in periods:
        # Calculate position values using cumulative returns
        portfolio[f'AAPL_{period}_Position'] = capital_aapl * (1 + aapl_returns[period]).cumprod()
        portfolio[f'AMZN_{period}_Position'] = capital_amzn * (1 + amzn_returns[period]).cumprod()
        
        # Total portfolio value for this period
        portfolio[f'{period}_Value'] = portfolio[f'AAPL_{period}_Position'] + portfolio[f'AMZN_{period}_Position']
        portfolio[f'{period}_Return'] = portfolio[f'{period}_Value'].pct_change()
    
    # Buy & Hold strategy (using cumulative returns of daily total returns)
    aapl_daily_return = (1 + aapl_returns).prod(axis=1) - 1
    amzn_daily_return = (1 + amzn_returns).prod(axis=1) - 1
    
    portfolio['AAPL_BH_Position'] = capital_aapl * (1 + aapl_daily_return).cumprod()
    portfolio['AMZN_BH_Position'] = capital_amzn * (1 + amzn_daily_return).cumprod()
    portfolio['Buy_Hold_Value'] = portfolio['AAPL_BH_Position'] + portfolio['AMZN_BH_Position']
    portfolio['Buy_Hold_Return'] = portfolio['Buy_Hold_Value'].pct_change()
    
    return portfolio

def calculate_metrics(portfolio, rf_rate=2.0):
    """
    Calculate performance metrics using the same methodology as the notebook
    """
    strategies = ['Night_Return', 'AM_Return', 'Mid_Return', 'PM_Return', 'Buy_Hold']
    
    metrics = pd.DataFrame(index=['Total_Return(%)', 'Annualized_Return(%)', 'Sharpe_Ratio', 
                                'Max_Drawdown(%)', 'Win_Rate(%)', 'Profit_Factor', 
                                'Number_of_Trades', 'Daily_Std(%)'])
    
    rf_daily = rf_rate/100/252  # Convert annual percentage to daily rate
    
    for strategy in strategies:
        returns = portfolio[f'{strategy}_Return'].dropna()
        values = portfolio[f'{strategy}_Value'].dropna()
        
        # Total return
        total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
        
        # Annualized return
        ann_return = ((1 + returns.mean()) ** 252 - 1) * 100
        
        # Sharpe ratio
        excess_returns = returns - rf_daily
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        # Maximum drawdown
        drawdown = (values / values.expanding().max() - 1).min() * 100
        
        # Win rate
        win_rate = (returns > 0).mean() * 100
        
        # Profit factor
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns != 0 else np.inf
        
        # Daily standard deviation
        daily_std = returns.std() * 100
        
        metrics[strategy] = [
            total_return,
            ann_return,
            sharpe,
            drawdown,
            win_rate,
            profit_factor,
            len(returns),
            daily_std
        ]
    
    return metrics.round(2)

# Load data
aapl, amzn = load_data()

# Sidebar inputs
st.sidebar.header("Parameters")

# Date range selector
min_date = aapl['timestamp'].min().date()
max_date = aapl['timestamp'].max().date()

start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Weight sliders
st.sidebar.subheader("Portfolio Weights")
aapl_weight = st.sidebar.slider("AAPL Weight", 0.0, 1.0, 0.5, 0.1)
amzn_weight = 1 - aapl_weight
st.sidebar.write(f"AMZN Weight: {amzn_weight:.1f}")

# In the sidebar inputs section, add this after the portfolio weights
st.sidebar.subheader("Risk Parameters")
rf_rate = st.sidebar.number_input(
    "Risk-free Rate (%)", 
    min_value=0.0, 
    max_value=20.0, 
    value=2.0, 
    step=0.1,
    help="Annual risk-free rate used for Sharpe ratio calculation"
)

# Main content
st.title("Trading Hours Analysis")

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

if st.sidebar.button("Analyze"):
    # Filter data by date range
    mask = (aapl['timestamp'].dt.date >= start_date) & (aapl['timestamp'].dt.date <= end_date)
    aapl_filtered = aapl[mask].copy()
    amzn_filtered = amzn[mask].copy()
    
    # Calculate returns for both stocks
    aapl_returns = calculate_returns(aapl_filtered)
    amzn_returns = calculate_returns(amzn_filtered)
    
    # Calculate portfolio returns and metrics
    st.session_state.portfolio = calculate_period_strategy_returns(aapl_returns, amzn_returns, aapl_weight, amzn_weight)
    st.session_state.metrics = calculate_metrics(st.session_state.portfolio, rf_rate)

if st.session_state.portfolio is not None:
    # Create PnL plot
    st.subheader("Portfolio Value Over Time")
    
    fig_pnl = go.Figure()
    strategies = ['Night_Return', 'AM_Return', 'Mid_Return', 'PM_Return', 'Buy_Hold']
    
    # Add traces for each strategy
    col1, col2, col3, col4, col5 = st.columns(5)
    visibility = {}
    
    with col1:
        visibility['Night_Return'] = st.checkbox("Show Night Return", value=True, key="pnl_Night_Return")
    with col2:
        visibility['AM_Return'] = st.checkbox("Show AM Return", value=True, key="pnl_AM_Return")
    with col3:
        visibility['Mid_Return'] = st.checkbox("Show Mid Return", value=True, key="pnl_Mid_Return")
    with col4:
        visibility['PM_Return'] = st.checkbox("Show PM Return", value=True, key="pnl_PM_Return")
    with col5:
        visibility['Buy_Hold'] = st.checkbox("Show Buy & Hold", value=True, key="pnl_Buy_Hold")
    
    for strategy in strategies:
        fig_pnl.add_trace(
            go.Scatter(
                x=st.session_state.portfolio.index,
                y=st.session_state.portfolio[f'{strategy}_Value'],
                name=strategy.replace('_', ' '),
                visible=visibility[strategy]
            )
        )
    
    fig_pnl.update_layout(
        title="Portfolio Value Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=600
    )
    
    st.plotly_chart(fig_pnl, use_container_width=True)
    
    # Create Drawdown plot
    st.subheader("Drawdown Analysis")
    
    fig_dd = go.Figure()
    
    # Calculate and plot drawdowns for each strategy
    col1, col2, col3, col4, col5 = st.columns(5)
    dd_visibility = {}
    
    with col1:
        dd_visibility['Night_Return'] = st.checkbox("Show Night Return DD", value=True, key="dd_Night_Return")
    with col2:
        dd_visibility['AM_Return'] = st.checkbox("Show AM Return DD", value=True, key="dd_AM_Return")
    with col3:
        dd_visibility['Mid_Return'] = st.checkbox("Show Mid Return DD", value=True, key="dd_Mid_Return")
    with col4:
        dd_visibility['PM_Return'] = st.checkbox("Show PM Return DD", value=True, key="dd_PM_Return")
    with col5:
        dd_visibility['Buy_Hold'] = st.checkbox("Show Buy & Hold DD", value=True, key="dd_Buy_Hold")
    
    for strategy in strategies:
        values = st.session_state.portfolio[f'{strategy}_Value']
        drawdown = (values / values.expanding().max() - 1) * 100
        
        fig_dd.add_trace(
            go.Scatter(
                x=st.session_state.portfolio.index,
                y=drawdown,
                name=f"{strategy.replace('_', ' ')} Drawdown",
                visible=dd_visibility[strategy]
            )
        )
    
    fig_dd.update_layout(
        title="Strategy Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=400
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # Display metrics table
    st.subheader("Performance Metrics")
    st.dataframe(st.session_state.metrics, use_container_width=True)

else:
    st.info("Please click 'Analyze' to view the results.") 