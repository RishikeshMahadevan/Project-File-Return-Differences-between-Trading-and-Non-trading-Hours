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

def calculate_returns(df):
    """Calculate returns for different time periods"""
    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter for trading hours (9:30 AM to 4:00 PM)
    df['time'] = df['timestamp'].dt.time
    df = df[
        (df['time'] >= pd.to_datetime('09:30').time()) &
        (df['time'] <= pd.to_datetime('16:00').time())
    ]
    
    df.set_index('timestamp', inplace=True)
    
    # Create daily groups
    daily_groups = df.groupby(df.index.date)
    
    # Initialize lists to store returns
    night_returns = []
    am_returns = []
    mid_returns = []
    pm_returns = []
    dates = []
    
    prev_close = None
    
    for date, group in daily_groups:
        group = group.sort_index()
        
        # Only process if we have data for the full trading day
        first_time = group.index[0].time()
        last_time = group.index[-1].time()
        
        if first_time <= pd.to_datetime('09:31').time() and last_time >= pd.to_datetime('15:59').time():
            # Night return (previous close to today's open)
            if prev_close is not None:
                night_return = (group['o'].iloc[0] / prev_close) - 1
                
                # AM return (9:30 to 10:30)
                am_data = group.between_time('09:30', '10:30')
                if len(am_data) > 0:
                    am_return = (am_data['c'].iloc[-1] / group['o'].iloc[0]) - 1
                    
                    # Mid return (10:30 to 15:00)
                    mid_data = group.between_time('10:30', '15:00')
                    if len(mid_data) > 0:
                        mid_return = (mid_data['c'].iloc[-1] / am_data['c'].iloc[-1]) - 1
                        
                        # PM return (15:00 to close)
                        pm_data = group.between_time('15:00', '16:00')
                        if len(pm_data) > 0:
                            pm_return = (group['c'].iloc[-1] / mid_data['c'].iloc[-1]) - 1
                            
                            # Append all returns only if we have complete data
                            night_returns.append(night_return)
                            am_returns.append(am_return)
                            mid_returns.append(mid_return)
                            pm_returns.append(pm_return)
                            dates.append(date)
            
            prev_close = group['c'].iloc[-1]
    
    # Create returns DataFrame
    returns_df = pd.DataFrame({
        'Night_Return': night_returns,
        'AM_Return': am_returns,
        'Mid_Return': mid_returns,
        'PM_Return': pm_returns
    }, index=dates)
    
    return returns_df

def calculate_period_strategy_returns(aapl_returns, amzn_returns, aapl_weight=0.5, amzn_weight=0.5, initial_capital=100000):
    """
    Calculate strategy returns using vectorized operations with custom weights
    """
    capital_aapl = initial_capital * aapl_weight
    capital_amzn = initial_capital * amzn_weight
    periods = ['Night_Return', 'AM_Return', 'Mid_Return', 'PM_Return']
    
    # Initialize portfolio DataFrame
    portfolio = pd.DataFrame(index=aapl_returns.index)
    
    # Vectorized calculations for each period
    for period in periods:
        portfolio[f'AAPL_{period}_Position'] = capital_aapl * (1 + aapl_returns[period]).cumprod()
        portfolio[f'AMZN_{period}_Position'] = capital_amzn * (1 + amzn_returns[period]).cumprod()
        portfolio[f'{period}_Value'] = portfolio[f'AAPL_{period}_Position'] + portfolio[f'AMZN_{period}_Position']
        portfolio[f'{period}_Return'] = portfolio[f'{period}_Value'].pct_change()
    
    # Buy & Hold strategy
    aapl_daily_return = (1 + aapl_returns).prod(axis=1) - 1
    amzn_daily_return = (1 + amzn_returns).prod(axis=1) - 1
    
    portfolio['AAPL_BH_Position'] = capital_aapl * (1 + aapl_daily_return).cumprod()
    portfolio['AMZN_BH_Position'] = capital_amzn * (1 + amzn_daily_return).cumprod()
    portfolio['Buy_Hold_Value'] = portfolio['AAPL_BH_Position'] + portfolio['AMZN_BH_Position']
    portfolio['Buy_Hold_Return'] = portfolio['Buy_Hold_Value'].pct_change()
    
    return portfolio

def calculate_metrics(portfolio):
    """
    Calculate performance metrics
    """
    strategies = ['Night_Return', 'AM_Return', 'Mid_Return', 'PM_Return', 'Buy_Hold']
    
    metrics = pd.DataFrame(index=['Total_Return(%)', 'Annualized_Return(%)', 'Sharpe_Ratio', 
                                'Max_Drawdown(%)', 'Win_Rate(%)', 'Profit_Factor', 
                                'Number_of_Trades', 'Daily_Std(%)'])
    
    rf_daily = 0.02/252
    
    for strategy in strategies:
        returns = portfolio[f'{strategy}_Return'].dropna()
        values = portfolio[f'{strategy}_Value'].dropna()
        
        total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
        excess_returns = returns - rf_daily
        sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        drawdown = (values / values.expanding().max() - 1).min() * 100
        win_rate = (returns > 0).mean() * 100
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns != 0 else np.inf
        
        metrics[strategy] = [
            total_return,
            ((1 + returns.mean()) ** 252 - 1) * 100,
            sharpe,
            drawdown,
            win_rate,
            profit_factor,
            len(returns),
            returns.std() * 100
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
    st.session_state.metrics = calculate_metrics(st.session_state.portfolio)

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