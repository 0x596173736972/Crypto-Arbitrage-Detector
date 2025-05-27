import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
from modules.data_fetcher import CryptoDataFetcher
from modules.arbitrage_engine import ArbitrageEngine
from modules.database import DatabaseManager
from modules.statistical_analysis import StatisticalAnalyzer
from modules.ml_model import MLPredictor
from modules.backtesting import BacktestEngine
from modules.utils import format_currency, calculate_profit_metrics

# Page configuration
st.set_page_config(
    page_title="Crypto Arbitrage Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = CryptoDataFetcher()
if 'arbitrage_engine' not in st.session_state:
    st.session_state.arbitrage_engine = ArbitrageEngine()
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if 'statistical_analyzer' not in st.session_state:
    st.session_state.statistical_analyzer = StatisticalAnalyzer()
if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = MLPredictor()
if 'backtest_engine' not in st.session_state:
    st.session_state.backtest_engine = BacktestEngine()
if 'running' not in st.session_state:
    st.session_state.running = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Main title
st.title("🚀 Cryptocurrency Arbitrage Dashboard")
st.markdown("Real-time arbitrage opportunity detection with quantitative analysis")

# Sidebar configuration
st.sidebar.header("Configuration")

# Data fetching settings
st.sidebar.subheader("Data Fetching")
fetch_interval = st.sidebar.slider("Fetch Interval (seconds)", 10, 300, 30)

# Get all supported cryptocurrencies
all_cryptos = st.session_state.data_fetcher.get_supported_cryptocurrencies()
selected_cryptos = st.sidebar.multiselect(
    "Select Cryptocurrencies",
    all_cryptos,
    default=["BTC", "ETH", "ADA"]
)

# Arbitrage settings
st.sidebar.subheader("Arbitrage Parameters")
min_spread = st.sidebar.slider("Minimum Spread (%)", 0.01, 5.0, 0.01)
max_risk_score = st.sidebar.slider("Maximum Risk Score", 1, 10, 10)
min_profit = st.sidebar.number_input("Minimum Profit ($)", 0.0, 100.0, 0.1)

# Exchange fees (%)
st.sidebar.subheader("Exchange Fees")
coingecko_fee = st.sidebar.number_input("CoinGecko Fee (%)", 0.0, 1.0, 0.1)
coinpaprika_fee = st.sidebar.number_input("CoinPaprika Fee (%)", 0.0, 1.0, 0.15)
cryptocompare_fee = st.sidebar.number_input("CryptoCompare Fee (%)", 0.0, 1.0, 0.2)
coinlore_fee = st.sidebar.number_input("CoinLore Fee (%)", 0.0, 1.0, 0.1)
nomics_fee = st.sidebar.number_input("Nomics Fee (%)", 0.0, 1.0, 0.25)

fees = {
    'coingecko': coingecko_fee,
    'coinpaprika': coinpaprika_fee,
    'cryptocompare': cryptocompare_fee,
    'coinlore': coinlore_fee,
    'nomics': nomics_fee
}

# Control buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Start", disabled=st.session_state.running):
        st.session_state.running = True
        st.rerun()

with col2:
    if st.button("Stop", disabled=not st.session_state.running):
        st.session_state.running = False
        st.rerun()

# Main content tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Real-time Opportunities",
    "📈 Backtesting",
    "💾 Historical Data"
])

with tab1:
    st.header("Real-time Arbitrage Opportunities")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "🟢 Running" if st.session_state.running else "🔴 Stopped"
        st.metric("Status", status)
    with col2:
        if st.session_state.last_update:
            st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
    with col3:
        st.metric("Active Pairs", len(selected_cryptos))

    # Fetch and display real-time data
    if st.session_state.running or st.button("Fetch Current Prices"):
        with st.spinner("Fetching real-time data..."):
            try:
                # Fetch current prices
                price_data = st.session_state.data_fetcher.fetch_all_prices(selected_cryptos)
                
                if price_data:
                    # Display current prices first
                    st.subheader("📊 Current Prices")
                    
                    # Create a nice price display
                    price_display = []
                    for crypto, exchanges in price_data.items():
                        for exchange, price in exchanges.items():
                            price_display.append({
                                'Cryptocurrency': crypto,
                                'Exchange': exchange,
                                'Price (USD)': f"${price:.4f}"
                            })
                    
                    if price_display:
                        df_prices = pd.DataFrame(price_display)
                        st.dataframe(df_prices, use_container_width=True)
                        
                        # Show price differences
                        st.subheader("💰 Price Analysis")
                        for crypto, exchanges in price_data.items():
                            if len(exchanges) >= 2:
                                prices = list(exchanges.values())
                                min_price = min(prices)
                                max_price = max(prices)
                                spread = ((max_price - min_price) / min_price) * 100
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(f"{crypto} Min", f"${min_price:.4f}")
                                with col2:
                                    st.metric(f"{crypto} Max", f"${max_price:.4f}")
                                with col3:
                                    st.metric(f"{crypto} Spread", f"{spread:.2f}%")
                                with col4:
                                    if spread >= min_spread:
                                        st.success("✅ Opportunity!")
                                    else:
                                        st.info(f"Need {min_spread:.1f}%+")
                    
                    # Find arbitrage opportunities
                    opportunities = st.session_state.arbitrage_engine.find_opportunities(
                        price_data, fees, min_spread
                    )
                    
                    # Store in database
                    st.session_state.db_manager.store_prices(price_data)
                    
                    # Store opportunities for statistical analysis and ML
                    for opportunity in opportunities:
                        st.session_state.db_manager.store_opportunity(opportunity)
                    
                    # Display opportunities
                    if opportunities:
                        df_opportunities = pd.DataFrame(opportunities)
                        
                        # Filter by risk score and minimum profit
                        df_filtered = df_opportunities[
                            (df_opportunities['risk_score'] <= max_risk_score) & 
                            (df_opportunities['net_profit'] >= min_profit)
                        ]
                        
                        if not df_filtered.empty:
                            st.subheader(f"🎯 Found {len(df_filtered)} Arbitrage Opportunities")
                            
                            # Add action recommendations
                            def get_action_recommendation(row):
                                if row['net_profit'] > 50:
                                    return "🟢 EXECUTE - High profit potential"
                                elif row['net_profit'] > 10:
                                    return "🟡 CONSIDER - Moderate profit"
                                elif row['spread_pct'] > 2.0:
                                    return "🔴 INVESTIGATE - High spread may indicate data error"
                                else:
                                    return "🟡 MONITOR - Small profit margin"
                            
                            df_filtered['Action'] = df_filtered.apply(get_action_recommendation, axis=1)
                            
                            # Reorder columns for better display
                            display_columns = ['crypto', 'buy_exchange', 'sell_exchange', 'buy_price', 'sell_price', 
                                             'spread_pct', 'net_profit', 'risk_score', 'Action']
                            df_display = df_filtered[display_columns].copy()
                            
                            # Style the dataframe
                            styled_df = df_display.style.format({
                                'buy_price': '${:.4f}',
                                'sell_price': '${:.4f}',
                                'spread_pct': '{:.2f}%',
                                'net_profit': '${:.2f}',
                                'risk_score': '{:.0f}'
                            })
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Show detailed analysis for each opportunity
                            st.subheader("📋 Detailed Action Plan")
                            
                            for idx, row in df_filtered.iterrows():
                                with st.expander(f"{row['crypto']} - {row['buy_exchange']} → {row['sell_exchange']} (${row['net_profit']:.2f} profit)"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Trade Details:**")
                                        st.write(f"• Buy at {row['buy_exchange']}: ${row['buy_price']:.4f}")
                                        st.write(f"• Sell at {row['sell_exchange']}: ${row['sell_price']:.4f}")
                                        st.write(f"• Spread: {row['spread_pct']:.2f}%")
                                        st.write(f"• Gross Profit: ${row['gross_profit']:.2f}")
                                        st.write(f"• Net Profit: ${row['net_profit']:.2f}")
                                    
                                    with col2:
                                        st.write("**Risk Assessment:**")
                                        st.write(f"• Risk Score: {row['risk_score']}/10")
                                        st.write(f"• Opportunity Score: {row['opportunity_score']:.1f}/10")
                                        
                                        # Action recommendation
                                        action = get_action_recommendation(row)
                                        st.write(f"**Recommendation:** {action}")
                                        
                                        # Additional warnings
                                        if row['spread_pct'] > 5.0:
                                            st.warning("⚠️ Very high spread - verify data quality")
                                        if row['risk_score'] > 7:
                                            st.warning("⚠️ High risk - proceed with caution")
                            
                            # Export button
                            csv = df_filtered.to_csv(index=False)
                            st.download_button(
                                label="📥 Export All Opportunities",
                                data=csv,
                                file_name=f"arbitrage_opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime='text/csv'
                            )
                        else:
                            st.warning(f"No opportunities found with profit ≥ ${min_profit:.2f} and risk ≤ {max_risk_score}")
                    else:
                        st.info("No arbitrage opportunities detected - try lowering the minimum spread or profit threshold")
                    
                    st.session_state.last_update = datetime.now()
                else:
                    st.error("Failed to fetch price data")
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

    # Real-time price display
    if st.session_state.running:
        placeholder = st.empty()
        
        # Auto-refresh mechanism
        if st.session_state.running:
            time.sleep(fetch_interval)
            st.rerun()

with tab2:
    st.header("📈 Backtesting Engine")
    
    # Backtesting parameters
    st.subheader("Backtest Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        backtest_days = st.number_input("Backtest Period (days)", 1, 90, 30)
        initial_capital = st.number_input("Initial Capital ($)", 1000, 100000, 10000)
    
    with col2:
        slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.1)
        position_size = st.number_input("Position Size (%)", 1, 100, 10)
    
    with col3:
        latency_ms = st.number_input("Latency (ms)", 0, 1000, 100)
    
    # Run backtest
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                # Get historical data for backtesting
                end_date = datetime.now()
                start_date = end_date - timedelta(days=backtest_days)
                
                historical_data = st.session_state.db_manager.get_historical_data(
                    start_date, end_date
                )
                
                if not historical_data.empty:
                    # Run backtest
                    results = st.session_state.backtest_engine.run_backtest(
                        historical_data=historical_data,
                        initial_capital=initial_capital,
                        slippage=slippage / 100,
                        position_size=position_size / 100,
                        fees=fees,
                        latency_ms=latency_ms
                    )
                    
                    # Display results
                    st.subheader("Backtest Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{results['total_return']:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                    with col4:
                        st.metric("Win Rate", f"{results['win_rate']:.2%}")
                    
                    # Plot equity curve
                    if 'equity_curve' in results:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=results['equity_curve'].index,
                            y=results['equity_curve'],
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='green')
                        ))
                        fig.update_layout(
                            title="Portfolio Equity Curve",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade summary
                    if 'trades' in results:
                        st.subheader("Trade Summary")
                        trades_df = pd.DataFrame(results['trades'])
                        st.dataframe(trades_df, use_container_width=True)
                        
                        # Export trades
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Export Trade History",
                            data=csv,
                            file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                else:
                    st.warning("Insufficient historical data for backtesting")
                    
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")

with tab3:
    st.header("💾 Historical Data")
    
    # Database statistics
    stats = st.session_state.db_manager.get_database_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", stats.get('total_records', 0))
    with col2:
        st.metric("Unique Cryptos", stats.get('unique_cryptos', 0))
    with col3:
        st.metric("Date Range", f"{stats.get('date_range', 'N/A')} days")
    
    # Data visualization
    st.subheader("Historical Price Data")
    
    if stats.get('total_records', 0) > 0:
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Get historical data
        historical_data = st.session_state.db_manager.get_historical_data(
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.min.time())
        )
        
        if not historical_data.empty:
            # Plot price trends
            crypto_to_plot = st.selectbox("Select crypto to plot", 
                                        historical_data['crypto'].unique())
            
            crypto_data = historical_data[historical_data['crypto'] == crypto_to_plot]
            
            fig = go.Figure()
            
            for exchange in crypto_data['exchange'].unique():
                exchange_data = crypto_data[crypto_data['exchange'] == exchange]
                fig.add_trace(go.Scatter(
                    x=exchange_data['timestamp'],
                    y=exchange_data['price'],
                    mode='lines',
                    name=exchange.title(),
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=f"{crypto_to_plot} Price Comparison",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data export
            st.subheader("Export Historical Data")
            csv = historical_data.to_csv(index=False)
            st.download_button(
                label="📥 Export All Historical Data",
                data=csv,
                file_name=f"historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
    else:
        st.info("No historical data available. Start the data fetching process to collect data.")
    
    # Database management
    st.subheader("Database Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Old Data (>30 days)"):
            deleted_count = st.session_state.db_manager.cleanup_old_data(days=30)
            st.success(f"Deleted {deleted_count} old records")
    
    with col2:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                st.session_state.db_manager.clear_all_data()
                st.success("All data cleared")

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tip**: This application performs real-time web scraping. "
    "Ensure stable internet connection for optimal performance."
)
