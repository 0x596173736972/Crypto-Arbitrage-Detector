import streamlit as st
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import DEFAULT_CONFIG
from components.sidebar import render_sidebar
from components.price_table import render_price_table
from components.opportunity_alerts import render_opportunity_alerts
from components.historical_stats import render_historical_stats
from components.status_indicators import render_status_indicators
from services.arbitrage_service import ArbitrageService

# Page configuration
st.set_page_config(
    page_title="Crypto Arbitrage Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'arbitrage_service' not in st.session_state:
    st.session_state.arbitrage_service = ArbitrageService()
    
if 'config' not in st.session_state:
    st.session_state.config = DEFAULT_CONFIG
    
if 'opportunities' not in st.session_state:
    st.session_state.opportunities = []
    
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 16px;
    }
    .profit-positive {
        color: #00cc96;
        font-weight: bold;
    }
    .profit-negative {
        color: #ef553b;
        font-weight: bold;
    }
    .exchange-connected {
        color: #00cc96;
    }
    .exchange-disconnected {
        color: #ef553b;
    }
    .card {
        background-color: #1e2130;
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    st.title("Crypto Arbitrage Trading System")
    
    # Sidebar with configuration
    render_sidebar()
    
    # Main content
    tabs = st.tabs(["Dashboard", "History", "Settings"])
    
    with tabs[0]:  # Dashboard tab
        col1, col2 = st.columns([7, 3])
        
        with col1:
            st.subheader("Exchange Status")
            render_status_indicators(st.session_state.arbitrage_service)
            
            st.subheader("Current Price Comparison")
            render_price_table(st.session_state.arbitrage_service)
            
        with col2:
            st.subheader("Active Opportunities")
            render_opportunity_alerts(st.session_state.opportunities)
    
    with tabs[1]:  # History tab
        render_historical_stats(st.session_state.opportunities)
    
    with tabs[2]:  # Settings tab
        st.subheader("Advanced Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Slippage Estimate (%)", 0.0, 5.0, st.session_state.config['slippage_estimate'], 0.1, 
                     help="Estimated slippage when executing trades")
            
            st.slider("Minimum Profit Threshold (%)", 0.1, 10.0, st.session_state.config['min_profit_threshold'], 0.1,
                     help="Minimum profit percentage to consider an opportunity valid")
            
        with col2:
            st.slider("Risk Tolerance", 1, 10, st.session_state.config['risk_tolerance'],
                     help="Higher values will show more risky opportunities")
            
            st.slider("Order Book Depth", 1, 100, st.session_state.config['order_book_depth'],
                     help="Depth of order book to consider for volume availability")

        st.subheader("Exchange Fees")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Binance Trading Fee (%)", 0.0, 1.0, st.session_state.config['exchange_fees']['binance'], 0.01)
        with col2:
            st.number_input("Kraken Trading Fee (%)", 0.0, 1.0, st.session_state.config['exchange_fees']['kraken'], 0.01)
            
        st.subheader("Transfer Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Network Transfer Time (minutes)", 1, 120, st.session_state.config['network_transfer_time'], 1)
        with col2:
            st.number_input("Network Transfer Fee", 0.0, 100.0, st.session_state.config['network_transfer_fee'], 0.1)
            
    # Auto-refresh the app every 5 seconds when running
    if st.session_state.is_running:
        st.empty()
        st.experimental_rerun()

if __name__ == "__main__":
    main()