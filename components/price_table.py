import streamlit as st
import pandas as pd
from datetime import datetime

def render_price_table(arbitrage_service):
    """Render the price comparison table with current exchange rates"""
    # Get price data from the arbitrage service
    price_data = arbitrage_service.get_current_prices()
    
    if not price_data or len(price_data) == 0:
        # If no data, show placeholder
        st.info("Waiting for price data... Start monitoring to fetch prices.")
        return
    
    # Convert to DataFrame for display
    df = pd.DataFrame(price_data)
    
    # Format the dataframe
    if not df.empty:
        # Ensure all required columns exist
        required_cols = ['symbol', 'binance_price', 'kraken_price', 'difference_pct', 'potential_profit']
        for col in required_cols:
            if col not in df.columns:
                if col == 'symbol':
                    df['symbol'] = 'N/A'
                else:
                    df[col] = 0.0
        
        # Calculate timestamp age
        now = datetime.now()
        if 'timestamp' in df.columns:
            df['age'] = df['timestamp'].apply(lambda x: (now - x).seconds)
        else:
            df['age'] = 0
        
        # Format the display table
        display_df = df[['symbol', 'binance_price', 'kraken_price', 'difference_pct', 'potential_profit']].copy()
        
        # Format columns
        display_df['binance_price'] = display_df['binance_price'].apply(lambda x: f"${x:.2f}" if x > 1 else f"${x:.6f}")
        display_df['kraken_price'] = display_df['kraken_price'].apply(lambda x: f"${x:.2f}" if x > 1 else f"${x:.6f}")
        display_df['difference_pct'] = display_df['difference_pct'].apply(lambda x: f"{x:.2f}%")
        
        # Format profit column with colors
        def format_profit(profit):
            if profit > 0:
                return f"<span class='profit-positive'>+{profit:.2f}%</span>"
            elif profit < 0:
                return f"<span class='profit-negative'>{profit:.2f}%</span>"
            else:
                return f"{profit:.2f}%"
        
        display_df['potential_profit'] = display_df['potential_profit'].apply(format_profit)
        
        # Rename columns for display
        display_df.columns = ['Trading Pair', 'Binance Price', 'Kraken Price', 'Difference', 'Potential Profit']
        
        # Display the table with custom formatting
        st.markdown(
            display_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
        
        # Show last update time
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.info("No price data available. Start monitoring to fetch prices.")