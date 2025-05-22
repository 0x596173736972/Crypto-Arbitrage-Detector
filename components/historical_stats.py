import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def render_historical_stats(opportunities):
    """Render historical statistics and charts for arbitrage opportunities"""
    if not opportunities or len(opportunities) == 0:
        st.info("No historical data available yet.")
        return
    
    # Convert opportunities to DataFrame
    df = pd.DataFrame(opportunities)
    
    # Ensure required columns exist
    required_cols = ['symbol', 'potential_profit', 'timestamp', 'entry_exchange', 'exit_exchange']
    for col in required_cols:
        if col not in df.columns:
            if col == 'timestamp':
                df['timestamp'] = datetime.now()
            elif col == 'potential_profit':
                df['potential_profit'] = 0.0
            elif col == 'symbol':
                df['symbol'] = 'Unknown'
            else:
                df[col] = 'Unknown'
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Opportunities", 
            len(df),
            delta=None
        )
    
    with col2:
        avg_profit = df['potential_profit'].mean() if 'potential_profit' in df.columns else 0
        st.metric(
            "Avg. Profit Potential", 
            f"{avg_profit:.2f}%",
            delta=None
        )
    
    with col3:
        max_profit = df['potential_profit'].max() if 'potential_profit' in df.columns else 0
        st.metric(
            "Max Profit Potential", 
            f"{max_profit:.2f}%",
            delta=None
        )
    
    with col4:
        # Count opportunities in the last hour
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        recent_count = len(df[df['timestamp'] > one_hour_ago]) if 'timestamp' in df.columns else 0
        
        st.metric(
            "Last Hour", 
            recent_count,
            delta=None
        )
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Opportunities Timeline", "By Trading Pair", "By Exchange"])
    
    with tabs[0]:
        # Timeline of opportunities
        if 'timestamp' in df.columns and 'potential_profit' in df.columns:
            fig = px.scatter(
                df,
                x='timestamp',
                y='potential_profit',
                color='symbol',
                size='potential_profit',
                hover_data=['entry_exchange', 'exit_exchange'],
                title='Arbitrage Opportunities Timeline',
                labels={
                    'timestamp': 'Time',
                    'potential_profit': 'Profit Potential (%)',
                    'symbol': 'Trading Pair'
                }
            )
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Opportunities by trading pair
        if 'symbol' in df.columns and 'potential_profit' in df.columns:
            pair_summary = df.groupby('symbol')['potential_profit'].agg(['mean', 'max', 'count']).reset_index()
            pair_summary.columns = ['Trading Pair', 'Avg. Profit (%)', 'Max Profit (%)', 'Count']
            
            fig = px.bar(
                pair_summary,
                x='Trading Pair',
                y='Count',
                color='Avg. Profit (%)',
                text='Count',
                title='Opportunities by Trading Pair',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table with pair statistics
            st.dataframe(pair_summary)
    
    with tabs[2]:
        # Opportunities by exchange pair
        if 'entry_exchange' in df.columns and 'exit_exchange' in df.columns:
            df['exchange_pair'] = df['entry_exchange'] + ' → ' + df['exit_exchange']
            exchange_summary = df.groupby('exchange_pair')['potential_profit'].agg(['mean', 'max', 'count']).reset_index()
            exchange_summary.columns = ['Exchange Route', 'Avg. Profit (%)', 'Max Profit (%)', 'Count']
            
            fig = px.bar(
                exchange_summary,
                x='Exchange Route',
                y='Count',
                color='Avg. Profit (%)',
                text='Count',
                title='Opportunities by Exchange Route',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show table with exchange statistics
            st.dataframe(exchange_summary)