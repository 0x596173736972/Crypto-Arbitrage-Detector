import streamlit as st
from datetime import datetime

def render_opportunity_alerts(opportunities):
    """Render the arbitrage opportunity alerts section"""
    if not opportunities or len(opportunities) == 0:
        st.info("No arbitrage opportunities detected yet.")
        return
    
    # Sort opportunities by profitability (highest first)
    sorted_opportunities = sorted(
        opportunities, 
        key=lambda x: x.get('potential_profit', 0), 
        reverse=True
    )
    
    # Show only the most recent opportunities (last 5)
    recent_opportunities = sorted_opportunities[:5]
    
    # Display each opportunity as a card
    for opp in recent_opportunities:
        with st.container():
            st.markdown(f"""
            <div class="card">
                <h3>{opp.get('symbol', 'Unknown')}</h3>
                <p><strong>Strategy:</strong> {opp.get('strategy', 'Buy/Sell')}</p>
                <p><strong>Entry:</strong> {opp.get('entry_exchange', 'Unknown')} @ ${opp.get('entry_price', 0):.2f}</p>
                <p><strong>Exit:</strong> {opp.get('exit_exchange', 'Unknown')} @ ${opp.get('exit_price', 0):.2f}</p>
                <p><strong>Profit:</strong> <span class="{'profit-positive' if opp.get('potential_profit', 0) > 0 else 'profit-negative'}">{opp.get('potential_profit', 0):.2f}%</span></p>
                <p><strong>Volume:</strong> ${opp.get('required_volume', 0):.2f}</p>
                <p><strong>Risk:</strong> {get_risk_indicator(opp.get('risk_score', 5))}</p>
                <p><small>Detected: {opp.get('timestamp', datetime.now()).strftime('%H:%M:%S')}</small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Show a message if there are more opportunities
    if len(opportunities) > 5:
        st.caption(f"Showing 5 of {len(opportunities)} opportunities. See History tab for all.")

def get_risk_indicator(risk_score):
    """Generate a visual risk indicator based on the risk score"""
    if risk_score <= 3:
        return "🟢 Low"
    elif risk_score <= 7:
        return "🟠 Medium"
    else:
        return "🔴 High"