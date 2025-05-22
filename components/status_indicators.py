import streamlit as st
from datetime import datetime, timedelta

def render_status_indicators(arbitrage_service):
    """Render status indicators for exchange connections and system health"""
    col1, col2, col3 = st.columns(3)
    
    # Get connection status from the arbitrage service
    status = arbitrage_service.get_connection_status()
    
    with col1:
        # Binance connection status
        binance_status = status.get('binance', {})
        is_connected = binance_status.get('connected', False)
        last_update = binance_status.get('last_update', datetime.now() - timedelta(minutes=10))
        latency = binance_status.get('latency', 0)
        
        st.markdown(f"""
        <div style="border-left: 4px solid {'#00cc96' if is_connected else '#ef553b'}; padding-left: 10px;">
            <h4>Binance</h4>
            <p>Status: <span class="{'exchange-connected' if is_connected else 'exchange-disconnected'}">
                {'Connected' if is_connected else 'Disconnected'}
            </span></p>
            <p>Last Update: {last_update.strftime('%H:%M:%S')}</p>
            <p>Latency: {latency}ms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Kraken connection status
        kraken_status = status.get('kraken', {})
        is_connected = kraken_status.get('connected', False)
        last_update = kraken_status.get('last_update', datetime.now() - timedelta(minutes=10))
        latency = kraken_status.get('latency', 0)
        
        st.markdown(f"""
        <div style="border-left: 4px solid {'#00cc96' if is_connected else '#ef553b'}; padding-left: 10px;">
            <h4>Kraken</h4>
            <p>Status: <span class="{'exchange-connected' if is_connected else 'exchange-disconnected'}">
                {'Connected' if is_connected else 'Disconnected'}
            </span></p>
            <p>Last Update: {last_update.strftime('%H:%M:%S')}</p>
            <p>Latency: {latency}ms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # System status
        system_status = status.get('system', {})
        is_running = system_status.get('running', st.session_state.is_running)
        uptime = system_status.get('uptime', 0)
        memory_usage = system_status.get('memory_usage', 0)
        
        st.markdown(f"""
        <div style="border-left: 4px solid {'#00cc96' if is_running else '#ef553b'}; padding-left: 10px;">
            <h4>System</h4>
            <p>Status: <span class="{'exchange-connected' if is_running else 'exchange-disconnected'}">
                {'Running' if is_running else 'Stopped'}
            </span></p>
            <p>Uptime: {format_uptime(uptime)}</p>
            <p>Memory: {memory_usage}MB</p>
        </div>
        """, unsafe_allow_html=True)

def format_uptime(seconds):
    """Format uptime in seconds to a readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"