import streamlit as st

def render_sidebar():
    """Render the sidebar with controls and configuration options"""
    with st.sidebar:
        st.header("Arbitrage Controls")
        
        # Start/Stop Button
        if st.session_state.is_running:
            if st.button("⏹️ Stop Monitoring", use_container_width=True):
                st.session_state.is_running = False
                st.session_state.arbitrage_service.stop()
                st.experimental_rerun()
        else:
            if st.button("▶️ Start Monitoring", use_container_width=True):
                st.session_state.is_running = True
                st.session_state.arbitrage_service.start()
                st.experimental_rerun()
        
        # Trading Pair Selection
        st.subheader("Trading Pairs")
        all_pairs = st.session_state.config['trading_pairs']
        selected_pairs = st.multiselect(
            "Select trading pairs to monitor",
            options=all_pairs,
            default=all_pairs[:3]  # Default to first 3 pairs
        )
        
        if st.button("Update Trading Pairs", use_container_width=True):
            if selected_pairs:
                st.session_state.config['trading_pairs'] = selected_pairs
                st.session_state.arbitrage_service.update_trading_pairs(selected_pairs)
                st.success(f"Updated trading pairs: {', '.join(selected_pairs)}")
            else:
                st.error("Please select at least one trading pair")
        
        # Exchange Selection
        st.subheader("Exchanges")
        binance_enabled = st.checkbox("Binance", value=st.session_state.config['exchanges']['binance']['enabled'])
        kraken_enabled = st.checkbox("Kraken", value=st.session_state.config['exchanges']['kraken']['enabled'])
        
        if st.button("Update Exchanges", use_container_width=True):
            if binance_enabled or kraken_enabled:
                st.session_state.config['exchanges']['binance']['enabled'] = binance_enabled
                st.session_state.config['exchanges']['kraken']['enabled'] = kraken_enabled
                st.session_state.arbitrage_service.update_exchanges(
                    binance=binance_enabled,
                    kraken=kraken_enabled
                )
                st.success(f"Updated exchanges: {'Binance ' if binance_enabled else ''}{'Kraken' if kraken_enabled else ''}")
            else:
                st.error("Please enable at least one exchange")
        
        # Profit Threshold Slider
        st.subheader("Profit Settings")
        profit_threshold = st.slider(
            "Min. Profit Threshold (%)", 
            min_value=0.1, 
            max_value=5.0, 
            value=st.session_state.config['min_profit_threshold'], 
            step=0.1
        )
        
        if st.button("Update Profit Threshold", use_container_width=True):
            st.session_state.config['min_profit_threshold'] = profit_threshold
            st.session_state.arbitrage_service.update_profit_threshold(profit_threshold)
            st.success(f"Updated profit threshold to {profit_threshold}%")
        
        # Information section
        st.markdown("---")
        st.markdown("""
        **How it works:**
        1. Select trading pairs to monitor
        2. Choose exchanges to compare
        3. Set minimum profit threshold
        4. Start monitoring for opportunities
        """)
        
        # Credits
        st.markdown("---")
        st.caption("© 2025 Crypto Arbitrage Tracker")