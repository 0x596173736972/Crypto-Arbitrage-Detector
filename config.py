# Default configuration settings

DEFAULT_CONFIG = {
    # Trading pairs to monitor
    'trading_pairs': [
        'BTC/USDT',
        'ETH/USDT',
        'SOL/USDT',
        'ADA/USDT',
        'XRP/USDT',
    ],
    
    # Exchange specific settings
    'exchanges': {
        'binance': {
            'enabled': True,
            'api_key': '',  # Will be populated from environment variables
            'api_secret': '',  # Will be populated from environment variables
        },
        'kraken': {
            'enabled': True,
            'api_key': '',  # Will be populated from environment variables
            'api_secret': '',  # Will be populated from environment variables
        }
    },
    
    # Fee configuration
    'exchange_fees': {
        'binance': 0.1,  # 0.1% trading fee
        'kraken': 0.16,  # 0.16% trading fee
    },
    
    # Arbitrage parameters
    'min_profit_threshold': 0.5,  # Minimum profit percentage
    'slippage_estimate': 0.2,  # Estimated slippage percentage
    'network_transfer_time': 20,  # Estimated time to transfer assets (minutes)
    'network_transfer_fee': 0.001,  # Fixed fee for network transfers (in BTC equivalent)
    
    # Risk settings
    'risk_tolerance': 5,  # Scale 1-10
    'min_volume': 1000,  # Minimum USD volume required
    'max_exposure': 10000,  # Maximum USD exposure per trade
    'order_book_depth': 10,  # Depth of order book to analyze
    
    # System settings
    'update_interval': 5,  # Data refresh interval in seconds
    'log_level': 'INFO',
    'history_max_items': 100,  # Maximum number of historical items to keep
}

# Exchange connection settings - will be loaded from .env file
EXCHANGE_SETTINGS = {
    'binance': {
        'websocket_url': 'wss://stream.binance.com:9443/ws',
        'rest_url': 'https://api.binance.com',
    },
    'kraken': {
        'websocket_url': 'wss://ws.kraken.com',
        'rest_url': 'https://api.kraken.com',
    }
}