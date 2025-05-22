import asyncio
import ccxt.async_support as ccxt
import websockets
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('arbitrage_service')

class ArbitrageService:
    def __init__(self):
        self.exchanges = {}
        self.websockets = {}
        self.is_running = False
        self.current_prices = {}
        self.opportunities = []
        self.connection_status = {
            'binance': {
                'connected': False,
                'last_update': datetime.now() - timedelta(minutes=10),
                'latency': 0
            },
            'kraken': {
                'connected': False,
                'last_update': datetime.now() - timedelta(minutes=10),
                'latency': 0
            },
            'system': {
                'running': False,
                'uptime': 0,
                'memory_usage': 0
            }
        }
        self.start_time = datetime.now()
        self.trading_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.min_profit_threshold = 0.5
        self.config = self._load_config()
        
        # Initialize exchange instances
        self._init_exchanges()
        
    def _load_config(self) -> dict:
        """Load configuration from environment variables"""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        return {
            'exchange_fees': {
                'binance': float(os.getenv('BINANCE_TRADING_FEE', '0.1')),
                'kraken': float(os.getenv('KRAKEN_TRADING_FEE', '0.16')),
            },
            'api_keys': {
                'binance': {
                    'api_key': os.getenv('BINANCE_API_KEY'),
                    'api_secret': os.getenv('BINANCE_API_SECRET')
                },
                'kraken': {
                    'api_key': os.getenv('KRAKEN_API_KEY'),
                    'api_secret': os.getenv('KRAKEN_API_SECRET')
                }
            },
            'slippage_estimate': 0.2,
            'network_transfer_time': 20,
            'network_transfer_fee': 0.001,
        }
    
    def _init_exchanges(self):
        """Initialize exchange connections"""
        # Initialize Binance
        self.exchanges['binance'] = ccxt.binance({
            'apiKey': self.config['api_keys']['binance']['api_key'],
            'secret': self.config['api_keys']['binance']['api_secret'],
            'enableRateLimit': True,
        })
        
        # Initialize Kraken
        self.exchanges['kraken'] = ccxt.kraken({
            'apiKey': self.config['api_keys']['kraken']['api_key'],
            'secret': self.config['api_keys']['kraken']['api_secret'],
            'enableRateLimit': True,
        })
    
    async def start(self):
        """Start the arbitrage detection service"""
        if self.is_running:
            logger.warning("Service is already running")
            return
        
        self.is_running = True
        self.connection_status['system']['running'] = True
        self.start_time = datetime.now()
        
        try:
            # Start WebSocket connections
            await self._start_websockets()
            
            # Start price monitoring
            await self._monitor_prices()
            
        except Exception as e:
            logger.error(f"Error starting service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the arbitrage detection service"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.connection_status['system']['running'] = False
        
        # Close WebSocket connections
        for exchange, ws in self.websockets.items():
            if ws:
                await ws.close()
                logger.info(f"Closed WebSocket connection for {exchange}")
        
        # Close exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        logger.info("Arbitrage detection service stopped")
    
    async def _start_websockets(self):
        """Initialize WebSocket connections to exchanges"""
        try:
            # Start Binance WebSocket
            binance_pairs = [p.lower().replace('/', '') for p in self.trading_pairs]
            binance_streams = [f"{pair}@ticker" for pair in binance_pairs]
            binance_ws_url = f"wss://stream.binance.com:9443/ws/{'/'.join(binance_streams)}"
            
            self.websockets['binance'] = await websockets.connect(binance_ws_url)
            self.connection_status['binance']['connected'] = True
            logger.info("Connected to Binance WebSocket")
            
            # Start Kraken WebSocket
            kraken_ws_url = "wss://ws.kraken.com"
            self.websockets['kraken'] = await websockets.connect(kraken_ws_url)
            
            # Subscribe to Kraken ticker
            subscribe_message = {
                "event": "subscribe",
                "pair": self.trading_pairs,
                "subscription": {"name": "ticker"}
            }
            await self.websockets['kraken'].send(json.dumps(subscribe_message))
            self.connection_status['kraken']['connected'] = True
            logger.info("Connected to Kraken WebSocket")
            
        except Exception as e:
            logger.error(f"Error establishing WebSocket connections: {e}")
            raise
    
    async def _monitor_prices(self):
        """Monitor price updates from WebSocket connections"""
        while self.is_running:
            try:
                # Create tasks for each exchange
                binance_task = asyncio.create_task(self._process_binance_messages())
                kraken_task = asyncio.create_task(self._process_kraken_messages())
                
                # Wait for both tasks
                await asyncio.gather(binance_task, kraken_task)
                
            except Exception as e:
                logger.error(f"Error in price monitoring: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_binance_messages(self):
        """Process incoming Binance WebSocket messages"""
        try:
            while self.is_running:
                message = await self.websockets['binance'].recv()
                data = json.loads(message)
                
                # Update price data
                symbol = data['s'].upper()
                normalized_symbol = f"{symbol[:3]}/{symbol[3:]}"
                
                if normalized_symbol in self.trading_pairs:
                    self.current_prices[f"binance_{normalized_symbol}"] = {
                        'price': float(data['c']),
                        'timestamp': datetime.now()
                    }
                    
                    # Update connection status
                    self.connection_status['binance']['last_update'] = datetime.now()
                    self.connection_status['binance']['latency'] = float(data['E']) - float(data['T'])
                    
                    # Check for arbitrage opportunities
                    await self._check_arbitrage(normalized_symbol)
                    
        except Exception as e:
            logger.error(f"Error processing Binance messages: {e}")
            self.connection_status['binance']['connected'] = False
            raise
    
    async def _process_kraken_messages(self):
        """Process incoming Kraken WebSocket messages"""
        try:
            while self.is_running:
                message = await self.websockets['kraken'].recv()
                data = json.loads(message)
                
                # Handle heartbeat messages
                if isinstance(data, list) and len(data) >= 2:
                    pair_name = data[3]
                    ticker_data = data[1]
                    
                    normalized_symbol = self._normalize_kraken_pair(pair_name)
                    if normalized_symbol in self.trading_pairs:
                        self.current_prices[f"kraken_{normalized_symbol}"] = {
                            'price': float(ticker_data['c'][0]),
                            'timestamp': datetime.now()
                        }
                        
                        # Update connection status
                        self.connection_status['kraken']['last_update'] = datetime.now()
                        self.connection_status['kraken']['latency'] = 0  # Kraken doesn't provide latency info
                        
                        # Check for arbitrage opportunities
                        await self._check_arbitrage(normalized_symbol)
                    
        except Exception as e:
            logger.error(f"Error processing Kraken messages: {e}")
            self.connection_status['kraken']['connected'] = False
            raise
    
    async def _check_arbitrage(self, symbol: str):
        """Check for arbitrage opportunities for a given symbol"""
        try:
            binance_data = self.current_prices.get(f"binance_{symbol}")
            kraken_data = self.current_prices.get(f"kraken_{symbol}")
            
            if not (binance_data and kraken_data):
                return
            
            binance_price = Decimal(str(binance_data['price']))
            kraken_price = Decimal(str(kraken_data['price']))
            
            # Calculate price difference percentage
            price_diff = ((binance_price - kraken_price) / kraken_price) * 100
            
            # Determine buy and sell exchanges
            if binance_price > kraken_price:
                buy_exchange = 'kraken'
                sell_exchange = 'binance'
                buy_price = kraken_price
                sell_price = binance_price
            else:
                buy_exchange = 'binance'
                sell_exchange = 'kraken'
                buy_price = binance_price
                sell_price = kraken_price
            
            # Calculate net profit after fees
            buy_fee = Decimal(str(self.config['exchange_fees'][buy_exchange]))
            sell_fee = Decimal(str(self.config['exchange_fees'][sell_exchange]))
            slippage = Decimal(str(self.config['slippage_estimate']))
            
            gross_profit = ((sell_price - buy_price) / buy_price) * 100
            net_profit = gross_profit - buy_fee - sell_fee - slippage
            
            # If profit exceeds threshold, create opportunity
            if net_profit > Decimal(str(self.min_profit_threshold)):
                opportunity = self._create_opportunity({
                    'symbol': symbol,
                    'buy_exchange': buy_exchange,
                    'sell_exchange': sell_exchange,
                    'buy_price': float(buy_price),
                    'sell_price': float(sell_price),
                    'net_profit': float(net_profit)
                })
                
                self.opportunities.append(opportunity)
                # Keep only recent opportunities
                if len(self.opportunities) > 100:
                    self.opportunities.pop(0)
                
                logger.info(f"New arbitrage opportunity detected: {opportunity}")
            
        except Exception as e:
            logger.error(f"Error checking arbitrage for {symbol}: {e}")
    
    def _create_opportunity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an arbitrage opportunity object"""
        # Calculate required volume based on symbol
        base_volume = self._calculate_base_volume(data['symbol'])
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(data)
        
        return {
            'symbol': data['symbol'],
            'entry_exchange': data['buy_exchange'].capitalize(),
            'exit_exchange': data['sell_exchange'].capitalize(),
            'entry_price': data['buy_price'],
            'exit_price': data['sell_price'],
            'potential_profit': data['net_profit'],
            'required_volume': base_volume,
            'risk_score': risk_score,
            'timestamp': datetime.now(),
            'strategy': f"Buy on {data['buy_exchange'].capitalize()}, Sell on {data['sell_exchange'].capitalize()}",
            'status': 'active'
        }
    
    def _calculate_base_volume(self, symbol: str) -> float:
        """Calculate the base volume for a trading pair"""
        base_volumes = {
            'BTC/USDT': 10000,
            'ETH/USDT': 5000,
            'SOL/USDT': 2000,
        }
        return base_volumes.get(symbol, 1000)
    
    def _calculate_risk_score(self, data: Dict[str, Any]) -> int:
        """Calculate risk score for an opportunity"""
        # Factors: profit margin, volume, exchange reliability
        profit_factor = min(data['net_profit'] / 2, 5)
        volume_factor = 2  # Default medium risk for volume
        exchange_factor = 1 if data['buy_exchange'] == 'binance' and data['sell_exchange'] == 'kraken' else 2
        
        risk_score = round(10 - (profit_factor + volume_factor + exchange_factor))
        return max(1, min(10, risk_score))
    
    def _normalize_kraken_pair(self, pair: str) -> str:
        """Normalize Kraken pair names to standard format"""
        # Example: 'XXBTZUSD' -> 'BTC/USDT'
        mappings = {
            'XXBTZUSD': 'BTC/USDT',
            'XETHZUSD': 'ETH/USDT',
            'SOLUSDT': 'SOL/USDT',
        }
        return mappings.get(pair, pair)
    
    def get_current_prices(self) -> List[Dict[str, Any]]:
        """Get current prices for all trading pairs"""
        result = []
        for symbol in self.trading_pairs:
            binance_data = self.current_prices.get(f"binance_{symbol}")
            kraken_data = self.current_prices.get(f"kraken_{symbol}")
            
            if binance_data and kraken_data:
                binance_price = binance_data['price']
                kraken_price = kraken_data['price']
                
                price_diff = ((binance_price - kraken_price) / kraken_price) * 100
                net_profit = abs(price_diff) - sum(self.config['exchange_fees'].values()) - self.config['slippage_estimate']
                
                result.append({
                    'symbol': symbol,
                    'binance_price': binance_price,
                    'kraken_price': kraken_price,
                    'difference_pct': abs(price_diff),
                    'potential_profit': net_profit,
                    'timestamp': datetime.now()
                })
        
        return result
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        if self.is_running:
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            self.connection_status['system']['uptime'] = int(uptime_seconds)
        
        return self.connection_status
    
    def update_trading_pairs(self, pairs: List[str]):
        """Update trading pairs and restart WebSocket connections"""
        self.trading_pairs = pairs
        if self.is_running:
            asyncio.create_task(self._restart_websockets())
    
    async def _restart_websockets(self):
        """Restart WebSocket connections with updated pairs"""
        # Close existing connections
        for ws in self.websockets.values():
            if ws:
                await ws.close()
        
        # Start new connections
        await self._start_websockets()
    
    def update_profit_threshold(self, threshold: float):
        """Update minimum profit threshold"""
        self.min_profit_threshold = threshold
        logger.info(f"Updated profit threshold to {threshold}%")