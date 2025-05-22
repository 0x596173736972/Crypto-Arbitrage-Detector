
"""
Crypto Arbitrage Detector
Scanner les différences de prix entre exchanges (Binance vs Kraken)
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import websockets
import aiohttp
import ccxt.async_support as ccxt
from dataclasses import dataclass, asdict
from collections import defaultdict
import signal
import sys

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arbitrage.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Structure pour stocker les données de prix"""
    exchange: str
    symbol: str
    bid: float
    ask: float
    timestamp: float
    volume: float = 0.0

@dataclass
class ArbitrageOpportunity:
    """Structure pour une opportunité d'arbitrage"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    profit_percentage: float
    profit_after_fees: float
    volume: float
    timestamp: float

class ExchangeConnector:
    """Connecteur pour un exchange spécifique"""
    
    def __init__(self, exchange_name: str, config: Dict):
        self.name = exchange_name
        self.config = config
        self.prices = {}
        self.last_update = {}
        self.exchange = None
        self.websocket = None
        self.running = False
        
    async def initialize_ccxt(self):
        """Initialise la connexion CCXT"""
        try:
            if self.name.lower() == 'binance':
                self.exchange = ccxt.binance({
                    'apiKey': self.config.get('api_key', ''),
                    'secret': self.config.get('secret', ''),
                    'sandbox': self.config.get('sandbox', True),
                    'enableRateLimit': True,
                })
            elif self.name.lower() == 'kraken':
                self.exchange = ccxt.kraken({
                    'apiKey': self.config.get('api_key', ''),
                    'secret': self.config.get('secret', ''),
                    'sandbox': self.config.get('sandbox', True),
                    'enableRateLimit': True,
                })
            
            await self.exchange.load_markets()
            logger.info(f"Exchange {self.name} initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation {self.name}: {e}")
            
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Récupère les frais de trading pour un symbole"""
        try:
            if self.exchange:
                fees = await self.exchange.fetch_trading_fees()
                return fees.get(symbol, {'maker': 0.001, 'taker': 0.001})
            else:
                # Frais par défaut si pas d'API
                default_fees = {
                    'binance': {'maker': 0.001, 'taker': 0.001},
                    'kraken': {'maker': 0.0016, 'taker': 0.0026}
                }
                return default_fees.get(self.name.lower(), {'maker': 0.001, 'taker': 0.001})
        except:
            return {'maker': 0.001, 'taker': 0.001}
    
    async def connect_websocket_binance(self, symbols: List[str]):
        """Connexion WebSocket pour Binance"""
        streams = [f"{symbol.lower().replace('/', '')}@ticker" for symbol in symbols]
        url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
        
        try:
            async with websockets.connect(url) as websocket:
                self.websocket = websocket
                logger.info(f"WebSocket Binance connecté pour {len(symbols)} symboles")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if 'stream' in data:
                            await self._process_binance_ticker(data['data'])
                        else:
                            await self._process_binance_ticker(data)
                    except Exception as e:
                        logger.error(f"Erreur traitement message Binance: {e}")
                        
        except Exception as e:
            logger.error(f"Erreur WebSocket Binance: {e}")
    
    async def connect_websocket_kraken(self, symbols: List[str]):
        """Connexion WebSocket pour Kraken"""
        url = "wss://ws.kraken.com"
        
        try:
            async with websockets.connect(url) as websocket:
                self.websocket = websocket
                logger.info(f"WebSocket Kraken connecté")
                
                # Subscribe to ticker
                subscribe_msg = {
                    "event": "subscribe",
                    "pair": symbols,
                    "subscription": {"name": "ticker"}
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if isinstance(data, list) and len(data) >= 4:
                            await self._process_kraken_ticker(data)
                    except Exception as e:
                        logger.error(f"Erreur traitement message Kraken: {e}")
                        
        except Exception as e:
            logger.error(f"Erreur WebSocket Kraken: {e}")
    
    async def _process_binance_ticker(self, data: Dict):
        """Traite les données ticker de Binance"""
        try:
            symbol = data['s']
            # Convertir le format Binance vers format standard
            if len(symbol) == 6:  # BTCUSDT -> BTC/USDT
                base = symbol[:3]
                quote = symbol[3:]
                standard_symbol = f"{base}/{quote}"
            else:
                standard_symbol = symbol
            
            price_data = PriceData(
                exchange=self.name,
                symbol=standard_symbol,
                bid=float(data['b']),
                ask=float(data['a']),
                timestamp=float(data['E']) / 1000,
                volume=float(data['v'])
            )
            
            self.prices[standard_symbol] = price_data
            self.last_update[standard_symbol] = time.time()
            
        except Exception as e:
            logger.error(f"Erreur processing Binance ticker: {e}")
    
    async def _process_kraken_ticker(self, data: List):
        """Traite les données ticker de Kraken"""
        try:
            if len(data) >= 4 and data[2] == "ticker":
                ticker_data = data[1]
                symbol = data[3]
                
                # Convertir format Kraken vers standard
                standard_symbol = symbol.replace('XBT', 'BTC')
                if '/' not in standard_symbol:
                    # Essayer de deviner la séparation
                    if 'USD' in standard_symbol:
                        standard_symbol = standard_symbol.replace('USD', '/USD')
                    elif 'EUR' in standard_symbol:
                        standard_symbol = standard_symbol.replace('EUR', '/EUR')
                
                price_data = PriceData(
                    exchange=self.name,
                    symbol=standard_symbol,
                    bid=float(ticker_data['b'][0]),
                    ask=float(ticker_data['a'][0]),
                    timestamp=time.time(),
                    volume=float(ticker_data['v'][1])
                )
                
                self.prices[standard_symbol] = price_data
                self.last_update[standard_symbol] = time.time()
                
        except Exception as e:
            logger.error(f"Erreur processing Kraken ticker: {e}")
    
    async def start_websocket(self, symbols: List[str]):
        """Démarre la connexion WebSocket"""
        self.running = True
        
        if self.name.lower() == 'binance':
            await self.connect_websocket_binance(symbols)
        elif self.name.lower() == 'kraken':
            await self.connect_websocket_kraken(symbols)
    
    async def stop(self):
        """Arrête les connexions"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
        if self.exchange:
            await self.exchange.close()

class ArbitrageDetector:
    """Détecteur principal d'opportunités d'arbitrage"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        self.exchanges = {}
        self.opportunities = []
        self.running = False
        
        # Symboles à surveiller
        self.symbols = self.config.get('symbols', [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT'
        ])
        
        # Seuils
        self.min_profit_percentage = self.config.get('min_profit_percentage', 0.5)
        self.max_data_age = self.config.get('max_data_age_seconds', 30)
        
    def _load_config(self, config_file: str) -> Dict:
        """Charge la configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Fichier config {config_file} non trouvé, utilisation config par défaut")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "exchanges": {
                "binance": {
                    "api_key": "",
                    "secret": "",
                    "sandbox": True
                },
                "kraken": {
                    "api_key": "",
                    "secret": "",
                    "sandbox": True
                }
            },
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT"],
            "min_profit_percentage": 0.5,
            "max_data_age_seconds": 30,
            "notification": {
                "enabled": False,
                "webhook_url": ""
            }
        }
    
    async def initialize_exchanges(self):
        """Initialise tous les exchanges"""
        exchange_configs = self.config.get('exchanges', {})
        
        for exchange_name, exchange_config in exchange_configs.items():
            connector = ExchangeConnector(exchange_name, exchange_config)
            await connector.initialize_ccxt()
            self.exchanges[exchange_name] = connector
            
        logger.info(f"Initialisé {len(self.exchanges)} exchanges")
    
    async def start_monitoring(self):
        """Démarre le monitoring"""
        self.running = True
        logger.info("Démarrage du monitoring d'arbitrage...")
        
        # Démarrer les WebSockets de tous les exchanges
        tasks = []
        for exchange_name, connector in self.exchanges.items():
            task = asyncio.create_task(
                connector.start_websocket(self.symbols)
            )
            tasks.append(task)
        
        # Task de détection d'arbitrage
        detection_task = asyncio.create_task(self._arbitrage_detection_loop())
        tasks.append(detection_task)
        
        # Task de nettoyage périodique
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        tasks.append(cleanup_task)
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Erreur dans le monitoring: {e}")
    
    async def _arbitrage_detection_loop(self):
        """Boucle principale de détection d'arbitrage"""
        while self.running:
            try:
                await self._detect_opportunities()
                await asyncio.sleep(1)  # Vérification chaque seconde
            except Exception as e:
                logger.error(f"Erreur détection arbitrage: {e}")
                await asyncio.sleep(5)
    
    async def _detect_opportunities(self):
        """Détecte les opportunités d'arbitrage"""
        current_time = time.time()
        
        for symbol in self.symbols:
            exchange_prices = {}
            
            # Collecter les prix de tous les exchanges pour ce symbole
            for exchange_name, connector in self.exchanges.items():
                if (symbol in connector.prices and 
                    symbol in connector.last_update and
                    current_time - connector.last_update[symbol] < self.max_data_age):
                    
                    exchange_prices[exchange_name] = connector.prices[symbol]
            
            # Analyser les opportunités pour ce symbole
            if len(exchange_prices) >= 2:
                await self._analyze_symbol_arbitrage(symbol, exchange_prices)
    
    async def _analyze_symbol_arbitrage(self, symbol: str, prices: Dict[str, PriceData]):
        """Analyse l'arbitrage pour un symbole donné"""
        exchanges = list(prices.keys())
        
        for i in range(len(exchanges)):
            for j in range(len(exchanges)):
                if i != j:
                    buy_exchange = exchanges[i]
                    sell_exchange = exchanges[j]
                    
                    buy_price = prices[buy_exchange].ask  # Prix d'achat (ask)
                    sell_price = prices[sell_exchange].bid  # Prix de vente (bid)
                    
                    if sell_price > buy_price:
                        # Calculer le profit brut
                        profit_percentage = ((sell_price - buy_price) / buy_price) * 100
                        
                        if profit_percentage >= self.min_profit_percentage:
                            # Calculer le profit net après frais
                            profit_after_fees = await self._calculate_profit_after_fees(
                                symbol, buy_exchange, sell_exchange, buy_price, sell_price
                            )
                            
                            if profit_after_fees > 0:
                                opportunity = ArbitrageOpportunity(
                                    symbol=symbol,
                                    buy_exchange=buy_exchange,
                                    sell_exchange=sell_exchange,
                                    buy_price=buy_price,
                                    sell_price=sell_price,
                                    profit_percentage=profit_percentage,
                                    profit_after_fees=profit_after_fees,
                                    volume=min(prices[buy_exchange].volume, prices[sell_exchange].volume),
                                    timestamp=time.time()
                                )
                                
                                await self._handle_opportunity(opportunity)
    
    async def _calculate_profit_after_fees(self, symbol: str, buy_exchange: str, 
                                         sell_exchange: str, buy_price: float, sell_price: float) -> float:
        """Calcule le profit après déduction des frais"""
        try:
            # Récupérer les frais des deux exchanges
            buy_fees = await self.exchanges[buy_exchange].get_trading_fees(symbol)
            sell_fees = await self.exchanges[sell_exchange].get_trading_fees(symbol)
            
            # Calculer les coûts
            buy_cost = buy_price * (1 + buy_fees['taker'])  # Frais d'achat
            sell_revenue = sell_price * (1 - sell_fees['taker'])  # Revenus après frais de vente
            
            # Profit net en pourcentage
            profit_after_fees = ((sell_revenue - buy_cost) / buy_cost) * 100
            
            return profit_after_fees
            
        except Exception as e:
            logger.error(f"Erreur calcul frais pour {symbol}: {e}")
            return 0.0
    
    async def _handle_opportunity(self, opportunity: ArbitrageOpportunity):
        """Gère une opportunité détectée"""
        # Éviter les doublons récents
        recent_opportunities = [
            opp for opp in self.opportunities 
            if (time.time() - opp.timestamp < 60 and 
                opp.symbol == opportunity.symbol and
                opp.buy_exchange == opportunity.buy_exchange and
                opp.sell_exchange == opportunity.sell_exchange)
        ]
        
        if not recent_opportunities:
            self.opportunities.append(opportunity)
            
            # Log de l'opportunité
            logger.info(
                f"🚀 ARBITRAGE DÉTECTÉ: {opportunity.symbol} | "
                f"Acheter sur {opportunity.buy_exchange} à {opportunity.buy_price:.8f} | "
                f"Vendre sur {opportunity.sell_exchange} à {opportunity.sell_price:.8f} | "
                f"Profit: {opportunity.profit_percentage:.2f}% "
                f"(Net: {opportunity.profit_after_fees:.2f}%)"
            )
            
            # Notification
            await self._send_notification(opportunity)
    
    async def _send_notification(self, opportunity: ArbitrageOpportunity):
        """Envoie une notification"""
        notification_config = self.config.get('notification', {})
        
        if notification_config.get('enabled', False):
            webhook_url = notification_config.get('webhook_url')
            if webhook_url:
                try:
                    message = {
                        "text": f"🚀 Arbitrage {opportunity.symbol}: "
                               f"Buy {opportunity.buy_exchange} @ {opportunity.buy_price:.8f}, "
                               f"Sell {opportunity.sell_exchange} @ {opportunity.sell_price:.8f}, "
                               f"Profit: {opportunity.profit_after_fees:.2f}%"
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        await session.post(webhook_url, json=message)
                        
                except Exception as e:
                    logger.error(f"Erreur notification: {e}")
    
    async def _cleanup_loop(self):
        """Nettoie les anciennes opportunités"""
        while self.running:
            try:
                current_time = time.time()
                # Garder seulement les opportunités des dernières 24h
                self.opportunities = [
                    opp for opp in self.opportunities 
                    if current_time - opp.timestamp < 86400
                ]
                await asyncio.sleep(3600)  # Nettoyage chaque heure
            except Exception as e:
                logger.error(f"Erreur nettoyage: {e}")
                await asyncio.sleep(3600)
    
    def get_recent_opportunities(self, hours: int = 1) -> List[ArbitrageOpportunity]:
        """Récupère les opportunités récentes"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            opp for opp in self.opportunities 
            if opp.timestamp > cutoff_time
        ]
    
    def get_statistics(self) -> Dict:
        """Récupère les statistiques"""
        recent_opps = self.get_recent_opportunities(24)
        
        if not recent_opps:
            return {"message": "Aucune opportunité détectée dans les dernières 24h"}
        
        # Statistiques par symbole
        symbol_stats = defaultdict(list)
        for opp in recent_opps:
            symbol_stats[opp.symbol].append(opp.profit_after_fees)
        
        stats = {
            "total_opportunities": len(recent_opps),
            "symbols_tracked": len(self.symbols),
            "exchanges_connected": len(self.exchanges),
            "average_profit": sum(opp.profit_after_fees for opp in recent_opps) / len(recent_opps),
            "max_profit": max(opp.profit_after_fees for opp in recent_opps),
            "by_symbol": {
                symbol: {
                    "count": len(profits),
                    "avg_profit": sum(profits) / len(profits),
                    "max_profit": max(profits)
                }
                for symbol, profits in symbol_stats.items()
            }
        }
        
        return stats
    
    async def stop(self):
        """Arrête le détecteur"""
        logger.info("Arrêt du détecteur d'arbitrage...")
        self.running = False
        
        for connector in self.exchanges.values():
            await connector.stop()

async def main():
    """Fonction principale"""
    detector = ArbitrageDetector()
    
    # Gestionnaire de signal pour arrêt propre
    def signal_handler(sig, frame):
        logger.info("Signal d'arrêt reçu")
        asyncio.create_task(detector.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialiser les exchanges
        await detector.initialize_exchanges()
        
        # Démarrer le monitoring
        await detector.start_monitoring()
        
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
    finally:
        await detector.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Programme interrompu")
