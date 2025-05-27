import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    def __init__(self):
        # Exchanges that provide public price data without API keys
        self.exchanges = {
            'coingecko': {
                'base_url': 'https://api.coingecko.com/api/v3/simple/price',
                'name': 'CoinGecko',
                'requires_api': False
            },
            'coinpaprika': {
                'base_url': 'https://api.coinpaprika.com/v1/tickers',
                'name': 'CoinPaprika', 
                'requires_api': False
            },
            'cryptocompare': {
                'base_url': 'https://min-api.cryptocompare.com/data/price',
                'name': 'CryptoCompare',
                'requires_api': False
            },
            'coinlore': {
                'base_url': 'https://api.coinlore.net/api/ticker',
                'name': 'CoinLore',
                'requires_api': False
            },
            'nomics': {
                'base_url': 'https://api.nomics.com/v1/currencies/ticker',
                'name': 'Nomics',
                'requires_api': False
            }
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Supported cryptocurrencies with their identifiers for different APIs
        self.crypto_mappings = {
            'BTC': {
                'coingecko': 'bitcoin',
                'coinpaprika': 'btc-bitcoin',
                'cryptocompare': 'BTC',
                'coinlore': '90',
                'nomics': 'BTC'
            },
            'ETH': {
                'coingecko': 'ethereum',
                'coinpaprika': 'eth-ethereum',
                'cryptocompare': 'ETH',
                'coinlore': '80',
                'nomics': 'ETH'
            },
            'ADA': {
                'coingecko': 'cardano',
                'coinpaprika': 'ada-cardano',
                'cryptocompare': 'ADA',
                'coinlore': '257',
                'nomics': 'ADA'
            },
            'DOT': {
                'coingecko': 'polkadot',
                'coinpaprika': 'dot-polkadot',
                'cryptocompare': 'DOT',
                'coinlore': '33285',
                'nomics': 'DOT'
            },
            'LINK': {
                'coingecko': 'chainlink',
                'coinpaprika': 'link-chainlink',
                'cryptocompare': 'LINK',
                'coinlore': '1975',
                'nomics': 'LINK'
            },
            'MATIC': {
                'coingecko': 'matic-network',
                'coinpaprika': 'matic-polygon',
                'cryptocompare': 'MATIC',
                'coinlore': '8347',
                'nomics': 'MATIC'
            },
            'SOL': {
                'coingecko': 'solana',
                'coinpaprika': 'sol-solana',
                'cryptocompare': 'SOL',
                'coinlore': '48543',
                'nomics': 'SOL'
            },
            'AVAX': {
                'coingecko': 'avalanche-2',
                'coinpaprika': 'avax-avalanche',
                'cryptocompare': 'AVAX',
                'coinlore': '50139',
                'nomics': 'AVAX'
            },
            'ATOM': {
                'coingecko': 'cosmos',
                'coinpaprika': 'atom-cosmos',
                'cryptocompare': 'ATOM',
                'coinlore': '3794',
                'nomics': 'ATOM'
            },
            'XRP': {
                'coingecko': 'ripple',
                'coinpaprika': 'xrp-xrp',
                'cryptocompare': 'XRP',
                'coinlore': '58',
                'nomics': 'XRP'
            },
            'ALGO': {
                'coingecko': 'algorand',
                'coinpaprika': 'algo-algorand',
                'cryptocompare': 'ALGO',
                'coinlore': '4030',
                'nomics': 'ALGO'
            },
            'FTM': {
                'coingecko': 'fantom',
                'coinpaprika': 'ftm-fantom',
                'cryptocompare': 'FTM',
                'coinlore': '3513',
                'nomics': 'FTM'
            },
            'NEAR': {
                'coingecko': 'near',
                'coinpaprika': 'near-near-protocol',
                'cryptocompare': 'NEAR',
                'coinlore': '6535',
                'nomics': 'NEAR'
            },
            'VET': {
                'coingecko': 'vechain',
                'coinpaprika': 'vet-vechain',
                'cryptocompare': 'VET',
                'coinlore': '1027',
                'nomics': 'VET'
            },
            'THETA': {
                'coingecko': 'theta-token',
                'coinpaprika': 'theta-theta-network',
                'cryptocompare': 'THETA',
                'coinlore': '2416',
                'nomics': 'THETA'
            },
            'ICP': {
                'coingecko': 'internet-computer',
                'coinpaprika': 'icp-internet-computer',
                'cryptocompare': 'ICP',
                'coinlore': '8916',
                'nomics': 'ICP'
            },
            'MANA': {
                'coingecko': 'decentraland',
                'coinpaprika': 'mana-decentraland',
                'cryptocompare': 'MANA',
                'coinlore': '1966',
                'nomics': 'MANA'
            },
            'SAND': {
                'coingecko': 'the-sandbox',
                'coinpaprika': 'sand-the-sandbox',
                'cryptocompare': 'SAND',
                'coinlore': '6758',
                'nomics': 'SAND'
            },
            'AXS': {
                'coingecko': 'axie-infinity',
                'coinpaprika': 'axs-axie-infinity',
                'cryptocompare': 'AXS',
                'coinlore': '6783',
                'nomics': 'AXS'
            },
            'LUNA': {
                'coingecko': 'terra-luna-2',
                'coinpaprika': 'luna-terra',
                'cryptocompare': 'LUNA',
                'coinlore': '4172',
                'nomics': 'LUNA'
            }
        }

    def _fetch_price_coingecko(self, crypto: str) -> Optional[float]:
        """Fetch price from CoinGecko API."""
        try:
            crypto_id = self.crypto_mappings.get(crypto, {}).get('coingecko')
            if not crypto_id:
                return None
            
            url = f"{self.exchanges['coingecko']['base_url']}?ids={crypto_id}&vs_currencies=usd"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if crypto_id in data and 'usd' in data[crypto_id]:
                price = float(data[crypto_id]['usd'])
                logger.info(f"CoinGecko {crypto}: ${price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"CoinGecko API failed for {crypto}: {str(e)}")
            return None

    def _fetch_price_coinpaprika(self, crypto: str) -> Optional[float]:
        """Fetch price from CoinPaprika API."""
        try:
            crypto_id = self.crypto_mappings.get(crypto, {}).get('coinpaprika')
            if not crypto_id:
                return None
            
            url = f"{self.exchanges['coinpaprika']['base_url']}/{crypto_id}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'quotes' in data and 'USD' in data['quotes']:
                price = float(data['quotes']['USD']['price'])
                logger.info(f"CoinPaprika {crypto}: ${price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"CoinPaprika API failed for {crypto}: {str(e)}")
            return None

    def _fetch_price_cryptocompare(self, crypto: str) -> Optional[float]:
        """Fetch price from CryptoCompare API."""
        try:
            crypto_symbol = self.crypto_mappings.get(crypto, {}).get('cryptocompare')
            if not crypto_symbol:
                return None
            
            url = f"{self.exchanges['cryptocompare']['base_url']}?fsym={crypto_symbol}&tsyms=USD"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'USD' in data:
                price = float(data['USD'])
                logger.info(f"CryptoCompare {crypto}: ${price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"CryptoCompare API failed for {crypto}: {str(e)}")
            return None

    def _fetch_price_coinlore(self, crypto: str) -> Optional[float]:
        """Fetch price from CoinLore API."""
        try:
            crypto_id = self.crypto_mappings.get(crypto, {}).get('coinlore')
            if not crypto_id:
                return None
            
            url = f"{self.exchanges['coinlore']['base_url']}/?id={crypto_id}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0 and 'price_usd' in data[0]:
                price = float(data[0]['price_usd'])
                logger.info(f"CoinLore {crypto}: ${price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"CoinLore API failed for {crypto}: {str(e)}")
            return None

    def _fetch_price_nomics(self, crypto: str) -> Optional[float]:
        """Fetch price from Nomics API."""
        try:
            crypto_symbol = self.crypto_mappings.get(crypto, {}).get('nomics')
            if not crypto_symbol:
                return None
            
            url = f"{self.exchanges['nomics']['base_url']}?ids={crypto_symbol}&convert=USD"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0 and 'price' in data[0]:
                price = float(data[0]['price'])
                logger.info(f"Nomics {crypto}: ${price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"Nomics API failed for {crypto}: {str(e)}")
            return None

    def fetch_price(self, exchange: str, crypto: str) -> Optional[float]:
        """Fetch price for a specific crypto from an exchange."""
        # Map exchange names to fetch functions
        fetch_functions = {
            'coingecko': self._fetch_price_coingecko,
            'coinpaprika': self._fetch_price_coinpaprika,
            'cryptocompare': self._fetch_price_cryptocompare,
            'coinlore': self._fetch_price_coinlore,
            'nomics': self._fetch_price_nomics
        }
        
        fetch_func = fetch_functions.get(exchange)
        if fetch_func:
            return fetch_func(crypto)
        
        logger.warning(f"Unknown exchange: {exchange}")
        return None

    def fetch_all_prices(self, crypto_list: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch prices for all cryptos from all exchanges."""
        all_prices = {}
        
        for crypto in crypto_list:
            all_prices[crypto] = {}
            
            for exchange in self.exchanges.keys():
                try:
                    price = self.fetch_price(exchange, crypto)
                    if price:
                        all_prices[crypto][exchange] = price
                        logger.info(f"Fetched {crypto} from {exchange}: ${price}")
                    else:
                        logger.warning(f"Failed to fetch {crypto} from {exchange}")
                        
                except Exception as e:
                    logger.error(f"Error fetching {crypto} from {exchange}: {str(e)}")
                
                # Add delay between requests to avoid rate limiting
                time.sleep(random.uniform(0.5, 1.5))
        
        return all_prices

    def get_supported_cryptocurrencies(self) -> List[str]:
        """Get list of supported cryptocurrencies."""
        return list(self.crypto_mappings.keys())

    def get_exchange_status(self) -> Dict[str, bool]:
        """Check the status of all exchanges."""
        status = {}
        
        for exchange in self.exchanges.keys():
            try:
                # Try to fetch BTC price as a health check
                price = self.fetch_price(exchange, 'BTC')
                status[exchange] = price is not None
            except Exception:
                status[exchange] = False
        
        return status

    def test_connection(self) -> Dict[str, str]:
        """Test connection to all exchanges and return status messages."""
        results = {}
        
        for exchange in self.exchanges.keys():
            try:
                price = self.fetch_price(exchange, 'BTC')
                if price:
                    results[exchange] = f"✅ Connected - BTC: ${price}"
                else:
                    results[exchange] = "❌ Failed to fetch price"
            except Exception as e:
                results[exchange] = f"❌ Connection error: {str(e)}"
        
        return results
