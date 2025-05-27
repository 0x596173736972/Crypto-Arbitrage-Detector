import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "crypto_arbitrage.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prices table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        crypto TEXT NOT NULL,
                        exchange TEXT NOT NULL,
                        price REAL NOT NULL,
                        volume REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Arbitrage opportunities table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS opportunities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        crypto TEXT NOT NULL,
                        buy_exchange TEXT NOT NULL,
                        sell_exchange TEXT NOT NULL,
                        buy_price REAL NOT NULL,
                        sell_price REAL NOT NULL,
                        spread_pct REAL NOT NULL,
                        gross_profit REAL NOT NULL,
                        net_profit REAL NOT NULL,
                        opportunity_score REAL NOT NULL,
                        risk_score INTEGER NOT NULL,
                        trade_amount REAL NOT NULL,
                        executed BOOLEAN DEFAULT FALSE,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Trades table for backtesting and actual trades
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        crypto TEXT NOT NULL,
                        trade_type TEXT NOT NULL, -- 'backtest' or 'live'
                        buy_exchange TEXT NOT NULL,
                        sell_exchange TEXT NOT NULL,
                        buy_price REAL NOT NULL,
                        sell_price REAL NOT NULL,
                        amount REAL NOT NULL,
                        gross_profit REAL NOT NULL,
                        net_profit REAL NOT NULL,
                        fees_paid REAL NOT NULL,
                        slippage REAL DEFAULT 0,
                        execution_time REAL DEFAULT 0,
                        success BOOLEAN DEFAULT TRUE,
                        notes TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Market analysis table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        crypto TEXT NOT NULL,
                        analysis_type TEXT NOT NULL, -- 'cointegration', 'zscore', etc.
                        result_data TEXT NOT NULL, -- JSON string
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_prices_crypto_exchange ON prices(crypto, exchange)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_prices_timestamp ON prices(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_opportunities_crypto ON opportunities(crypto)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_opportunities_timestamp ON opportunities(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def store_prices(self, price_data: Dict[str, Dict[str, float]], timestamp: datetime = None):
        """Store price data in the database."""
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for crypto, exchanges in price_data.items():
                    for exchange, price in exchanges.items():
                        cursor.execute('''
                            INSERT INTO prices (timestamp, crypto, exchange, price)
                            VALUES (?, ?, ?, ?)
                        ''', (timestamp, crypto, exchange, price))
                
                conn.commit()
                logger.info(f"Stored price data for {len(price_data)} cryptos")
                
        except Exception as e:
            logger.error(f"Error storing prices: {str(e)}")

    def store_opportunity(self, opportunity: Dict):
        """Store an arbitrage opportunity in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO opportunities (
                        timestamp, crypto, buy_exchange, sell_exchange,
                        buy_price, sell_price, spread_pct, gross_profit,
                        net_profit, opportunity_score, risk_score, trade_amount
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    opportunity['timestamp'],
                    opportunity['crypto'],
                    opportunity['buy_exchange'],
                    opportunity['sell_exchange'],
                    opportunity['buy_price'],
                    opportunity['sell_price'],
                    opportunity['spread_pct'],
                    opportunity['gross_profit'],
                    opportunity['net_profit'],
                    opportunity['opportunity_score'],
                    opportunity['risk_score'],
                    opportunity['trade_amount']
                ))
                
                conn.commit()
                logger.info(f"Stored opportunity: {opportunity['crypto']} {opportunity['spread_pct']:.2f}%")
                
        except Exception as e:
            logger.error(f"Error storing opportunity: {str(e)}")

    def store_trade(self, trade: Dict):
        """Store a trade record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trades (
                        timestamp, crypto, trade_type, buy_exchange, sell_exchange,
                        buy_price, sell_price, amount, gross_profit, net_profit,
                        fees_paid, slippage, execution_time, success, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.get('timestamp', datetime.now()),
                    trade['crypto'],
                    trade.get('trade_type', 'backtest'),
                    trade['buy_exchange'],
                    trade['sell_exchange'],
                    trade['buy_price'],
                    trade['sell_price'],
                    trade['amount'],
                    trade['gross_profit'],
                    trade['net_profit'],
                    trade.get('fees_paid', 0),
                    trade.get('slippage', 0),
                    trade.get('execution_time', 0),
                    trade.get('success', True),
                    trade.get('notes', '')
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing trade: {str(e)}")

    def get_historical_prices(self, crypto: str = None, exchange: str = None,
                            start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get historical price data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM prices WHERE 1=1"
                params = []
                
                if crypto:
                    query += " AND crypto = ?"
                    params.append(crypto)
                
                if exchange:
                    query += " AND exchange = ?"
                    params.append(exchange)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                
                query += " ORDER BY timestamp DESC"
                
                df = pd.read_sql_query(query, conn, params=params)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting historical prices: {str(e)}")
            return pd.DataFrame()

    def get_historical_opportunities(self, crypto: str = None,
                                   start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Get historical arbitrage opportunities."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM opportunities WHERE 1=1"
                params = []
                
                if crypto:
                    query += " AND crypto = ?"
                    params.append(crypto)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                
                query += " ORDER BY timestamp DESC"
                
                df = pd.read_sql_query(query, conn, params=params)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting historical opportunities: {str(e)}")
            return pd.DataFrame()

    def get_historical_spreads(self, crypto: str = None) -> pd.DataFrame:
        """Get historical spread data for analysis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, crypto, spread_pct, opportunity_score, risk_score
                    FROM opportunities 
                    WHERE 1=1
                '''
                params = []
                
                if crypto:
                    query += " AND crypto = ?"
                    params.append(crypto)
                
                query += " ORDER BY timestamp"
                
                df = pd.read_sql_query(query, conn, params=params)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting historical spreads: {str(e)}")
            return pd.DataFrame()

    def get_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get all historical data for a date range."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, crypto, exchange, price
                    FROM prices 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(query, conn, params=[start_date, end_date])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()

    def get_trades(self, trade_type: str = None, start_date: datetime = None,
                   end_date: datetime = None) -> pd.DataFrame:
        """Get trade history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if trade_type:
                    query += " AND trade_type = ?"
                    params.append(trade_type)
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)
                
                query += " ORDER BY timestamp DESC"
                
                df = pd.read_sql_query(query, conn, params=params)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            return pd.DataFrame()

    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total records
                cursor.execute("SELECT COUNT(*) FROM prices")
                total_records = cursor.fetchone()[0]
                
                # Get unique cryptos
                cursor.execute("SELECT COUNT(DISTINCT crypto) FROM prices")
                unique_cryptos = cursor.fetchone()[0]
                
                # Get date range
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM prices")
                date_range = cursor.fetchone()
                
                if date_range[0] and date_range[1]:
                    start_date = datetime.fromisoformat(date_range[0])
                    end_date = datetime.fromisoformat(date_range[1])
                    days = (end_date - start_date).days
                else:
                    days = 0
                
                # Get opportunity count
                cursor.execute("SELECT COUNT(*) FROM opportunities")
                opportunity_count = cursor.fetchone()[0]
                
                return {
                    'total_records': total_records,
                    'unique_cryptos': unique_cryptos,
                    'date_range': days,
                    'opportunity_count': opportunity_count
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}

    def cleanup_old_data(self, days: int = 30) -> int:
        """Clean up data older than specified days."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old prices
                cursor.execute("DELETE FROM prices WHERE timestamp < ?", (cutoff_date,))
                deleted_prices = cursor.rowcount
                
                # Delete old opportunities
                cursor.execute("DELETE FROM opportunities WHERE timestamp < ?", (cutoff_date,))
                deleted_opportunities = cursor.rowcount
                
                conn.commit()
                
                total_deleted = deleted_prices + deleted_opportunities
                logger.info(f"Deleted {total_deleted} old records")
                
                return total_deleted
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            return 0

    def clear_all_data(self):
        """Clear all data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM prices")
                cursor.execute("DELETE FROM opportunities")
                cursor.execute("DELETE FROM trades")
                cursor.execute("DELETE FROM market_analysis")
                
                conn.commit()
                logger.info("All data cleared from database")
                
        except Exception as e:
            logger.error(f"Error clearing all data: {str(e)}")

    def store_analysis_result(self, crypto: str, analysis_type: str, result_data: Dict):
        """Store market analysis results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO market_analysis (crypto, analysis_type, result_data)
                    VALUES (?, ?, ?)
                ''', (crypto, analysis_type, json.dumps(result_data)))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")

    def get_latest_prices(self, crypto: str = None) -> pd.DataFrame:
        """Get the latest prices for each exchange."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if crypto:
                    query = '''
                        SELECT crypto, exchange, price, timestamp
                        FROM prices p1
                        WHERE crypto = ? AND timestamp = (
                            SELECT MAX(timestamp) 
                            FROM prices p2 
                            WHERE p2.crypto = p1.crypto AND p2.exchange = p1.exchange
                        )
                        ORDER BY exchange
                    '''
                    params = [crypto]
                else:
                    query = '''
                        SELECT crypto, exchange, price, timestamp
                        FROM prices p1
                        WHERE timestamp = (
                            SELECT MAX(timestamp) 
                            FROM prices p2 
                            WHERE p2.crypto = p1.crypto AND p2.exchange = p1.exchange
                        )
                        ORDER BY crypto, exchange
                    '''
                    params = []
                
                df = pd.read_sql_query(query, conn, params=params)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting latest prices: {str(e)}")
            return pd.DataFrame()
